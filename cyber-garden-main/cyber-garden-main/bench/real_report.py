"""Combine fresh real reports, keeping offline results and initial failures separate."""
from datetime import datetime, timezone
import json
import re
from pathlib import Path
import subprocess

from jsonschema import Draft202012Validator

from buyer.contracts import ROOT
from tests.live_support import percentiles, write_report


def read(name):
    return json.loads((ROOT / "bench/results" / name).read_text(encoding="utf-8"))


def lines(name):
    return [json.loads(line) for line in (ROOT / "bench/results" / name).read_text(encoding="utf-8").splitlines()]


def main():
    smoke = read("live-gemma-smoke.json")
    initial_cli = read("live-gemma-cli-summary.json")
    cli = read("live-gemma-cli-recheck-summary.json")
    quality = read("live-gemma-quality.json")
    recheck = read("live-gemma-quality-recheck.json")
    web = read("live-gemma-web.json")
    latency = read("live-gemma-latency.json")
    quality_cases = lines("live-gemma-cases.jsonl")
    corrected = {row["name"]: row for row in lines("live-gemma-quality-recheck-cases.jsonl")}
    effective_quality = [corrected.get(row["name"], row) for row in quality_cases]
    attempts, responses = [], []
    for name in ("live-gemma-cli.jsonl", "live-gemma-cli-recheck.jsonl", "live-gemma-cases.jsonl",
                 "live-gemma-quality-recheck-cases.jsonl"):
        for row in lines(name):
            attempts.extend(event for event in row["gateway_events"] if event["kind"] == "http_attempt")
            responses.extend(event for event in row["gateway_events"] if event["kind"] == "model_response")
    for report in (web, latency):
        observed = report.get("gateway_events", []) or [event for row in report.get("cases", []) for event in row["gateway_events"]]
        attempts.extend(event for event in observed if event["kind"] == "http_attempt")
        responses.extend(event for event in observed if event["kind"] == "model_response")
    times = sorted(datetime.fromisoformat(event["utc"]).timestamp() for event in attempts)
    maximum = max((sum(start <= value < start + 60 for value in times) for start in times), default=0)
    envelope_schema = Draft202012Validator(json.loads((ROOT / "schemas/llm-response.schema.json").read_bytes()))
    response_schema = Draft202012Validator(json.loads((ROOT / "schemas/response.schema.json").read_bytes()))
    schema_errors = []
    for index, event in enumerate(responses):
        if event.get("success"):
            schema_errors.extend(f"model response {index}: {error.message}" for error in envelope_schema.iter_errors(event["response"]))
    for row in initial_cli["cases"] + cli["cases"]:
        if "response" in row:
            schema_errors.extend(f"CLI {row['name']}: {error.message}" for error in response_schema.iter_errors(row["response"]))
    for row in web["cases"]:
        if row["body"].get("response") is not None:
            schema_errors.extend(f"web {row['name']}: {error.message}" for error in response_schema.iter_errors(row["body"]["response"]))
    usage = [event["response"].get("usage", {}) for event in responses if event.get("success")]
    local = dict(line.split("=", 1) for line in (ROOT / ".env").read_text().splitlines() if "=" in line)
    secret = local["LLM_API_KEY"].encode()
    leaks = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.name == ".env" or ".git" in path.relative_to(ROOT).parts:
            continue
        if secret in path.read_bytes():
            leaks.append(str(path.relative_to(ROOT)))
    ignored = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "check-ignore", ".env"],
                             cwd=ROOT, capture_output=True).returncode == 0
    offline = (ROOT / "bench/results/live-gemma-offline.txt").read_text(encoding="utf-8-sig")
    offline_run = re.search(r"Ran (\d+) tests in ([\d.]+)s", offline)
    report = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "started_at_utc": smoke["started_at_utc"],
        "model": smoke["model"], "endpoint": smoke["endpoint"], "configuration_source": "root .env",
        "offline": {"real_gemma": False, "original_tests": 92,
                    "current_tests": int(offline_run[1]) if offline_run else None,
                    "elapsed_s": float(offline_run[2]) if offline_run else None,
                    "passed": bool(offline_run) and "\nOK" in offline},
        "real_smoke": {"success": smoke["success"], "elapsed_ms": smoke["trace"]["elapsed_ms"]},
        "real_cli": {"initial": {"passed": initial_cli["passed"], "executed": initial_cli["executed"]},
                     "recheck": {"passed": cli["passed"], "executed": cli["executed"]},
                     "full_run_latency_ms": percentiles([row["elapsed_s"] * 1000 for row in cli["cases"]])},
        "real_quality": {"unique_cases": len(quality_cases), "initial_passed": quality["passed"],
                         "initial_requirements_passed": quality["requirements_passed"],
                         "initial_preferences_passed": quality["preferences_passed"],
                         "rechecked": recheck["executed"], "recheck_passed": recheck["passed"],
                         "effective_passed_after_targeted_rechecks": sum(row["passed"] for row in effective_quality),
                         "local_search_ms": quality["local_search_ms"],
                         "note": "36 initial cases plus targeted rechecks; not a second full 36-case run"},
        "real_web": {key: web[key] for key in ("real_requests", "real_generations", "passed", "failed",
                    "frontend_secret_absent", "http_502_source", "dialogue_wall_ms_including_pacing")},
        "latency_benchmark": {key: latency[key] for key in ("successes", "failures", "latency_ms", "prompt_tokens",
                                                           "completion_tokens", "total_dialogue_ms")},
        "all_real_calls": {"http_attempts": len(attempts) + 1, "successful_generations": sum(event.get("success", False) for event in responses) + 1,
                           "observed_http_errors": [event for event in attempts if event["status"] != 200],
                           "max_observed_attempts_in_60s_excluding_smoke": maximum,
                           "prompt_tokens": sum(item.get("prompt_tokens", 0) for item in usage) + smoke["trace"]["prompt_tokens"],
                           "completion_tokens": sum(item.get("completion_tokens", 0) for item in usage) + smoke["trace"]["completion_tokens"]},
        "schema_errors": schema_errors, "credential_scan": {"ignored_env": ignored, "leaking_files": leaks},
        "fixes": ["maximum budget modifier misclassified as an item", "comparison after conjunction misclassified as an item",
                  "literal misspelled computer preference lost after model spelling correction", "direct latency-script import shadowed llm.client"],
        "limitations": ["HTTP 502 web path exercised by local size validation, not an induced cloud outage",
                        "OpenSearch not contacted in this real Gemma audit", "No real 401/403/429 cloud response was forced"]}
    write_report(ROOT / "bench/results/live-gemma.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    assert not leaks and ignored and not schema_errors


if __name__ == "__main__":
    main()
