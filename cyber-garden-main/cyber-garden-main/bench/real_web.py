"""Exercise the actual demo process against the official Gemma gateway."""
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from buyer.contracts import ROOT
from tests.live_observer import events
from tests.live_support import Pacer, child_environment, denied, preflight, write_report


def main():
    metadata = preflight()
    began = time.perf_counter()
    trace_path = ROOT / "bench/results/live-gemma-web-http.jsonl"
    trace_path.write_text("", encoding="utf-8")
    rows = []
    pacer = Pacer()
    secret = dict(line.split("=", 1) for line in (ROOT / ".env").read_text().splitlines() if "=" in line)["LLM_API_KEY"]
    with tempfile.TemporaryDirectory() as directory:
        env = child_environment(directory, trace_path)
        env["PYTHONPATH"] = os.pathsep.join((directory, str(ROOT)))
        process = subprocess.Popen([sys.executable, "-m", "buyer.demo", "--port", "0"], cwd=ROOT, env=env,
                                   text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            url = process.stdout.readline().strip()
            if not url.startswith("http://127.0.0.1:"):
                raise RuntimeError("Demo did not start")
            port = int(url.rsplit(":", 1)[1])

            def post(messages):
                connection = HTTPConnection("127.0.0.1", port, timeout=115)
                start = time.perf_counter()
                try:
                    body = json.dumps({"messages": messages}, ensure_ascii=False).encode()
                    connection.request("POST", "/api/buy", body, {"Content-Type": "application/json"})
                    response = connection.getresponse()
                    raw = response.read()
                    assert secret.encode() not in raw
                    return {"http_status": response.status, "body": json.loads(raw),
                            "wall_ms": (time.perf_counter() - start) * 1000, "messages": messages}
                finally:
                    connection.close()

            def user(text):
                return {"role": "user", "text": text}

            def follow(history, body, text):
                response = body.get("response") or {}
                assistant = " ".join(question["text"] for question in response.get("questions", [])) or "Подобрал комплект."
                return history + [{"role": "assistant", "text": assistant}, user(text)]

            history = [user("Стол не шире 100 см. Таганрог.")]
            pacer.wait()
            with ThreadPoolExecutor(max_workers=1) as pool:
                first = pool.submit(post, history)
                time.sleep(.2)
                busy = post(history)
                busy.update(name="occupied_gateway", real_gemma=False)
                assert busy["http_status"] == 429, busy
                rows.append(busy)
                row = first.result(timeout=115)
            row.update(name="first_request", real_gemma=True)
            rows.append(row)
            if denied(row["body"].get("error", "")):
                raise RuntimeError("Real gateway denied access; stopped without another request")
            for name, text in (("budget_answer", "15000 рублей."), ("city_continuation", "Теперь в Азове."),
                               ("add_category", "Добавь кресло."), ("remove_category", "Кресло больше не нужно."),
                               ("unknown_item", "Стол, а также холодильник.")):
                history = follow(history, row["body"], text)
                pacer.wait()
                row = post(history)
                row.update(name=name, real_gemma=True)
                rows.append(row)
                pacer.after(events(trace_path))
                print(f"{name}: HTTP {row['http_status']}", flush=True)
                if row["http_status"] != 200:
                    if denied(row["body"].get("error", "")):
                        raise RuntimeError("Real gateway denied access; stopped without another request")
                    break
            pacer.wait()
            row = post([user("Стол шириной до 100. Азов. Бюджет 15000 рублей.")])
            row.update(name="ambiguous_measurement", real_gemma=True)
            rows.append(row)
            oversized = post([user("Стол " + "я" * 17000)])
            oversized.update(name="local_model_request_limit", real_gemma=False)
            assert oversized["http_status"] == 502 and "error" in oversized["body"]
            rows.append(oversized)
            frontend_clean = True
            for path in ("/", "/app.js", "/api/catalog"):
                connection = HTTPConnection("127.0.0.1", port, timeout=5)
                connection.request("GET", path)
                response = connection.getresponse()
                frontend_clean &= secret.encode() not in response.read()
                connection.close()
            assert frontend_clean
        finally:
            process.terminate()
            _, error_log = process.communicate(timeout=10)
    observed = events(trace_path)
    for row in rows:
        body = row["body"]
        response = body.get("response") or {}
        if not row["real_gemma"]:
            row["passed"] = row["http_status"] == (429 if row["name"] == "occupied_gateway" else 502)
        elif row["http_status"] != 200 or body.get("trace", {}).get("model") != metadata["model"]:
            row["passed"] = False
        elif row["name"] in ("first_request", "unknown_item"):
            expected_field = "budget_rub" if row["name"] == "first_request" else "categories"
            row["passed"] = response.get("status") == "clarify" and expected_field in [
                question["field"] for question in response.get("questions", [])]
        elif row["name"] == "ambiguous_measurement":
            row["passed"] = body["response"] is None and bool(body["clarification"])
        else:
            expected = {"table", "chair"} if row["name"] == "add_category" else {"table"}
            row["passed"] = response.get("status") == "ok" and body["budget_rub"] == 15000 and all(
                {body["items"][ident]["category_id"] for ident in option["listing_ids"]} == expected
                for option in response.get("options", []))
    report = metadata | {"cases": rows, "gateway_events": observed, "frontend_secret_absent": frontend_clean,
        "real_requests": sum(row["real_gemma"] for row in rows),
        "real_generations": sum(event["kind"] == "model_response" and event.get("success", False) for event in observed),
        "gateway_technical_error_injected": False,
        "dialogue_wall_ms_including_pacing": (time.perf_counter() - began) * 1000,
        "passed": sum(row["passed"] for row in rows), "failed": [row["name"] for row in rows if not row["passed"]],
        "http_502_source": "local request-size validation before transmission", "server_stderr": error_log}
    write_report(ROOT / "bench/results/live-gemma-web.json", report)
    print(json.dumps({"real_requests": report["real_requests"], "real_generations": report["real_generations"],
                      "frontend_secret_absent": frontend_clean}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
