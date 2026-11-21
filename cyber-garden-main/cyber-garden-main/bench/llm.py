"""Gateway-only latency on the first ten labelled dialogues, one call at a time."""
import json
from pathlib import Path
import sys
import time
sys.path[0] = str(Path(__file__).resolve().parents[1])
from buyer.model import ModelError, extract
from buyer.contracts import ROOT
from tests.test_conversation import CASES
from tests.live_observer import install, events
from tests.live_support import Pacer, denied, percentiles, preflight, write_report

if __name__ == "__main__":
    metadata = preflight()
    trace_path = ROOT / "bench/results/live-gemma-latency-http.jsonl"
    trace_path.write_text("", encoding="utf-8")
    install(trace_path)
    pacer = Pacer()
    times, errors = [], []
    rows = []
    for name, messages, _ in CASES[:10]:
        pacer.wait()
        start = time.monotonic()
        before = len(events(trace_path))
        row = {"name": name, "messages": messages}
        try:
            _, trace = extract(messages, full_history=True)
            times.append(trace.elapsed_ms)
            row.update(success=True, extract_ms=trace.elapsed_ms, prompt_tokens=trace.prompt_tokens,
                       completion_tokens=trace.completion_tokens)
        except ModelError as exc:
            errors.append(str(exc))
            row.update(success=False, error=str(exc))
        row.update(dialogue_wall_ms=(time.monotonic() - start) * 1000, gateway_events=events(trace_path)[before:])
        pacer.after(row["gateway_events"])
        rows.append(row)
        print(f"{name}: {'PASS' if row['success'] else 'FAIL'}", file=sys.stderr, flush=True)
        if denied(row.get("error", "")):
            break
    gateway_times = [event["gateway_ms"] for row in rows for event in row["gateway_events"]
                     if event["kind"] == "model_response" and event.get("success")]
    report = metadata | dict(success_ms=times, latency_ms=percentiles(gateway_times), errors=errors, cases=rows,
        successes=len(times), failures=len(errors),
        prompt_tokens=sum(row.get("prompt_tokens") or 0 for row in rows),
        completion_tokens=sum(row.get("completion_tokens") or 0 for row in rows),
        total_dialogue_ms=sum(row["dialogue_wall_ms"] for row in rows))
    write_report(ROOT / "bench/results/live-gemma-latency.json", report)
    print(json.dumps(report, ensure_ascii=False))
