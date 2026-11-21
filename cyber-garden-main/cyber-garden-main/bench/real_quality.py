"""Sequential real Gemma quality evaluation against frozen, independent states."""
from dataclasses import asdict
import argparse
import time

from buyer import model
from buyer.contracts import ROOT, decode_json, read_catalog
from buyer.conversation import ConversationManager
from buyer.ranking import tokens
from buyer.selection import select
from tests.live_observer import events, install
from tests.live_support import Pacer, denied, percentiles, preflight, write_report
from tests.real_nlu_cases import CASES


def preference_keys(preferences):
    return {(pref.category, pref.attribute, tokens(pref.text)) for pref in preferences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output-prefix", default="live-gemma-quality")
    args = parser.parse_args()
    cases = [case for case in CASES if not args.case or case[0] in args.case]
    if not cases:
        parser.error("No matching frozen cases")
    metadata = preflight()
    output = ROOT / "bench/results" / ("live-gemma-cases.jsonl" if args.output_prefix == "live-gemma-quality"
                                       else args.output_prefix + "-cases.jsonl")
    trace_path = ROOT / "bench/results" / (args.output_prefix + "-http.jsonl")
    output.write_text("", encoding="utf-8")
    trace_path.write_text("", encoding="utf-8")
    install(trace_path)
    manager = ConversationManager()
    catalog = read_catalog(decode_json((ROOT / "data/catalog.json").read_bytes()))
    records = []
    pacer = Pacer()
    for name, messages, gold, wishes in cases:
        pacer.wait()
        row = {"name": name, "messages": messages, "expected_requirements": asdict(gold),
               "expected_preferences": [asdict(pref) for pref in wishes], "real_gemma": True}
        began = time.perf_counter()
        local = manager.local(messages)
        row["local_nlu_ms"] = (time.perf_counter() - began) * 1000
        before = len(events(trace_path))
        try:
            started = time.perf_counter()
            query, trace = manager.resolve(messages, model.extract, require_model=True)
            row.update(resolve_ms=(time.perf_counter() - started) * 1000, actual_requirements=asdict(query.requirements),
                       actual_preferences=[asdict(pref) for pref in query.preferences], trace=asdict(trace),
                       requirements_pass=query.requirements == gold,
                       preferences_pass=preference_keys(query.preferences) == preference_keys(wishes),
                       attribute_questions=query.attribute_questions())
            started = time.perf_counter()
            row["selection"] = select(catalog, query.requirements, query.preferences).response
            row["local_search_ms"] = (time.perf_counter() - started) * 1000
            row["success"] = True
        except model.ModelError as error:
            row.update(success=False, error=str(error), requirements_pass=False, preferences_pass=False)
        row["gateway_events"] = events(trace_path)[before:]
        pacer.after(row["gateway_events"])
        row["passed"] = row["requirements_pass"] and row["preferences_pass"]
        records.append(row)
        write_report(output, row, append=True)
        print(("PASS " if row["passed"] else "FAIL ") + name, flush=True)
        if denied(row.get("error", "")):
            break
    gateway_times = [event["gateway_ms"] for row in records for event in row["gateway_events"]
                     if event["kind"] == "model_response" and event.get("success")]
    report = metadata | {"planned": len(cases), "executed": len(records), "passed": sum(row["passed"] for row in records),
        "requirements_passed": sum(row["requirements_pass"] for row in records),
        "preferences_passed": sum(row["preferences_pass"] for row in records),
        "transport_errors": sum(not row["success"] for row in records), "gateway_latency_ms": percentiles(gateway_times),
        "tokens_in": sum(row.get("trace", {}).get("prompt_tokens") or 0 for row in records),
        "tokens_out": sum(row.get("trace", {}).get("completion_tokens") or 0 for row in records),
        "local_search_ms": percentiles([row["local_search_ms"] for row in records if row["success"]]),
        "failures": [row for row in records if not row["passed"]]}
    write_report(ROOT / "bench/results" / (args.output_prefix + ".json"), report)
    print(f"{report['passed']}/{report['executed']}; requirements {report['requirements_passed']}; preferences {report['preferences_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
