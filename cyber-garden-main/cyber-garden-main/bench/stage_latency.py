"""Real Web/Gemma stage latency. Reports intentionally contain no request text."""
import argparse
from datetime import datetime, timezone
from http.client import HTTPConnection
import json
from pathlib import Path
import tempfile
import threading
import time
import urllib.error
import urllib.request

from jsonschema import Draft202012Validator
from bench.dialogue_live import BASE, CASES as DIALOGUE_CASES, FACTS, ROWS, dialog
from buyer.contracts import ROOT, read_catalog
from buyer.demo import DemoServer
from tests.live_support import Pacer, denied, percentiles, preflight, write_report

SELECTED = {"initial_bundle", "off_topic", "resume_after_off_topic", "only_table", "remove_chair",
            "budget_update", "missing_lamps_without_budget", "infeasible_bundle_budget"}
CASES = [case for case in DIALOGUE_CASES if case[0] in SELECTED] + [
    ("single_lamp", dialog("Только лампа в Таганроге, бюджет 5000 рублей."),
     FACTS | {"categories": ["lamp"], "budget_rub": 5000, "width_cm": []}, "ok", ["l"], "full"),
    ("pancake_recipe", dialog(BASE, "Подобрал комплект.", "Скажи рецепт блинчиков"),
     FACTS, "clarify", None, "full"),
    ("pythagorean_theorem", dialog(BASE, "Подобрал комплект.", "Посчитай теорему Пифагора"),
     FACTS, "clarify", None, "full"),
    ("unsupported_refrigerator", dialog(BASE, "Подобрал комплект.", "Найди холодильник"),
     FACTS | {"categories": []}, "clarify", None, "full"),
    ("unsupported_wardrobe", dialog(BASE, "Подобрал комплект.", "Хочу купить шкаф"),
     FACTS | {"categories": []}, "clarify", None, "full"),
]
CASES += [
    ("organizer_single_table", dialog("Только стол не шире 100 см. Таганрог. Бюджет 15000 рублей."),
     FACTS | {"categories": ["table"]}, "ok", None, "organizer"),
    ("organizer_full_bundle", dialog(BASE), FACTS, "ok", None, "organizer"),
]
STAGES = ("gemma_ms", "resolve_ms", "search_ms", "total_ms")


def timings(header):
    values = {part.strip().split(";dur=")[0] + "_ms": float(part.split(";dur=")[1])
              for part in header.split(",")}
    assert set(values) == set(STAGES) and all(value >= 0 for value in values.values())
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "bench/results/final-stage-latency.json")
    parser.add_argument("--case-id", action="append", default=[])
    args = parser.parse_args()
    cases = [case for case in CASES if not args.case_id or case[0] in args.case_id]
    if not cases or any(name not in {case[0] for case in CASES} for name in args.case_id):
        parser.error("Unknown stage scenario")
    if args.output.exists():
        raise FileExistsError("Choose a new report path; previous results are preserved")
    meta = preflight()
    validator = Draft202012Validator(json.loads((ROOT / "schemas/response.schema.json").read_bytes()))
    attempts = []
    original = urllib.request.OpenerDirector.open

    def observed_open(opener, request, *args, **kwargs):
        url = request.full_url if isinstance(request, urllib.request.Request) else request
        if url != meta["endpoint"] + "/chat/completions":
            return original(opener, request, *args, **kwargs)
        event = {"kind": "http_attempt", "utc": datetime.now(timezone.utc).isoformat(), "url": url}
        try:
            response = original(opener, request, *args, **kwargs)
            event["status"] = response.status
            return response
        except urllib.error.HTTPError as error:
            event.update(status=error.code, retry_after=error.headers.get("Retry-After"))
            raise
        except (OSError, urllib.error.URLError):
            event["status"] = "connection_error"
            raise
        finally:
            attempts.append(event)

    urllib.request.OpenerDirector.open = observed_open
    records = []
    pacer = Pacer()
    try:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory, "catalog.json")
            for name, messages, gold, expected, first_ids, catalogue_name in cases:
                rows = ROWS if catalogue_name == "full" else [row for row in ROWS if row["category"] != "lamp"]
                if catalogue_name == "organizer":
                    rows = json.loads((ROOT / "data/catalog.json").read_text(encoding="utf-8"))
                read_catalog(rows)
                catalog.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
                with DemoServer(0, catalog) as server:
                    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
                    worker.start()
                    pacer.wait()
                    before = len(attempts)
                    record = {"case_id": name, "expected_status": expected, "real_gemma": True,
                              "catalogue_kind": catalogue_name, "catalogue_records": len(rows)}
                    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=115)
                    started = time.perf_counter()
                    stop = False
                    try:
                        connection.request("POST", "/api/buy", json.dumps({"messages": messages}, ensure_ascii=False).encode(),
                                           {"Content-Type": "application/json"})
                        response = connection.getresponse()
                        raw = response.read()
                        record["client_wall_ms"] = round((time.perf_counter() - started) * 1000, 3)
                        record["http_status"] = response.status
                        body = json.loads(raw)
                        stop = denied(body.get("error", ""))
                        record.update(timings(response.getheader("Server-Timing", "")))
                        assert response.status == 200
                        assert body["trace"]["model"] == meta["model"]
                        record["model_trace_elapsed_ms"] = body["trace"]["elapsed_ms"]
                        result = body["response"]
                        validator.validate(result)
                        record["response_status"] = result["status"]
                        assert result["status"] == expected
                        if expected == "ok":
                            if first_ids is not None:
                                assert result["options"][0]["listing_ids"] == first_ids
                            record["listing_ids"] = [option["listing_ids"] for option in result["options"]]
                            by_id = {row["id"]: row for row in rows}
                            for option in result["options"]:
                                items = [by_id[ident] for ident in option["listing_ids"]]
                                assert {item["category"] for item in items} == set(gold["categories"])
                                assert all(item["city_id"] == gold["city_id"] for item in items)
                                assert sum(item["price_rub"] for item in items) == option["total_rub"] <= gold["budget_rub"]
                        else:
                            assert "options" not in result and body["items"] == {}
                        wording = " ".join(q["text"] for q in result.get("questions", []))
                        if name in {"off_topic", "pancake_recipe", "pythagorean_theorem"}:
                            assert "не отвечаю на вопросы по другим темам" in wording
                        if name.startswith("unsupported_"):
                            assert ("Холодильники" if name.endswith("refrigerator") else "Шкафы") in wording
                        assert record["gemma_ms"] <= record["resolve_ms"] + .002
                        assert record["resolve_ms"] + record["search_ms"] <= record["total_ms"] + .002
                        assert attempts[before:] and any(e["status"] == 200 for e in attempts[before:])
                        record["passed"] = True
                    except Exception as error:
                        # Do not persist bodies, prompts, exception text or headers.
                        record.update(passed=False, error_type=type(error).__name__)
                    finally:
                        connection.close()
                        server.shutdown(); worker.join()
                    record["http_attempts"] = attempts[before:]
                    pacer.after(record["http_attempts"])
                    records.append(record)
                    write_report(args.output, meta | {"complete": False, "planned": len(cases), "cases": records})
                    print(("PASS " if record["passed"] else "FAIL ") + name + " " +
                          json.dumps({k:record[k] for k in STAGES if k in record}), flush=True)
                    if stop:
                        break
    finally:
        urllib.request.OpenerDirector.open = original
    completed = [row for row in records if all(k in row for k in STAGES)]
    successful = [row for row in records if row["passed"]]
    report = meta | {"complete": len(records) == len(cases), "planned": len(cases), "executed": len(records),
        "passed": len(successful), "errors": len(records) - len(successful),
        "successful_real_http_generations": sum(e["status"] == 200 for e in attempts),
        "http_attempts": len(attempts), "stages_ms": {k:percentiles([row[k] for row in completed]) for k in STAGES},
        "stage_max_ms": {k:max((row[k] for row in completed), default=None) for k in STAGES},
        "search_performed_ms": percentiles([row["search_ms"] for row in completed if row["search_ms"] > 0]),
        "search_skipped_count": sum(row["search_ms"] == 0 for row in completed),
        "latency_30_70s_reproduced": any(30000 <= row["total_ms"] <= 70000 for row in completed),
        "requests_over_30s": [row["case_id"] for row in completed if row["total_ms"] >= 30000],
        "clock": "perf_counter for all nested Web stages; existing ModelTrace clock unchanged",
        "gemma_definition": "extract call including preparation, actual gateway request and response parsing; no search",
        "resolve_definition": "full resolve including Gemma; nested, not additive to gemma_ms",
        "total_definition": "handler start through response serialization, immediately before sending headers/body",
        "privacy": {"request_text_saved": False, "authorization_saved": False, "response_body_saved": False},
        "mock_measurements_included": False, "cases": records}
    write_report(args.output, report)
    print(json.dumps({"passed": report["passed"], "executed": report["executed"], "stages_ms": report["stages_ms"]}), flush=True)
    return int(len(successful) != len(cases))


if __name__ == "__main__":
    raise SystemExit(main())
