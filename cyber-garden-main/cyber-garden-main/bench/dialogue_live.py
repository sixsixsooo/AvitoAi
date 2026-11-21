"""Fifteen frozen dialogue checks via real HTTP Web API and official Gemma."""
import argparse
from dataclasses import asdict
from http.client import HTTPConnection
import json
from pathlib import Path
import tempfile
import threading
import time

from jsonschema import Draft202012Validator
from buyer.contracts import ROOT, read_catalog, read_requirements
from buyer.demo import DemoServer
from tests.fixtures import product
from tests.live_observer import events, install
from tests.live_support import Pacer, denied, preflight, write_report

BASE = "Стол не шире метра, кресло и лампа. Таганрог. До 15 000 ₽ за всё."
FACTS = dict(city_id="taganrog", budget_rub=15000, categories=["table", "chair", "lamp"],
             width_cm=[["lte", 100]], adjustable_height=None, color_temperature_k=[])
ROWS = [product("t80", "table", 2000, 80), product("t100", "table", 3000, 100),
        product("wide", "table", 1000, 110), product("c", "chair", 1000, True),
        product("l", "lamp", 500, 3000), product("azT", "table", 2000, 100, "azov"),
        product("azC", "chair", 1000, True, "azov"), product("azL", "lamp", 500, 3000, "azov")]


def dialog(*texts):
    return [{"role": "user" if i % 2 == 0 else "assistant", "text": text} for i, text in enumerate(texts)]


# Independent expected states: never inferred from actual model output.
CASES = [
    ("initial_bundle", dialog(BASE), FACTS, "ok", ["t80", "c", "l"], "full"),
    ("width_update", dialog(BASE, "Подобрал комплект.", "нужен стол 100см таганрог"),
     FACTS | {"width_cm": [["eq", 100]]}, "ok", ["t100", "c", "l"], "full"),
    ("table_alias", dialog(BASE, "Подобрал комплект.", "столик таганрог"),
     FACTS, "ok", ["t80", "c", "l"], "full"),
    ("ambiguous_item_price", dialog(BASE, "Подобрал комплект.", "стул таганрог 1400"),
     FACTS, "clarify", None, "full"),
    ("off_topic", dialog(BASE, "Подобрал комплект.", "рецепт гуляша"), FACTS, "clarify", None, "full"),
    ("resume_after_off_topic", dialog(BASE, "Подобрал комплект.", "рецепт гуляша",
        "Я помогаю подбирать товары из каталога. Уточните, что вы хотите приобрести: например, стол, кресло или лампу.",
        "нужен стол"), FACTS, "ok", ["t80", "c", "l"], "full"),
    ("only_table", dialog(BASE, "Подобрал комплект.", "оставь только стол"),
     FACTS | {"categories": ["table"]}, "ok", ["t80"], "full"),
    ("remove_chair", dialog(BASE, "Подобрал комплект.", "убери кресло"),
     FACTS | {"categories": ["table", "lamp"]}, "ok", ["t80", "l"], "full"),
    ("cancel_wish_add_lamp", dialog("Деревянный стол в Таганроге. Бюджет 15000 рублей.",
        "Подобрал стол.", "Отмени предыдущее пожелание, добавь лампу."),
     FACTS | {"categories": ["table", "lamp"], "width_cm": []}, "ok", ["wide", "l"], "full"),
    ("budget_update", dialog(BASE, "Подобрал комплект.", "бюджет теперь 12000"),
     FACTS | {"budget_rub": 12000}, "ok", ["t80", "c", "l"], "full"),
    ("city_update", dialog(BASE, "Подобрал комплект.", "город теперь Азов"),
     FACTS | {"city_id": "azov"}, "ok", ["azT", "azC", "azL"], "full"),
    ("clear_width", dialog(BASE, "Подобрал комплект.", "неважно, какая ширина"),
     FACTS | {"width_cm": []}, "ok", ["wide", "c", "l"], "full"),
    ("missing_lamps_without_budget", dialog("Нужна лампа в Таганроге"),
     FACTS | {"categories": ["lamp"], "budget_rub": None, "width_cm": []}, "no_match", None, "no_lamps"),
    ("new_dialogue", dialog(BASE, "Подобрал комплект.", "новый диалог"),
     FACTS | {"city_id": None, "budget_rub": None, "categories": [], "width_cm": []}, "clarify", None, "full"),
    ("infeasible_bundle_budget", dialog(BASE.replace("15 000", "100")),
     FACTS | {"budget_rub": 100}, "no_match", None, "full"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", default="dialogue-live-web")
    args = parser.parse_args()
    meta = preflight()
    prefix = ROOT / "bench/results" / args.output_prefix
    output, summary, trace_path = (Path(str(prefix) + suffix) for suffix in ("-cases.jsonl", ".json", "-http.jsonl"))
    if any(p.exists() for p in (output, summary, trace_path)):
        raise FileExistsError("Use a new report prefix")
    output.touch(); trace_path.touch()
    install(trace_path)
    validator = Draft202012Validator(json.loads((ROOT / "schemas/response.schema.json").read_bytes()))
    pacer = Pacer()
    records = []
    secret = dict(line.split("=", 1) for line in (ROOT / ".env").read_text().splitlines() if "=" in line)["LLM_API_KEY"]
    frontend_secret_absent = True
    with tempfile.TemporaryDirectory() as directory:
        catalog = Path(directory, "catalog.json")
        for name, messages, gold, status, first_ids, catalogue_name in CASES:
            rows = ROWS if catalogue_name == "full" else [row for row in ROWS if row["category"] != "lamp"]
            read_catalog(rows)
            catalog.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            with DemoServer(0, catalog) as server:
                worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
                worker.start()
                pacer.wait()
                before = len(events(trace_path))
                start = time.perf_counter()
                expected = json.loads(json.dumps(asdict(read_requirements(gold)), default=str))
                record = {"name": name, "messages": messages, "expected_requirements": expected,
                          "expected_preferences": [], "expected_status": status, "catalogue": catalogue_name,
                          "real_gemma": True, "actual_http_post": True}
                connection = HTTPConnection("127.0.0.1", server.server_port, timeout=115)
                try:
                    connection.request("POST", "/api/buy", json.dumps({"messages": messages}, ensure_ascii=False).encode(),
                                       {"Content-Type": "application/json"})
                    response = connection.getresponse(); raw = response.read()
                    assert secret.encode() not in raw
                    body = json.loads(raw)
                    record.update(http_status=response.status, body=body)
                    observed = events(trace_path)[before:]
                    record["gateway_events"] = observed
                    assert response.status == 200, body
                    assert body["trace"]["model"] == meta["model"], body
                    validator.validate(body["response"])
                    states = [e for e in observed if e["kind"] == "resolved_state"]
                    attempts = [e for e in observed if e["kind"] == "http_attempt"]
                    assert len(states) == 1 and attempts, observed
                    assert all(e["url"] == meta["endpoint"] + "/chat/completions" for e in attempts)
                    record["actual_requirements"] = states[0]["requirements"]
                    assert states[0]["requirements"] == expected, states[0]
                    assert states[0]["preferences"] == [], states[0]
                    result = body["response"]
                    assert result["status"] == status, result
                    if status == "ok":
                        assert result["options"][0]["listing_ids"] == first_ids, result
                        by_id = {row["id"]: row for row in rows}
                        for option in result["options"]:
                            items = [by_id[i] for i in option["listing_ids"]]
                            assert {row["category"] for row in items} == set(gold["categories"])
                            assert all(row["city_id"] == gold["city_id"] for row in items)
                            assert sum(row["price_rub"] for row in items) == option["total_rub"] <= gold["budget_rub"]
                    else:
                        assert "options" not in result and body["items"] == {}, body
                    record["passed"] = True
                except Exception as error:
                    record.update(passed=False, error=str(error))
                finally:
                    connection.close()
                    server.shutdown(); worker.join()
                record.setdefault("gateway_events", events(trace_path)[before:])
                record["wall_ms"] = (time.perf_counter() - start) * 1000
                pacer.after(record["gateway_events"])
                write_report(output, record, append=True); records.append(record)
                print(("PASS " if record["passed"] else "FAIL ") + name, flush=True)
                if denied(json.dumps(record.get("body", {}))) or denied(record.get("error", "")):
                    break
        # Static assets are never allowed to contain backend credentials.
        with DemoServer(0, catalog) as server:
            worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}); worker.start()
            try:
                for path in ("/", "/app.js", "/api/catalog"):
                    c = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
                    c.request("GET", path); response = c.getresponse()
                    frontend_secret_absent &= response.status == 200 and secret.encode() not in response.read()
                    c.close()
            finally:
                server.shutdown(); worker.join()
    report = meta | {"planned": len(CASES), "executed": len(records), "passed": sum(r["passed"] for r in records),
        "real_generations": sum(e["kind"] == "model_response" and e.get("success", False)
                                for row in records for e in row["gateway_events"]),
        "frontend_secret_absent": frontend_secret_absent, "cases": records}
    write_report(summary, report)
    print(f"{report['passed']}/{report['executed']}; real generations {report['real_generations']}", flush=True)
    return int(report["passed"] != len(CASES))


if __name__ == "__main__":
    raise SystemExit(main())
