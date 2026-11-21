"""Reproduce the reported purposes through the real web endpoint and Gemma."""
import json
from threading import Thread
from urllib.request import Request, urlopen

from buyer.contracts import ROOT
from buyer.demo import DemoServer
from tests.live_support import Pacer, preflight, write_report


def main():
    report = preflight()
    report["cases"] = []
    pacer = Pacer()
    with DemoServer(0, ROOT / "data/catalog.json") as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            for text, category, purpose in [
                ("Мне нужно кресло, на котором удобно сидеть моей собаке, покажи варианты.", "chair", "собаке"),
                ("Какой стол подойдет для груминга собак?", "table", "груминга"),
            ]:
                messages = [{"role": "user", "text": text}]
                for step in range(2):
                    pacer.wait()
                    request = Request(f"http://127.0.0.1:{server.server_port}/api/buy",
                        data=json.dumps({"messages": messages}).encode(),
                        headers={"Content-Type": "application/json"})
                    with urlopen(request, timeout=115) as response:
                        result = json.load(response)
                    assert any(p["category"] == category and purpose in p["text"]
                               for p in result["preferences"]), result
                    status = result["response"]["status"]
                    assert status == ("clarify" if step == 0 else "ok"), result
                    if step == 0:
                        assert {q["field"] for q in result["response"]["questions"]} == {"city_id", "budget_rub"}, result
                        question = " ".join(q["text"] for q in result["response"]["questions"])
                        messages.extend([{"role": "assistant", "text": question},
                                         {"role": "user", "text": "Таганрог, бюджет 15000 рублей."}])
                    report["cases"].append({"query": text, "step": step, "result": result})
        finally:
            server.shutdown()
            thread.join(timeout=5)
    write_report(ROOT / "bench/results/purpose-live.json", report)
    print("4/4 real HTTP + Gemma checks passed")


if __name__ == "__main__":
    main()
