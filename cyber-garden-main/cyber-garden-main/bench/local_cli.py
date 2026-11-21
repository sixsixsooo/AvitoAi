"""Reproducible offline CLI timings with a real local HTTP mock Gemma gateway.

python -m bench.local_cli --repeats 30 --output bench/results/local-cli.json
Requires tests/requirements.txt only for schema validation, not for ./run.
"""

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from jsonschema import Draft202012Validator

from buyer import selection
from buyer.contracts import ROOT, decode_json, read_catalog, read_messages
from buyer.conversation import ConversationManager
from buyer.model import extract
from tests.cli_support import cli_command
from tests.fixtures import product

MESSAGES = [{"role": "user", "text": "Деревянный стол, мягкое кресло и настольная лампа. Таганрог. До 15000 рублей."}]
MODEL = dict(city_id="taganrog", budget_rub=15000, categories=["table", "chair", "lamp"],
             width_cm=[], adjustable_height=None, color_temperature_k=[], preferences=[
                 {"category": "table", "text": "деревянный"}, {"category": "chair", "text": "мягкое"},
                 {"category": "lamp", "text": "настольная"}])


def percentiles(samples):
    ordered = sorted(samples)
    return {f"p{p}_ms": round(ordered[math.ceil(p * len(ordered) / 100) - 1], 4) for p in (50, 95)}


@contextmanager
def gateway():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(MODEL)}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    with patch("socket.getfqdn", return_value="localhost"):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .05})
    worker.start()
    settings = dict(LLM_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1", LLM_MODEL="mock-gemma",
                    LLM_API_KEY="mock-only", LLM_USER_AGENT="smartset-bench/1.0")
    try:
        with patch.dict(os.environ, settings):
            yield calls
    finally:
        server.shutdown()
        worker.join()
        server.server_close()


def measured(path):
    timings = {}

    def timed(name, function, materialize=False):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = function(*args, **kwargs)
            if materialize:
                result = list(result)
            timings[name] = (time.perf_counter() - start) * 1000
            return result
        return wrapper

    start = time.perf_counter()
    request = read_messages(decode_json(json.dumps({"messages": MESSAGES})))
    catalog_start = time.perf_counter()
    listings = read_catalog(decode_json(path.read_bytes()))
    timings["catalog_read_validate"] = (time.perf_counter() - catalog_start) * 1000
    nlu_start = time.perf_counter()
    query, _ = ConversationManager().resolve(request, extract, require_model=True)
    timings["nlu_mock_gateway"] = (time.perf_counter() - nlu_start) * 1000
    with patch.object(selection, "filter_listings", timed("strict_filter", selection.filter_listings)), \
         patch.object(selection, "listing_scores", timed("relevance", selection.listing_scores)), \
         patch.object(selection, "ranked_bundles", timed("bundle_search", selection.ranked_bundles, True)), \
         patch.object(selection, "_cheapest", timed("bundle_search", selection._cheapest, True)):
        result = selection.select(listings, query.requirements, query.preferences)
        encoded = json.dumps(result.response, ensure_ascii=False, allow_nan=False)
    timings["python_total"] = (time.perf_counter() - start) * 1000
    timings["response_and_orchestration"] = max(0, timings["python_total"] - sum(
        timings.get(key, 0) for key in ("catalog_read_validate", "nlu_mock_gateway", "strict_filter", "relevance", "bundle_search")))
    return timings, json.loads(encoded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, default=ROOT / "bench/results/local-cli.json")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("At least two repetitions are required")
    validator = Draft202012Validator(json.loads((ROOT / "schemas/response.schema.json").read_text(encoding="utf-8")))
    report = {"environment": {"platform": platform.platform(), "python": platform.python_version(),
                              "processor": platform.processor(), "cpu_count": os.cpu_count()},
              "repeats": args.repeats, "gateway": "local HTTP mock, no real Gemma inference",
              "percentile_method": "nearest rank", "datasets": {}}
    with tempfile.TemporaryDirectory() as directory, gateway() as calls:
        synthetic = Path(directory) / "balanced-250.json"
        rows = []
        for index in range(250):
            category = ("table", "chair", "lamp")[index % 3]
            title = {"table": "Деревянный стол", "chair": "Мягкое кресло", "lamp": "Настольная лампа"}[category]
            rows.append(product(f"{category}-{index:03}", category, 1000 + index) | {"title": title})
        synthetic.write_text(json.dumps(rows), encoding="utf-8")
        for name, path in (("organizers", ROOT / "data/catalog.json"), ("balanced_250", synthetic)):
            catalog = read_catalog(decode_json(path.read_bytes()))
            sizes = [sum(item.category == category and item.city_id == "taganrog" for item in catalog)
                     for category in MODEL["categories"]]
            samples, process_samples = [], []
            measured(path)  # One warm-up; subprocess measurements always start a fresh interpreter.
            for _ in range(args.repeats):
                phases, answer = measured(path)
                validator.validate(answer)
                assert answer["status"] == "ok"
                samples.append(phases)
                started = time.perf_counter()
                done = subprocess.run(cli_command(ROOT / "run", "--catalog", path), cwd=directory,
                                      input=json.dumps({"messages": MESSAGES}), capture_output=True,
                                      text=True, encoding="utf-8", env=os.environ.copy(), timeout=15)
                process_samples.append((time.perf_counter() - started) * 1000)
                assert done.returncode == 0 and not done.stderr, done.stderr
                output = json.loads(done.stdout)
                validator.validate(output)
                assert output == answer
            report["datasets"][name] = {
                "listing_count": len(catalog), "eligible_category_sizes": sizes,
                "combination_count": math.prod(sizes), "warmup_runs": 1,
                "phases": {key: percentiles([sample.get(key, 0) for sample in samples]) for key in samples[0]},
                "fresh_run_process": percentiles(process_samples), "samples_ms": samples,
                "process_samples_ms": process_samples, "example_response": answer,
            }
        report["mock_gateway_calls"] = len(calls)
        report["stdout_and_schema_validated"] = True
        report["catalog_text_sent_to_gateway"] = any("description" in json.dumps(call) for call in calls)
        assert not report["catalog_text_sent_to_gateway"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in row.items() if key not in
                           ("samples_ms", "process_samples_ms", "example_response")}
                      for name, row in report["datasets"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
