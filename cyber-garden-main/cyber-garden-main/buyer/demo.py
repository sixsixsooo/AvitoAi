"""Local browser demo. The CLI and browser share the same interpretation and solver."""

import argparse
import errno
import json
import socket
import sys
import time
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore

from buyer.contracts import CATEGORIES, CITIES, ROOT, decode_json, read_catalog, read_messages
from buyer.conversation import ConversationManager
from buyer.indexed import SearchError, SearchProvider
from buyer.lab import QUERY, compare
from buyer.model import ModelError, ModelTrace, extract
from buyer.selection import LABELS, Selection, actual_feature, describe_condition, select

UI = Path(__file__).with_name("ui")
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/workspace.webp": ("workspace.webp", "image/webp"),
}
CSP = "default-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64

    def __init__(self, port, catalog, provider=None):
        self.provider = provider
        self.catalog = [] if provider else decode_json(catalog.read_bytes())
        self.listings = read_catalog(self.catalog)
        self.titles = {item["id"]: item["title"] for item in self.catalog}
        self.gateway_slot = BoundedSemaphore(1)
        super().__init__(("127.0.0.1", port), Handler)


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def _send(self, status, raw, mime):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(raw)))
        stages = getattr(self, "_stages", None)
        if stages is not None:
            stages["total_ms"] = (time.perf_counter() - self._request_started) * 1000
            measured = {key: round(value, 3) for key, value in stages.items()}
            self.send_header("Server-Timing", ", ".join(
                key.removesuffix("_ms") + ";dur=" + str(value) for key, value in measured.items()))
            # Deliberately exclude request text, headers, URLs and credentials.
            sys.stderr.write(json.dumps({"event": "buyer_stage_latency", "status": int(status), **measured}) + "\n")
            sys.stderr.flush()
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _json(self, status, data):
        self._send(status, json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _local_host(self):
        port = self.server.server_port
        return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

    def do_GET(self):
        asset = ASSETS.get(self.path)
        if not self._local_host():
            self._json(HTTPStatus.FORBIDDEN, {"error": "Демо доступно через localhost."})
        elif self.path == "/api/catalog":
            try:
                provider = self.server.provider
                total = provider.cluster.call(provider.index + "/_count")["count"] if provider else len(self.server.listings)
                self._json(HTTPStatus.OK, {"total": total, "cities": list(CITIES.values())})
            except SearchError:
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Каталог временно недоступен."})
        elif asset and (UI / asset[0]).is_file():
            self._send(HTTPStatus.OK, (UI / asset[0]).read_bytes(), asset[1])
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Страница не найдена."})

    def _request(self):
        if self.headers.get_content_type() != "application/json":
            raise ValueError
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= 65_536:
            raise ValueError
        return decode_json(self.rfile.read(size))

    def do_POST(self):
        self._request_started = time.perf_counter()
        self._stages = dict(gemma_ms=0.0, resolve_ms=0.0, search_ms=0.0, total_ms=0.0) if self.path == "/api/buy" else None
        if not self._local_host():
            self._json(HTTPStatus.FORBIDDEN, {"error": "Демо доступно через localhost."})
            return
        if self.path not in ("/api/buy", "/api/lab"):
            self._json(HTTPStatus.NOT_FOUND, {"error": "Страница не найдена."})
            return
        try:
            data = self._request()
            if self.path == "/api/lab":
                if data != {}:
                    raise ValueError
                messages = [{"role": "user", "text": QUERY}]
            else:
                messages = read_messages(data)
        except (ValueError, RecursionError, socket.timeout):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "Проверьте формат и длину диалога."})
            return

        try:
            resolve_started = time.perf_counter()
            try:
                query, trace = ConversationManager().resolve(
                    messages, self._gemma, require_model=True
                )
            finally:
                if self._stages is not None:
                    self._stages["resolve_ms"] = (time.perf_counter() - resolve_started) * 1000
        except GatewayBusy:
            self._json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "Gemma уже обрабатывает запрос. Повторите чуть позже."})
            return
        except ModelError as error:
            self._json(HTTPStatus.BAD_GATEWAY, {"error": str(error)})
            return
        requirements = query.requirements
        if trace is None:
            trace = ModelTrace("Локальный разбор", None, None, 0, "локальный разбор")
        elif "categories" in query.confirmed:
            trace = ModelTrace(trace.model, trace.prompt_tokens, trace.completion_tokens,
                               trace.elapsed_ms, "локальный разбор")
        if self.path == "/api/lab":
            data = compare(requirements)
            data.update({"query": QUERY, "trace": asdict(trace)})
        else:
            try:
                data = self._buyer_result(query, trace)
            except (SearchError, ValueError):
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Не удалось получить полный результат поиска."})
                return
            data["preferences"] = [asdict(item) for item in query.preferences]
        self._json(HTTPStatus.OK, data)

    def _gemma(self, messages, *, full_history):
        if not self.server.gateway_slot.acquire(blocking=False):
            raise GatewayBusy
        started = time.perf_counter()
        try:
            return extract(messages, full_history=full_history)
        finally:
            if self._stages is not None:
                # One high-resolution clock for all nested Web stages. The
                # existing client trace uses monotonic and stays unchanged.
                self._stages["gemma_ms"] = (time.perf_counter() - started) * 1000
            self.server.gateway_slot.release()

    def _buyer_result(self, query, trace):
        requirements = query.requirements
        clarification = query.attribute_questions() if not query.questions() else ()
        listings, titles = self.server.listings, self.server.titles
        if query.decision_questions():
            clarification = ()
            result = Selection({"status": "clarify", "questions": list(query.decision_questions())}, {}, None)
        elif clarification:
            result = None
        else:
            search_started = time.perf_counter()
            try:
                if self.server.provider:
                    found = self.server.provider.search(requirements, query.preferences)
                    result, listings, titles = found.selection, found.listings, found.titles
                else:
                    result = select(listings, requirements, query.preferences)
            finally:
                if self._stages is not None:
                    self._stages["search_ms"] = (time.perf_counter() - search_started) * 1000
        items = {
            listing_id for option in (result.response.get("options", []) if result else [])
            for listing_id in option["listing_ids"]
        }
        core_fields = [
            {"label": "Город самовывоза", "value": CITIES.get(requirements.city_id)},
            {"label": "Бюджет", "value": requirements.budget_rub},
            {"label": "Состав", "value": ", ".join(CATEGORIES[key] for key in requirements.categories) or None},
        ]
        conditions = []
        for key in requirements.categories:
            condition = requirements.condition(key)
            if condition is not None and condition != ():
                conditions.append({"label": LABELS[key], "value": describe_condition(key, condition)})

        return {
            "response": result.response if result else None,
            "clarification": clarification,
            "trace": asdict(trace),
            "requirements": core_fields + conditions,
            "budget_rub": requirements.budget_rub,
            "eligible": [{"category": CATEGORIES[key], "count": count}
                         for key, count in result.eligible_counts.items()] if result else [],
            "minimum_total_rub": result.minimum_total_rub if result else None,
            "items": {
                item.id: {"category": CATEGORIES[item.category], "category_id": item.category,
                          "title": titles[item.id], "price_rub": item.price_rub,
                          "feature": actual_feature(item), "city": CITIES[item.city_id]}
                for item in listings if item.id in items
            },
        }


class GatewayBusy(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.json")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--index", help="Имя готового индекса OpenSearch")
    parser.add_argument("--search-url", default="http://127.0.0.1:19200")
    parser.add_argument("--semantic-model", type=Path, help="Папка с ONNX E5 и tokenizer.json")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port должен быть от 0 до 65535")
    provider = SearchProvider(args.search_url, args.index) if args.index else None
    if args.semantic_model:
        if not provider:
            parser.error("--semantic-model требует --index")
        from buyer.semantic import SemanticRanker
        provider.semantic = SemanticRanker(args.semantic_model)

    # A previous demo process is often still running during a presentation.
    # Pick the next port instead of failing with an opaque bind error.
    server = None
    for port in range(args.port, min(args.port + 10, 65536)):
        try:
            server = DemoServer(port, args.catalog, provider)
            break
        except OSError as error:
            if error.errno != errno.EADDRINUSE and getattr(error, "winerror", None) != 10048:
                raise
    if server is None:
        parser.error(f"Порты {args.port}–{args.port + 9} заняты. Остановите старое демо или задайте --port.")
    with server:
        print(f"http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
