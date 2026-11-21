"""Mock HTTP tests for mandatory NLU, retries and the total gateway deadline."""
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import unittest
from unittest.mock import Mock, patch
from urllib.request import Request

from buyer import demo, model
from buyer.contracts import ROOT
from buyer.model import ModelError, ModelQuery, ModelTrace
from tests.test_buyer import requirements

REAL_SLEEP = time.sleep


@contextmanager
def http_gateway(statuses, retry_after=None, body_delay=0):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            status = statuses[min(len(calls) - 1, len(statuses) - 1)]
            body = b'{"mock":"ok"}'
            self.send_response(status)
            if retry_after is not None:
                self.send_header("Retry-After", retry_after)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                if body_delay:
                    for char in body:
                        REAL_SLEEP(body_delay)
                        self.wfile.write(bytes([char]))
                        self.wfile.flush()
                else:
                    self.wfile.write(body)
            except (OSError, ConnectionError):
                pass

        def log_message(self, *_args):
            pass

    with patch("socket.getfqdn", return_value="localhost"):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
    worker.start()
    try:
        yield Request(f"http://127.0.0.1:{server.server_port}/chat/completions", data=b"{}"), calls
    finally:
        server.shutdown()
        worker.join()
        server.server_close()


class DemoModelTest(unittest.TestCase):
    def setUp(self):
        with patch("socket.getfqdn", return_value="localhost"):
            self.server = demo.DemoServer(0, ROOT / "data/catalog.json")
        self.worker = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01})
        self.worker.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.worker.join)
        self.addCleanup(self.server.shutdown)
        self.mock = Mock(return_value=(ModelQuery(requirements(), ()), ModelTrace("mock", 1, 1, 1)))
        self.patch = patch.object(demo, "extract", self.mock)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def buy(self):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        body = json.dumps({"messages": [{"role": "user", "text": "Стол в Таганроге. Бюджет 6000 рублей."}]}).encode()
        try:
            connection.request("POST", "/api/buy", body=body, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            self.last_headers = dict(response.getheaders())
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_each_simple_buy_calls_gemma(self):
        for _ in range(2):
            status, body = self.buy()
            self.assertEqual(status, 200)
            self.assertEqual(body["trace"]["model"], "mock")
        self.assertEqual(self.mock.call_count, 2)

    def test_unavailable_gemma_is_technical_error(self):
        self.mock.side_effect = ModelError("mock gateway unavailable")
        status, body = self.buy()
        self.assertEqual(status, 502)
        self.assertIn("error", body)
        self.assertNotIn("response", body)

    def test_occupied_gateway_slot_stays_busy(self):
        self.server.gateway_slot.acquire()
        try:
            status, body = self.buy()
            self.assertEqual(status, 429)
            self.assertIn("Gemma", body["error"])
            self.mock.assert_not_called()
        finally:
            self.server.gateway_slot.release()


    def test_busy_response_has_zero_gemma_and_search_timings(self):
        self.server.gateway_slot.acquire()
        try:
            status, body = self.buy()
        finally:
            self.server.gateway_slot.release()
        self.assertEqual(status, 429)
        self.assertIn("error", body)
        timings = {p.strip().split(";dur=")[0]: float(p.split(";dur=")[1])
                   for p in self.last_headers["Server-Timing"].split(",")}
        self.assertEqual(timings["gemma"], 0)
        self.assertEqual(timings["search"], 0)
        self.assertGreaterEqual(timings["total"], timings["resolve"])
        self.mock.assert_not_called()

    def test_model_error_has_timings_without_search_or_no_match(self):
        self.mock.side_effect = ModelError("offline model error")
        status, body = self.buy()
        self.assertEqual(status, 502)
        self.assertIn("error", body)
        self.assertNotIn("response", body)
        timings = {p.strip().split(";dur=")[0]: float(p.split(";dur=")[1])
                   for p in self.last_headers["Server-Timing"].split(",")}
        self.assertEqual(timings["search"], 0)
        self.assertGreaterEqual(timings["resolve"] + .001, timings["gemma"])
        self.assertGreaterEqual(timings["total"], timings["resolve"])
        self.mock.assert_called_once()


class GatewayDeadlineTest(unittest.TestCase):
    def test_retryable_http_statuses_retry_exactly_once(self):
        for status, header, delay in ((429, "1", 1), (502, None, 2), (503, None, 2)):
            with self.subTest(status=status), http_gateway([status, 200], header) as (request, calls):
                with patch.object(model.time, "sleep") as sleep:
                    self.assertEqual(json.loads(model._gateway(request)), {"mock": "ok"})
                sleep.assert_called_once_with(delay)
                self.assertEqual(len(calls), 2)

    def test_repeated_error_exhausts_retry_not_no_match(self):
        with http_gateway([503, 503, 200]) as (request, calls), patch.object(model.time, "sleep"):
            with self.assertRaisesRegex(ModelError, "503"):
                model._gateway(request)
            self.assertEqual(len(calls), 2)

    def test_retry_after_above_cap_is_not_retried(self):
        with http_gateway([429, 200], "90") as (request, calls), patch.object(model.time, "sleep") as sleep:
            with self.assertRaisesRegex(ModelError, "429"):
                model._gateway(request)
            self.assertEqual(len(calls), 1)
            sleep.assert_not_called()

    def test_no_retry_when_total_deadline_has_too_little_time(self):
        # First call sets deadline=106; HTTP response arrives with only 7 seconds left.
        clock = iter([0, 0, 99])
        with http_gateway([503, 200]) as (request, calls), \
             patch.object(model.time, "monotonic", side_effect=lambda: next(clock, 99)), \
             patch.object(model.time, "sleep") as sleep:
            with self.assertRaises(ModelError):
                model._gateway(request)
            self.assertEqual(len(calls), 1)
            sleep.assert_not_called()

    def test_deadline_expired_before_connection(self):
        with patch.object(model.time, "monotonic", side_effect=[0, 107]), \
             patch.object(model.urllib.request, "build_opener") as build:
            with self.assertRaises(ModelError):
                model._gateway(Request("http://localhost/test", data=b"{}"))
            build.return_value.open.assert_not_called()

    def test_slow_body_cannot_outlive_total_deadline(self):
        # Scale the same monotonic clock by 1000: 106 budget seconds = .106 real seconds.
        # Body sends a byte every .03 seconds, so the socket never hits its idle timeout.
        monotonic = time.monotonic
        start = monotonic()
        with http_gateway([200], body_delay=.03) as (request, calls), \
             patch.object(model.time, "monotonic", side_effect=lambda: (monotonic() - start) * 1000):
            with self.assertRaises(ModelError):
                model._gateway(request)
            self.assertEqual(len(calls), 1)
