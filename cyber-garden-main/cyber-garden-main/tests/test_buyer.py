import itertools
import json
import os
import random
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from buyer.anchors import categories_from_dialog
from buyer.contracts import ROOT, decode_json, read_catalog, read_messages, read_requirements
from buyer.lab import CATALOG as LAB_CATALOG, compare, mutated_catalog
from buyer.model import MAX_TEXT_BYTES, ModelError, prepare_body
from buyer.selection import select
from tests.fixtures import CATALOG, product
from tests.cli_support import cli_command


def requirements(**updates):
    return read_requirements(dict(city_id="taganrog", budget_rub=6000, categories=["table"],
                                  width_cm=[], adjustable_height=None, color_temperature_k=[]) | updates)


class InputsTest(unittest.TestCase):
    def test_category_anchors_respect_changes_and_ambiguity(self):
        def dialog(*turns):
            return [{"role": "user" if index % 2 == 0 else "assistant", "text": text}
                    for index, text in enumerate(turns)]

        cases = [
            (("Стол не шире метра, кресло и лампа в Таганроге.",),
             ("table", "chair", "lamp")),
            (("Нужна настольная лампа.",), ("lamp",)),
            (("Стол и кресло.", "Подобрал комплект.", "Лампу добавь, стол больше не нужен."),
             ("chair", "lamp")),
            (("Кресло и лампа.", "Какой бюджет?", "Только стол, до 5000."),
             ("table",)),
            (("Стол и диван.",), None),
            (("Стол или кресло.",), None),
        ]
        for turns, expected in cases:
            with self.subTest(turns=turns):
                self.assertEqual(categories_from_dialog(dialog(*turns)).categories, expected)

    def test_model_receives_user_history_and_latest_assistant(self):
        messages = [
            {"role": "user", "text": "Стол и лампа. Таганрог."},
            {"role": "assistant", "text": "Старое длинное объяснение."},
            {"role": "user", "text": "Без лампы."},
            {"role": "assistant", "text": "Какой бюджет на весь комплект?"},
            {"role": "user", "text": "Пять тысяч."},
        ]
        body = decode_json(prepare_body(messages, "model"))
        self.assertEqual([item["content"] for item in body["messages"][1:]],
                         [messages[0]["text"], messages[2]["text"],
                          messages[3]["text"], messages[4]["text"]])
        self.assertLessEqual(body["max_tokens"], 384)

    def test_json_rejects_duplicates_nonfinite(self):
        for raw in ('{"n": 1, "n": 2}', '{"n": NaN}', '{"n": -Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                decode_json(raw)

    def test_message_structure_and_stateless_history(self):
        user = {"role": "user", "text": "Лампа"}
        assistant = {"role": "assistant", "text": "Где вы ищете лампу?"}
        for case in ([], [assistant], [user, assistant], [user, user, user]):
            with self.subTest(case=case), self.assertRaises(ValueError):
                read_messages({"messages": case})
        self.assertEqual(read_messages({"messages": [user, assistant, user]}), [user, assistant, user])

    def test_duplicate_listing_id_and_bad_boolean(self):
        item = CATALOG[0]
        with self.assertRaises(ValueError):
            read_catalog([item, item])
        with self.assertRaises(ValueError):
            read_catalog([product("chair", "chair", 100, 0)])

    def test_model_types_fail_closed(self):
        for fields in ({"budget_rub": True}, {"budget_rub": -1}, {"city_id": "unknown"},
                       {"categories": ["table", "table"]}, {"adjustable_height": 0},
                       {"width_cm": [["le", 100]]}, {"width_cm": [["lte", "100"]]}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                requirements(**fields)

    def test_gateway_utf8_budget_and_field_mapping(self):
        messages = [{"role": "user", "text": "м" * (MAX_TEXT_BYTES // 2)}]
        with self.assertRaises(ModelError):
            prepare_body(messages, "model")
        body = decode_json(prepare_body([{"role": "user", "text": "стол"}], "model"))
        self.assertEqual(body["messages"][-1], {"role": "user", "content": "стол"})
        self.assertEqual(body["response_format"], {"type": "json_object"})


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.catalog = read_catalog(CATALOG)

    def result(self, **changes):
        return select(self.catalog, requirements(**changes)).response

    def test_all_missing_fields_precede_catalog_search(self):
        response = self.result(city_id=None, budget_rub=None, categories=None)
        self.assertEqual(response["status"], "clarify")
        self.assertEqual({entry["field"] for entry in response["questions"]},
                         {"budget_rub", "city_id", "categories"})

    def test_exact_budget_and_width_with_strict_boundary(self):
        self.assertEqual(self.result(budget_rub=3500, width_cm=[["eq", 100]])["options"][0]["listing_ids"],
                         ["t100"])
        self.assertEqual(self.result(budget_rub=4000, width_cm=[["gt", 100], ["lt", 110]])["options"][0]["listing_ids"],
                         ["t105"])

    def test_false_is_known_and_unknown_survives_without_filter(self):
        false = self.result(categories=["chair"], adjustable_height=False)
        self.assertEqual([option["listing_ids"] for option in false["options"]], [["cFixed"]])
        unrestricted = self.result(categories=["chair"])
        self.assertEqual(unrestricted["options"][0]["listing_ids"], ["cUnknown"])

    def test_known_missing_constraint_has_data_driven_suggestion(self):
        response = self.result(categories=["lamp"], city_id="azov", color_temperature_k=[["eq", 4500]])
        self.assertEqual(set(response), {"status", "message"})
        self.assertEqual(response["status"], "no_match")
        self.assertIn("3300 К", response["message"])
        self.assertNotIn("azLamp", response["message"])

    def test_budget_zero_reports_minimum_and_delta(self):
        response = self.result(city_id="azov", budget_rub=0)
        self.assertEqual(response["status"], "no_match")
        self.assertIn("2 000 ₽", response["message"])

    def test_empty_catalog_explains_absence(self):
        response = select((), requirements()).response
        self.assertEqual(response["status"], "no_match")
        self.assertIn("отсутствует", response["message"])

    def test_listing_text_is_irrelevant_even_on_eligible_listing(self):
        self.assertEqual(read_catalog(LAB_CATALOG), read_catalog(mutated_catalog()))
        result = compare(requirements(width_cm=[["lte", 100]]))
        self.assertTrue(result["passed"])
        self.assertEqual(result["after"]["options"][0]["total_rub"], 5400)

    def test_heap_matches_exhaustive_top_three_on_ties(self):
        rng = random.Random(21)
        for _ in range(80):
            categories = ["table", "chair", "lamp"][:rng.randint(1, 3)]
            groups = [[product(f"{category}{i}", category, rng.randint(1, 9))
                       for i in range(rng.randint(1, 8))] for category in categories]
            budget = rng.randint(0, 23)
            listing_dicts = [item for group in groups for item in group]
            rng.shuffle(listing_dicts)
            response = select(read_catalog(listing_dicts),
                              requirements(categories=categories, budget_rub=budget)).response
            baseline = sorted(sum(item["price_rub"] for item in bundle)
                              for bundle in itertools.product(*groups))[:3]
            expected = [total for total in baseline if total <= budget]
            options = response.get("options", [])
            self.assertEqual([item["total_rub"] for item in options], expected)
            self.assertEqual(len({tuple(item["listing_ids"]) for item in options}), len(options))


class CliTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.status = 200
        self.model_output = {
            "city_id": "taganrog", "budget_rub": 6000, "categories": ["table"],
            "width_cm": [["lte", 100]], "adjustable_height": None, "color_temperature_k": [],
        }
        test = self

        class Gateway(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                test.calls.append((self.path, self.headers, body))
                content = json.dumps({"choices": [{"finish_reason": "stop",
                                                    "message": {"content": json.dumps(test.model_output)}}]})
                self.send_response(test.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(content.encode())

            def log_message(self, *_args):
                pass

        with patch("socket.getfqdn", return_value="localhost"):
            self.server = ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
        self.worker = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .05})
        self.worker.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.worker.join)
        self.addCleanup(self.server.shutdown)

    def invoke(self, catalog=CATALOG, text="table please", messages=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "random-name.json"
            path.write_text(json.dumps(catalog), encoding="utf-8")
            config = os.environ | {
                "LLM_API_KEY": "stub-only", "LLM_MODEL": "stub-model", "LLM_USER_AGENT": "team-test/1.0",
                "LLM_BASE_URL": f"http://127.0.0.1:{self.server.server_port}/v1",
            }
            return subprocess.run(cli_command(ROOT / "run", "--catalog", path), cwd=directory,
                                  input=json.dumps({"messages": messages or [{"role": "user", "text": text}]}),
                                  text=True, encoding="utf-8", capture_output=True, timeout=10, env=config)

    def test_end_to_end_uses_current_catalog_and_environment(self):
        catalog = [dict(CATALOG[0], title="private-catalog-text", description="SELLER_INSTRUCTION"),
                   dict(CATALOG[1], id="opaque-新-100")]
        done = self.invoke(catalog)
        self.assertEqual((done.returncode, done.stderr), (0, ""))
        response = json.loads(done.stdout)
        self.assertEqual([option["listing_ids"] for option in response["options"]], [["t80"], ["opaque-新-100"]])
        path, headers, raw = self.calls[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer stub-only")
        self.assertEqual(headers["User-Agent"], "team-test/1.0")
        self.assertEqual(json.loads(raw)["model"], "stub-model")
        self.assertNotIn(b"private-catalog-text", raw)
        self.assertNotIn(b"SELLER_INSTRUCTION", raw)

    def test_gateway_error_is_technical_failure(self):
        self.status = 403
        done = self.invoke()
        self.assertNotEqual(done.returncode, 0)
        self.assertEqual(done.stdout, "")
        self.assertIn("HTTP 403", done.stderr)
        self.assertNotIn("stub-only", done.stderr)

    def test_invalid_model_output_is_technical_failure(self):
        self.model_output["city_id"] = []
        done = self.invoke()
        self.assertNotEqual(done.returncode, 0)
        self.assertEqual(done.stdout, "")

    def test_single_flat_comparison_is_repaired_without_changing_meaning(self):
        self.model_output["width_cm"] = ["lte", 100]
        done = self.invoke()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout)["options"][0]["listing_ids"], ["t80"])

    def test_supported_category_anchor_does_not_hide_unsupported_product(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(json.dumps(CATALOG), encoding="utf-8")
            config = os.environ | {
                "LLM_API_KEY": "stub-only", "LLM_MODEL": "stub-model", "LLM_USER_AGENT": "team-test/1.0",
                "LLM_BASE_URL": f"http://127.0.0.1:{self.server.server_port}/v1",
            }
            done = subprocess.run(
                cli_command(ROOT / "run", "--catalog", path), cwd=directory,
                input='{"messages":[{"role":"user","text":"Нужны стол и диван в Таганроге до 6000 ₽"}]}',
                text=True, encoding="utf-8", capture_output=True, timeout=10, env=config,
            )
        self.assertEqual(done.returncode, 0, done.stderr)
        response = json.loads(done.stdout)
        self.assertEqual(response["status"], "clarify")
        self.assertEqual([question["field"] for question in response["questions"]], ["categories"])

    def test_invalid_catalog_fails_before_network(self):
        done = self.invoke([CATALOG[0], CATALOG[0]])
        self.assertNotEqual(done.returncode, 0)
        self.assertEqual(done.stdout, "")
        self.assertFalse(self.calls)


if __name__ == "__main__":
    unittest.main()
