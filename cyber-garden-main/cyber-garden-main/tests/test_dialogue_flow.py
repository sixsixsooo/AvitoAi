"""Regression chains through resolve, actual CLI and actual Web API."""
import json
import io
import sys
import time
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock, patch

from jsonschema import Draft202012Validator
from buyer import demo
from buyer.contracts import ROOT, read_requirements
from buyer.conversation import ConversationManager
from buyer.indexed import SearchError
from buyer.model import ModelQuery, ModelTrace
from tests.cli_support import cli_command
from tests.fixtures import product
from tests.test_conversation import dialog

BASE = "Стол не шире метра, кресло и лампа. Таганрог. До 15 000 ₽ за всё."
ROWS = [product("t80", "table", 2000, 80), product("t100", "table", 3000, 100),
        product("c", "chair", 1000, True), product("l", "lamp", 500, 3000)]
FACTS = dict(city_id="taganrog", budget_rub=15000, categories=["table", "chair", "lamp"],
             width_cm=[["lte", 100]], adjustable_height=None, color_temperature_k=[])


class DialogueFlowTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name, "catalog.json")
        self.path.write_text(json.dumps(ROWS), encoding="utf-8")
        self.reply = FACTS | {"preferences": []}
        self.calls = []
        self.gateway_delay = 0
        owner = self

        class Gateway(BaseHTTPRequestHandler):
            def do_POST(self):
                owner.calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                raw = json.dumps({"model": "offline-gemma", "choices": [{"finish_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps(owner.reply)}}]}).encode()
                time.sleep(owner.gateway_delay)
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *_):
                pass

        with patch("socket.getfqdn", return_value="localhost"):
            self.gateway = ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
            self.web = demo.DemoServer(0, self.path)
        for server in (self.gateway, self.web):
            worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
            worker.start()
            self.addCleanup(server.server_close)
            self.addCleanup(worker.join)
            self.addCleanup(server.shutdown)
        config = {"LLM_BASE_URL": f"http://127.0.0.1:{self.gateway.server_port}/v1",
                  "LLM_MODEL": "offline-gemma", "LLM_API_KEY": "offline-dialogue-key",
                  "LLM_USER_AGENT": "offline-dialogue/1.0"}
        self.env = os.environ | config
        setting = patch.dict(os.environ, config)
        setting.start()
        self.addCleanup(setting.stop)
        self.validator = Draft202012Validator(json.loads((ROOT / "schemas/response.schema.json").read_bytes()))

    def resolve(self, messages):
        fields = {k: v for k, v in self.reply.items() if k not in {"preferences", "intent"}}
        gemma = Mock(return_value=(ModelQuery(read_requirements(fields), (), self.reply.get("intent")), ModelTrace("offline", 1, 1, 1)))
        state, _ = ConversationManager().resolve(messages, gemma, require_model=True)
        gemma.assert_called_once()
        return state

    def post(self, messages):
        connection = HTTPConnection("127.0.0.1", self.web.server_port, timeout=10)
        try:
            connection.request("POST", "/api/buy", json.dumps({"messages": messages}).encode(),
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            self.last_headers = dict(response.getheaders())
            body = json.loads(response.read())
            self.assertEqual(response.status, 200, body)
            if body["response"] is not None:
                self.validator.validate(body["response"])
            return body
        finally:
            connection.close()

    def cli(self, messages, rows=ROWS):
        self.path.write_text(json.dumps(rows), encoding="utf-8")
        done = subprocess.run(cli_command(ROOT / "run", "--catalog", self.path),
                              input=json.dumps({"messages": messages}), cwd=ROOT, env=self.env,
                              text=True, encoding="utf-8", capture_output=True, timeout=12)
        self.assertEqual((done.returncode, done.stderr), (0, ""))
        self.assertEqual(len(done.stdout.strip().splitlines()), 1)
        response = json.loads(done.stdout)
        self.validator.validate(response)
        return response

    def assert_options(self, response, categories, budget=15000, city="taganrog"):
        self.assertEqual(response["status"], "ok")
        by_id = {r["id"]: r for r in ROWS}
        for option in response["options"]:
            records = [by_id[i] for i in option["listing_ids"]]
            self.assertEqual({r["category"] for r in records}, set(categories))
            self.assertTrue(all(r["city_id"] == city for r in records))
            self.assertEqual(sum(r["price_rub"] for r in records), option["total_rub"])
            self.assertLessEqual(option["total_rub"], budget)

    def test_off_topic_preserves_state_and_resume_works_through_web(self):
        first = self.post(dialog(BASE))
        self.assert_options(first["response"], FACTS["categories"])
        history = dialog(BASE, "Подобрал комплект.", "рецепт гуляша")
        self.assertEqual(self.resolve(history).requirements, read_requirements(FACTS))
        aside = self.post(history)
        self.assertEqual(aside["response"]["status"], "clarify")
        self.assertEqual([q["field"] for q in aside["response"]["questions"]], ["categories"])
        self.assertNotIn("options", aside["response"])
        self.assertEqual(aside["items"], {})
        self.assertNotIn("гуляш", json.dumps(aside["preferences"], ensure_ascii=False))
        history += [{"role": "assistant", "text": aside["response"]["questions"][0]["text"]},
                    {"role": "user", "text": "нужен стол"}]
        resumed = self.post(history)
        self.assert_options(resumed["response"], FACTS["categories"])
        self.assertEqual(len(self.calls), 3)
        sent = [m["content"] for m in self.calls[-1]["messages"] if m["role"] == "user"]
        self.assertEqual(sent, [BASE, "рецепт гуляша", "нужен стол"])

    def test_model_semantics_preserve_state_for_unlisted_off_topic_language(self):
        messages = dialog(BASE, "Подобрал комплект.", "хочу узнать историю города")
        model = Mock(return_value=(ModelQuery(read_requirements(FACTS), (), "off_topic"), object()))
        state, _ = ConversationManager().resolve(messages, model, require_model=True)
        self.assertEqual(state.requirements, read_requirements(FACTS))
        self.assertTrue(state.decision_questions())
        self.reply["intent"] = "off_topic"
        response = self.post(messages)
        self.assertEqual(response["response"]["status"], "clarify")
        self.assertEqual(response["items"], {})

    def test_off_topic_preserves_budget_and_wishes_resolved_only_by_gemma(self):
        from buyer.query_parser import Preference
        messages = dialog("Красный стол в Таганроге. Бюджет два с половиной тысячи.",
                          "Подобрал стол.", "рецепт гуляша")
        expected = read_requirements(FACTS | {"budget_rub": 2500, "categories": ["table"], "width_cm": []})
        wish = Preference("table", "Красный")
        model = Mock(return_value=(ModelQuery(expected, (wish,), "off_topic"), object()))
        state, _ = ConversationManager().resolve(messages, model, require_model=True)
        self.assertEqual(state.requirements, expected)
        self.assertIn(wish, state.preferences)
        self.assertTrue(state.decision_questions())

    def test_off_topic_through_real_cli(self):
        response = self.cli(dialog(BASE, "Подобрал комплект.", "рецепт гуляша"))
        self.assertEqual(response["status"], "clarify")
        self.assertNotIn("options", response)

    def test_measurement_change_reassembles_current_bundle(self):
        messages = dialog(BASE, "Подобрал комплект.", "нужен стол 100см таганрог")
        state = self.resolve(messages)
        self.assertEqual(state.requirements.budget_rub, 15000)
        self.assertEqual(state.requirements.categories, ("table", "chair", "lamp"))
        self.assertEqual([(c.operation, c.value) for c in state.requirements.width_cm], [("eq", 100)])
        response = self.cli(messages)
        self.assert_options(response, FACTS["categories"])
        self.assertEqual(response["options"][0]["listing_ids"], ["t100", "c", "l"])

    def test_table_alias_and_accessory_do_not_replace_composition(self):
        for text in ("столик Таганрог", "стол с ящиком", "стол и кресло"):
            with self.subTest(text=text):
                messages = dialog(BASE, "Подобрал комплект.", text)
                self.assertEqual(self.resolve(messages).requirements.categories, ("table", "chair", "lamp"))
                self.assert_options(self.cli(messages), FACTS["categories"])

    def test_bare_item_price_is_not_total_budget(self):
        for text in ("стул 1400", "стул таганрог 1400", "стол 1400"):
            with self.subTest(text=text):
                messages = dialog(BASE, "Подобрал комплект.", text)
                state = self.resolve(messages)
                self.assertEqual(state.requirements.budget_rub, 15000)
                response = self.cli(messages)
                self.assertEqual(response["status"], "clarify")
                self.assertIn("budget_rub", [q["field"] for q in response["questions"]])
                self.assertNotIn("options", response)

    def test_composition_edits_and_clear_width(self):
        for text, categories, width in (("оставь только стол", ("table",), True),
                                        ("убери кресло", ("table", "lamp"), True),
                                        ("неважно, какая ширина", ("table", "chair", "lamp"), False)):
            with self.subTest(text=text):
                messages = dialog(BASE, "Подобрал комплект.", text)
                state = self.resolve(messages)
                self.assertEqual(state.requirements.categories, categories)
                self.assertEqual(bool(state.requirements.width_cm), width)
                self.assert_options(self.cli(messages), categories)

    def test_add_category_budget_and_city_edits(self):
        first = "Только стол. Таганрог. Бюджет 15000 рублей."
        for text, fields in (("добавь лампу", {"categories": ("table", "lamp")}),
                             ("бюджет теперь 12000", {"budget_rub": 12000}),
                             ("город теперь Азов", {"city_id": "azov"})):
            with self.subTest(text=text):
                state = self.resolve(dialog(first, "Подобрал стол.", text))
                for key, value in fields.items():
                    self.assertEqual(getattr(state.requirements, key), value)

    def test_cancel_last_preference_does_not_resurrect_from_model(self):
        from buyer.query_parser import Preference
        from buyer.preferences import grounded
        messages = dialog("Деревянный стол в Таганроге. Бюджет 15000 рублей.",
                          "Подобрал стол.", "отмени предыдущее пожелание")
        model = Mock(return_value=(ModelQuery(read_requirements(FACTS),
                                  (Preference("table", "деревянный"),)), object()))
        state, _ = ConversationManager().resolve(messages, model, require_model=True)
        self.assertEqual(state.preferences, ())
        self.assertEqual(grounded((Preference("table", "деревянный"),), messages), ())

    def test_new_dialogue_resets_even_if_model_repeats_old_state(self):
        messages = dialog(BASE, "Подобрал комплект.", "новый диалог")
        state = self.resolve(messages)
        self.assertIsNone(state.requirements.city_id)
        self.assertIsNone(state.requirements.budget_rub)
        self.assertEqual(state.requirements.categories, ())
        response = self.cli(messages)
        self.assertEqual(response["status"], "clarify")
        self.assertEqual({q["field"] for q in response["questions"]}, {"city_id", "budget_rub", "categories"})
        self.assertEqual([m["content"] for m in self.calls[-1]["messages"] if m["role"] == "user"], ["новый диалог"])

    def test_missing_supported_category_before_budget_question(self):
        self.reply = FACTS | {"categories": ["lamp"], "budget_rub": None, "width_cm": [], "preferences": []}
        response = self.cli(dialog("Нужна лампа в Таганроге"), [ROWS[0]])
        self.assertEqual(response["status"], "no_match")
        self.assertIn("Лампа", response["message"])
        self.assertNotIn("options", response)

    def test_missing_in_city_and_hard_attribute_before_budget_question(self):
        self.reply = FACTS | {"categories": ["table"], "budget_rub": None, "width_cm": [], "preferences": []}
        for rows, text in (([product("az", "table", 1000, 80, "azov")], "Стол в Таганроге"),
                           ([product("wide", "table", 1000, 100)], "Стол не шире 90 см в Таганроге")):
            with self.subTest(text=text):
                response = self.cli(dialog(text), rows)
                self.assertEqual(response["status"], "no_match")
                self.assertNotIn("options", response)

    def test_full_bundle_is_never_partial_or_over_budget(self):
        scenarios = [([product("wide", "table", 1000, 110), *ROWS[2:]], BASE),
                     (ROWS, BASE.replace("15 000", "100")),
                     (ROWS[:3], BASE)]
        for rows, text in scenarios:
            with self.subTest(text=text, ids=[r["id"] for r in rows]):
                response = self.cli(dialog(text), rows)
                self.assertEqual(response["status"], "no_match")
                self.assertNotIn("options", response)

    def test_unsupported_item_is_clarify_not_catalog_absence(self):
        response = self.cli(dialog("Стол и принтер в Таганроге, бюджет 15000 рублей"))
        self.assertEqual(response["status"], "clarify")
        self.assertIn("categories", [q["field"] for q in response["questions"]])

    def test_search_error_is_technical_not_no_match(self):
        self.web.provider = Mock()
        self.web.provider.search.side_effect = SearchError("offline search unavailable")
        c = HTTPConnection("127.0.0.1", self.web.server_port, timeout=5)
        try:
            c.request("POST", "/api/buy", json.dumps({"messages": dialog(BASE)}),
                      {"Content-Type": "application/json"})
            response = c.getresponse()
            self.assertEqual(response.status, 503)
            body = json.loads(response.read())
            self.assertIn("error", body)
            self.assertNotIn("response", body)
        finally:
            c.close()

    def test_polish_off_topic_wording_preserves_context(self):
        for text in ("Скажи рецепт блинчиков", "Посчитай теорему Пифагора", "Расскажи анекдот", "Какая сегодня погода?"):
            with self.subTest(text=text):
                self.reply["intent"] = "off_topic"
                messages = dialog(BASE, "Подобрал комплект.", text)
                self.assertEqual(self.resolve(messages).requirements, read_requirements(FACTS))
                body = self.post(messages)
                self.assertEqual(body["response"]["status"], "clarify")
                wording = " ".join(q["text"] for q in body["response"]["questions"])
                self.assertIn("не отвечаю на вопросы по другим темам", wording)
                self.assertNotIn("options", body["response"])
                self.assertEqual(body["items"], {})

    def test_polish_unsupported_goods_have_safe_specific_wording(self):
        for text, name, removal in (("Найди холодильник", "Холодильники", "Холодильник больше не нужен"),
                                    ("Нужен диван", "Диваны", "Диван больше не нужен"),
                                    ("Хочу купить шкаф", "Шкафы", "Шкаф больше не нужен")):
            with self.subTest(text=text):
                messages = dialog(BASE, "Подобрал комплект.", text)
                self.reply["intent"] = "shopping"
                body = self.post(messages)
                self.assertEqual(body["response"]["status"], "clarify")
                wording = " ".join(q["text"] for q in body["response"]["questions"])
                self.assertIn(name, wording)
                self.assertIn("нет категории", wording)
                self.assertEqual(body["items"], {})
                restored = self.resolve(messages + [{"role": "assistant", "text": wording},
                                                    {"role": "user", "text": removal}])
                self.assertEqual(restored.requirements, read_requirements(FACTS))

    def test_polish_unrecognized_word_does_not_invent_a_product_name(self):
        body = self.post(dialog(BASE, "Подобрал комплект.", "Найди абракадабру"))
        self.assertEqual(body["response"]["status"], "clarify")
        self.assertNotIn("абракадабр", json.dumps(body["response"], ensure_ascii=False).lower())
        self.assertEqual(body["items"], {})

    def test_polish_supported_absent_lamp_still_no_match(self):
        self.web.listings = tuple(item for item in self.web.listings if item.category != "lamp")
        self.reply = FACTS | {"categories": ["lamp"], "budget_rub": None, "width_cm": [], "preferences": []}
        body = self.post(dialog("Нужна лампа в Таганроге"))
        self.assertEqual(body["response"]["status"], "no_match")
        self.assertNotIn("нет категории", body["response"]["message"])

    def test_stage_timings_exclude_gemma_from_search_and_log_no_input(self):
        self.gateway_delay = .08
        original_select = demo.select
        def delayed_select(*args, **kwargs):
            time.sleep(.015)
            return original_select(*args, **kwargs)
        logs = io.StringIO()
        marker = "private-user-utterance-marker"
        with patch.object(demo, "select", side_effect=delayed_select), patch.object(sys, "stderr", logs):
            body = self.post(dialog(BASE + " " + marker))
        self.assertEqual(body["response"]["status"], "ok")
        header = self.last_headers["Server-Timing"]
        timings = {part.strip().split(";dur=")[0] + "_ms": float(part.split(";dur=")[1])
                   for part in header.split(",")}
        self.assertGreaterEqual(timings["gemma_ms"], 60)
        self.assertGreaterEqual(timings["search_ms"], 10)
        self.assertLess(timings["search_ms"], timings["gemma_ms"])
        self.assertGreaterEqual(timings["resolve_ms"] + 1, timings["gemma_ms"])
        self.assertGreaterEqual(timings["total_ms"] + 1, timings["resolve_ms"] + timings["search_ms"])
        entries = [json.loads(line) for line in logs.getvalue().splitlines() if line.startswith('{')]
        self.assertEqual(len(entries), 1)
        self.assertEqual(set(entries[0]), {"event", "status", "gemma_ms", "resolve_ms", "search_ms", "total_ms"})
        for private in (marker, "Authorization", "offline-dialogue-key"):
            self.assertNotIn(private, logs.getvalue())
