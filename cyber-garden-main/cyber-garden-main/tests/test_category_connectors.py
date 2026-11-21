"""Whole-resolution and real-CLI regressions for natural composition markers."""
import unittest
from unittest.mock import Mock

from buyer.anchors import categories_from_dialog
from buyer.contracts import read_catalog
from buyer.conversation import ConversationManager
from buyer.model import ModelQuery
from buyer.preferences import category_text
from buyer.selection import select
from tests import test_final_cli as cli_tests
from tests.fixtures import product
from tests.test_buyer import requirements
from tests.test_conversation import dialog

POSITIVE = ["Стол и кресло", "Стол, а также кресло", "Стол, дополнительно кресло",
            "Стол вместе с креслом", "Стол плюс кресло", "Стол, также кресло",
            "Стол, ещё кресло", "Стол, добавь кресло"]
NEGATIVE = ["Стол, а также холодильник", "Нужен стол, дополнительно велосипед",
            "Стол вместе с принтером", "Стол, дополнительно сканер",
            "Стол также сканер", "Стол плюс сканер", "Стол, ещё сканер", "Стол, добавь сканер"]
FEATURES = ["Стол с ящиком", "Стол с металлическими ножками", "Кресло с подлокотниками",
            "Стол с регулируемой высотой"]


class CategoryConnectorTest(unittest.TestCase):
    def test_category_projection_preserves_connectors_and_separators(self):
        text = "Стол, а также холодильник; дополнительно велосипед вместе с принтером"
        self.assertEqual(category_text(text), text.lower())

    def test_resolve_unknown_items_overrides_neither_null_nor_partial_model_set(self):
        manager = ConversationManager()
        for text in NEGATIVE:
            for model_categories in (None, ["table"]):
                with self.subTest(text=text, model_categories=model_categories):
                    messages = dialog(text + ". Таганрог. Бюджет 15000 рублей.")
                    local = manager.local(messages)
                    self.assertNotIn("categories", local.confirmed)
                    self.assertIsNone(categories_from_dialog(messages).categories)
                    gemma = Mock(return_value=(ModelQuery(requirements(categories=model_categories), ()), None))
                    query, _ = manager.resolve(messages, gemma, require_model=True)
                    gemma.assert_called_once()
                    self.assertEqual(select([], query.requirements, query.preferences).response["status"], "clarify")

    def test_resolve_supported_additions_and_item_features(self):
        rows = [product("table", "table", 100, 90), product("chair", "chair", 100, True)]
        for text in POSITIVE + FEATURES:
            with self.subTest(text=text):
                expected = ["chair"] if text.startswith("Кресло") else (["table", "chair"] if text in POSITIVE else ["table"])
                gemma = Mock(return_value=(ModelQuery(requirements(categories=expected), ()), None))
                query, _ = ConversationManager().resolve(dialog(text + ". Таганрог. Бюджет 15000 рублей."),
                                                         gemma, require_model=True)
                response = select(read_catalog(rows), query.requirements, query.preferences).response
                self.assertEqual(response["status"], "ok")
                self.assertEqual(response["options"][0]["listing_ids"], ["table", "chair"] if text in POSITIVE else expected)


class CategoryConnectorCliTest(unittest.TestCase):
    setUp = cli_tests.FinalCliTest.setUp
    invoke = cli_tests.FinalCliTest.invoke
    answer = cli_tests.FinalCliTest.answer

    def test_real_run_unknown_additions_clarify(self):
        for text in NEGATIVE:
            with self.subTest(text=text):
                self.reply["categories"] = None
                response = self.answer([product("t", "table", 100)],
                                       dialog(text + ". Таганрог. Бюджет 15000 рублей."))
                self.assertEqual(response["status"], "clarify")
                self.assertIn("categories", [question["field"] for question in response["questions"]])

    def test_real_run_supported_additions_and_features(self):
        rows = [product("table", "table", 100, 90), product("chair", "chair", 100, True)]
        for text in POSITIVE + FEATURES:
            with self.subTest(text=text):
                expected = ["chair"] if text.startswith("Кресло") else (["table", "chair"] if text in POSITIVE else ["table"])
                self.reply["categories"] = expected
                response = self.answer(rows, dialog(text + ". Таганрог. Бюджет 15000 рублей."))
                self.assertEqual(response["status"], "ok")
                self.assertEqual(response["options"][0]["listing_ids"], expected)
