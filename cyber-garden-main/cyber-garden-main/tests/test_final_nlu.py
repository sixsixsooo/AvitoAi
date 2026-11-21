"""Regression cases written before the final targeted audit fixes."""
from copy import deepcopy
import unittest
from unittest.mock import Mock

from buyer.anchors import categories_from_dialog
from buyer.conversation import ConversationManager, EMPTY
from buyer.model import ModelQuery
from tests.test_buyer import requirements
from tests.test_conversation import dialog


class FinalNluTest(unittest.TestCase):
    def setUp(self):
        self.manager = ConversationManager()

    def test_live_unscoped_number_cannot_become_a_model_width(self):
        messages = dialog("Стол. До 100. Таганрог.")
        inferred = requirements(budget_rub=None, width_cm=[["lte", 100]])
        gemma = Mock(return_value=(ModelQuery(inferred, ()), object()))
        query, _ = self.manager.resolve(messages, gemma, require_model=True)
        self.assertEqual(query.requirements.width_cm, ())
        self.assertIsNone(query.requirements.budget_rub)
        self.assertEqual(query.requirements.categories, ("table",))
        self.assertIn("width_cm", query.unresolved)
        self.assertTrue(query.attribute_questions())

    def resolve(self, messages, **fields):
        model = Mock(return_value=(ModelQuery(requirements(**fields), ()), object()))
        state, _ = self.manager.resolve(messages, model, require_model=False)
        return state, model

    def test_unsupported_items_do_not_confirm_partial_composition(self):
        for text in ("Стол и диван в Таганроге", "Стол и принтер в Таганроге",
                     "Стол и холодильник в Таганроге", "Кресло и велосипед в Таганроге",
                     "Стол и сканер в Таганроге"):
            with self.subTest(text=text):
                messages = dialog(text)
                self.assertIsNone(categories_from_dialog(messages).categories)
                local = self.manager.local(messages)
                self.assertNotIn("categories", local.confirmed)
                self.assertTrue(local.needs_model)
                state, model = self.resolve(messages, categories=None)
                model.assert_called_once()
                self.assertIn("categories", state.questions())

    def test_unitless_measurement_is_not_a_budget_or_confirmed_bound(self):
        for text, attr in (("Стол шириной до 100", "width_cm"),
                           ("Стол шириной от 80 до 100", "width_cm"),
                           ("Лампа. Температура света до 3000", "color_temperature_k")):
            with self.subTest(text=text):
                messages = dialog(text)
                local = self.manager.local(messages)
                self.assertIsNone(local.requirements.budget_rub)
                self.assertNotIn("budget_rub", local.confirmed)
                self.assertIn(attr, local.unresolved)
                self.assertTrue(local.attribute_questions())
                state, model = self.resolve(messages, budget_rub=15000,
                                            categories=["lamp"] if attr != "width_cm" else ["table"])
                model.assert_called_once()
                self.assertEqual(state.requirements.budget_rub, 15000)

    def test_naked_do_number_requires_context_but_explicit_units_work(self):
        cases = [
            ("Стол. До 100", None, ()),
            ("Стол до 100 см", None, (("lte", 100),)),
            ("Стол шириной до 100 см. Бюджет 15000 рублей", 15000, (("lte", 100),)),
            ("Ширина от 80 до 100 см, бюджет 15000 рублей", 15000, (("gte", 80), ("lte", 100))),
        ]
        for text, budget, width in cases:
            with self.subTest(text=text):
                messages = dialog("Стол в Таганроге.", "Какие требования?", text)
                local = self.manager.local(messages)
                self.assertEqual(local.requirements.budget_rub, budget)
                self.assertEqual(tuple((rule.operation, rule.value) for rule in local.requirements.width_cm), width)
                self.assertEqual(local.requirements.categories, ("table",))

    def test_budget_forms_and_model_fallback_preserve_context(self):
        cases = [("25 тысяч", 25000, 25000), ("три тысячи", 3000, 3000),
                 ("полторы тысячи", 1500, 1500), ("два с половиной тысячи", None, 2500),
                 ("тысячи три", None, 3000), ("15 тысяч", 15000, 15000)]
        for text, local_amount, correct in cases:
            with self.subTest(text=text):
                messages = dialog("Стол в Азове.", "Какой бюджет на весь комплект?", text)
                local = self.manager.local(messages)
                self.assertEqual(local.requirements.budget_rub, local_amount)
                state, model = self.resolve(messages, city_id="azov", budget_rub=correct)
                self.assertEqual(state.requirements.budget_rub, correct)
                self.assertEqual(state.requirements.city_id, "azov")
                self.assertEqual(state.requirements.categories, ("table",))
                self.assertEqual(model.call_count, int(local_amount is None))
                unprompted = dialog("Стол в Азове.", "Какие ещё условия?", text)
                state, model = self.resolve(unprompted, city_id="azov", budget_rub=correct)
                self.assertEqual(state.requirements.budget_rub, correct)
                self.assertEqual(model.call_count, int(local_amount is None))
        messages = dialog("Стол в Азове.", "Какой бюджет?", "Шириной до 100")
        state = self.manager.local(messages)
        self.assertIsNone(state.requirements.budget_rub)
        self.assertIn("width_cm", state.unresolved)

    def test_explicit_width_survives_even_before_category_is_known(self):
        state = self.manager.local(dialog("Ширина от 80 до 100 см, бюджет 15000 рублей"))
        self.assertEqual(state.requirements.budget_rub, 15000)
        self.assertEqual(state.requirements.categories, ())
        self.assertEqual(tuple((rule.operation, rule.value) for rule in state.requirements.width_cm),
                         (("gte", 80), ("lte", 100)))

    def test_unsupported_item_removal_restores_remembered_categories(self):
        for noun in ("Диван", "Принтер", "Холодильник", "Сканер"):
            with self.subTest(noun=noun):
                base = dialog("Стол в Азове, бюджет 5000", "Подобрал стол.", "Добавь " + noun)
                local = self.manager.local(base)
                self.assertIn("categories", local.questions())
                resolved = base + [{"role": "assistant", "text": "Уточните состав комплекта?"},
                                   {"role": "user", "text": noun + " больше не нужен"}]
                self.assertEqual(self.manager.local(resolved).requirements.categories, ("table",))
                state, _ = self.resolve(resolved, city_id="azov", budget_rub=5000)
                self.assertEqual(state.requirements.categories, ("table",))
                self.assertEqual(state.requirements.budget_rub, 5000)
                self.assertEqual(state.requirements.city_id, "azov")
                only = base + [{"role": "assistant", "text": "Уточните состав комплекта?"},
                               {"role": "user", "text": "Только стол"}]
                self.assertEqual(self.manager.local(only).requirements.categories, ("table",))
        messages = dialog("Стол, кресло и лампа в Азове, бюджет 15000", "Подобрал комплект.", "Без кресла")
        self.assertEqual(self.manager.local(messages).requirements.categories, ("table", "lamp"))

    def test_empty_defaults_are_not_mutated_by_independent_dialogues(self):
        before = deepcopy(EMPTY)
        first = self.manager.local(dialog("Стол от 80 до 100 см и лампа 3000 К в Азове. Бюджет 15000",
                                         "Подобрал комплект.", "Ширина не важна. Без лампы.",
                                         "Подобрал стол.", "Добавь лампу. Стол до 120 см."))
        self.assertTrue(first.requirements.width_cm)
        second = self.manager.local(dialog("Стол и лампа в Таганроге. Бюджет 5000"))
        self.assertFalse(second.requirements.width_cm)
        self.assertFalse(second.requirements.color_temperature_k)
        self.assertEqual(EMPTY, before)
