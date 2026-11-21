import unittest

from buyer.conversation import ConversationManager
from buyer.model import ModelQuery
from buyer.query_parser import Preference
from buyer.contracts import read_catalog, read_requirements
from buyer.selection import select
from tests.fixtures import CATALOG


def history():
    return [
        {"role": "user", "text": "Стол не шире метра, кресло и лампа. Таганрог. До 100 ₽ за всё."},
        {"role": "assistant", "text": "Самый доступный полный комплект стоит 4 900 ₽."},
        {"role": "user", "text": "а что насчет стола для груминга собак в таганроге до 15к"},
    ]


class ProductSwitchTest(unittest.TestCase):
    def test_unknown_category_is_explicitly_unavailable(self):
        state = ConversationManager().local([
            {"role": "user", "text": "Хочу велосипед в Таганроге до 10000 рублей"}])
        self.assertTrue(state.blocked_categories)
        self.assertIn("категории нет", state.decision_questions()[0]["text"])

    def test_million_ruble_budget(self):
        for amount in ("2 миллиона рублей", "два миллиона рублей", "2 млн"):
            with self.subTest(amount=amount):
                state = ConversationManager().local([
                    {"role": "user", "text": f"Стол в Таганроге, бюджет {amount}"}])
                self.assertEqual(state.requirements.budget_rub, 2_000_000)

    def test_new_product_replaces_bundle_and_old_width(self):
        manager = ConversationManager()
        state = manager.local(history())
        self.assertEqual(state.requirements.categories, ("table",))
        self.assertEqual(state.requirements.width_cm, ())
        self.assertEqual(state.requirements.budget_rub, 15000)
        self.assertEqual(state.requirements.city_id, "taganrog")
        response = select(read_catalog(CATALOG), state.requirements).response
        self.assertEqual(response["status"], "ok")
        self.assertTrue(all(len(option["listing_ids"]) == 1 for option in response["options"]))

    def test_model_cannot_restore_old_bundle(self):
        manager = ConversationManager()
        stale = read_requirements({"city_id":"taganrog", "budget_rub":15000,
            "categories":["table","chair","lamp"], "width_cm":[["lte",100]],
            "adjustable_height":None, "color_temperature_k":[]})
        result = manager.merge(manager.local(history()), ModelQuery(stale,
            (Preference("table", "для груминга собак"),), "shopping"), history())
        self.assertEqual(result.requirements.categories, ("table",))
        self.assertEqual(result.requirements.width_cm, ())
        self.assertTrue(any("груминга" in pref.text for pref in result.preferences))

    def test_rejection_requests_clarification_instead_of_repeating_cards(self):
        state = ConversationManager().local(history() + [
            {"role": "assistant", "text": "Подобрал комплект."},
            {"role": "user", "text": "нахера мне комплект"},
        ])
        self.assertEqual(state.decision_questions()[0]["field"], "categories")

    def test_explicit_addition_still_keeps_existing_product(self):
        state = ConversationManager().local([
            {"role":"user", "text":"Стол в Таганроге, бюджет 15000 рублей."},
            {"role":"assistant", "text":"Показал стол."},
            {"role":"user", "text":"Добавь кресло."},
        ])
        self.assertEqual(state.requirements.categories, ("table", "chair"))


if __name__ == "__main__":
    unittest.main()
