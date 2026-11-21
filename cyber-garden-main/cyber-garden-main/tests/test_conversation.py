import unittest
from unittest.mock import Mock

from buyer.contracts import read_requirements
from buyer.conversation import EMPTY, ConversationManager
from buyer.model import ModelQuery
from buyer.query_parser import Preference


def dialog(*turns):
    return [{"role": "user" if index % 2 == 0 else "assistant", "text": text}
            for index, text in enumerate(turns)]


def expected(**fields):
    return read_requirements(EMPTY | fields)


# Expected states are authored independently of both parsers.
CASES = [
    ("Три категории и синонимы", dialog(
        "Компьютерный стол не больше метра, удобный стул и настольный светильник. "
        "Таганрог. На всё 15 000 рублей."),
     expected(city_id="taganrog", budget_rub=15000, categories=["table", "chair", "lamp"],
              width_cm=[["lte", 100]])),
    ("Десятичные метры и сокращение суммы", dialog("Стол уже 1,05 м. Азов. Бюджет 12,5 тыс."),
     expected(city_id="azov", budget_rub=12500, categories=["table"], width_cm=[["lt", 105]])),
    ("Миллиметры и две границы", dialog("Нужен стол от 850 до 1050 мм в Таганроге до 5000 ₽."),
     expected(city_id="taganrog", budget_rub=5000, categories=["table"],
              width_cm=[["gte", 85], ["lte", 105]])),
    ("Отсутствие регулировки", dialog("Стул без регулировки по высоте в Азове. Полторы тысячи."),
     expected(city_id="azov", budget_rub=1500, categories=["chair"], adjustable_height=False)),
    ("Только мягкая характеристика", dialog("Азов. Стол, желательно до 90 см. Бюджет 6500."),
     expected(city_id="azov", budget_rub=6500, categories=["table"])),
    ("Кельвины и сокращённый бюджет", dialog("Лампа 3000 К. Ростов-на-Дону. До 4к."),
     expected(city_id="rostov-na-donu", budget_rub=4000, categories=["lamp"],
              color_temperature_k=[["eq", 3000]])),
    ("Короткий ответ на вопрос", dialog("Стол и лампа. В Азове.",
           "Какой бюджет на весь комплект?", "Две тысячи."),
     expected(city_id="azov", budget_rub=2000, categories=["table", "lamp"])),
    ("Замена только верхней границы", dialog("Стол от 80 до 100 см в Азове, бюджет 5000 ₽.",
           "Нашёл комплект.", "Стол теперь не шире 1,2 м."),
     expected(city_id="azov", budget_rub=5000, categories=["table"],
              width_cm=[["gte", 80], ["lte", 120]])),
    ("Снятие ограничения и удаление товара", dialog(
           "Стол не шире метра, кресло и лампа в Азове. Бюджет 8000 ₽.",
           "Нашёл комплект.", "Ширина не важна. Без кресла и лампы."),
     expected(city_id="azov", budget_rub=8000, categories=["table"])),
    ("Изменить город и бюджет", dialog("Стол до 100 см и кресло. Азов. Бюджет 6к.",
           "Стол и кресло за 4200 ₽.", "Теперь в Таганроге, бюджет 7000."),
     expected(city_id="taganrog", budget_rub=7000, categories=["table", "chair"], width_cm=[["lte", 100]])),
    ("Неоднозначный город", dialog("Лампа. До 3000 рублей. Таганрог или Азов."),
     expected(budget_rub=3000, categories=["lamp"])),
    ("Уточнение состава после альтернативы", dialog("Стол или кресло в Азове до 4000 ₽.",
           "Какой состав нужен?", "Только стол."),
     expected(city_id="azov", budget_rub=4000, categories=["table"])),
    ("Относительный бюджет", dialog("Лампа в Азове, бюджет 10 тысяч.",
           "Подобрал лампу.", "На две тысячи меньше."),
     expected(city_id="azov", budget_rub=8000, categories=["lamp"])),
    ("Неподдерживаемый товар", dialog("Стол и диван в Таганроге, бюджет 9000."),
     expected(city_id="taganrog", budget_rub=9000)),
    ("Строго меньше по цене", dialog("Кресло в Азове, бюджет строго меньше 3500 рублей."),
     expected(city_id="azov", budget_rub=3499, categories=["chair"])),
]


class ConversationTest(unittest.TestCase):
    def setUp(self):
        self.manager = ConversationManager()

    def test_live_city_and_maximum_budget_edit_preserves_composition(self):
        messages = dialog("Стол до 100 см и кресло, Таганрог, до 5 тыс.",
                          "Подобрал вариант за 4200 ₽.", "Лучше в Азове и максимум 3200.")
        gold = expected(city_id="azov", budget_rub=3200, categories=["table", "chair"], width_cm=[["lte", 100]])
        gemma = Mock(return_value=(ModelQuery(gold, ()), object()))
        state, _ = self.manager.resolve(messages, gemma, require_model=True)
        self.assertEqual(state.requirements, gold)
        self.assertFalse(state.blocked_categories)

    def test_live_comparison_after_and_is_not_an_extra_item(self):
        messages = dialog("Стол не шире 100 см и шире 120 см. Таганрог. Бюджет 15000 рублей.")
        gold = expected(city_id="taganrog", budget_rub=15000, categories=["table"],
                        width_cm=[["gt", 120], ["lte", 100]])
        gemma = Mock(return_value=(ModelQuery(gold, ()), object()))
        query, _ = self.manager.resolve(messages, gemma, require_model=True)
        self.assertEqual(query.requirements, gold)
        self.assertFalse(query.blocked_categories)

    def test_local_corpus(self):
        for label, messages, gold in CASES:
            with self.subTest(label=label):
                state = self.manager.local(messages)
                if label == "Относительный бюджет":
                    self.assertEqual(state.unresolved, {"budget_rub"})
                    self.assertIsNone(state.requirements.budget_rub)
                else:
                    self.assertEqual(state.requirements, gold)

    def test_local_fast_path_and_cli_model_policy(self):
        _, messages, gold = CASES[0]
        gemma = Mock(return_value=(ModelQuery(expected(city_id="azov", budget_rub=1), ()), object()))
        query, trace = self.manager.resolve(messages, gemma, require_model=False)
        self.assertEqual(query.requirements, gold)
        self.assertIsNone(trace)
        gemma.assert_not_called()
        query, trace = self.manager.resolve(messages, gemma, require_model=True)
        self.assertEqual(query.requirements, gold)
        self.assertIsNotNone(trace)
        gemma.assert_called_once()

    def test_model_updates_only_uncertain_budget(self):
        _, messages, gold = next(case for case in CASES if case[0] == "Относительный бюджет")
        gemma = Mock(return_value=(ModelQuery(expected(budget_rub=8000), ()), object()))
        query, _ = self.manager.resolve(messages, gemma, require_model=False)
        self.assertEqual(query.requirements, gold)
        gemma.assert_called_once_with(messages, full_history=True)

    def test_preferences_stay_soft_even_if_gemma_promotes_them(self):
        messages = dialog("Азов. Стол, желательно до 90 см. Бюджет 6500.")
        gemma = Mock(return_value=(ModelQuery(expected(width_cm=[["lte", 90]]), ()), object()))
        query, _ = self.manager.resolve(messages, gemma, require_model=True)
        self.assertFalse(query.requirements.width_cm)
        self.assertIn(Preference("table", "до 90 см", "width_cm"), query.preferences)

    def test_removing_and_adding_back_category_does_not_restore_old_width(self):
        messages = dialog("Стол до 90 см и лампа в Азове. До 8000 ₽.",
                          "Подобрал комплект.", "Без стола.", "Подобрал лампу.", "Добавь стол.")
        query = self.manager.local(messages)
        self.assertEqual(query.requirements.categories, ("table", "lamp"))
        self.assertEqual(query.requirements.width_cm, ())

    def test_explicit_city_alternative_cannot_be_filled_by_gemma(self):
        messages = dialog("Кресло до 5000 рублей. Таганрог или Азов.")
        gemma = Mock(return_value=(ModelQuery(expected(city_id="azov"), ()), object()))
        query, _ = self.manager.resolve(messages, gemma, require_model=True)
        self.assertEqual(query.questions(), ("city_id",))

    def test_unresolved_width_is_asked_and_a_bare_answer_updates_only_width(self):
        messages = dialog("Стол шириной 80 см или 100 см в Азове. Бюджет 6500 ₽.")
        gemma = Mock(return_value=(ModelQuery(expected(
            city_id="azov", budget_rub=6500, categories=["table"], width_cm=[["eq", 80]]
        ), ()), object()))
        query, _ = self.manager.resolve(messages, gemma, require_model=False)
        self.assertEqual(query.requirements.width_cm, ())
        question, = query.attribute_questions()
        answered = messages + [{"role": "assistant", "text": question}, {"role": "user", "text": "100"}]
        query, _ = self.manager.resolve(answered, gemma, require_model=False)
        self.assertEqual(query.requirements, expected(city_id="azov", budget_rub=6500,
                                                       categories=["table"], width_cm=[["eq", 100]]))

    def test_new_city_outside_dictionary_does_not_keep_old_city(self):
        messages = dialog("Стол в Азове. Бюджет 5000.", "Подобрал стол.", "Я сейчас в Краснодаре.")
        gemma = Mock(return_value=(ModelQuery(expected(), ()), object()))
        query, _ = self.manager.resolve(messages, gemma, require_model=False)
        self.assertEqual(query.requirements, expected(budget_rub=5000, categories=["table"]))
        self.assertEqual(query.questions(), ("city_id",))

    def test_negating_category_with_pronoun_preserves_other_categories(self):
        messages = dialog("Стол и кресло в Азове. Бюджет 8000.", "Подобрал комплект.",
                          "Стол мне не нужен.")
        self.assertEqual(self.manager.local(messages).requirements,
                         expected(city_id="azov", budget_rub=8000, categories=["chair"]))

    def test_preferred_budget_does_not_replace_existing_hard_maximum(self):
        messages = dialog("Лампа в Азове. Бюджет 7000 ₽.", "Подобрал лампу.",
                          "Желательно уложиться в 5000 ₽.")
        gemma = Mock(return_value=(ModelQuery(expected(budget_rub=5000), ()), object()))
        query, _ = self.manager.resolve(messages, gemma, require_model=True)
        self.assertEqual(query.requirements.budget_rub, 7000)
        self.assertIn(Preference(None, "желательно уложиться в 5000 ₽", "budget_rub"), query.preferences)
