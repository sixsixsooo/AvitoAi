"""Frozen expected states authored before fresh real Gemma evaluation."""
from buyer.query_parser import Preference
from tests.test_conversation import CASES as BASE, dialog, expected

CASES = [(name, messages, gold,
          (Preference("table", "Компьютерный"), Preference("chair", "удобный"), Preference("lamp", "настольный"))
          if index == 0 else ((Preference("table", "до 90 см", "width_cm"),) if index == 4 else ()))
         for index, (name, messages, gold) in enumerate(BASE)]


def add(name, turns, preferences=(), **fields):
    CASES.append((name, dialog(*turns), expected(**fields), preferences))


COMMON = dict(city_id="taganrog", budget_rub=15000)
for label, sentence in (("а также неизвестный товар", "Стол, а также холодильник"),
                        ("дополнительно неизвестный товар", "Нужен стол, дополнительно велосипед"),
                        ("вместе с неизвестным товаром", "Стол вместе с принтером")):
    add(label, [sentence + ". Таганрог. Бюджет 15000 рублей."], **COMMON)
add("пожелание ящика", ["Желательно стол с ящиком. Таганрог. Бюджет 15000 рублей."],
    [Preference("table", "с ящиком")], categories=["table"], **COMMON)
add("опечатка в назначении", ["Нужен кампьютерный стол в Таганроге, бюджет 15000 рублей."],
    [Preference("table", "кампьютерный")], categories=["table"], **COMMON)
for text, amount in (("25 тысяч", 25000), ("три тысячи", 3000), ("полторы тысячи", 1500),
                     ("два с половиной тысячи", 2500), ("тысячи три", 3000)):
    add("бюджет: " + text, ["Стол в Азове.", "Какой бюджет на комплект?", text],
        city_id="azov", budget_rub=amount, categories=["table"])
add("число ширины без единицы", ["Стол шириной до 100. Таганрог. Бюджет 15000 рублей."],
    categories=["table"], **COMMON)
add("отрицание материала", ["Деревянный стол и кресло. Таганрог. Бюджет 15000 рублей.",
    "Подобрал комплект.", "Не хочу деревянный стол."], categories=["table", "chair"], **COMMON)
add("добавление совместимого пожелания", ["Деревянный стол и удобное кресло. Таганрог. Бюджет 15000 рублей.",
    "Подобрал комплект.", "Всё тот же комплект, но теперь желательно мягкое кресло."],
    [Preference("table", "деревянный"), Preference("chair", "удобное"), Preference("chair", "мягкое")],
    categories=["table", "chair"], **COMMON)
add("удаление материала ради цены", ["Деревянный стол. Таганрог. Бюджет 15000 рублей.",
    "Подобрал стол.", "Деревянный стол больше не нужен, теперь главное цена."], categories=["table"], **COMMON)
add("замена пожелания", ["Мягкое кресло. Таганрог. Бюджет 15000 рублей.", "Подобрал кресло.",
    "Вместо мягкого теперь удобное кресло."], [Preference("chair", "удобное")], categories=["chair"], **COMMON)
add("противоречивые границы", ["Стол не шире 100 см и шире 120 см. Таганрог. Бюджет 15000 рублей."],
    categories=["table"], width_cm=[["gt", 120], ["lte", 100]], **COMMON)
add("уменьшение бюджета", ["Стол в Азове, бюджет 5000 рублей.", "Подобрал стол.", "Теперь бюджет 1000 рублей."],
    city_id="azov", budget_rub=1000, categories=["table"])
add("две верхние границы", ["Стол до 90 см и до 100 см. Таганрог. Бюджет 15000 рублей."],
    categories=["table"], width_cm=[["lte", 90]], **COMMON)
add("неизвестная категория велосипед", ["Кресло и велосипед в Таганроге, бюджет 5000 рублей."],
    city_id="taganrog", budget_rub=5000)
add("голое число не бюджет", ["Стол. До 100. Таганрог."], city_id="taganrog", categories=["table"])
add("краткий ответ о ширине", ["Стол. Таганрог. Бюджет 15000 рублей.",
    "Какую ширину стола указать в сантиметрах?", "100"], categories=["table"], width_cm=[["eq", 100]], **COMMON)

assert len(CASES) == 36
