"""Contracts for buyer messages, AI interpretation and projected catalog records."""

import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("В JSON повторяется ключ объекта.")
        result[key] = value
    return result


def _constant(_value):
    raise ValueError("В JSON допустимы только конечные числа.")


def decode_json(data):
    if isinstance(data, bytes):
        data = data.decode("utf-8-sig")
    return json.loads(data, object_pairs_hook=_object, parse_constant=_constant, parse_float=Decimal)


CITIES = {record["id"]: record["name"] for record in decode_json((ROOT / "data/cities.json").read_bytes())}
CATEGORIES = {
    record["id"]: record["name"]
    for record in decode_json((ROOT / "data/categories.json").read_bytes())
}
FEATURES = {"table": "width_cm", "chair": "adjustable_height", "lamp": "color_temperature_k"}
OPERATORS = {"eq", "gte", "lte", "gt", "lt"}


def _integer(value, minimum):
    if type(value) is Decimal and value.is_finite() and value == value.to_integral_value():
        value = int(value)
    if type(value) is not int or value < minimum:
        raise ValueError("Некорректное целое число.")
    return value


def read_messages(request):
    if not isinstance(request, dict) or set(request) != {"messages"}:
        raise ValueError("Ожидается объект с полем messages.")
    messages = request["messages"]
    if not isinstance(messages, list) or not 1 <= len(messages) <= 20 or len(messages) % 2 == 0:
        raise ValueError("Диалог должен начинаться и заканчиваться репликой покупателя.")
    for index, message in enumerate(messages):
        role = "assistant" if index % 2 else "user"
        if (not isinstance(message, dict) or set(message) != {"role", "text"}
                or message["role"] != role or not isinstance(message["text"], str)
                or not message["text"]):
            raise ValueError("Некорректная реплика диалога.")
    return messages


@dataclass(frozen=True)
class Listing:
    id: str
    category: str
    city_id: str
    price_rub: int
    feature: int | bool | None
    # Passive search material, never a source of structured facts. Excluded from
    # fact equality so existing catalog-isolation checks retain their meaning.
    title: str = field(default="", compare=False)
    description: str = field(default="", compare=False)


def read_catalog(data):
    if not isinstance(data, list) or len(data) > 250:
        raise ValueError("Ожидается массив из не более 250 объявлений.")
    seen = set()
    listings = []
    fields = {"id", "category", "city_id", "price_rub", "title", "description", "attributes"}
    for item in data:
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError("Некорректная структура объявления.")
        listing_id, category, city = (item[key] for key in ("id", "category", "city_id"))
        if (not all(isinstance(value, str) for value in (listing_id, category, city))
                or not listing_id or listing_id in seen or category not in CATEGORIES or city not in CITIES):
            raise ValueError("Некорректный ID, категория или город объявления.")
        attributes = item["attributes"]
        if (not isinstance(item["title"], str) or not item["title"]
                or not isinstance(item["description"], str) or not isinstance(attributes, dict)
                or any(type(value) not in (str, int, bool, Decimal) for value in attributes.values())):
            raise ValueError("Некорректный текст или характеристики объявления.")

        values = {}
        for field in set(FEATURES.values()) & attributes.keys():
            value = attributes[field]
            if field == "adjustable_height":
                if type(value) is not bool:
                    raise ValueError("Регулировка высоты должна быть логическим значением.")
                values[field] = value
            else:
                values[field] = _integer(value, 1)
        listings.append(Listing(listing_id, category, city, _integer(item["price_rub"], 1),
                                values.get(FEATURES[category]), item["title"], item["description"]))
        seen.add(listing_id)
    return tuple(listings)


@dataclass(frozen=True)
class Comparison:
    operation: str
    value: Decimal


@dataclass(frozen=True)
class Requirements:
    city_id: str | None
    budget_rub: int | None
    categories: tuple[str, ...]
    width_cm: tuple[Comparison, ...]
    adjustable_height: bool | None
    color_temperature_k: tuple[Comparison, ...]

    def condition(self, category):
        return getattr(self, FEATURES[category])


def _comparisons(entries):
    if not isinstance(entries, list) or len(entries) > 8:
        raise ValueError("Ожидается список условий на числовую характеристику.")
    comparisons = []
    for entry in entries:
        if (not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], str)
                or entry[0] not in OPERATORS or type(entry[1]) not in (int, Decimal)):
            raise ValueError("Некорректное сравнение числовой характеристики.")
        value = Decimal(entry[1])
        if not value.is_finite() or abs(value) > Decimal(10**12):
            raise ValueError("Некорректное значение числовой характеристики.")
        comparisons.append(Comparison(entry[0], value))
    return tuple(comparisons)


def read_requirements(data):
    fields = {"city_id", "budget_rub", "categories", *FEATURES.values()}
    if not isinstance(data, dict) or set(data) != fields:
        raise ValueError("Некорректная структура извлечённых условий.")
    city = data["city_id"]
    if city is not None and (not isinstance(city, str) or city not in CITIES):
        raise ValueError("Модель вернула неизвестный идентификатор города.")
    budget = data["budget_rub"]
    if budget is not None:
        budget = _integer(budget, 0)
    categories = data["categories"]
    if categories is None:
        categories = []
    if (not isinstance(categories, list) or len(categories) > len(CATEGORIES)
            or any(not isinstance(category, str) or category not in CATEGORIES for category in categories)
            or len(set(categories)) != len(categories)):
        raise ValueError("Модель вернула некорректные категории.")
    adjustable = data["adjustable_height"]
    if adjustable is not None and type(adjustable) is not bool:
        raise ValueError("Модель вернула некорректное условие на регулировку высоты.")
    return Requirements(city, budget, tuple(key for key in CATEGORIES if key in categories),
                        _comparisons(data["width_cm"]), adjustable, _comparisons(data["color_temperature_k"]))
