"""Deterministic filtering, ranking and explanations over structured catalog facts."""

import heapq
import operator
from dataclasses import dataclass

from buyer.contracts import CATEGORIES, CITIES
from buyer.ranking import listing_scores, ranked_bundles

CHECK = {"eq": operator.eq, "lt": operator.lt, "lte": operator.le,
         "gt": operator.gt, "gte": operator.ge}
SYMBOLS = {"eq": "=", "lt": "<", "lte": "≤", "gt": ">", "gte": "≥"}
UNITS = {"table": "см", "lamp": "К"}
LABELS = {"table": "Ширина стола", "chair": "Регулировка высоты", "lamp": "Температура света"}
QUESTIONS = {
    "budget_rub": "Какой у вас максимальный бюджет на весь комплект?",
    "city_id": "В каком городе нужен самовывоз — Таганрог, Ростов-на-Дону или Азов?",
    "categories": "Что подобрать из каталога — стол, кресло или лампу? Можно выбрать несколько.",
}


def rubles(amount):
    return f"{amount:,}".replace(",", " ") + " ₽"


def number(value):
    text = format(value, "f")
    return (text.rstrip("0").rstrip(".") if "." in text else text).replace(".", ",")


def describe_condition(category, condition):
    if category == "chair":
        return "есть" if condition else "нет"
    return " и ".join(f"{SYMBOLS[item.operation]} {number(item.value)} {UNITS[category]}"
                      for item in condition)


def actual_feature(item):
    if item.feature is None:
        return None
    if item.category == "chair":
        return "Регулировка высоты есть" if item.feature else "Без регулировки высоты"
    return f"{LABELS[item.category]} {item.feature} {UNITS[item.category]}"


def matches(item, condition):
    if condition is None or condition == ():
        return True
    if item.feature is None:
        return False
    if item.category == "chair":
        return item.feature is condition
    return all(CHECK[rule.operation](item.feature, rule.value) for rule in condition)


@dataclass(frozen=True)
class Selection:
    response: dict
    eligible_counts: dict
    minimum_total_rub: int | None


def _cheapest(groups, budget):
    """Monotone price grid. The cheapest unseen tuple is on the heap frontier."""
    start = (0,) * len(groups)
    frontier = [(sum(group[0].price_rub for group in groups),
                 tuple(group[0].id for group in groups), start)]
    seen = {start}
    emitted = 0
    while frontier and emitted < 3:
        total, _, indices = heapq.heappop(frontier)
        if total > budget:
            break
        yield total, tuple(group[i] for group, i in zip(groups, indices))
        emitted += 1
        for axis, (group, index) in enumerate(zip(groups, indices)):
            if index + 1 == len(group):
                continue
            next_indices = indices[:axis] + (index + 1,) + indices[axis + 1:]
            if next_indices in seen:
                continue
            seen.add(next_indices)
            next_total = total + group[index + 1].price_rub - group[index].price_rub
            ids = tuple(group[i].id for group, i in zip(groups, next_indices))
            heapq.heappush(frontier, (next_total, ids, next_indices))


def _explain(items, total, requirements, preferences=()):
    facts = [
        f"{CATEGORIES[item.category]} — {rubles(item.price_rub)}"
        + (f" ({feature.lower()})" if (feature := actual_feature(item)) else "")
        for item in items
    ]
    reserve = requirements.budget_rub - total
    budget_note = f"Запас бюджета {rubles(reserve)}." if reserve else "Ровно в пределах бюджета."
    from buyer.preferences import dimension
    purposes = list(dict.fromkeys(pref.text for pref in preferences if dimension(pref) == "purpose"))
    purpose_note = (" Назначение «" + "», «".join(purposes)
                    + "» учтено как пожелание. Пригодность для этого использования "
                    "не подтверждена структурированными характеристиками каталога.") if purposes else ""
    return (f"Самовывоз · {CITIES[requirements.city_id]}. " + ", ".join(facts)
            + f". Итого {rubles(total)}. {budget_note}" + purpose_note)


def _no_candidates(by_category, unavailable, requirements, catalog_categories=None):
    city = CITIES[requirements.city_id]
    reasons = []
    for category in unavailable:
        items = by_category[category]
        name = CATEGORIES[category]
        if not items:
            scope = "каталоге" if catalog_categories is not None and category not in catalog_categories else "каталоге этого города"
            reasons.append(f"«{name}» отсутствует в {scope}")
            continue
        condition = requirements.condition(category)
        label = LABELS[category].lower()
        requested = describe_condition(category, condition)
        reason = f"для «{name}» нет подтверждённого соответствия условию «{label} {requested}»"
        known = [item.feature for item in items if item.feature is not None]
        if not known:
            reason += " (у всех объявлений значение неизвестно)"
        elif category == "chair":
            reason += f" (среди подтверждённых значений: {'есть' if any(known) else 'нет'})"
        else:
            def distance(value):
                return sum(abs(value - rule.value) for rule in condition
                           if not CHECK[rule.operation](value, rule.value))

            closest = min(known, key=lambda value: (distance(value), value))
            reason += f" (ближайшее подтверждённое значение: {closest} {UNITS[category]})"
        reasons.append(reason)
    if all(by_category[category] for category in unavailable):
        suggestion = "Какую характеристику можно изменить или рассмотреть другой город?"
    else:
        suggestion = "Подойдёт другой город или другой состав комплекта?"
    return f"Город самовывоза — {city}. " + ". ".join(s[0].upper() + s[1:] for s in reasons) + ". " + suggestion


def filter_listings(listings, requirements):
    """Strict structured facts only; text and preferences never enter this step."""
    by_category = {key: [] for key in requirements.categories}
    for item in listings:
        if item.city_id == requirements.city_id and item.category in by_category:
            by_category[item.category].append(item)
    eligible = {
        key: sorted((item for item in items if matches(item, requirements.condition(key))),
                    key=lambda item: (item.price_rub, item.id))
        for key, items in by_category.items()
    }
    return by_category, eligible


def clarification_questions(requirements):
    return [
        {"field": field, "text": question}
        for field, question in QUESTIONS.items()
        if getattr(requirements, field) is None or (field == "categories" and not requirements.categories)
    ]


def select(listings, requirements, preferences=()):
    questions = clarification_questions(requirements)
    # A known city and composition suffice to prove an empty hard-filtered
    # category even before the total budget has been supplied.
    if any(question["field"] != "budget_rub" for question in questions):
        return Selection({"status": "clarify", "questions": questions}, {}, None)

    by_category, eligible = filter_listings(listings, requirements)
    counts = {key: len(items) for key, items in eligible.items()}
    unavailable = [key for key, count in counts.items() if count == 0]
    if unavailable:
        return Selection({"status": "no_match",
                          "message": _no_candidates(by_category, unavailable, requirements,
                                                    {item.category for item in listings})}, counts, None)
    if questions:
        return Selection({"status": "clarify", "questions": questions}, counts, None)

    groups = tuple(eligible.values())
    minimum = sum(group[0].price_rub for group in groups)
    if minimum > requirements.budget_rub:
        delta = minimum - requirements.budget_rub
        message = (f"В городе {CITIES[requirements.city_id]} самый доступный полный комплект "
                   f"с вашими характеристиками стоит {rubles(minimum)}. "
                   f"До него не хватает {rubles(delta)}. "
                   "Можно увеличить бюджет до этой суммы или изменить одно из требований. Что вам удобнее?")
        return Selection({"status": "no_match", "message": message}, counts, minimum)

    scores = listing_scores(eligible, preferences) if preferences else {}
    bundles = (ranked_bundles(groups, requirements.budget_rub, scores) if any(scores.values())
               else _cheapest(groups, requirements.budget_rub))
    options = [
        {"listing_ids": [item.id for item in items], "total_rub": total,
         "explanation": _explain(items, total, requirements, preferences)}
        for total, items in bundles
    ]
    return Selection({"status": "ok", "options": options}, counts, minimum)
