"""Conservative extraction of typed facts from one Russian-language buyer turn."""

import re
import unicodedata
from bisect import bisect_right
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

from buyer.anchors import AMBIGUOUS, WORDS
from buyer.contracts import CITIES, FEATURES

CORE = ("city_id", "budget_rub", "categories")
ATTRIBUTES = ("width_cm", "adjustable_height", "color_temperature_k")
FIELDS = CORE + ATTRIBUTES
OWNER = {attribute: category for category, attribute in FEATURES.items()}
NUMBER = r"(?:\d{1,3}(?: \d{3})+|\d+)(?:[.,]\d+)?"
NUM_START = r"(?<!\w)(?:(?<=\d-)|(?<![-−]))"
CITY_FORMS = {
    "taganrog": r"таганрог(?:е|а|у|ом)?",
    "azov": r"азов(?:е|а|у|ом)?",
    "rostov-na-donu": r"ростов(?:е|а|у|ом)?(?:[- ]на[- ]дону)?(?!\s+велик)",
}
CITIES_RE = {key: re.compile(r"\b(?:" + CITY_FORMS.get(key, re.escape(name.lower())) + r")\b")
             for key, name in CITIES.items()}
ALTERNATIVE = re.compile(r"\b(?:или|либо|люб[оы]\w*)\b")
SOFT = re.compile(r"\b(?:желательно|предпочтительно|по возможности|в идеале|было бы хорошо)\b")
CLEAR = r"(?:не\s+важ\w*|не\s+принципиаль\w*|люб\w*|сними\s+ограничение)"
FEATURE_WORDS = {
    "width_cm": r"ширин\w*|шире|уже|пошире|поуже",
    "adjustable_height": r"регулиров\w*|высот\w*",
    "color_temperature_k": r"температур\w*|кельвин\w*",
}
OPERATIONS = (
    ("lte", r"не\s+(?:больше|выше|шире|дороже)|не\s+более|до|максимум"),
    ("gte", r"не\s+(?:меньше|ниже|уже)|не\s+менее|от|минимум|хотя\s+бы"),
    ("lt", r"(?:строго\s+)?(?:меньше|ниже|уже|дешевле)"),
    ("gt", r"(?:строго\s+)?(?:больше|выше|шире|дороже)"),
    ("eq", r"ровно|точно"),
)
OP = re.compile(r"\b(?:" + "|".join(f"(?P<{key}>{pattern})" for key, pattern in OPERATIONS) + r")\s*$")
WIDTH_UNIT = r"миллиметр\w*|мм|сантиметр\w*|см|метр\w*|м"
TEMP_UNIT = r"кельвин\w*|[кk]"
WIDTH = re.compile(rf"{NUM_START}(?P<n>{NUMBER})\s*(?P<u>{WIDTH_UNIT})(?!\w)|\b(?P<m>метр(?:а|у|ом|е)?)(?!\w)")
TEMPERATURE = re.compile(rf"{NUM_START}(?P<n>{NUMBER})\s*(?P<u>{TEMP_UNIT})(?!\w)")
SPACED_TEMPERATURE = re.compile(rf"{NUM_START}(?P<n>{NUMBER})\s+(?P<u>{TEMP_UNIT})(?!\w)")
RANGE_START = re.compile(rf"{NUM_START}(?P<n>{NUMBER})\s*(?:до|[-–—])\s*$")
MONEY_UNIT = r"миллион\w*|млн\.?|тыс(?:яч\w*)?\.?|[кk]|₽|руб(?:л\w*)?\.?"
MONEY = re.compile(rf"(?<![\w-])(?P<n>{NUMBER})(?:\s*(?P<u>{MONEY_UNIT})(?!\w))?")
BARE_MEASURE = re.compile(rf"{NUM_START}(?P<n>{NUMBER})(?P<u>)")
PLAIN_MEASURE = re.compile(rf"(?:(?:{'|'.join(pattern for _, pattern in OPERATIONS)})\s+|"
                          rf"[<>=≤≥]+\s*)?{NUMBER}\s*[.!?]*")
CHAIR = re.compile(r"\b(?:без\s+регулиров\w*|не\s+регулиру\w*|фиксированн\w*\s+высот\w*|"
                   r"с\s+регулиров\w*|регулиров\w*\s+высот\w*\s+(?:нуж\w*|обязательн\w*)|"
                   r"регулируем\w*|регулируется)\b")
SPELLED = {
    "полтора": "1.5", "полторы": "1.5", "одна": "1", "одну": "1", "две": "2",
    "два": "2", "три": "3", "четыре": "4", "пять": "5", "шесть": "6", "семь": "7",
    "восемь": "8", "девять": "9", "десять": "10", "пятнадцать": "15",
    "двадцать": "20", "тридцать": "30", "пятьдесят": "50", "сто": "100",
}
WORD_MONEY = re.compile(r"\b(" + "|".join(SPELLED) + r")(\s+(?:миллион\w*|млн|тыс(?:яч\w*)?|руб\w*))\b")
BUDGET_WORD = re.compile(r"\b(?:бюджет\w*|на\s+все|за\s+все|общая\s+сумма|"
                         r"уложиться|максимум|лимит\w*|ценов\w*\s+потолок)\b")
PRICE_PER_ITEM = re.compile(r"\b(?:за\s+(?:стол\w*|кресл\w*|ламп\w*|стул\w*|светильник\w*)|"
                            r"на\s+(?:стол\w*|кресл\w*|ламп\w*)|за\s+штуку)\b")
RELATIVE_BUDGET = re.compile(r"\b(?:на\s+[^,.!?]{1,32}\s+(?:меньше|больше|дешевле|дороже)|"
                             r"(?:половин\w*|треть)\s+(?:от\s+)?(?:прошл\w*|прежн\w*)\s+бюджет\w*|"
                             r"(?:прибавь|вычти|снизь|увеличь)\b[^,.!?]*\bбюджет\w*)\b")
SPLIT = re.compile(r"[!?;\n]+|\.(?!\d)|,(?!\d)|\s+\b(?:а|но)\b\s+")
QUALITIES = {
    "table": r"к[оа]мпьютерн\w*|письменн\w*|деревянн\w*|углов\w*",
    "chair": r"удобн\w*|эргономичн\w*|мягк\w*|офисн\w*|кожан\w*",
    "lamp": r"настольн\w*|светодиодн\w*|тепл\w*|холодн\w*",
}
CONTEXT = re.compile(r"\b(?:как\s+раньше|как\s+в\s+первом|(?:первый|второй|третий)\s+вариант|"
                     r"такой\s+же|те\s+же\s+условия|котор\w*\s+ты\s+предложил)\b")
CITY_CHANGE = re.compile(r"\b(?:город\s*[:—-]|(?:теперь|лучше|только|ищи|поищи|ищу|перенеси)\s+в\s+|"
                         r"(?:город|самовывоз|переезд)\s+(?:в|из)\s+)")
PROPER_LOCATION = re.compile(r"(?i:\b(?:в|из|город(?:е)?)\s+)[А-ЯЁ][а-яё-]+")
NEGATED_CITY = re.compile(r"\bне\s+(?:в\s+|из\s+)?$")
NEGATED_MEASURE = re.compile(rf"\bне\s+(?:(?:ровно|точно|до|от)\s+)?{NUMBER}")


@dataclass(frozen=True)
class Preference:
    category: str | None
    text: str
    attribute: str | None = None


def typed_preference(pref):
    """Recover the internal attribute tag from grounded Gemma soft text."""
    if pref.attribute is not None:
        return pref
    text = normalize(pref.text)
    attr = None
    if pref.category == "table" and WIDTH.search(text):
        attr = "width_cm"
    elif pref.category == "lamp" and TEMPERATURE.search(text):
        attr = "color_temperature_k"
    elif pref.category == "chair" and CHAIR.search(text):
        attr = "adjustable_height"
    return Preference(pref.category, pref.text, attr)


@dataclass
class Turn:
    values: dict = field(default_factory=dict)
    uncertain: set = field(default_factory=set)
    preferences: list = field(default_factory=list)
    ambiguous_attributes: set = field(default_factory=set)
    needs_model: bool = False


@dataclass(frozen=True)
class BudgetReading:
    amount: int | None = None
    uncertain: bool = False
    preference: str | None = None


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    text = re.sub(r"\bневажно\b", "не важно", text)
    return re.sub(r"[^\S\n]+", " ", text).strip()


def _decimal(value):
    return Decimal(value.replace(" ", "").replace(",", "."))


def _value(match, kind):
    if match.groupdict().get("m"):
        return Decimal(100)
    value = _decimal(match["n"])
    unit = match["u"] or ""
    if kind == "width_cm":
        if unit == "м" or unit.startswith("метр"):
            return value * 100
        if unit == "мм" or unit.startswith("миллиметр"):
            return value / 10
    if kind == "budget_rub":
        if unit.startswith(("миллион", "млн")):
            return value * 1_000_000
        if unit in {"к", "k"} or unit.startswith("тыс"):
            return value * 1000
    return value


def _operator(prefix, default):
    match = OP.search(prefix.rstrip())
    if match:
        return match.lastgroup
    symbol = re.search(r"(<=|>=|[=<>≤≥])\s*$", prefix)
    return {"<": "lt", ">": "gt", "<=": "lte", ">=": "gte", "≤": "lte",
            "≥": "gte", "=": "eq"}[symbol[1]] if symbol else default


def _numeric_rules(clause, pattern, kind):
    hits = list(pattern.finditer(clause))
    rules = []
    for index, match in enumerate(hits):
        prefix = clause[hits[index - 1].end() if index else 0:match.start()]
        value = _value(match, kind)
        if not 0 < value <= 10**12:
            return None
        # A bare first endpoint inherits the unit of the second endpoint.
        bound = RANGE_START.search(prefix)
        if bound:
            original = match.groupdict()
            scale = value / _decimal(original["n"]) if original.get("n") else Decimal(100)
            rules.extend([["gte", _decimal(bound["n"]) * scale], ["lte", value]])
        elif index and re.fullmatch(r"\s*[-–—]\s*", prefix):
            rules[-1][0] = "gte"
            rules.append(["lte", value])
        else:
            rules.append([_operator(prefix, "eq"), value])
    return tighten(rules)


def tighten(rules):
    """Keep the strongest lower and upper bound and retain exact values."""
    lower = [rule for rule in rules if rule[0] in {"gt", "gte"}]
    upper = [rule for rule in rules if rule[0] in {"lt", "lte"}]
    equal = list(dict.fromkeys(end for op, end in rules if op == "eq"))
    return ([max(lower, key=lambda rule: (rule[1], rule[0] == "gt"))] if lower else []) + (
        [min(upper, key=lambda rule: (rule[1], rule[0] != "lt"))] if upper else []
    ) + [["eq", value] for value in equal]


def _budget(text, numeric_spans, prompted, first):
    if RELATIVE_BUDGET.search(text):
        return BudgetReading(uncertain=True)
    explicit = bool(BUDGET_WORD.search(text))
    matches = list(MONEY.finditer(text))
    spelled = [(word.start(), word.end(), _decimal(SPELLED[word[1]]) *
                (1_000_000 if word[2].strip().startswith(("миллион", "млн")) else
                 1000 if word[2].strip().startswith("тыс") else 1), True)
               for word in WORD_MONEY.finditer(text)]
    candidates = spelled + [
        (match.start(), match.end(), _value(match, "budget_rub"), bool(match["u"]))
        for match in matches if not any(lo <= match.start() < hi for lo, hi in numeric_spans)
    ]
    boundaries = [0, *(separator.end() for separator in SPLIT.finditer(text)), len(text)]
    qualified = []
    preference = None
    for start, end, value, unit in candidates:
        prefix, suffix = text[:start], text[end:]
        index = bisect_right(boundaries, start) - 1
        clause = text[boundaries[index]:boundaries[index + 1]]
        if PRICE_PER_ITEM.search(clause) and not BUDGET_WORD.search(clause):
            continue
        if re.match(r"\s*(?:%|кг\b|штук\w*\b)", suffix):
            continue
        if re.match(r"\s*(?:[$€]|usd\b|eur\b|доллар\w*\b|евро\b)", suffix):
            return BudgetReading(uncertain=True)
        op = _operator(text[max(0, start - 40):start], None)
        if unit or explicit or prompted:
            if not explicit and _categories(prefix) and _categories(suffix):
                return BudgetReading(uncertain=True)
            if SOFT.search(clause) and not re.search(r"\b(?:строго|предел|обязательн\w*)\b", clause):
                preference = clause.strip(" ,.")
                continue
            qualified.append((start, value, op, clause))
    if not qualified:
        return BudgetReading(uncertain=(explicit or prompted) and not preference,
                             preference=preference)
    qualified.sort()
    marked = [item for item in qualified if BUDGET_WORD.search(item[3])]
    if marked:
        qualified = marked
    if len(qualified) == 2 and [item[2] for item in qualified] == ["gte", "lte"]:
        qualified = qualified[1:]
    if len(qualified) != 1:
        return BudgetReading(uncertain=True, preference=preference)
    _, value, op, clause = qualified[0]
    if re.search(r"\b(?:примерно|около|доплат\w*)\b", clause):
        return BudgetReading(uncertain=True, preference=preference)
    if op in {"gte", "gt"} or not 0 <= value <= 10**12:
        return BudgetReading(uncertain=True, preference=preference)
    if op == "lt":
        amount = max(0, int(value.to_integral_value(rounding=ROUND_CEILING)) - 1)
    else:
        amount = int(value.to_integral_value(rounding=ROUND_FLOOR))
    return BudgetReading(amount, preference=preference)


def _categories(text):
    return {key for key, pattern in WORDS.items() if pattern.search(text)}


def _feature_spans(text, regex):
    for hit in regex.finditer(text):
        window_start = max(0, hit.start() - 48)
        start = RANGE_START.search(text[window_start:hit.start()])
        yield (window_start + start.start() if start else hit.start(), hit.end())


def parse_turn(message, pending, current):
    text = normalize(message)
    turn = Turn(needs_model=bool(CONTEXT.search(text) or AMBIGUOUS.search(text)))
    city_hits, rejected_cities = set(), set()
    for key, pattern in CITIES_RE.items():
        for match in pattern.finditer(text):
            preceding = text[max(0, match.start() - 16):match.start()]
            (rejected_cities if NEGATED_CITY.search(preceding) else city_hits).add(key)
    if len(city_hits) == 1:
        turn.values["city_id"] = city_hits.pop()
    elif city_hits or CITY_CHANGE.search(text) or PROPER_LOCATION.search(message) or "city_id" in pending:
        if len(city_hits) > 1 and ALTERNATIVE.search(text):
            turn.values["city_id"] = None
        else:
            turn.uncertain.add("city_id")
    elif current.city_id in rejected_cities:
        turn.values["city_id"] = None
    mentioned = _categories(text)
    active = mentioned | set(current.categories)
    pending_attr = next(iter(pending)) if len(pending) == 1 and pending <= set(ATTRIBUTES) else None
    if (pending_attr in {"width_cm", "color_temperature_k"}
            and PLAIN_MEASURE.fullmatch(text)
            and not any(pattern.search(text) for pattern in CITIES_RE.values())
            and not BUDGET_WORD.search(text)):
        measurements = _numeric_rules(text, BARE_MEASURE, pending_attr)
        if measurements:
            turn.values[pending_attr] = measurements
            return turn
    if pending_attr == "adjustable_height" and text.strip(" .!?") in {"да", "нет"}:
        turn.values[pending_attr] = text.startswith("да")
        return turn
    spans = list(_feature_spans(text, WIDTH))
    if "lamp" in active or re.search(FEATURE_WORDS["color_temperature_k"], text):
        spans.extend(_feature_spans(text, SPACED_TEMPERATURE))
        for clause in SPLIT.split(text):
            if re.search(r"\bтемператур\w*", clause) or "lamp" in _categories(clause) and not BUDGET_WORD.search(clause):
                for match in TEMPERATURE.finditer(clause):
                    if _value(match, "color_temperature_k") >= 1000:
                        start = text.find(clause)
                        spans.append((start + match.start(), start + match.end()))
    # Reserve numbers in explicitly named measurements before reading money.
    # Without units they need clarification, not an assumed currency or scale.
    for clause in SPLIT.split(text):
        budget_marker = BUDGET_WORD.search(clause)
        measurement = clause[:budget_marker.start()] if budget_marker else clause
        offset = text.find(clause)
        for attr in ("width_cm", "color_temperature_k"):
            if not re.search(r"\b(?:" + FEATURE_WORDS[attr] + r")\b", measurement):
                continue
            bare = [hit for hit in BARE_MEASURE.finditer(measurement)
                    if not any(lo <= offset + hit.start() < hi for lo, hi in spans)
                    and not ((money := MONEY.match(measurement, hit.start())) and money["u"])]
            if bare:
                spans.extend((offset + hit.start(), offset + hit.end()) for hit in bare)
                turn.uncertain.add(attr)
                turn.ambiguous_attributes.add(attr)
    budget = _budget(text, spans, "budget_rub" in pending, current.budget_rub is None)
    if budget.preference and len(budget.preference) <= 120:
        turn.preferences.append(Preference(None, budget.preference, "budget_rub"))
    if budget.amount is not None:
        turn.values["budget_rub"] = budget.amount
    elif budget.uncertain or (not budget.preference and
                              re.search(r"\b(?:дешевле|дороже|подешевле|потолок|сумм\w*)\b", text)):
        turn.uncertain.add("budget_rub")
    elif not budget.preference and re.search(r"\bтыс(?:яч\w*)?\b", text):
        # Inverted/compound colloquial amounts are left to Gemma as a whole.
        turn.uncertain.add("budget_rub")

    if (len(active) == 1 and "budget_rub" not in pending
            and not BUDGET_WORD.search(text) and not SOFT.search(text)):
        category = next(iter(active))
        attr = FEATURES[category]
        if attr in {"width_cm", "color_temperature_k"} and not current.condition(category):
            for hit in BARE_MEASURE.finditer(text):
                money = MONEY.match(text, hit.start())
                if (not any(lo <= hit.start() < hi for lo, hi in spans)
                        and not (money and money["u"])
                        and _operator(text[:hit.start()], None) is not None):
                    # A bare "до 100" supplies neither a verified unit nor an
                    # attribute. Do not accept Gemma's guessed hard bound.
                    turn.uncertain.add(attr)
                    turn.ambiguous_attributes.add(attr)

    for attr, words in FEATURE_WORDS.items():
        if re.search(rf"\b{CLEAR}[ ,]*(?:(?:какая|какой|какое)\s+)?(?:{words})\b", text):
            turn.values[attr] = None if attr == "adjustable_height" else []
    owner = next(iter(current.categories), None) if len(current.categories) == 1 else None
    for clause in filter(str.strip, SPLIT.split(text)):
        for fragment in re.split(r"\s+и\s+(?=(?:(?:нуж\w*|еще|также)\s+)?(?:"
                                 + "|".join(p.pattern for p in WORDS.values()) + r"))", clause):
            categories = _categories(fragment)
            if len(categories) == 1:
                owner = next(iter(categories))
            hint = SOFT.search(fragment)
            for key, pattern in QUALITIES.items():
                if key in categories or (not categories and owner == key) or (
                        not categories and hint and key in active):
                    turn.preferences.extend(Preference(key, hit[0]) for hit in re.finditer(pattern, fragment))
            for attr, words in FEATURE_WORDS.items():
                category = OWNER[attr]
                reset = re.search(rf"\b(?:{words})\b.*?\b{CLEAR}\b", fragment)
                if reset and (not categories or category in categories):
                    turn.values[attr] = None if attr == "adjustable_height" else []
                    continue
                explicit = bool(re.search(rf"\b(?:{words})\b", fragment))
                owns_value = category in categories or (not categories and
                    (owner == category or explicit))
                if not owns_value:
                    continue
                if attr == "adjustable_height":
                    hits = list(CHAIR.finditer(fragment))
                    options = {not hit[0].startswith(("без", "не ", "фиксированн")) for hit in hits}
                    value = next(iter(options)) if len(options) == 1 and "не без" not in fragment else None
                else:
                    pattern = WIDTH if attr == "width_cm" else TEMPERATURE
                    value = _numeric_rules(fragment, pattern, attr)
                    if attr == "color_temperature_k" and not explicit:
                        value = _numeric_rules(fragment, SPACED_TEMPERATURE, attr)
                        if not value and not BUDGET_WORD.search(fragment):
                            compact = _numeric_rules(fragment, TEMPERATURE, attr)
                            if compact and all(limit >= 1000 for _, limit in compact):
                                value = compact
                if hint and (explicit or value):
                    preference_text = fragment[hint.end():].strip(" ,.")
                    if 0 < len(preference_text) <= 120:
                        turn.preferences.append(Preference(category, preference_text, attr))
                elif ((ALTERNATIVE.search(fragment) and (explicit or value))
                      or (attr != "adjustable_height" and value and NEGATED_MEASURE.search(fragment))):
                    turn.uncertain.add(attr)
                    turn.ambiguous_attributes.add(attr)
                elif value is not None and value != []:
                    if isinstance(value, list):
                        turn.values.setdefault(attr, []).extend(value)
                    else:
                        turn.values[attr] = value
                elif explicit or (value and ALTERNATIVE.search(fragment)):
                    turn.uncertain.add(attr)
    if re.search(r"\b(?:как\s+в\s+первом|(?:первый|второй|третий)\s+вариант)\b", text):
        turn.uncertain.update(set(FIELDS) - turn.values.keys())
    if re.search(r"\b(?:если|вместо|не\s+без|то\s+ли)\b", text):
        turn.uncertain.update(attr for attr in ATTRIBUTES if attr in turn.values)
        for attr in turn.uncertain:
            turn.values.pop(attr, None)
    return turn
