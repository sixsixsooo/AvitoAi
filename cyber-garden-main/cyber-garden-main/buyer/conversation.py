"""Replay buyer turns and reconcile local facts with a single Gemma result."""

import re
from dataclasses import dataclass, replace

from buyer.anchors import WORDS, accessory_context, categories_from_dialog
from buyer.contracts import Requirements, read_requirements
from buyer.query_parser import (ATTRIBUTES, BUDGET_WORD, CITIES_RE, CORE, MONEY, OWNER,
                                TEMPERATURE, WIDTH, Preference, normalize, parse_turn, tighten)
from buyer.preferences import category_text, evolve, grounded, purpose_wishes

EMPTY = dict(city_id=None, budget_rub=None, categories=None, width_cm=[],
             adjustable_height=None, color_temperature_k=[])
MISSING_CATEGORY = "В запросе есть товар вне каталога"
UNSUPPORTED_NAMES = {"холодильник": "Холодильники", "диван": "Диваны", "шкаф": "Шкафы"}
LIMIT_OPERATIONS = {"gte": "lower", "gt": "lower", "lte": "upper", "lt": "upper", "eq": "equal"}
QUESTIONS = {
    "city_id": r"город\w*|самовывоз\w*",
    "budget_rub": r"бюджет\w*|верхн\w+\s+цен\w*|максималь\w+\s+цен\w*|сколько\s+готов\w*",
    "categories": r"что\s+(?:найти|подобрать|купить)|категори\w*|состав\w*",
}
ANSWER_QUESTIONS = {**QUESTIONS, "width_cm": r"ширин\w*", "adjustable_height": r"регулиров\w*",
                    "color_temperature_k": r"температур\w*|кельвин\w*"}
ATTRIBUTE_QUESTIONS = {
    "width_cm": "Какую ширину стола указать? Можно назвать точное значение или диапазон в сантиметрах.",
    "adjustable_height": "Нужна ли регулировка высоты кресла?",
    "color_temperature_k": "Какую температуру света указать? Можно назвать значение или диапазон в кельвинах.",
}
CATEGORY_ALTERNATIVE = re.compile(r"\b(?:или|либо)\b")
CITY_NAMES = "(?:" + "|".join(pattern.pattern for pattern in CITIES_RE.values()) + ")"
CITY_ALTERNATIVE = re.compile(rf"{CITY_NAMES}\s+(?:или|либо)\s+(?:в\s+)?{CITY_NAMES}")
SAFE_WORDS = re.compile(
    r"(?:(?:мне|нам|я|ты|бы|в|во|на|из|для|под|с|со|и|а|но|за|по|к|у|не|да|"
    r"до|от|см|мм|м|р|руб|тыс|лишь|только|еще|уже|ровно|точно|строго|чуть|"
    r"было|хорошо|теперь|лучше|все|то|же|очень|также|плюс|минимум|максимум|"
    r"больше|меньше|шире|выше|ниже|более|менее|хотя|убери|добавь|помоги|найди|подбери|"
    r"хочу|ищу|нуж\w*|бюджет\w*|город\w*|самовывоз\w*|рубл\w*|тысяч\w*|"
    r"миллиметр\w*|сантиметр\w*|метр\w*|кельвин\w*|температур\w*|ширин\w*|"
    r"высот\w*|регулиров\w*|регулиру\w*|фиксированн\w*|комплект\w*|покупк\w*|"
    r"желательн\w*|предпочтительн\w*|возможност\w*|идеал\w*|важ\w*|"
    r"сними|ограничени\w*|принципиальн\w*|без|компьютерн\w*|письменн\w*|"
    r"офисн\w*|удобн\w*|эргономичн\w*|мягк\w*|настольн\w*|тепл\w*|"
    r"холодн\w*|свет\w*|деревянн\w*|углов\w*|одн\w*|две|два|три|"
    r"пять|десять|пятнадцать|полтор\w*|двадцать|полностью))$"
)


RESET_DIALOG = re.compile(r"^(?:(?:начни|начать|начнем)\s+)?новый\s+диалог[.!?\s]*$")
OFF_TOPIC = re.compile(r"\b(?:рецепт\w*|приготов\w*|погод\w*|анекдот\w*|гороскоп\w*|теорем\w*|пифагор\w*|переведи|напиши\s+код)\b")
REJECT_BUNDLE = re.compile(r"\b(?:нахера|зачем|нафига|почему)\s+(?:мне\s+)?комплект\b|\bне\s+нужен\s+комплект\b")
SHOPPING_CONTEXT = re.compile(
    r"\b(?:нуж\w*|хочу|ищу|подбери|подобрать|найди|куп\w*|добавь|убери|оставь|"
    r"бюджет\w*|город\w*|ширин\w*|регулиров\w*|температур\w*|пожелан\w*|"
    r"комплект\w*|вариант\w*|как\s+раньше|тот\s+же|все\s+то\s+же|table|desk|chair|armchair|lamp)\b")
OFF_TOPIC_QUESTION = ("Я специализируюсь на подборе товаров из каталога и не отвечаю на вопросы "
                      "по другим темам. Могу помочь выбрать стол, кресло или лампу. Что вас интересует?")
DOMAIN_QUESTION = ("Я помогаю подбирать товары из каталога. Уточните, что вы хотите "
                   "приобрести: например, стол, кресло или лампу.")


def active_dialog(messages):
    """A buyer's explicit reset is a boundary for both local replay and Gemma."""
    boundary = 0
    for index, message in enumerate(messages):
        if message["role"] == "user" and RESET_DIALOG.fullmatch(normalize(message["text"])):
            boundary = index
    return messages[boundary:] if boundary else messages


def _unscoped_price(text):
    if BUDGET_WORD.search(text) or not any(pattern.search(text) for pattern in WORDS.values()):
        return False
    spans = [hit.span() for pattern in (WIDTH, TEMPERATURE) for hit in pattern.finditer(text)]
    for hit in MONEY.finditer(text):
        if hit["u"] or any(lo <= hit.start() < hi for lo, hi in spans):
            continue
        # Comparisons without a unit are handled by attribute ambiguity rules.
        if re.search(r"\b(?:до|от|ровно|больше|меньше|шире|уже)\s*$", text[:hit.start()]):
            continue
        return True
    return False


@dataclass(frozen=True)
class LocalState:
    requirements: Requirements
    preferences: tuple[Preference, ...]
    confirmed: frozenset[str]
    unresolved: frozenset[str]
    ambiguous_attributes: frozenset[str]
    needs_model: bool
    blocked_categories: bool = False
    turn_issue: str | None = None
    unsupported_items: tuple[str, ...] = ()

    def decision_questions(self):
        if self.turn_issue == "bundle_rejected":
            return ({"field": "categories", "text":
                     "Понял, вам нужен один товар. Что именно показать — стол, кресло или лампу?"},)
        if self.turn_issue == "off_topic":
            return ({"field": "categories", "text": OFF_TOPIC_QUESTION},)
        if self.turn_issue == "ambiguous":
            return ({"field": "categories", "text": DOMAIN_QUESTION},)
        if self.blocked_categories:
            # Labels come only from reviewed local names, never an invented
            # model label or an arbitrary word from a buyer message.
            from buyer.selection import clarification_questions
            names = [UNSUPPORTED_NAMES[item] for item in self.unsupported_items if item in UNSUPPORTED_NAMES]
            text = ("К сожалению, в текущем каталоге нет " + ("категории" if len(names) == 1 else "категорий") + " «" + "», «".join(names)
                    + "». Сейчас доступны столы, кресла и лампы. Может быть, вас интересует что-то из них?"
                    if names else "Запрошенной категории нет в текущем каталоге. Доступны столы, кресла и лампы. Что из них вам показать?")
            questions = clarification_questions(self.requirements)
            return tuple(dict(question, text=text) if question["field"] == "categories" else question
                         for question in questions)
        if self.turn_issue in {"price", "chair_price"}:
            questions = [{"field": "budget_rub", "text":
                          "Указанное число — цена одного товара или общий бюджет на весь комплект?"}]
            if self.turn_issue == "chair_price":
                questions.append({"field": "categories", "text":
                    "В каталоге поддерживается категория «Кресло». Подойдёт ли вам кресло вместо стула?"})
            return tuple(questions)
        return ()

    def questions(self):
        values = self.requirements
        return tuple(key for key in CORE if getattr(values, key) in (None, ()))

    def attribute_questions(self):
        return tuple(question for attr, question in ATTRIBUTE_QUESTIONS.items()
                     if OWNER[attr] in self.requirements.categories and attr in self.unresolved
                     and getattr(self.requirements, attr) in (None, ()))


def _pending(text):
    return {key for sentence in normalize(text).split("?")[:-1]
            for key, pattern in ANSWER_QUESTIONS.items() if re.search(pattern, sentence)}


def _category_update(text, categories, prompted):
    text = CITY_ALTERNATIVE.sub(" ", text)
    mentions = {key for key, pattern in WORDS.items() if pattern.search(text)}
    anchor = categories_from_dialog([{"role": "user", "text": text}], categories)
    if anchor.reason == MISSING_CATEGORY:
        return True, None
    if not mentions:
        return False, None
    if CATEGORY_ALTERNATIVE.search(text):
        left, right = CATEGORY_ALTERNATIVE.split(text, maxsplit=1)
        if any(pattern.search(left) for pattern in WORDS.values()) and any(
                pattern.search(right) for pattern in WORDS.values()):
            return True, None
    if any(accessory_context(text[:match.start()]) for pattern in WORDS.values()
           for match in pattern.finditer(text)):
        return False, None
    if prompted:
        anchor = categories_from_dialog([{"role": "user", "text": text}])
    if anchor.categories:
        return True, list(anchor.categories)
    if categories and re.search(r"\b(?:без|убери|не\s+нуж\w*)\b", text) and mentions == set(categories):
        return True, []
    return False, None


def _new_bounds(previous, incoming):
    incoming = tighten(incoming)
    if not incoming or any(op == "eq" for op, _ in incoming):
        return incoming
    directions = {LIMIT_OPERATIONS[op] for op, _ in incoming}
    if directions == {"lower", "upper"}:
        return incoming
    return [rule for rule in previous if LIMIT_OPERATIONS[rule[0]] not in directions | {"equal"}] + incoming


def _filter_preferences(preferences, categories):
    return tuple(dict.fromkeys(item for item in preferences
                               if item.category is None or item.category in categories))


class ConversationManager:
    """Stateless replay also works when CLI launches a fresh process per turn."""

    def local(self, messages):
        messages = active_dialog(messages)
        values = EMPTY.copy()
        confirmed = set()
        unresolved = set()
        ambiguous_attributes = set()
        preferences = ()
        pending = set()
        needs_model = False
        composition_history = []
        blocked_categories = False
        unsupported_items = ()
        turn_issue = None
        for message in messages:
            text = normalize(message["text"])
            if message["role"] == "assistant":
                pending = _pending(text)
                continue

            turn_issue = None
            if REJECT_BUNDLE.search(text):
                turn_issue = "bundle_rejected"
                needs_model = True
                pending = set()
                continue
            if RESET_DIALOG.fullmatch(text):
                confirmed.update(EMPTY)
                needs_model = True
                pending = set()
                continue
            current = read_requirements(values)
            turn = parse_turn(message["text"], pending, current)
            turn.preferences.extend(purpose_wishes(text, current.categories))
            mentions_product = any(pattern.search(text) for pattern in WORDS.values())
            shopping_signal = (mentions_product or turn.values or turn.preferences or turn.uncertain
                               or SHOPPING_CONTEXT.search(text))
            if (OFF_TOPIC.search(text) and not mentions_product) or not shopping_signal:
                turn_issue = "off_topic" if OFF_TOPIC.search(text) else "ambiguous"
                needs_model = True
                pending = set()
                continue
            if _unscoped_price(text):
                turn_issue = "chair_price" if re.search(r"\bстул\w*\b", text) else "price"
                turn.values.pop("budget_rub", None)
                turn.uncertain.discard("budget_rub")
                needs_model = True
            composition = category_text(text)
            switching_product = bool(re.search(
                r"\b(?:а\s+)?что\s+(?:насчет|насчёт|по\s+поводу)\s+", composition))
            composition_history.append({"role": "user", "text": composition})
            anchor = categories_from_dialog(composition_history)
            recognized, categories = _category_update(composition, current.categories, "categories" in pending)
            if switching_product:
                replacement = categories_from_dialog([{"role": "user", "text": composition}])
                if replacement.categories:
                    recognized, categories = True, list(replacement.categories)
                    for field in ATTRIBUTES:
                        if field not in turn.values:
                            turn.values[field] = EMPTY[field]
                    preferences = ()
            mentions = any(pattern.search(composition) for pattern in WORDS.values())
            was_blocked = blocked_categories
            blocked_categories = bool(anchor.pending_unknown)
            unsupported_items = anchor.pending_unknown
            if blocked_categories:
                # Retain the known part for replay, but never confirm a partial set.
                values["categories"] = list(anchor.supported)
                turn.uncertain.add("categories")
                turn.values.pop("categories", None)
            elif was_blocked and anchor.categories:
                turn.values["categories"] = list(anchor.categories)
            elif recognized:
                turn.values["categories"] = categories
            elif mentions or "categories" in pending:
                turn.uncertain.add("categories")
            pending = set()

            # Unknown prose or unsupported syntax is sent to Gemma with the original dialogue.
            room = text
            for pattern in (*CITIES_RE.values(), *WORDS.values()):
                room = pattern.sub(" ", room)
            needs_model |= turn.needs_model or any(
                not SAFE_WORDS.fullmatch(word) for word in re.findall(r"[а-яa-z]+", room)
            )
            if turn.uncertain:
                needs_model = True

            unresolved.update(turn.uncertain - turn.values.keys())
            ambiguous_attributes.update(turn.ambiguous_attributes - turn.values.keys())
            confirmed.difference_update(turn.uncertain)
            for item in turn.preferences:
                attr = item.attribute
                if attr and attr not in turn.values and attr not in unresolved:
                    confirmed.add(attr)
            new_categories = turn.values.get("categories")
            if new_categories is not None:
                removed = set(current.categories) - set(new_categories)
                for field, category in OWNER.items():
                    if category in removed and field not in turn.values:
                        turn.values[field] = EMPTY[field]
            for field, value in turn.values.items():
                if field in ("width_cm", "color_temperature_k"):
                    value = _new_bounds(values[field], value)
                values[field] = value
                confirmed.add(field)
                unresolved.discard(field)
                ambiguous_attributes.discard(field)
                if field in ATTRIBUTES and (value is None or value == []):
                    preferences = tuple(item for item in preferences if item.attribute != field)
            preferences = evolve(preferences, turn.preferences, text)
            if new_categories is not None:
                preferences = _filter_preferences(preferences, new_categories)

        # An uncertain last edit cannot reuse the old value in a local-only result.
        for field in unresolved:
            values[field] = EMPTY[field]
        requirements = read_requirements(values)
        return LocalState(requirements, _filter_preferences(preferences, requirements.categories),
                          frozenset(confirmed), frozenset(unresolved),
                          frozenset(ambiguous_attributes), needs_model, blocked_categories, turn_issue, unsupported_items)

    def merge(self, local, model, messages=()):
        if local.turn_issue:
            # Ground preserved model-only wishes in earlier buyer turns, never
            # in the unrelated/ambiguous current text or assistant suggestions.
            history = messages[:-2] if len(messages) > 1 else []
            previous_texts = [normalize(message["text"]) for message in history if message["role"] == "user"]
            budget_evidence = any(re.search(r"\bтыс(?:яч\w*)?\b", text)
                or any(hit["u"] for hit in MONEY.finditer(text))
                or (BUDGET_WORD.search(text) and MONEY.search(text)) for text in previous_texts)
            requirements, unresolved = local.requirements, local.unresolved
            if ("budget_rub" in unresolved and budget_evidence
                    and local.turn_issue not in {"price", "chair_price"}
                    and model.requirements.budget_rub is not None):
                requirements = replace(requirements, budget_rub=model.requirements.budget_rub)
                unresolved = unresolved - {"budget_rub"}
            preferences = grounded((*local.preferences, *model.preferences), history)
            preferences = tuple(pref for pref in local.preferences if pref.category is None) + preferences
            issue = "off_topic" if model.intent == "off_topic" and local.turn_issue == "ambiguous" else local.turn_issue
            return replace(local, requirements=requirements, unresolved=unresolved, turn_issue=issue,
                           preferences=_filter_preferences(preferences, requirements.categories))
        fields = {field: getattr(local.requirements, field)
                  for field in local.confirmed | local.ambiguous_attributes}
        requirements = replace(model.requirements, **fields)
        if local.blocked_categories:
            requirements = replace(requirements, categories=())
        preferences = (evolve(tuple(pref for pref in local.preferences if pref.attribute),
                              grounded((*local.preferences, *model.preferences), messages), "")
                       if messages else evolve(local.preferences, model.preferences, ""))
        return replace(local, requirements=requirements, preferences=_filter_preferences(
            preferences, requirements.categories
        ))

    def resolve(self, messages, gemma, *, require_model):
        messages = active_dialog(messages)
        local = self.local(messages)
        if not require_model and not local.needs_model:
            return local, None
        model, trace = gemma(messages, full_history=local.needs_model)
        if model.intent in {"off_topic", "ambiguous"} and not local.turn_issue:
            previous_messages = messages[:-2] if len(messages) > 1 else []
            previous = self.local(previous_messages)
            last = messages[-1]["text"]
            turn = parse_turn(last, set(), previous.requirements)
            explicit = turn.values or turn.preferences or any(pattern.search(last) for pattern in WORDS.values())
            if not explicit:
                # Semantic classification covers topics outside the small local
                # guard vocabulary. Neither intent nor a guessed category can
                # erase previously confirmed purchase facts.
                preserved = replace(previous, turn_issue=model.intent, needs_model=True)
                return self.merge(preserved, model, messages), trace
        return self.merge(local, model, messages), trace
