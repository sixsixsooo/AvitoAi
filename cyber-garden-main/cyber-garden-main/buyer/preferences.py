"""Ground positive wishes in buyer turns and replay their edits conservatively."""

import re

from buyer.anchors import ADDITIONS, WORDS, categories_from_dialog
from buyer.query_parser import CITIES_RE, Preference, QUALITIES, SPLIT, normalize, typed_preference
from buyer.ranking import tokens

RESET = re.compile(r"\b(?:без пожеланий|пожелания\s+(?:больше\s+)?не\s+важны|"
                   r"(?:теперь\s+)?главное\s+цена|убери\s+все\s+пожелания)\b")
BEFORE_REJECT = re.compile(r"\b(?:(?:не\s+(?:хочу|нуж\w*|надо)|без|убери|удали|"
                           r"отмен\w*|вместо)\s+(?:\w+\s+){0,3}|"
                           r"не\s+(?:(?:очень|слишком)\s+)?)$")
AFTER_REJECT = re.compile(r"^\s*(?:\w+\s+){0,3}(?:(?:больше|уже)\s+)?"
                          r"не\s+(?:нуж\w*|важ\w*|хочу|надо)")
DIMENSIONS = {
    "purpose": r"для\b|чтобы\b|котор\w*",
    "material": r"дерев\w*|метал\w*|стеклян\w*|пластик\w*|кожан\w*|ткан\w*|текстиль\w*",
    "color": r"красн\w*|син\w*|зелен\w*|бел\w*|черн\w*|сер\w*|желт\w*",
    "softness": r"мягк\w*|жестк\w*",
    "light": r"тепл\w*|холодн\w*",
    "shape": r"углов\w*|прям\w*",
}


def purpose_wishes(text, categories=()):
    """Keep literal use cases, scoped to the preceding product or sole active category."""
    text = normalize(text)
    mentions = sorted((hit.start(), hit.end(), key) for key, pattern in WORDS.items()
                      for hit in pattern.finditer(text))
    result = []
    for hit in re.finditer(r"\b(?:для\s+|чтобы\s+|на\s+котор\w*\s+)", text):
        previous = [mention for mention in mentions if mention[1] <= hit.start()]
        owner = previous[-1][2] if previous else (categories[0] if len(categories) == 1 else None)
        if owner is None:
            continue
        # An explicit next product or sentence terminates this product's use case.
        end = next((start for start, _, _ in mentions if start > hit.start()), len(text))
        phrase = re.split(r"[,;.!?\n]|\s+и\s+(?=кресл|стол|ламп)|\b(?:покажи|покажите|предложи|бюджет|город)\b",
                          text[hit.start():end])[0].strip(" ,")
        # Location and total price describe the search, not the product's purpose.
        city_starts = [match.start() for pattern in CITIES_RE.values()
                       for match in pattern.finditer(phrase)]
        if city_starts:
            phrase = re.sub(r"\s+в\s*$", "", phrase[:min(city_starts)]).strip()
        phrase = re.split(r"\s+(?:до|за|не\s+дороже)\s+\d+[\d\s]*(?:к|тыс\.?|руб\w*|₽)?\b",
                          phrase, maxsplit=1)[0].strip()
        phrase = re.sub(r"\s+и$", "", phrase)
        if 3 < len(phrase) <= 120:
            result.append(Preference(owner, phrase))
    return tuple(result)


def dimension(pref):
    if pref.attribute:
        return pref.attribute
    return next((key for key, pattern in DIMENSIONS.items()
                 if re.search(r"\b(?:" + pattern + r")\b", normalize(pref.text))), None)


def rejected(pref, text):
    """Negative references remove wishes; they never create exclusion filters."""
    if pref.attribute:
        return False
    wanted = tokens(pref.text)
    if not wanted:
        return True
    for clause in SPLIT.split(normalize(text)):
        for hit in re.finditer(r"[а-яa-z0-9]+", clause):
            prefix = re.split(r"\b(?:теперь|желательно|предпочтительно|и|а|но)\b", clause[:hit.start()])[-1]
            if tokens(hit[0]) & wanted and (
                    BEFORE_REJECT.search(prefix)
                    or AFTER_REJECT.search(clause[hit.end():])):
                return True
    return False


def category_text(text):
    # Mask only the rejected item segment. Preserve separators and composition
    # connectors verbatim: SPLIT would consume 'а' in 'а также' and lose intent.
    parts = re.split(r"([,;.!?\n]+|\b(?:" + ADDITIONS + r"|и)\b)", normalize(text))
    for index in range(0, len(parts), 2):
        clause = parts[index]
        qualified_rejection = any(
            WORDS[category].search(clause) and any(
                rejected(Preference(category, hit[0]), clause)
                for hit in re.finditer(pattern, clause))
            for category, pattern in QUALITIES.items())
        if qualified_rejection:
            parts[index] = " " * len(clause)
    return "".join(parts)


def evolve(previous, incoming, text):
    text = normalize(text)
    previous = () if RESET.search(text) else previous
    if re.search(r"\bотмен\w*\s+предыдущ\w*\s+пожелани\w*\b", text):
        previous = previous[:-1]
        incoming = ()
    new = tuple(typed_preference(pref) for pref in incoming if isinstance(pref.text, str)
                and 0 < len(pref.text.strip()) <= 120 and not rejected(typed_preference(pref), text))
    previous = tuple(pref for pref in previous if not rejected(pref, text))
    if (re.search(r"\b(?:теперь|вместо|замени|лучше)\b", text)
            and not re.search(r"\b(?:добавь|также|еще)\b", text)):
        explicit_replace = bool(re.search(r"\b(?:вместо|замени)\b", text))
        # Compatible qualities accumulate. A new color/material replaces the
        # old value of that dimension; an explicit replacement may replace all.
        previous = tuple(old for old in previous if not any(
            old.category == pref.category and (explicit_replace or (
                dimension(pref) is not None and dimension(old) == dimension(pref))) for pref in new))
    result = []
    for pref in (*previous, *new):
        key = (pref.category, pref.attribute, tokens(pref.text))
        if not any((old.category, old.attribute, tokens(old.text)) == key for old in result):
            result.append(pref)
    return tuple(result)


def grounded(preferences, messages):
    """Require literal buyer evidence, correct nearby category, and replay edits.

    Model output cannot resurrect an old wish by quoting an earlier buyer turn.
    Assistant text is never evidence. Negative wishes create no hard filters.
    """
    candidates = tuple(pref for pref in preferences if isinstance(pref, Preference)
                       and pref.category in WORDS and isinstance(pref.text, str)
                       and 0 < len(pref.text.strip()) <= 120)
    result = ()
    history = []
    for message in messages:
        if message["role"] != "user":
            continue
        history.append({"role": "user", "text": category_text(message["text"])})
        active = categories_from_dialog(history).categories or ()
        text = normalize(message["text"])
        incoming = []
        for pref in candidates:
            phrase = normalize(pref.text)
            if not phrase or phrase not in text or not tokens(phrase):
                continue
            inherited_owner = active[0] if len(active) == 1 else None
            for clause in SPLIT.split(text):
                clause_categories = [category for category, pattern in WORDS.items() if pattern.search(clause)]
                if len(clause_categories) == 1:
                    inherited_owner = clause_categories[0]
                occurrence = re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", clause)
                if not occurrence:
                    continue
                position = occurrence.start()
                mentions = [(min(abs(hit.start() - (position + len(phrase))),
                                 abs(hit.end() - position)), category)
                            for category, pattern in WORDS.items() for hit in pattern.finditer(clause)]
                owner = min(mentions)[1] if mentions else inherited_owner
                quality_owner = {category for category, pattern in QUALITIES.items()
                                 if re.search(pattern, phrase)}
                if quality_owner and not mentions:
                    owner = next(iter(quality_owner)) if len(quality_owner) == 1 else None
                if owner == pref.category:
                    incoming.append(Preference(pref.category, pref.text.strip(), pref.attribute))
                    break
        result = evolve(result, incoming, text)
        if active:
            result = tuple(pref for pref in result if pref.category in active)
    return result
