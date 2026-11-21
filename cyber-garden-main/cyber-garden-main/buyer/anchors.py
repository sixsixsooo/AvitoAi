"""Conservative local category anchors for the three supported products."""

import re
from dataclasses import dataclass

from buyer.contracts import CATEGORIES

FORMS = {
    "table": r"стол(?:ик(?:а|у|ом|и|ов)?|а|у|ом|е|ы|ов|ами|ах)?",
    "chair": r"кресл(?:о|а|у|ом|е|ы|ам|ами|ах)?|стул(?:а|у|ом|е|ы|ья|ьев|ьям|ьями|ьях)?",
    "lamp": r"ламп(?:а|у|ы|е|ой|ами|ах)?|светильник(?:а|у|ом|е|и|ов|ами|ах)?",
}
WORDS = {key: re.compile(r"(?<!\w)(?:" + forms + r")(?!\w)", re.I)
         for key, forms in FORMS.items()}
UNSUPPORTED = re.compile(
    r"(?<!\w)(?:диван|шкаф|кровать|полк[ауи]|тумб[ауеы]|монитор|компьютер|ноутбук|"
    r"табурет|пуфик|зеркало|телевизор)(?:а|у|ом|е|ы|ов|ами|ах)?(?!\w)", re.I
)
REMOVE = re.compile(r"\b(?:без|убери|исключи|удали|вычеркни|отмени)\s*$", re.I)
AFTER_REMOVE = re.compile(r"^\s*(?:(?:(?:мне|нам|теперь|уже|больше)\s+)*не\s+нуж|уже\s+есть\b)")
ADDITIONS = r"а\s+также|также|дополнительно|вместе\s+с|плюс|еще|добавь|добавить"
ADD = re.compile(r"\b(?:" + ADDITIONS + r")\b", re.I)
ONLY = re.compile(r"\b(?:только|лишь)\s+(?:(?:компьютерн\w*|письменн\w*|"
                  r"удобн\w*|офисн\w*|настольн\w*)\s+)?$", re.I)
SWITCH = re.compile(r"\b(?:а\s+)?что\s+(?:насчет|насчёт|по\s+поводу)\s*$", re.I)
AMBIGUOUS = re.compile(r"\b(?:или|либо|может|возможно|если|вместо|лучше|предпочтительнее)\b", re.I)


@dataclass(frozen=True)
class CategoryAnchor:
    categories: tuple[str, ...] | None
    reason: str
    supported: tuple[str, ...] = ()
    pending_unknown: tuple[str, ...] = ()


ITEM_CUE = re.compile(r"\b(?:" + ADDITIONS + r"|и|без|убери|исключи|удали|"
                      r"нуж\w*|хочу|ищу|найди|подбери|купить|закажи|только)\s+"
                      r"(?:(?:\w+(?:ый|ий|ой|ая|яя|ое|ые|ую|ого|ому|ыми|ых)|очень|еще|мне)\s+){0,3}$")
NON_ITEMS = re.compile(r"^(?:а|и|но|чтобы|котор\w*|где|покажи|покажите|предложи|предложите|расскажи|вариант\w*|вместе|в|на|до|от|не|еще|также|все|тот|те|же|теперь|лучше|минимум|максимум|"
                       r"шире|уже|больше|меньше|выше|ниже|более|менее|"
                       r"желательн\w*|предпочтительн\w*|обязательн\w*|очень|нуж\w*|"
                       r"хочу|ищу|добавь|добавить|убери|без|только|вместо|подбери|найди|купить|закажи|"
                       r"бюджет\w*|ширин\w*|температур\w*|свет\w*|город\w*|самовывоз\w*|"
                       r"таганрог\w*|азов\w*|ростов\w*|регулиров\w*|комплект\w*|"
                       r"тысяч\w*|рубл\w*|цен\w*|главное|как|услови\w*)$")


def _unknown_items(text):
    """Conservative item slots, not a dictionary of all unsupported products.

    An unfamiliar word in an item position is unresolved and goes to Gemma.
    Adjectives/constraint clauses are not treated as additional requested goods.
    """
    hits = {(hit.start(), hit.end()): hit for hit in UNSUPPORTED.finditer(text)}
    for hit in re.finditer(r"[а-яa-z][а-яa-z-]*", text):
        before = text[:hit.start()]
        initial_removal = not before.strip() and AFTER_REMOVE.search(text[hit.end():])
        if not (initial_removal or re.search(r"[,;]\s*$", before) or ITEM_CUE.search(before)):
            continue
        word = hit[0]
        if (NON_ITEMS.fullmatch(word) or ADD.fullmatch(word)
                or any(pattern.fullmatch(word) for pattern in WORDS.values())
                or re.match(r"\s+(?:тыс\w*|руб\w*)\b", text[hit.end():])
                or re.search(r"(?:ый|ий|ой|ая|яя|ое|ее|ые|ие|ую|ого|ому|ыми|ых)$", word)):
            continue
        hits[hit.span()] = hit
    return sorted(hits.values(), key=lambda hit: hit.start())


def _item_key(word):
    return re.sub(r"(?:ами|ах|ом|ов|а|у|е|ы)$", "", word)


def accessory_context(prefix):
    """Plain 'с' describes an item; 'вместе с' explicitly adds another item."""
    return bool(re.search(r"\b(?:для|под|к|от|на|с|со)\s*$", prefix)
                and not re.search(r"\bвместе\s+с\s*$", prefix))


def categories_from_dialog(messages, initial=()):
    """Return an anchor only when all category changes have an explicit local reading."""
    selected = set(initial)
    unsupported_requested = set()
    for message in messages:
        if message["role"] != "user":
            continue
        text = message["text"].lower().replace("ё", "е")
        mentions = sorted(
            (match.start(), match.end(), key)
            for key, pattern in WORDS.items() for match in pattern.finditer(text)
        )
        only_category = any(ONLY.search(text[:start]) or SWITCH.search(text[:start])
                            for start, _, _ in mentions)
        if only_category:
            unsupported_requested.clear()
        for match in _unknown_items(text):
            before = text[max(0, match.start() - 32):match.start()]
            after = text[match.end():match.end() + 32]
            if accessory_context(before):
                continue
            removed = (REMOVE.search(before) or re.search(r"\bне\s+нуж\w*\s*$", before)
                       or AFTER_REMOVE.search(after))
            if removed:
                unsupported_requested.discard(_item_key(match[0]))
                continue
            unsupported_requested.add(_item_key(match[0]))
        if not mentions:
            continue
        if AMBIGUOUS.search(text):
            return CategoryAnchor(None, "Неоднозначное сравнение товаров")

        adds = set()
        removes = set()
        previous_end = 0
        previous_removed = False
        for start, end, key in mentions:
            before = text[max(0, start - 32):start]
            after = text[end:end + 32]
            removed = bool(REMOVE.search(before) or re.search(r"\bне\s+(?:нуж\w*|хочу)\s*$", before)
                           or AFTER_REMOVE.search(after))
            if previous_removed and re.fullmatch(r"\s*(?:,\s*)?(?:и\s+)?", text[previous_end:start]):
                removed = True
            if removed:
                removes.add(key)
            else:
                adds.add(key)
            previous_end, previous_removed = end, removed
        if adds & removes:
            return CategoryAnchor(None, "Противоречивое изменение состава")

        if not selected or only_category:
            selected = adds
        elif ADD.search(text) or removes or adds <= selected:
            selected.update(adds)
        else:
            return CategoryAnchor(None, "Неясно, заменён ли состав комплекта")
        selected.difference_update(removes)

    if unsupported_requested:
        return CategoryAnchor(None, "В запросе есть товар вне каталога",
                              tuple(key for key in CATEGORIES if key in selected),
                              tuple(sorted(unsupported_requested)))
    if not selected:
        return CategoryAnchor(None, "Категории нужно уточнить")
    result = tuple(key for key in CATEGORIES if key in selected)
    return CategoryAnchor(result, "Явные категории и изменения", result)
