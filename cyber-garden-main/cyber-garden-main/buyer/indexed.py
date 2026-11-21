"""OpenSearch retrieval over one search snapshot. Only the seller title is searchable."""
import json
import re
from dataclasses import dataclass, replace
from decimal import ROUND_CEILING, ROUND_FLOOR
from itertools import product
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from buyer.contracts import CATEGORIES, FEATURES, Comparison, read_catalog
from buyer.selection import Selection, _explain, clarification_questions, matches, select
from buyer.ranking import ALIASES, _soft_condition, listing_scores
from buyer.query_parser import typed_preference


class SearchError(RuntimeError):
    pass


class Cluster:
    def __init__(self, url="http://127.0.0.1:19200"):
        self.url = url.rstrip("/")

    def call(self, path, body=None, method=None, timeout=30):
        payload = body if isinstance(body, bytes) else (
            json.dumps(body, ensure_ascii=False).encode() if body is not None else None)
        mime = "application/x-ndjson" if isinstance(body, bytes) else "application/json"
        request = Request(self.url + "/" + path, data=payload, method=method,
                          headers={"Content-Type": mime})
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            raise SearchError(f"OpenSearch HTTP {exc.code}: {exc.read(600).decode(errors='replace')}") from exc
        except (OSError, ValueError) as exc:
            raise SearchError("Ошибка связи с OpenSearch") from exc


def index_name(name):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", name):
        raise ValueError("Ожидается имя одного индекса")
    return name


def limits(req, category):
    field = "attributes." + FEATURES[category]
    condition = req.condition(category)
    filters = [{"term": {"category": category}}]
    if category == "chair" and condition is not None:
        filters.append({"term": {field: condition}})
    elif category != "chair":
        for comparison in condition:
            value = comparison.value
            floor = int(value.to_integral_value(rounding=ROUND_FLOOR))
            ceil = int(value.to_integral_value(rounding=ROUND_CEILING))
            op = comparison.operation
            if op == "eq":
                filters.append({"term": {field: floor}} if floor == ceil else {"match_none": {}})
            else:
                bound = floor if op in ("lte", "gt") else ceil
                filters.append({"range": {field: {op: bound}}})
    return filters


# Small Russian vocabulary validated by bench.quality, used only for positive wishes.
ORDER = [{"price_rub": "asc"}, {"id": "asc"}]


def terms(text, expand):
    return text + " " + " ".join(words for stem, words in ALIASES.items()
                                   if expand and stem in text.lower())


@dataclass
class Retrieved:
    selection: Selection
    listings: tuple
    titles: dict
    engine_ms: int


class SearchProvider:
    def __init__(self, url, index, semantic=None):
        self.semantic = semantic
        self.cluster = Cluster(url)
        self.index = index_name(index)

    def search(self, req, preferences=(), expand=True, cache=None):
        try:
            return self._search(req, preferences, expand, cache)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            raise SearchError("Некорректный или неполный ответ поискового индекса") from None

    def _search(self, req, preferences=(), expand=True, cache=None):
        if not req.categories or req.city_id is None or req.budget_rub is None:
            # No catalogue request has run: absence cannot be inferred here.
            response = {"status": "clarify", "questions": clarification_questions(req)}
            return Retrieved(Selection(response, {}, None), (), {}, 0)
        buckets = {}
        wishes = {}
        preferences = tuple(typed_preference(pref) for pref in preferences)
        numeric = tuple(pref for pref in preferences if pref.attribute and _soft_condition(pref) is not None)
        soft_requirements = {}
        for category in req.categories:
            scope = limits(req, category)
            aggs = {"cheap": {"top_hits": {"size": 3, "sort": ORDER}}}
            words = " ".join(p.text for p in preferences if p.category == category and not p.attribute
                             and not re.search(r"\b(?:не|без)\b", p.text.lower()))
            if words.strip():
                wishes[category] = words
                if self.semantic:
                    aggs["cheap"]["top_hits"]["size"] = 16
                aggs["relevant"] = {"filter": {"match": {"title": terms(words, expand)}},
                                    "aggs": {"hits": {"top_hits": {"size": 8, "sort": ["_score", *ORDER]}}}}
            soft_requirements[category] = {}
            for position, pref in enumerate(numeric):
                if pref.category != category:
                    continue
                condition = _soft_condition(pref)
                value = (condition[0][1] if category == "chair" else
                         tuple(Comparison(op, bound) for op, bound in condition))
                soft_requirements[category][f"p{position}"] = replace(req, **{pref.attribute: value})
            if soft_requirements[category]:
                # These buckets add matching candidates inside the hard scope;
                # they do not narrow the main query or remove cheap alternatives.
                aggs["numeric"] = {"filters": {"filters": {
                    key: {"bool": {"filter": limits(soft, category)}}
                    for key, soft in soft_requirements[category].items()}},
                    "aggs": {"hits": {"top_hits": {"size": 8, "sort": ORDER}}}}
            buckets[category] = {"filter": {"bool": {"filter": scope}}, "aggs": aggs}
        # Score the same positive terms in query context (filter aggregations do not score).
        should = [{"bool": {"filter": {"term": {"category": key}},
                            "must": agg["aggs"]["relevant"]["filter"]}}
                  for key, agg in buckets.items() if "relevant" in agg["aggs"]]
        query = {"filter": [{"term": {"city_id": req.city_id}}, {"terms": {"category": req.categories}}],
                 "should": should, "minimum_should_match": 0}
        path = self.index + "/_search" + ("" if cache is None else "?request_cache=" + str(cache).lower())
        raw = self.cluster.call(path, {"size": 0, "track_total_hits": False,
                                      "query": {"bool": query}, "aggs": buckets})
        if raw["timed_out"] or raw["_shards"]["failed"]:
            raise SearchError("Неполный ответ поискового индекса")
        if set(raw["aggregations"]) != set(req.categories):
            raise SearchError("Индекс не вернул все запрошенные категории")
        groups, titles, scores, counts = {}, {}, {}, {}
        for category, group in raw["aggregations"].items():
            counts[category] = group["doc_count"]
            hits = group["cheap"]["hits"]["hits"]
            ranked = group.get("relevant", {}).get("hits", {}).get("hits", {}).get("hits", [])
            if (type(counts[category]) is not int or counts[category] < 0
                    or not isinstance(hits, list) or not isinstance(ranked, list)
                    or (counts[category] > 0 and not hits)
                    or (counts[category] == 0 and (hits or ranked))):
                raise SearchError("Несогласованное число результатов индекса")
            by_id = {}
            candidates = [(hit, None) for hit in hits + ranked]
            if soft_requirements[category]:
                soft_buckets = group["numeric"]["buckets"]
                if set(soft_buckets) != set(soft_requirements[category]):
                    raise SearchError("Индекс не вернул все числовые пожелания")
                for key, soft in soft_requirements[category].items():
                    bucket = soft_buckets[key]
                    soft_hits = bucket["hits"]["hits"]["hits"]
                    count = bucket["doc_count"]
                    if (type(count) is not int or not 0 <= count <= counts[category]
                            or not isinstance(soft_hits, list) or bool(count) != bool(soft_hits)
                            or len(soft_hits) > count):
                        raise SearchError("Неполная выборка по числовому пожеланию")
                    candidates.extend((hit, soft.condition(category)) for hit in soft_hits)
            for hit, soft_condition in candidates:
                row = hit["_source"]
                item, = read_catalog([row])
                if (hit["_id"] != item.id or item.city_id != req.city_id or item.category != category
                        or not matches(item, req.condition(category))):
                    raise SearchError("Индекс вернул несогласованные данные")
                if soft_condition is not None and not matches(item, soft_condition):
                    raise SearchError("Индекс вернул неподтверждённое числовое совпадение")
                if item.id in by_id and by_id[item.id] != item:
                    raise SearchError("Индекс вернул противоречивые данные одного ID")
                by_id[item.id] = item
                titles[item.id] = row["title"]
            if len(by_id) > counts[category]:
                raise SearchError("Число ID превышает размер категории индекса")
            order = [hit["_id"] for hit in ranked]
            if self.semantic and category in wishes:
                lexical = list(dict.fromkeys(order + [hit["_id"] for hit in hits]))
                order = self.semantic.fuse(wishes[category], by_id, titles, lexical)
            for rank, ident in enumerate(order):
                scores[ident] = 8 - rank
            groups[category] = tuple(by_id.values())
        if numeric:
            numeric_scores = listing_scores(groups, numeric)
            numeric_categories = {pref.category for pref in numeric}
            for category, items in groups.items():
                has_numeric = category in numeric_categories
                has_text = category in wishes
                for item in items:
                    # Equal category scales; retain the existing indexed text
                    # order and combine it with the backend's structured score.
                    lexical = 1250 * max(0, scores.get(item.id, 0))
                    scores[item.id] = (lexical + numeric_scores[item.id]) // max(1, has_numeric + has_text)
        listings = tuple(item for group in groups.values() for item in group)
        baseline = select(listings, req)
        response = baseline.response
        if any(not count for count in counts.values()):
            missing = ", ".join(CATEGORIES[k] for k, count in counts.items() if not count)
            response = {"status":"no_match","message":f"В выбранном городе нет подходящих товаров: {missing}. Можно изменить город или условие на характеристику?"}
        elif scores and response["status"] == "ok":
            options = []
            for items in product(*(groups[key] for key in req.categories)):
                total = sum(i.price_rub for i in items)
                if total <= req.budget_rub:
                    ids = tuple(i.id for i in items)
                    options.append((-sum(scores.get(i, 0) for i in ids), total, ids, items))
            response = {"status":"ok","options":[{"listing_ids":list(ids), "total_rub":t,
                        "explanation":_explain(items, t, req, preferences)} for _, t, ids, items in sorted(options)[:3]]}
        return Retrieved(Selection(response, counts, baseline.minimum_total_rub), listings, titles, raw["took"])
