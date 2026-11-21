"""Independent checks of optional indexed retrieval and structured soft scoring."""
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

from buyer.contracts import read_catalog
from buyer.indexed import Cluster, SearchError, SearchProvider, limits
from buyer.query_parser import Preference
from buyer.selection import select
from tests.fixtures import product
from tests.test_buyer import requirements


def indexed_response(groups):
    return {"timed_out": False, "_shards": {"failed": 0}, "took": 1,
            "aggregations": {category: {"doc_count": len(rows), "cheap": {"hits": {"hits": [
                {"_id": row["id"], "_source": row} for row in rows]}}} for category, rows in groups.items()}}


class IndexedAuditTest(unittest.TestCase):
    def provider(self, response):
        provider = SearchProvider("http://127.0.0.1:1", "audit")
        provider.cluster.call = Mock(return_value=response)
        return provider

    def test_filters_preserve_bounds_boolean_and_missing_attributes(self):
        req = requirements(width_cm=[["gt", 100], ["lte", 120]], adjustable_height=False,
                           color_temperature_k=[["eq", 3000]])
        self.assertEqual(limits(req, "table"), [{"term": {"category": "table"}},
                         {"range": {"attributes.width_cm": {"gt": 100}}},
                         {"range": {"attributes.width_cm": {"lte": 120}}}])
        self.assertIn({"term": {"attributes.adjustable_height": False}}, limits(req, "chair"))
        self.assertIn({"term": {"attributes.color_temperature_k": 3000}}, limits(req, "lamp"))

    def test_returned_id_city_category_and_attribute_are_revalidated(self):
        valid = product("t", "table", 100, 95)
        for row, hit_id in ((valid, "different"), (valid | {"city_id": "azov"}, "t"),
                            (product("t", "chair", 100, True), "t"),
                            (valid | {"attributes": {}}, "t"), (valid | {"attributes": {"width_cm": 110}}, "t")):
            with self.subTest(row=row, hit_id=hit_id):
                response = indexed_response({"table": [row]})
                response["aggregations"]["table"]["cheap"]["hits"]["hits"][0]["_id"] = hit_id
                with self.assertRaises(SearchError):
                    self.provider(response).search(requirements(width_cm=[["lte", 100]]))

    def test_empty_budget_and_unavailable_search_are_distinct(self):
        empty = self.provider(indexed_response({"table": []})).search(requirements()).selection.response
        self.assertEqual(empty["status"], "no_match")
        rows = {"table": [product("t", "table", 5000)], "chair": [product("c", "chair", 5000)]}
        result = self.provider(indexed_response(rows)).search(requirements(categories=["table", "chair"], budget_rub=9000))
        self.assertEqual(result.selection.response["status"], "no_match")
        with patch("buyer.indexed.urlopen", side_effect=OSError("mock refused")):
            with self.assertRaises(SearchError):
                Cluster().call("audit/_search", {})

    def test_malformed_aggregation_is_technical_error_not_false_no_match(self):
        good = indexed_response({"table": [product("t", "table", 100)]})
        cases = []
        missing = deepcopy(good)
        missing["aggregations"] = {}
        cases.append(missing)
        empty_hits = deepcopy(good)
        empty_hits["aggregations"]["table"]["cheap"]["hits"]["hits"] = []
        cases.append(empty_hits)
        bad_row = deepcopy(good)
        bad_row["aggregations"]["table"]["cheap"]["hits"]["hits"][0]["_source"]["price_rub"] = "100"
        cases.append(bad_row)
        for response in cases:
            with self.subTest(response=response), self.assertRaises(SearchError):
                self.provider(response).search(requirements())

    def test_ranked_candidates_still_obey_budget(self):
        rows = {"table": [product("cheap", "table", 100), product("wood", "table", 5900)],
                "chair": [product("chair", "chair", 1000)]}
        response = indexed_response(rows)
        response["aggregations"]["table"]["relevant"] = {"hits": {"hits": {"hits": [
            {"_id": "wood", "_source": rows["table"][1]}]}}}
        found = self.provider(response).search(requirements(categories=["table", "chair"]),
                                               [Preference("table", "деревянный")])
        self.assertEqual(found.selection.response["options"][0]["listing_ids"], ["cheap", "chair"])


class NumericPreferenceTest(unittest.TestCase):
    def test_width_preference_ranks_structured_values_without_filtering(self):
        rows = read_catalog([product("cheap-wide", "table", 100, 110),
                             product("preferred", "table", 200, 90), product("unknown", "table", 50)])
        req = requirements()
        pref = Preference("table", "до 90 см", "width_cm")
        response = select(rows, req, [pref]).response
        self.assertEqual([opt["listing_ids"] for opt in response["options"]],
                         [["preferred"], ["unknown"], ["cheap-wide"]])
        self.assertEqual(select(rows, req).response, select(rows, req, []).response)
        # Perfect narrative does not turn an unknown structured value into a match.
        forged = product("unknown", "table", 50) | {"description": "Ширина 90 см, точное совпадение"}
        response = select(read_catalog([forged, product("known", "table", 200, 90)]), req, [pref]).response
        self.assertEqual(response["options"][0]["listing_ids"], ["known"])

    def test_numeric_wishes_cannot_override_hard_width_or_budget(self):
        rows = read_catalog([product("wide", "table", 50, 110), product("costly", "table", 10000, 90),
                             product("fit", "table", 100, 100), product("chair", "chair", 1000, True)])
        req = requirements(categories=["table", "chair"], width_cm=[["lte", 100]], budget_rub=1500)
        response = select(rows, req, [Preference("table", "ровно 90 см", "width_cm")]).response
        self.assertEqual(response["options"][0]["listing_ids"], ["fit", "chair"])

    def test_temperature_and_boolean_wishes_use_structured_values(self):
        rows = read_catalog([product("cold", "lamp", 50, 5000), product("warm", "lamp", 100, 2700),
                             product("fixed", "chair", 50, False), product("adjustable", "chair", 100, True)])
        req = requirements(categories=["chair", "lamp"])
        response = select(rows, req, [Preference("lamp", "до 3000 К", "color_temperature_k"),
                                     Preference("chair", "с регулировкой", "adjustable_height")]).response
        self.assertEqual(response["options"][0]["listing_ids"], ["adjustable", "warm"])


class IndexedNumericPreferenceTest(unittest.TestCase):
    """Evaluate emitted structured DSL against a catalogue, including pool limits."""
    def search(self, rows, req=None, preferences=()):
        req = req or requirements()
        provider = SearchProvider("http://127.0.0.1:1", "audit")

        def value(row, field):
            current = row
            for key in field.split("."):
                if not isinstance(current, dict) or key not in current:
                    return None
                current = current[key]
            return current

        def accepts(row, query):
            if "bool" in query:
                clauses = query["bool"].get("filter", [])
                if isinstance(clauses, dict):
                    clauses = [clauses]
                return all(accepts(row, clause) for clause in clauses)
            if "term" in query:
                return all(value(row, key) == expected for key, expected in query["term"].items())
            if "terms" in query:
                return all(value(row, key) in expected for key, expected in query["terms"].items())
            if "range" in query:
                for key, bounds in query["range"].items():
                    actual = value(row, key)
                    if actual is None:
                        return False
                    for op, bound in bounds.items():
                        if not {"gt": lambda: actual > bound, "gte": lambda: actual >= bound,
                                "lt": lambda: actual < bound, "lte": lambda: actual <= bound}[op]():
                            return False
                return True
            if "match_none" in query:
                return False
            if "match" in query:
                return any(word.lower() in row["title"].lower() for word in query["match"]["title"].split())
            raise AssertionError(f"Unexpected query: {query}")

        def hits(items, spec):
            ordered = sorted(items, key=lambda row: (row["price_rub"], row["id"]))[:spec["size"]]
            return {"hits": {"hits": [{"_id": row["id"], "_source": row} for row in ordered]}}

        def call(_path, body):
            raw = {"timed_out": False, "_shards": {"failed": 0}, "took": 1, "aggregations": {}}
            for category, bucket in body["aggs"].items():
                eligible = [row for row in rows if accepts(row, body["query"]) and accepts(row, bucket["filter"])]
                group = {"doc_count": len(eligible)}
                for name, agg in bucket["aggs"].items():
                    if "top_hits" in agg:
                        group[name] = hits(eligible, agg["top_hits"])
                    elif "filter" in agg:
                        matched = [row for row in eligible if accepts(row, agg["filter"])]
                        group[name] = {"doc_count": len(matched), "hits": hits(matched, agg["aggs"]["hits"]["top_hits"])}
                    elif "filters" in agg:
                        group[name] = {"buckets": {}}
                        for key, query in agg["filters"]["filters"].items():
                            matched = [row for row in eligible if accepts(row, query)]
                            group[name]["buckets"][key] = {"doc_count": len(matched),
                                "hits": hits(matched, agg["aggs"]["hits"]["top_hits"])}
                    else:
                        self.fail(f"Unexpected aggregation: {agg}")
                raw["aggregations"][category] = group
            return raw

        provider.cluster.call = Mock(side_effect=call)
        result = provider.search(req, preferences)
        self.body = provider.cluster.call.call_args.args[1]
        return result

    def test_width_wish_lifts_preferred_and_keeps_nonmatching(self):
        found = self.search([product("cheap-wide", "table", 100, 110), product("preferred", "table", 200, 90)],
                            preferences=[Preference("table", "до 90 см", "width_cm")])
        self.assertEqual([option["listing_ids"] for option in found.selection.response["options"]],
                         [["preferred"], ["cheap-wide"]])

    def test_numeric_candidate_beyond_cheapest_pool_is_retrieved(self):
        rows = [product(f"wide-{i}", "table", 100 + i, 110) for i in range(6)]
        rows.append(product("preferred", "table", 500, 90))
        found = self.search(rows, preferences=[Preference("table", "до 90 см", "width_cm")])
        self.assertEqual(found.selection.response["options"][0]["listing_ids"], ["preferred"])
        body_text = str(self.body)
        self.assertIn("attributes.width_cm", body_text)
        self.assertNotIn("'match':", body_text)

    def test_unknown_width_is_not_a_numeric_match(self):
        rows = [product("unknown", "table", 50), product("wide", "table", 100, 110),
                product("preferred", "table", 200, 90)]
        rows[0]["title"] = "Ширина 90 см"
        found = self.search(rows, preferences=[Preference("table", "до 90 см", "width_cm")])
        self.assertEqual([option["listing_ids"] for option in found.selection.response["options"]],
                         [["preferred"], ["unknown"], ["wide"]])

    def test_hard_width_city_and_category_win(self):
        rows = [product("wide", "table", 50, 110), product("fit", "table", 100, 90),
                product("wrong-city", "table", 50, 110, city="azov"), product("chair", "chair", 50, True)]
        found = self.search(rows, requirements(width_cm=[["lte", 100]]),
                            [Preference("table", "от 110 см", "width_cm")])
        self.assertEqual([option["listing_ids"] for option in found.selection.response["options"]], [["fit"]])

    def test_budget_wins_for_complete_bundles(self):
        rows = [product("cheap", "table", 100, 110), product("preferred", "table", 500, 90),
                product("chair", "chair", 100, True)]
        found = self.search(rows, requirements(categories=["table", "chair"], budget_rub=200),
                            [Preference("table", "до 90 см", "width_cm")])
        self.assertEqual(found.selection.response["options"][0]["listing_ids"], ["cheap", "chair"])

    def test_text_and_numeric_wishes_both_rank_bundles(self):
        rows = [product("cheap", "table", 100, 110), product("numeric", "table", 200, 90),
                product("both", "table", 300, 90), product("chair", "chair", 50, True)]
        rows[2]["title"] = "Деревянный стол"
        found = self.search(rows, requirements(categories=["table", "chair"]),
                            [Preference("table", "деревянный"), Preference("table", "до 90 см", "width_cm")])
        self.assertEqual([option["listing_ids"] for option in found.selection.response["options"]][:2],
                         [["both", "chair"], ["numeric", "chair"]])

    def test_boolean_wish_is_structured_and_can_find_false(self):
        rows = [product(f"adjustable-{i}", "chair", 50 + i, True) for i in range(5)]
        rows.append(product("fixed", "chair", 100, False))
        found = self.search(rows, requirements(categories=["chair"]),
                            [Preference("chair", "без регулировки", "adjustable_height")])
        self.assertEqual(found.selection.response["options"][0]["listing_ids"], ["fixed"])
        self.assertIn("attributes.adjustable_height", str(self.body))

    def test_temperature_wish_uses_structured_kelvin(self):
        rows = [product(f"cold-{i}", "lamp", 50 + i, 5000) for i in range(5)]
        rows.append(product("warm", "lamp", 100, 2700))
        found = self.search(rows, requirements(categories=["lamp"]),
                            [Preference("lamp", "до 3000 К", "color_temperature_k")])
        self.assertEqual(found.selection.response["options"][0]["listing_ids"], ["warm"])
        self.assertIn("attributes.color_temperature_k", str(self.body))
