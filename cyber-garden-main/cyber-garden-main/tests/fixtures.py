"""Small catalog with boundary values, unknown attributes and multiple cities."""

from buyer.contracts import FEATURES


def product(listing_id, category, price, value=None, city="taganrog"):
    return {
        "id": listing_id, "category": category, "price_rub": price, "city_id": city,
        "title": "Учебное объявление " + listing_id,
        "description": "Описание не содержит сведений для фильтрации.",
        "attributes": {} if value is None else {FEATURES[category]: value},
    }


CATALOG = [
    product("t80", "table", 2700, 80),
    product("t100", "table", 3500, 100),
    product("t105", "table", 3300, 105),
    product("t110", "table", 4000, 110),
    product("tUnknown", "table", 1000),
    product("tFake", "table", 14000, 90),
    product("cFixed", "chair", 1500, False),
    product("cAdj", "chair", 2000, True),
    product("cUnknown", "chair", 800),
    product("l2700", "lamp", 600, 2700),
    product("l3000", "lamp", 700, 3000),
    product("l4000", "lamp", 900, 4000),
    product("lUnknown", "lamp", 400),
    product("azTable", "table", 2000, 100, "azov"),
    product("azChair", "chair", 1200, False, "azov"),
    product("azLamp", "lamp", 500, 3300, "azov"),
    product("rdTable", "table", 1700, 90, "rostov-na-donu"),
    product("rdChair", "chair", 1000, True, "rostov-na-donu"),
    product("rdLamp", "lamp", 1200, 4000, "rostov-na-donu"),
]
