"""Create a fresh named index and stream gzipped JSON Lines via the Bulk API."""
import argparse
import gzip
import json
import time

from buyer.contracts import read_catalog
from buyer.indexed import Cluster, index_name


def create(cluster, index, shards=1, replicas=0):
    fields = {key: {"type": "keyword"} for key in ("id", "category", "city_id")}
    fields.update(title={"type": "text", "analyzer": "russian"},
                  description={"type": "text", "index": False},
                  price_rub={"type": "long"},
                  attributes={"type": "object", "dynamic": False, "properties": {
                      "width_cm": {"type": "long"}, "color_temperature_k": {"type": "long"},
                      "adjustable_height": {"type": "boolean"}}})
    cluster.call(index_name(index), {"settings": {"number_of_shards": shards,
                 "number_of_replicas": replicas, "refresh_interval": "-1"},
                 "mappings": {"dynamic": "strict", "properties": fields}}, "PUT")


def bulk(cluster, index, batch):
    result = cluster.call(index + "/_bulk?filter_path=errors,items.*.error", b"".join(batch), timeout=90)
    if result["errors"]:
        raise ValueError(str(result)[:800])


def load(cluster, index, catalog, batch_size, shards, replicas=0):
    start = time.perf_counter()
    create(cluster, index, shards, replicas)
    batch = []
    count = 0
    with gzip.open(catalog, "rb") as stream:
        for row in stream:
            doc = json.loads(row)
            item, = read_catalog([doc])
            header = json.dumps({"create": {"_id":item.id}}, separators=(",", ":")).encode()
            batch.append(header + b"\n" + row.rstrip(b"\r\n") + b"\n")
            count += 1
            if len(batch) == batch_size:
                bulk(cluster, index, batch)
                batch.clear()
        if batch:
            bulk(cluster, index, batch)
    cluster.call(index + "/_settings", {"index.refresh_interval": "1s"}, "PUT")
    cluster.call(index + "/_refresh", method="POST", timeout=120)
    actual = cluster.call(index + "/_count")["count"]
    if actual != count:
        raise ValueError((actual, count))
    return {"count": count, "index_s": round(time.perf_counter()-start, 3),
            "batch": batch_size, "shards": shards, "replicas": replicas,
            "bytes":cluster.call(index + "/_stats/store")["_all"]["total"]["store"]["size_in_bytes"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog")
    parser.add_argument("index")
    parser.add_argument("--batch",type=int,default=4000)
    parser.add_argument("--shards",type=int,default=1)
    parser.add_argument("--url", default="http://127.0.0.1:19200")
    args = parser.parse_args()
    print(json.dumps(load(Cluster(args.url), args.index, args.catalog, args.batch, args.shards)))
