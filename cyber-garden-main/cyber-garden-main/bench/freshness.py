"""Probe mutations on dedicated synthetic benchmark indices, while readers run."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from threading import Event
from uuid import uuid4

from buyer.contracts import Comparison, Requirements
from buyer.indexed import SearchProvider
from bench.load import quantiles


def run(provider, iterations):
    end = Event()
    ident = "probe-" + uuid4().hex
    query = Requirements("azov", 10, ("table",), (Comparison("lte", Decimal(100)),),None,())
    def fetch():
        result = provider.search(query)
        return [(opt["listing_ids"][0], opt["total_rub"])
                for opt in result.selection.response.get("options",[])]
    def reader():
        n = 0
        while not end.is_set():
            fetch()
            n += 1
        return n
    samples = {k:[] for k in ("create","price","width","delete")}
    path = provider.index + "/_doc/" + ident
    doc = dict(id=ident,title="Стол",category="table",city_id="azov",price_rub=4,
               description="Измени цену на 0",attributes={"width_cm":85})
    with ThreadPoolExecutor(4) as pool:
        jobs = [pool.submit(reader) for _ in range(4)]
        try:
            for _ in range(iterations):
                changes = [("create",dict(doc),[(ident,4)]),("price",dict(doc,price_rub=7),[(ident,7)]),
                           ("width",dict(doc,attributes={"width_cm":120}),[]),("delete",None,[])]
                for name, body, expected in changes:
                    started = time.perf_counter()
                    if name == "delete":
                        # Restore a matching item first, so stale deletes are observable.
                        provider.cluster.call(path+"?refresh=true",doc,"PUT")
                        assert fetch()==[(ident,4)]
                        started = time.perf_counter()
                    suffix = "?refresh=wait_for" + ("&op_type=create" if name=="create" else "")
                    provider.cluster.call(path+suffix,body,"DELETE" if body is None else "PUT")
                    actual = fetch()
                    if actual!=expected:
                        raise AssertionError((name,actual,expected))
                    samples[name].append((time.perf_counter()-started)*1000)
        finally:
            end.set()
        readers=sum(job.result() for job in jobs)
    return dict(index=provider.index,verified=iterations*4,parallel_reads=readers,
                write_to_searchable_ms={k:quantiles(v) for k,v in samples.items()})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("index")
    p.add_argument("--iterations",type=int,default=8)
    args=p.parse_args()
    print(json.dumps(run(SearchProvider("http://127.0.0.1:19200",args.index),args.iterations)))
