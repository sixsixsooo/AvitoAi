"""Closed-loop load on an already indexed catalogue. One in-flight request per worker."""
import argparse
import json
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from urllib.request import Request, urlopen

from buyer.conversation import ConversationManager
from buyer.demo import DemoServer
from buyer.indexed import SearchProvider


def quantiles(values):
    seq = sorted(values)
    return {f"p{p}":round(seq[math.ceil(p*len(seq)/100)-1],3)
            for p in (50,95,99)} if seq else {}


def dialogue(i):
    city = ("Таганрог", "Азов", "Ростов-на-Дону")[i % 3]
    return [{"role":"user","text":f"Стол шириной до {80+i%31} см, кресло и лампа. Город {city}, бюджет {8000+i%17*500} рублей."}]


def resources(provider):
    nodes = provider.cluster.call("_nodes/stats/process,jvm")["nodes"].values()
    cpu = sum(n["process"]["cpu"]["total_in_millis"] for n in nodes)
    heap = sum(n["jvm"]["mem"]["heap_used_in_bytes"] for n in nodes)
    return cpu, heap


def measure(provider, workers, n, mode, cache, port, pids=()):
    engine, lat, failures = [], [], []
    states = [ConversationManager().local(dialogue(i)).requirements for i in range(n)]
    def request(i):
        start = time.perf_counter()
        try:
            if mode == "http":
                body = json.dumps({"messages":dialogue(i)}).encode()
                with urlopen(Request(f"http://127.0.0.1:{port}/api/buy", data=body,
                             headers={"Content-Type":"application/json"}), timeout=35) as response:
                    answer = json.load(response)
                if answer["response"]["status"] != "ok" or answer["trace"]["model"] != "Локальный разбор":
                    raise ValueError("unexpected backend route or result")
            else:
                result = provider.search(states[i], cache=cache)
                if result.selection.response["status"] != "ok":
                    raise ValueError("unexpected status")
                engine.append(result.engine_ms)
            lat.append((time.perf_counter()-start)*1000)
        except Exception as error:
            failures.append(repr(error))
    cpu, heap = resources(provider)
    rss = []
    def sample():
        if pids:
            rows = subprocess.check_output(["ps","-o","rss=","-p",",".join(pids)],text=True)
            rss.append(sum(int(r) for r in rows.split())*1024)
    sample()
    start, pycpu = time.perf_counter(), time.process_time()
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(request,i) for i in range(n)]
        while not all(f.done() for f in futures):
            sample()
            time.sleep(.02)
        for f in futures:
            f.result()
    elapsed = time.perf_counter()-start
    after, endheap = resources(provider)
    return dict(index=provider.index, workers=workers, n=n, mode=mode, request_cache=cache,
                latency_ms=quantiles(lat), engine_ms=quantiles(engine), rps=round(len(lat)/elapsed,1),
                errors=len(failures),timeouts=sum("timed out" in x or "Timeout" in x for x in failures),
                first_error=next(iter(failures),None), engine_cpu_cores=round((after-cpu)/1000/elapsed,2),
                python_cpu_cores=round((time.process_time()-pycpu)/elapsed,2),
                heap_before_after=[heap,endheap], rss_sampled_max=max(rss,default=None))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("indices",nargs="+")
    p.add_argument("--n",type=int,default=150)
    p.add_argument("--workers",nargs="+",type=int,default=[1,8])
    p.add_argument("--pids",nargs="*",default=[])
    p.add_argument("--url",default="http://127.0.0.1:19200")
    a=p.parse_args()
    for index in a.indices:
        provider = SearchProvider(a.url,index)
        with DemoServer(0,None,provider) as server:
            Thread(target=server.serve_forever,daemon=True).start()
            for w in a.workers:
                for mode,cache in [("core",False),("core",True),("http",None)]:
                    print(json.dumps(measure(provider,w,a.n,mode,cache,server.server_port,a.pids)),flush=True)
            server.shutdown()
