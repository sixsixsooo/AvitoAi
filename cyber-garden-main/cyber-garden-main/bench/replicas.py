"""Measure one primary, then one primary plus one assigned replica on two nodes."""
import argparse
import json
from uuid import uuid4
from bench.index import load
from bench.load import measure
from buyer.indexed import SearchProvider


def main():
    p=argparse.ArgumentParser()
    p.add_argument("catalog")
    args=p.parse_args()
    name="smartset-replica-"+uuid4().hex[:12]
    s=SearchProvider("http://127.0.0.1:19200",name)
    if s.cluster.call("_cluster/health")["number_of_data_nodes"]<2:
        raise SystemExit("Two data nodes required")
    load(s.cluster,name,args.catalog,2000,1)
    for replicas in (0,1,0):
        s.cluster.call(name+"/_settings",{"index.number_of_replicas":replicas},"PUT")
        health=s.cluster.call("_cluster/health/"+name+"?wait_for_status=green&wait_for_no_relocating_shards=true&timeout=90s",timeout=100)
        assert not health["timed_out"] and health["active_shards"]==1+replicas
        stats=s.cluster.call(name+"/_stats/store")["_all"]["total"]["store"]
        for w in (1,8):
            print(json.dumps(dict(measure(s,w,150,"core",False,0),replicas=replicas,bytes=stats["size_in_bytes"])),flush=True)


if __name__=="__main__":
    main()
