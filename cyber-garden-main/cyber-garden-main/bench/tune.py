"""Paired bulk and shard trials. Each build uses a fresh disposable index."""
import argparse
import json
from uuid import uuid4

from bench.index import load
from bench.load import measure
from buyer.indexed import SearchProvider


def trials(catalog, url):
    configs=[(200,1),(1000,1),(2000,1),(4000,1),(2000,3)]
    for rep in range(2):
        for batch,shards in (configs if rep==0 else reversed(configs)):
            name="smartset-trial-"+uuid4().hex[:12]
            provider=SearchProvider(url,name)
            build=load(provider.cluster,name,catalog,batch,shards)
            print(json.dumps(dict(build,trial=rep,index=name)),flush=True)
            if batch==2000:
                for w in (1,8):
                    result=measure(provider,w,150,"core",False,0)
                    print(json.dumps(dict(result,shards=shards,trial=rep)),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("catalog")
    p.add_argument("--url",default="http://127.0.0.1:19200")
    args = p.parse_args()
    trials(args.catalog,args.url)
