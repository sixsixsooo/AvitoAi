"""Exercise lexical and hybrid modes with the same requests and indexed catalogue."""
import argparse
import json
import time
from bench.load import quantiles
from buyer.contracts import Requirements
from buyer.query_parser import Preference
from buyer.indexed import SearchProvider
from buyer.semantic import SemanticRanker
from bench.quality import CASES


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("indices",nargs="+")
    args=parser.parse_args()
    ranker=SemanticRanker(args.model)
    for index in args.indices:
        for mode in ("expanded","hybrid"):
            provider=SearchProvider("http://127.0.0.1:19200",index,ranker if mode=="hybrid" else None)
            for cycle in range(2):
                latency=[]
                for category,text,_ in CASES:
                    req=Requirements("azov",8000,(category,),(),None,())
                    start=time.perf_counter()
                    result=provider.search(req,(Preference(category,text),),cache=False)
                    assert result.selection.response["status"]=="ok"
                    latency.append((time.perf_counter()-start)*1000)
                print(json.dumps(dict(index=index,mode=mode,cycle=cycle,n=len(latency),
                                      latency_ms=quantiles(latency))),flush=True)
