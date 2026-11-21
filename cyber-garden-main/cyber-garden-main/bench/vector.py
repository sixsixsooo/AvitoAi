"""Offline exact cosine and reciprocal rank fusion on the same judged titles."""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from bench.quality import TITLES, CASES


def main():
    p=argparse.ArgumentParser()
    p.add_argument("folder",type=Path)
    p.add_argument("--lexical",type=Path,default=Path("bench/results/quality.jsonl"))
    args=p.parse_args()
    lexical={(row["category"],row["text"]):row["ids"] for row in
             map(json.loads,args.lexical.read_text().splitlines()) if row["mode"]=="expanded"}
    opts=ort.SessionOptions()
    opts.intra_op_num_threads=2
    opts.inter_op_num_threads=1
    model=ort.InferenceSession(str(args.folder/"model.onnx"),sess_options=opts,providers=["CPUExecutionProvider"])
    tok=Tokenizer.from_file(str(args.folder/"tokenizer.json"))
    def encode(text):
        encoded=tok.encode(text)
        ids=np.array([encoded.ids],dtype=np.int64)
        mask=np.ones_like(ids)
        inputs={"input_ids":ids,"attention_mask":mask,"token_type_ids":np.zeros_like(ids)}
        vectors=model.run(None,{inp.name:inputs[inp.name] for inp in model.get_inputs()})[0]
        pooled=vectors.mean(axis=1).flatten()
        return pooled/np.linalg.norm(pooled)
    catalog={key:np.stack([encode("passage: "+title) for title in titles]) for key,titles in TITLES.items()}
    for key,text,grades in CASES:
        start=time.perf_counter()
        similarity=catalog[key]@encode("query: "+text)
        dense=np.argsort(-similarity).tolist()
        elapsed=(time.perf_counter()-start)*1000
        # Fuse the emitted lexical top 3 and dense top 3; ties prefer lexical ordering.
        fuse={}
        for ranking in (lexical[key,text],dense[:3]):
            for pos,ident in enumerate(ranking):
                fuse[ident]=fuse.get(ident,0)+1/(60+pos+1)
        for mode,ids in (("dense",dense[:3]),("rrf",sorted(fuse,key=lambda i:-fuse[i])[:3])):
            def dcg(g):
                return sum((2**v-1)/math.log2(i+2) for i,v in enumerate(g))
            print(json.dumps(dict(text=text,category=key,mode=mode,ids=ids,
                      recall3=len(set(ids)&grades.keys())/len(grades),
                      ndcg3=dcg([grades.get(i,0) for i in ids])/dcg(sorted(grades.values(),reverse=True)[:3]),
                      encode_and_score_ms=round(elapsed,3)),ensure_ascii=False))


if __name__=="__main__":
    main()
