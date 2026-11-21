"""Optional CPU reranker. Dependencies and weights are loaded only in hybrid mode."""
from functools import lru_cache
from pathlib import Path
from threading import Lock


class SemanticRanker:
    def __init__(self, folder):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        self.np = np
        self.lock = Lock()
        self.session = ort.InferenceSession(str(Path(folder)/"model.onnx"), sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(Path(folder)/"tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=128)
        self.encode = lru_cache(maxsize=2048)(self._encode)

    def _encode(self, text):
        # Limit simultaneous inference and cap the CPU working set.
        with self.lock:
            ids = self.np.array([self.tokenizer.encode(text).ids], dtype=self.np.int64)
            tensors = {"input_ids":ids,"attention_mask":self.np.ones_like(ids),
                       "token_type_ids":self.np.zeros_like(ids)}
            hidden = self.session.run(None,{key.name:tensors[key.name]
                                      for key in self.session.get_inputs()})[0]
            vec = hidden.mean(axis=1).flatten()
            return vec/self.np.linalg.norm(vec)

    def fuse(self, text, candidates, titles, lexical):
        query = self.encode("query: " + text[:512])
        dense = sorted(candidates,key=lambda i: (-float(query @ self.encode(
            "passage: " + titles[i][:512])), i))
        combined = {}
        for ranking in (lexical[:3], dense[:3]):
            for position, ident in enumerate(ranking):
                combined[ident] = combined.get(ident,0) + 1/(61+position)
        return sorted(combined,key=lambda ident:-combined[ident])
