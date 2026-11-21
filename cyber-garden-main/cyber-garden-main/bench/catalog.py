"""Generate nested synthetic catalogues as gzip JSON Lines, bounded memory."""
import argparse
import gzip
import json
import random
from pathlib import Path

from buyer.contracts import CITIES, ROOT


def generate(folder, sizes, seed):
    templates = json.loads((ROOT / "data/catalog.json").read_text(encoding="utf8"))
    rng = random.Random(seed)
    folder.mkdir(parents=True, exist_ok=True)
    # Precompute templates. Price, template choice and id still vary per row.
    formats = []
    for sample in templates:
        for city in CITIES:
            for missing in (False, True):
                row = dict(sample, city_id=city, id="ID_MARKER", price_rub="PRICE_MARKER")
                if missing:
                    row["attributes"] = {}
                raw = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode()
                formats.append(raw.replace(b"%", b"%%").replace(b"ID_MARKER", b"%08d")
                               .replace(b'"PRICE_MARKER"', b"%d") + b"\n")
    files = {}
    try:
        for size in sorted(set(sizes)):
            files[size] = gzip.open(folder / f"{size}.jsonl.gz", "xb", compresslevel=1)
        for start in range(0, max(sizes), 1000):
            rows = [rng.choice(formats) % (i, rng.randint(100, 25000))
                    for i in range(start, min(max(sizes), start + 1000))]
            for size, handle in files.items():
                if start < size:
                    handle.write(b"".join(rows[:size-start]))
    finally:
        for handle in files.values():
            handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000,10000,100000,1000000,10000000])
    parser.add_argument("--seed", type=int, default=6006)
    args = parser.parse_args()
    generate(args.folder, args.sizes, args.seed)
