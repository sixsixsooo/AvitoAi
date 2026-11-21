"""Fresh real-generation preflight. Never persist credentials or request headers."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import sys

from buyer.contracts import ROOT
from buyer.model import ModelError, configuration, extract


def main():
    local = dict(line.split("=", 1) for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines()
                 if "=" in line and not line.lstrip().startswith("#"))
    settings = configuration()
    expiry = datetime.fromisoformat(local["LLM_EXPIRES_AT"])
    now = datetime.now(timezone.utc)
    if expiry <= now:
        print("Access expiry has passed; no request sent.", file=sys.stderr)
        return 2
    if (settings["LLM_BASE_URL"] != "https://sfedu-hackathon-ai-gateway.sfedu-hackathon-ai-gateway.workers.dev/v1"
            or settings["LLM_MODEL"] != "google/gemma-4-31B-it"):
        print("Official gateway/model required; no request sent.", file=sys.stderr)
        return 2
    messages = [{"role": "user", "text": "Стол не шире 100 см, кресло и лампа. Таганрог. Бюджет 15000 рублей."}]
    result = {"started_at_utc": now.isoformat(), "team": local["TEAM_NAME"],
              "endpoint": settings["LLM_BASE_URL"], "model": settings["LLM_MODEL"],
              "messages": messages, "real_gateway_attempted": True}
    try:
        query, trace = extract(messages, full_history=True)
        result.update(success=True, query=asdict(query), trace=asdict(trace))
    except ModelError as error:
        result.update(success=False, error=str(error))
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    # Defence in depth against accidental inclusion by future error formatting.
    raw = json.dumps(result, ensure_ascii=False, indent=2, default=str).replace(settings["LLM_API_KEY"], "[REDACTED]")
    path = ROOT / "bench/results/live-gemma-smoke.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw + "\n", encoding="utf-8")
    print(raw)
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
