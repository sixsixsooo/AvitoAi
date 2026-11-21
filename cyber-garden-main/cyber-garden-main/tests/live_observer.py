"""Optional live-test observer: delegates unchanged to the real HTTP client.

Never reads request headers. Activated only by the live harness, not by ./run.
"""
from datetime import datetime, timezone
from dataclasses import asdict
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

from buyer import model
from buyer.conversation import ConversationManager
from buyer.contracts import ROOT

OFFICIAL = "https://sfedu-hackathon-ai-gateway.sfedu-hackathon-ai-gateway.workers.dev/v1"


def install(path):
    local = dict(line.split("=", 1) for line in (ROOT / ".env").read_text().splitlines() if "=" in line)
    secret = local["LLM_API_KEY"]
    expires = datetime.fromisoformat(local["LLM_EXPIRES_AT"])
    lock = threading.Lock()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    def record(event):
        raw = json.dumps(event, ensure_ascii=False, default=str).replace(secret, "[REDACTED]")
        with lock, target.open("a", encoding="utf-8") as stream:
            stream.write(raw + "\n")

    real_open = urllib.request.OpenerDirector.open

    def observed_open(opener, request, *args, **kwargs):
        url = request.full_url if isinstance(request, urllib.request.Request) else request
        if not str(url).startswith(OFFICIAL + "/"):
            return real_open(opener, request, *args, **kwargs)
        if datetime.now(timezone.utc) >= expires:
            raise model.ModelError("Истёк указанный срок доступа; запрос не отправлен.")
        event = {"kind": "http_attempt", "utc": datetime.now(timezone.utc).isoformat(), "url": url}
        try:
            response = real_open(opener, request, *args, **kwargs)
            event["status"] = response.status
            return response
        except urllib.error.HTTPError as error:
            event.update(status=error.code, retry_after=error.headers.get("Retry-After"))
            raise
        except (OSError, urllib.error.URLError):
            event["status"] = "connection_error"
            raise
        finally:
            record(event)

    real_gateway = model._gateway

    def observed_gateway(request):
        start = time.perf_counter()
        event = {"kind": "model_response", "utc": datetime.now(timezone.utc).isoformat()}
        try:
            raw = real_gateway(request)
            event.update(success=True, response=json.loads(raw))
            return raw
        except model.ModelError as error:
            event.update(success=False, error=str(error))
            raise
        finally:
            event["gateway_ms"] = round((time.perf_counter() - start) * 1000, 3)
            record(event)

    urllib.request.OpenerDirector.open = observed_open
    model._gateway = observed_gateway

    real_resolve = ConversationManager.resolve

    def observed_resolve(manager, *args, **kwargs):
        state, trace = real_resolve(manager, *args, **kwargs)
        record({"kind": "resolved_state", "requirements": asdict(state.requirements),
                "preferences": [asdict(item) for item in state.preferences],
                "unresolved": sorted(state.unresolved),
                "ambiguous_attributes": sorted(state.ambiguous_attributes)})
        return state, trace

    ConversationManager.resolve = observed_resolve


def events(path):
    target = Path(path)
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()] if target.exists() else []
