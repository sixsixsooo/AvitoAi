"""Shared safe configuration, reports and pacing for real integration tests."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import math
import os
from pathlib import Path
import time

from buyer.contracts import ROOT
from buyer.model import SETTINGS, configuration
from tests.live_observer import OFFICIAL


def preflight():
    path = ROOT / ".env"
    local = dict(line.split("=", 1) for line in path.read_text(encoding="utf-8-sig").splitlines()
                 if "=" in line and not line.lstrip().startswith("#"))
    missing = [key for key in (*SETTINGS, "TEAM_NAME", "LLM_EXPIRES_AT") if not local.get(key)]
    if missing:
        raise RuntimeError("Missing local settings: " + ", ".join(missing))
    if datetime.fromisoformat(local["LLM_EXPIRES_AT"]) <= datetime.now(timezone.utc):
        raise RuntimeError("Configured access expiry has passed")
    settings = configuration()
    if any(settings[key] != local[key] for key in SETTINGS):
        raise RuntimeError("Environment overrides root .env; live test stopped")
    if settings["LLM_BASE_URL"] != OFFICIAL or settings["LLM_MODEL"] != "google/gemma-4-31B-it":
        raise RuntimeError("Official gateway and model required")
    return {"started_at_utc": datetime.now(timezone.utc).isoformat(), "team": local["TEAM_NAME"],
            "endpoint": OFFICIAL, "model": settings["LLM_MODEL"], "configuration_source": "root .env"}


def write_report(path, data, append=False):
    local = dict(line.split("=", 1) for line in (ROOT / ".env").read_text().splitlines() if "=" in line)
    raw = json.dumps(data, ensure_ascii=False, default=str, indent=None if append else 2)
    raw = raw.replace(local["LLM_API_KEY"], "[REDACTED]")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a" if append else "w", encoding="utf-8") as stream:
        stream.write(raw + "\n")


def denied(error):
    return "HTTP 401" in str(error) or "HTTP 403" in str(error)


class Pacer:
    # At most two transport attempts per outer call: <=18 attempts per 60s.
    def __init__(self):
        self.previous = None
        self.not_before = 0

    def wait(self):
        regular = self.previous + 7 if self.previous is not None else 0
        time.sleep(max(0, max(regular, self.not_before) - time.monotonic()))
        self.previous = time.monotonic()

    def after(self, observed):
        for event in observed:
            if event.get("kind") != "http_attempt" or event.get("status") != 429:
                continue
            value = event.get("retry_after") or ""
            try:
                delay = int(value) if value.isdigit() else (
                    parsedate_to_datetime(value).timestamp() - time.time())
                self.not_before = max(self.not_before, time.monotonic() + max(0, delay))
            except (ValueError, TypeError, OverflowError):
                pass


def percentiles(values):
    ordered = sorted(values)
    return {f"p{p}": round(ordered[math.ceil(p * len(ordered) / 100) - 1], 3)
            for p in (50, 95, 99)} if ordered else {}


def child_environment(observer_directory, trace):
    def shell_path(path):
        value = str(path).replace("\\", "/")
        return "/" + value[0].lower() + value[2:] if os.name == "nt" and value[1:2] == ":" else value
    env = os.environ.copy()
    env["PYTHONPATH"] = (":" if os.name == "nt" else os.pathsep).join(
        shell_path(path) for path in (observer_directory, ROOT))
    env["SMARTSET_REAL_TRACE"] = str(trace)
    Path(observer_directory, "sitecustomize.py").write_text(
        'import os\nfrom tests.live_observer import install\ninstall(os.environ["SMARTSET_REAL_TRACE"])\n',
        encoding="utf-8")
    return env
