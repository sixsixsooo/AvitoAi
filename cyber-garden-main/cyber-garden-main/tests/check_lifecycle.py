"""Repeat discovery in one process and audit server/thread/child cleanup.

python3 -m tests.check_lifecycle --repeats 2
This diagnostic does not change test or production timeouts.
"""
import argparse
from http.server import HTTPServer
import json
import socket
import subprocess
import threading
import time
import unittest
from unittest.mock import patch

from buyer.contracts import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    for iteration in range(args.repeats):
        servers, threads, processes = [], [], []
        original_bind = HTTPServer.server_bind
        original_start = threading.Thread.start
        original_popen = subprocess.Popen

        def bind(server):
            original_bind(server)
            servers.append(server)

        def start(thread):
            original_start(thread)
            threads.append(thread)

        class TrackedProcess(original_popen):
            def __init__(self, *values, **options):
                super().__init__(*values, **options)
                processes.append(self)

        began = time.perf_counter()
        with patch.object(HTTPServer, "server_bind", bind), patch.object(threading.Thread, "start", start), \
             patch.object(subprocess, "Popen", TrackedProcess):
            suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
            result = unittest.TextTestRunner(verbosity=1).run(suite)
        # All existing fixtures must have performed their shutdown/join/close.
        # The one short drip-response handler can be completing an already closed
        # client; join it to distinguish that from a permanently running worker.
        for thread in threads:
            thread.join(timeout=1)
        live = [thread.name for thread in threads if thread.is_alive()]
        open_servers = [server.server_address for server in servers if server.socket.fileno() != -1]
        reachable = []
        for server in servers:
            with socket.socket() as probe:
                probe.settimeout(.1)
                if probe.connect_ex(server.server_address) == 0:
                    reachable.append(server.server_address)
        children = [process.pid for process in processes if process.poll() is None]
        report = {"iteration": iteration + 1, "tests": result.testsRun, "success": result.wasSuccessful(),
                  "elapsed_s": round(time.perf_counter() - began, 3), "servers_created": len(servers),
                  "threads_created": len(threads), "children_created": len(processes),
                  "live_threads": live, "open_servers": open_servers, "reachable_servers": reachable,
                  "running_children": children}
        print(json.dumps(report), flush=True)
        if not result.wasSuccessful() or live or open_servers or reachable or children:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
