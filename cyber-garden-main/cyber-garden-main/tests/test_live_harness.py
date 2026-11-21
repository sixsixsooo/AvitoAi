"""Live entrypoint startup checks that do not contact any gateway."""
import subprocess
import sys
import unittest

from buyer.contracts import ROOT


class LiveEntrypointTest(unittest.TestCase):
    def test_latency_script_does_not_shadow_llm_client_package(self):
        code = ("import runpy,sys; "
                "sys.path[0] = sys.argv[1]; "
                "runpy.run_path(sys.argv[2], run_name='import_only'); print('imported')")
        done = subprocess.run([sys.executable, "-c", code, str(ROOT / "bench"), str(ROOT / "bench/llm.py")],
                              cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=10)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), "imported")
