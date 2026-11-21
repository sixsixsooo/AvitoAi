"""Run the unchanged bash entrypoint on Linux and on Windows with Git Bash."""

import os
from pathlib import Path
import shlex
import sys
import tempfile

_shim = None


def cli_command(entrypoint, *args):
    global _shim
    if os.name != "nt":
        return [str(entrypoint), *map(str, args)]
    bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
    python = shlex.quote(sys.executable.replace("\\", "/"))
    if _shim is None:
        _shim = tempfile.TemporaryDirectory(prefix="smartset-python-")
        # exec in ./run requires an executable shim, not a shell function.
        Path(_shim.name, "python3").write_text(
            '#!/usr/bin/env bash\nexport PYTHONPATH="$(cygpath -wp "$PYTHONPATH")"\n'
            + 'export PYTHONUTF8=1\nexec ' + python + ' "$@"\n', encoding="utf-8", newline="\n")
    script = 'export PATH="$(cygpath -u "$1"):$PATH"; shift; exec bash "$@"'
    return [str(bash), "-c", script, "smartset-test", _shim.name, str(entrypoint), *map(str, args)]
