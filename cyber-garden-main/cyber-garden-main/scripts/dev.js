#!/usr/bin/env node
"use strict";

const { spawn } = require("node:child_process");
const path = require("node:path");

const root = path.resolve(__dirname, "..");
const forwarded = process.argv.slice(2);
const pythonArgs = ["-m", "buyer.demo", ...forwarded];
const candidates = process.platform === "win32"
  ? [["py", ["-3", ...pythonArgs]], ["python", pythonArgs]]
  : [["python3", pythonArgs], ["python", pythonArgs]];

function launch(index) {
  if (index >= candidates.length) {
    console.error("Python 3.10+ was not found. Install Python and enable it in PATH, then run npm run dev again.");
    process.exitCode = 1;
    return;
  }

  const [command, args] = candidates[index];
  const child = spawn(command, args, {
    cwd: root,
    env: { ...process.env, PYTHONPATH: [root, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter) },
    stdio: "inherit",
    windowsHide: true,
  });

  child.once("error", (error) => {
    if (error.code === "ENOENT") return launch(index + 1);
    console.error(`Could not start ${command}: ${error.message}`);
    process.exitCode = 1;
  });
  child.once("exit", (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
      return;
    }
    if (code !== 0 && code !== null) process.exitCode = code;
  });
}

launch(0);
