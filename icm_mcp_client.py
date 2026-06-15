#!/usr/bin/env python3
"""
Minimal MCP stdio client (icm-core) — the mirror of icm_mcp.py.

Phase 2 of the plan: the flow engine uses this to call tools THROUGH the MCP server at
authored nodes, instead of calling the functions in-process. Tool calls are embedded in
the workflow (the model never elects them), but they travel the real MCP boundary — so the
same server is reusable by Claude Code, a local agent loop, or these flows, unchanged.

    with McpClient(icm_dir) as mcp:
        text, is_error = mcp.call("rust_verify", {"code": src})

Spawns `icm_mcp.py --icm <icm_dir>` as a subprocess, does the 2025-11-25 handshake, then
issues tools/call over newline-delimited JSON. Pure stdlib.
"""

import json
import os
import pathlib
import subprocess
import sys

CORE_DIR = pathlib.Path(__file__).parent.resolve()


class McpClient:
    def __init__(self, icm_dir, server=None, python=None):
        self.icm_dir = str(pathlib.Path(icm_dir).resolve())
        self.server = str(server or (CORE_DIR / "icm_mcp.py"))
        self.python = python or sys.executable
        self.proc = None
        self._id = 0

    # --- lifecycle ------------------------------------------------------------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()

    def start(self):
        self.proc = subprocess.Popen(
            [self.python, self.server, "--icm", self.icm_dir],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env={**os.environ, "ICM_DIR": self.icm_dir},
        )
        init = self._rpc("initialize", {
            "protocolVersion": "2025-11-25", "capabilities": {},
            "clientInfo": {"name": "icm_flow", "version": "0.1.0"}})
        if init.get("result", {}).get("protocolVersion") != "2025-11-25":
            raise RuntimeError(f"MCP handshake failed: {init}")
        self._notify("notifications/initialized")
        return self

    def close(self):
        if self.proc:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            self.proc.terminate()
            self.proc = None

    # --- wire -----------------------------------------------------------------
    def _send(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _rpc(self, method, params=None):
        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, "method": method,
                    "params": params or {}})
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"MCP server closed during {method}: "
                               f"{self.proc.stderr.read()[:300] if self.proc.stderr else ''}")
        return json.loads(line)

    def _notify(self, method, params=None):
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    # --- tools ----------------------------------------------------------------
    def list_tools(self):
        return self._rpc("tools/list").get("result", {}).get("tools", [])

    def call(self, name, arguments):
        """Returns (text, is_error). text is the concatenated text content blocks."""
        resp = self._rpc("tools/call", {"name": name, "arguments": arguments})
        if "error" in resp:
            return f"MCP error: {resp['error'].get('message')}", True
        result = resp.get("result", {})
        text = "\n".join(b.get("text", "") for b in result.get("content", [])
                         if b.get("type") == "text")
        return text, bool(result.get("isError"))
