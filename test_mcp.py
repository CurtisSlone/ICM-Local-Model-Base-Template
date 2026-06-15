#!/usr/bin/env python3
"""
MCP server protocol test — no Claude Code, no Ollama needed.

Spawns the stdio server pointed at an ICM, drives the full handshake (initialize →
initialized → tools/list), then calls `verify` (the deterministic oracle) on known-good
and known-broken code. Confirms the engine in icm-core drives the ICM's data correctly.

    python3 test_mcp.py [/path/to/icm]      # default: ../ICM-Rust
"""
import json
import pathlib
import subprocess
import sys

CORE = pathlib.Path(__file__).parent.resolve()
ICM = sys.argv[1] if len(sys.argv) > 1 else str(CORE.parent / "ICM-Rust")


def main():
    proc = subprocess.Popen(
        [sys.executable, str(CORE / "icm_mcp.py"), "--icm", ICM],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    def send(obj):
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()

    def recv():
        return json.loads(proc.stdout.readline())

    ok_all = True

    def check(label, cond):
        nonlocal ok_all
        ok_all = ok_all and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    # 1. handshake
    send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
          "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "test"}}})
    init = recv()
    check("initialize → protocolVersion 2025-11-25",
          init.get("result", {}).get("protocolVersion") == "2025-11-25")
    send({"jsonrpc": "2.0", "method": "notifications/initialized"})   # notification, no reply

    # 2. ping
    send({"jsonrpc": "2.0", "id": 2, "method": "ping"})
    check("ping → empty result", recv().get("result") == {})

    # 3. tools/list — find the verify tool generically (the one taking `code`)
    send({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
    tools = recv()["result"]["tools"]
    print("  tools:", [t["name"] for t in tools])
    verify_tools = [t["name"] for t in tools if "code" in t["inputSchema"].get("properties", {})]
    check("tools/list exposes a verify-style tool (takes `code`)", bool(verify_tools))
    check("each tool has an object inputSchema",
          all(t["inputSchema"]["type"] == "object" for t in tools))
    vname = verify_tools[0] if verify_tools else None

    # 4. tools/call verify — known-good
    send({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
          "params": {"name": vname,
                     "arguments": {"code": "fn main() { for _ in 0..3 { println!(\"hi\"); } }"}}})
    good = recv()["result"]
    print("  verify(good):", good["content"][0]["text"][:60])
    check("verify(good) → isError false", good.get("isError") is False)

    # 5. tools/call verify — known-broken
    send({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
          "params": {"name": vname,
                     "arguments": {"code": "fn main() { while true { println!(\"{}\", x); } }"}}})
    bad = recv()["result"]
    print("  verify(bad): ", bad["content"][0]["text"][:70].replace("\n", " "))
    check("verify(bad) → isError true", bad.get("isError") is True)

    # 6. unknown tool → JSON-RPC error
    send({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
          "params": {"name": "nope", "arguments": {}}})
    err = recv()
    check("unknown tool → JSON-RPC error -32602", err.get("error", {}).get("code") == -32602)

    proc.stdin.close()
    proc.terminate()
    print("\n", "ALL PASS ✓" if ok_all else "SOME FAILED ✗")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
