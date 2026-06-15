#!/usr/bin/env python3
"""
ICM MCP server (icm-core) — a thin, GENERIC doorway onto any ICM.

This is the reusable ENGINE. Code lives here; the DATA (manifest, kb, flows, refdocs,
config) lives in an ICM directory you point at with --icm. The server sets ICM_DIR so the
core runners read that ICM's data, but it imports the runners from HERE — the content dir
is never put on sys.path, so an ICM's files can never shadow the engine.

    python3 icm_mcp.py --icm /home/local/ollama/ICM-Rust
    python3 icm_mcp.py --icm /home/local/ollama/icm-core/icms/skeleton

Speaks MCP over stdio (JSON-RPC, spec 2025-11-25). Register with Claude Code:
    claude mcp add icm-rust -- python3 /home/local/ollama/icm-core/icm_mcp.py --icm /home/local/ollama/ICM-Rust
or a local model's agent loop speaks the same protocol.

Tool `kind`s (handlers) are fixed here; each ICM's config picks which to expose and names
them: verify → icm_verify, docs_search → icm_docs, kb_answer → flow 'answer-with-fallback',
generate_verify → flow 'code-from-kb'. stdout = protocol only; logging → stderr.
"""

import json
import os
import pathlib
import sys

PROTOCOL_VERSION = "2025-11-25"
CORE_DIR = pathlib.Path(__file__).parent.resolve()


def log(*a):
    print("[icm_mcp]", *a, file=sys.stderr, flush=True)


def _resolve_icm():
    """--icm sets the DATA dir (ICM_DIR env). We do NOT add it to sys.path — engine
    code always comes from icm-core; only data is read from the ICM dir."""
    argv = sys.argv[1:]
    icm = None
    for i, a in enumerate(argv):
        if a == "--icm":
            icm = argv[i + 1]
        elif a.startswith("--icm="):
            icm = a.split("=", 1)[1]
    if not icm:
        icm = os.environ.get("ICM_DIR")
    if not icm:
        raise SystemExit("usage: python3 icm_mcp.py --icm <icm-data-dir>")
    icm = str(pathlib.Path(icm).resolve())
    os.environ["ICM_DIR"] = icm
    return icm


ICM = _resolve_icm()
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))               # ensure engine modules resolve here
import icm_config                                   # noqa: E402
CFG = icm_config.load()                             # reads ICM_DIR/icm.config.json; applies model env
import icm_verify                                   # noqa: E402
import icm_docs                                     # noqa: E402
import icm_repair                                   # noqa: E402
import icm_flow                                     # noqa: E402

VCFG = CFG["verify"]

INPUT_SCHEMAS = {
    "verify": {"type": "object", "properties": {
        "code": {"type": "string", "description": "the full source to verify"},
        "expect": {"type": "array", "items": {"type": "string"},
                   "description": "optional tokens that must appear in program output"}},
        "required": ["code"]},
    "docs_search": {"type": "object", "properties": {
        "query": {"type": "string", "description": "natural-language search query"},
        "k": {"type": "integer", "description": "number of results (default 5)"}},
        "required": ["query"]},
    "kb_answer": {"type": "object", "properties": {
        "question": {"type": "string", "description": "the question to answer"}},
        "required": ["question"]},
    "generate_verify": {"type": "object", "properties": {
        "task": {"type": "string", "description": "what the code should do"},
        "max_repairs": {"type": "integer", "description": "repair attempts (default 4)"}},
        "required": ["task"]},
}


def _tools():
    return [{"name": t["name"], "description": t["description"],
             "inputSchema": INPUT_SCHEMAS[t["kind"]]}
            for t in CFG["tools"] if t["kind"] in INPUT_SCHEMAS]


def _kind_for(name):
    for t in CFG["tools"]:
        if t["name"] == name:
            return t["kind"]
    return None


# --- tool handlers: return (text, is_error) -----------------------------------
def _h_verify(args):
    ok, diag = icm_verify.verify_code(args["code"], VCFG, expect=args.get("expect"))
    if ok:
        return "PASS — compiles and runs clean under the configured oracle.", False
    return "FAIL — verification diagnostics:\n" + diag, True


def _h_docs_search(args):
    hits = icm_docs.search(args["query"], int(args.get("k", 5)))
    if not hits:
        return "No matching items found.", False
    return "\n\n".join(f"### {h['kind']} {h['title']}\n{h['text']}" for h in hits), False


def _h_kb_answer(args):
    flow = icm_flow.load_flow("answer-with-fallback")
    result, _ = icm_flow.run_flow(flow, args["question"], verbose=False)
    tier = {1: "KB", 2: "docs fallback", 0: "not covered"}.get(result.get("tier"), "?")
    return f"[tier: {tier}]\n{result.get('answer', '')}", False


def _h_generate_verify(args):
    flow = icm_flow.load_flow("code-from-kb")
    result, _ = icm_flow.run_flow(flow, args["task"], verbose=False)
    if result.get("ok"):
        return f"VERIFIED ({result.get('repairs', 0)} repair(s)):\n{result.get('code','')}", False
    return "FAILED to converge — oracle refused to ship. Last attempt:\n" + result.get("code", ""), True


HANDLERS = {"verify": _h_verify, "docs_search": _h_docs_search,
            "kb_answer": _h_kb_answer, "generate_verify": _h_generate_verify}


# --- JSON-RPC plumbing --------------------------------------------------------
def _ok(rid, result):
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _err(rid, code, message):
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def handle(msg):
    method, rid = msg.get("method"), msg.get("id")
    if method == "initialize":
        return _ok(rid, {"protocolVersion": PROTOCOL_VERSION,
                         "capabilities": {"tools": {"listChanged": False}},
                         "serverInfo": {"name": CFG.get("name", "icm"), "version": "0.1.0"}})
    if method == "notifications/initialized":
        return None
    if method == "ping":
        return _ok(rid, {})
    if method == "tools/list":
        return _ok(rid, {"tools": _tools()})
    if method == "tools/call":
        params = msg.get("params", {})
        name = params.get("name")
        kind = _kind_for(name)
        if kind is None or kind not in HANDLERS:
            return _err(rid, -32602, f"Unknown tool: {name}")
        try:
            text, is_error = HANDLERS[kind](params.get("arguments", {}))
        except Exception as ex:
            log(f"tool {name} raised: {ex!r}")
            return _ok(rid, {"content": [{"type": "text", "text": f"tool error: {ex}"}],
                             "isError": True})
        return _ok(rid, {"content": [{"type": "text", "text": text}], "isError": is_error})
    if rid is None:
        return None
    return _err(rid, -32601, f"Method not found: {method}")


def main():
    log(f"serving ICM '{CFG.get('name')}' data={ICM} "
        f"model={os.environ.get('OLLAMA_MODEL')} tools={[t['name'] for t in CFG['tools']]}")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
