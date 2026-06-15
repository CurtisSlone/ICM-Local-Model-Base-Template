#!/usr/bin/env python3
"""
ICM-Rust flow engine — authored, stateless multi-step chains (next-step #3).

The ProofLayer stateless-flow thesis, made concrete:

  - The ENGINE holds state on disk (a run dir of per-node JSON).
  - The GRAPH (flows/*.json) holds the plan — which nodes run, in what order, and
    EXACTLY which inputs each one declares.
  - Each NODE is a pure step: it is handed only its declared inputs and returns its
    declared outputs. No cumulative tape, no model deciding what to do next.

This is the inverse of a ReAct agent loop. Intelligence lives in the *structure*,
which is why a 7B stays reliable across multiple steps: no single call has to hold
the whole task in its head.

It does not reinvent the runners — it composes them as a library:
  route / read            <- icm_route.py   (the lookup axis: KB grounding)
  generate / verify+repair <- icm_repair.py (the generative axis: the cargo oracle)

The built-in `code-from-kb` flow:
  route (request -> entry_ids) -> ground (entry_ids -> context)
    -> generate (request + context -> code) -> verify (request + code -> ok, code, repairs)

Note what each node CANNOT see: `verify` never gets the routing rationale or KB context;
`generate` never gets the raw entry_ids. The engine enforces the isolation — it isn't
left to the model's good behaviour.

    python3 icm_flow.py "write a function that returns the nth Fibonacci number"
    python3 icm_flow.py --flow code-from-kb --show-code "..."
    python3 icm_flow.py                      # interactive REPL
"""

import json
import os
import pathlib
import re
import sys
import time

import icm_config                     # config (tool names, verify spec, models)
import icm_route                      # lookup axis (route, read_entry)
import icm_repair                     # generative axis (generate, verify+repair)
import icm_verify                     # oracle (direct-call fallback)
import icm_mcp_client                 # Phase 2: call tools THROUGH the MCP server

ICM_DIR = pathlib.Path(os.environ.get("ICM_DIR") or pathlib.Path(__file__).parent).resolve()
FLOWS_DIR = ICM_DIR / "flows"
RUNS_DIR = ICM_DIR / "runs"
MAX_REPAIRS = icm_repair.MAX_REPAIRS
_CFG = icm_config.load()
_VCFG = _CFG["verify"]
TOOL_NAMES = {t["kind"]: t["name"] for t in _CFG.get("tools", [])}   # kind → MCP tool name


# --- Phase 2: tool broker — authored nodes call tools THROUGH MCP --------------
# A node never elects a tool (no open agent loop); the graph calls a fixed tool by KIND.
# The broker resolves the kind to this ICM's tool name and dispatches over the MCP
# boundary, falling back to an identical in-process call if the server can't start.
def _direct_tool(kind, arguments):
    if kind == "docs_search":
        import icm_docs
        hits = icm_docs.search(arguments["query"], int(arguments.get("k", 5)))
        if not hits:
            return "No matching items found.", False
        return "\n\n".join(f"### {h['kind']} {h['title']}\n{h['text']}" for h in hits), False
    if kind == "verify":
        ok, diag = icm_verify.verify_code(arguments["code"], _VCFG, expect=arguments.get("expect"))
        return ("PASS", False) if ok else ("FAIL — verification diagnostics:\n" + diag, True)
    raise ValueError(f"no direct fallback for tool kind {kind}")


def _make_tool(mcp, verbose):
    def tool(kind, arguments):
        name = TOOL_NAMES.get(kind)
        if mcp and name:
            if verbose:
                print(f"  · tool[{kind}] → MCP {name}", file=sys.stderr)
            return mcp.call(name, arguments)
        if verbose:
            print(f"  · tool[{kind}] → in-process (no MCP)", file=sys.stderr)
        return _direct_tool(kind, arguments)
    return tool


# --- node handlers: each is (inputs: dict) -> outputs: dict, and uses ONLY inputs ---
def node_route(inputs, ctx):
    ids = icm_route.route(inputs["request"], verbose=ctx["verbose"])
    return {"entry_ids": ids}


def node_read(inputs, ctx):
    ids = inputs["entry_ids"]
    if not ids:
        if ctx["verbose"]:
            print("  · ground: no entry routed — generating without KB context", file=sys.stderr)
        return {"context": ""}
    context = "\n\n".join(f"### {cid}\n{icm_route.read_entry(cid)}" for cid in ids)
    if ctx["verbose"]:
        print(f"  · ground: read {ids} → {len(context)} chars of reference context",
              file=sys.stderr)
    return {"context": context}


def node_generate(inputs, ctx):
    context = inputs["context"].strip()
    reference = (
        f"\n\nUse these reference excerpts from a Rust knowledge base to follow correct "
        f"idioms. They are REFERENCE, not the task — implement the task below.\n"
        f"--- REFERENCE ---\n{context}\n--- END ---"
        if context else ""
    )
    prompt = (
        "You are a careful Rust engineer. Write ONE complete, self-contained program "
        "that compiles cleanly under `cargo clippy -- -D warnings` (no warnings). Rules: "
        "standard library ONLY (no external crates, no unused imports); no threads, "
        "channels, or sleep; include a real `fn main()` that exercises the code and "
        "prints results; the program must terminate. Return the FULL program in `code`."
        f"{reference}\n\nTask: {inputs['request']}"
    )
    # temp 0.3, not 0.1: with a long grounded prompt, near-greedy decode of the
    # constrained JSON reproducibly closes the code string early (truncating the
    # program). A little variance avoids that premature stop.
    code = icm_repair.generate(prompt, icm_repair.CODE_SCHEMA, temperature=0.3)["code"]
    if ctx["verbose"]:
        print(f"  · generate: {len(code)} chars proposed (grounded={bool(context)})",
              file=sys.stderr)
    return {"code": code}


def node_verify(inputs, ctx):
    # the repair loop's ORACLE call goes through the MCP `verify` tool (Phase 2):
    # the node calls a fixed tool; the model is only used for the repair proposals.
    def verify_fn(code):
        text, is_err = ctx["tool"]("verify", {"code": code, "expect": []})
        if not is_err:
            return True, ""
        # strip the tool's "FAIL — verification diagnostics:\n" prefix → raw diagnostics
        diag = text.split("diagnostics:\n", 1)[-1]
        return False, diag
    ok, code, repairs = icm_repair.generate_verify_repair(
        inputs["request"], MAX_REPAIRS, ctx["verbose"], ctx["show_code"],
        expect=[], seed=inputs["code"], verify_fn=verify_fn,
    )
    return {"ok": ok, "code": code, "repairs": repairs}


def _verified_answerable(request, context, verbose):
    """A ground-check with an ORACLE behind it. A bare 7B yes/no false-positives on
    topic overlap (a file-I/O question routed to the strings entry gets called
    'answerable'), so the model must QUOTE the supporting sentence — and code verifies
    that quote actually appears in the entry text. No real quote → not answerable →
    escalate to tier 2. Same 'model proposes, oracle verifies' move, applied to the
    uncertainty trigger itself."""
    schema = {
        "type": "object",
        "properties": {"answerable": {"type": "boolean"}, "evidence": {"type": "string"}},
        "required": ["answerable", "evidence"],
    }
    prompt = (
        "Decide if the ENTRY TEXT contains the SPECIFIC information needed to answer the "
        "QUESTION. Being on a related topic is NOT enough. If it does, set answerable=true "
        "and copy the EXACT sentence from the entry text that contains the answer into "
        "`evidence`, verbatim. If it does not specifically answer the question, set "
        "answerable=false.\n\n"
        f"--- ENTRY TEXT ---\n{context}\n--- END ---\n\nQUESTION: {request}"
    )
    v = icm_repair.generate(prompt, schema)
    if not v.get("answerable"):
        if verbose:
            print("  · ground-check: model says NOT answerable", file=sys.stderr)
        return False
    # ORACLE (fuzzy): what fraction of the evidence's significant words actually appear
    # in the entry text? A real quote scores ~1.0 even if the model reformatted backticks/
    # punctuation; a hallucinated quote about an absent topic scores low. Robust where
    # exact-substring was brittle (it false-rejected genuine quotes over formatting).
    words = lambda s: {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) >= 3}
    ev_words, ctx_words = words(v.get("evidence", "")), words(context)
    overlap = len(ev_words & ctx_words) / len(ev_words) if ev_words else 0.0
    grounded = len(ev_words) >= 4 and overlap >= 0.7
    if verbose:
        tag = (f"evidence VERIFIED ({overlap:.0%} in text)" if grounded
               else f"evidence WEAK ({overlap:.0%} in text) → reject (escalate)")
        print(f"  · ground-check: model says answerable; {tag}", file=sys.stderr)
    return grounded


def node_ground(inputs, ctx):
    """Read the routed entries AND run the verified ground-check. answerable=False is
    the uncertainty signal that triggers the tier-2 doc search downstream."""
    ids = inputs["entry_ids"]
    if not ids:
        return {"context": "", "answerable": False}
    context = "\n\n".join(f"### {cid}\n{icm_route.read_entry(cid)}" for cid in ids)
    answerable = _verified_answerable(inputs["request"], context, ctx["verbose"])
    return {"context": context, "answerable": answerable}


def node_docsearch(inputs, ctx):
    """Tier-2 fallback: only searches the docs when the KB ground-check failed. When the
    KB covered it, this node is a no-op. The search runs through the MCP docs_search tool."""
    if inputs["answerable"]:
        if ctx["verbose"]:
            print("  · doc-search: skipped (KB covered it)", file=sys.stderr)
        return {"doc_text": ""}
    k = int(os.environ.get("ICM_DOCS_K", "5"))
    text, _ = ctx["tool"]("docs_search", {"query": inputs["request"], "k": k})
    return {"doc_text": text}


def node_answer(inputs, ctx):
    """Answer from the KB (tier 1) if it covered the question; else from the std doc
    excerpts (tier 2); else decline. The answer is grounded in whichever text we have."""
    request = inputs["request"]
    if inputs["answerable"] and inputs["context"].strip():
        prompt = (
            "Answer the question using ONLY the Rust knowledge-base entry text below. "
            "Quote syntax and type names exactly. If it doesn't contain the answer, say so.\n\n"
            f"--- ENTRY TEXT ---\n{inputs['context']}\n--- END ---\n\nQuestion: {request}"
        )
        return {"answer": icm_repair.generate(prompt).strip(), "tier": 1}
    doc_text = inputs.get("doc_text", "")
    if doc_text.strip():
        prompt = (
            "The curated knowledge base did not cover this question, so here are excerpts "
            "from the reference docs. Answer using ONLY these excerpts and name the relevant "
            "item(s) (e.g. `Vec::pop`). If they don't answer it, say so.\n\n"
            f"--- DOC EXCERPTS ---\n{doc_text}\n--- END ---\n\nQuestion: {request}"
        )
        return {"answer": icm_repair.generate(prompt).strip(), "tier": 2}
    return {"answer": "That isn't covered in the knowledge base or the indexed docs.",
            "tier": 0}


HANDLERS = {
    "route": node_route,
    "read": node_read,
    "generate": node_generate,
    "verify": node_verify,
    "ground": node_ground,
    "docsearch": node_docsearch,
    "answer": node_answer,
}


# --- the engine: run the authored graph, persisting state to disk ------------------
def load_flow(name):
    path = FLOWS_DIR / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"no such flow: {name} (looked in {FLOWS_DIR})")
    return json.loads(path.read_text())


def run_flow(flow, request, verbose=True, show_code=False):
    run_id = time.strftime("run-%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / flow["name"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Phase 2: spawn ONE MCP server for the run; authored nodes call tools through it.
    mcp = None
    if os.environ.get("ICM_NO_MCP") != "1":
        try:
            mcp = icm_mcp_client.McpClient(str(ICM_DIR)).start()
            if verbose:
                print(f"  · MCP: spawned server for {ICM_DIR.name} "
                      f"(tools: {list(TOOL_NAMES.values())})", file=sys.stderr)
        except Exception as ex:
            if verbose:
                print(f"  · MCP unavailable ({ex}); tools run in-process", file=sys.stderr)
            mcp = None
    ctx = {"verbose": verbose, "show_code": show_code, "tool": _make_tool(mcp, verbose)}

    # the blackboard. Seeded with the request; nodes add ONLY their declared outputs.
    state = {"request": request}
    if verbose:
        print(f"▸ flow '{flow['name']}' → {run_dir.relative_to(ICM_DIR)}", file=sys.stderr)

    try:
        for i, node in enumerate(flow["nodes"]):
            # hand the node EXACTLY its declared inputs — nothing else from the tape
            missing = [k for k in node["inputs"] if k not in state]
            if missing:
                raise SystemExit(f"node '{node['id']}' needs {missing}, not produced upstream")
            node_inputs = {k: state[k] for k in node["inputs"]}
            if verbose:
                print(f"\n[{i}] {node['id']} ({node['kind']})  inputs={node['inputs']}",
                      file=sys.stderr)

            out = HANDLERS[node["kind"]](node_inputs, ctx)

            # keep only the declared outputs; the rest is discarded (no leakage downstream)
            declared = {k: out[k] for k in node["outputs"]}
            state.update(declared)
            (run_dir / f"{i:02d}-{node['id']}.json").write_text(
                json.dumps({"kind": node["kind"], "inputs": node_inputs, "outputs": declared},
                           indent=2)
            )
    finally:
        if mcp:
            mcp.close()

    result = {"request": request}
    for key in ("ok", "repairs", "code", "answer", "tier"):
        if key in state:
            result[key] = state[key]
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result, run_dir


def main():
    argv = sys.argv[1:]
    flow_name = "code-from-kb"
    show_code = False
    rest = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--flow":
            i += 1
            flow_name = argv[i]
        elif a.startswith("--flow="):
            flow_name = a.split("=", 1)[1]
        elif a == "--show-code":
            show_code = True
        else:
            rest.append(a)
        i += 1

    if icm_repair.health() is None:
        print(f"⚠ can't reach Ollama at {icm_repair.OLLAMA_URL} "
              f"(set OLLAMA_URL=http://<host>:11434)", file=sys.stderr)
        sys.exit(1)

    flow = load_flow(flow_name)

    def run_and_print(request):
        result, run_dir = run_flow(flow, request, verbose=True, show_code=show_code)
        print("\n" + "=" * 60)
        if "answer" in result:                        # lookup/answer flow
            tier = {1: "KB", 2: "std docs (fallback)", 0: "declined"}.get(result.get("tier"))
            print(f"answer (tier: {tier}) — artifacts in {run_dir.relative_to(ICM_DIR)}")
            print("=" * 60 + "\n")
            print(result["answer"])
        else:                                         # code-from-kb flow
            if result.get("ok"):
                print(f"✓ VERIFIED (correctness; lint advisory), {result['repairs']} repair(s) — "
                      f"artifacts in {run_dir.relative_to(ICM_DIR)}")
            else:
                print(f"✗ STILL FAILING after {result['repairs']} repair(s) — "
                      f"oracle refused to ship. artifacts in {run_dir.relative_to(ICM_DIR)}")
            print("=" * 60 + "\n")
            print(result["code"])

    if rest:
        run_and_print(" ".join(rest))
        return
    print(f"ICM-Rust flow '{flow_name}' → {icm_repair.OLLAMA_URL} "
          f"({icm_repair.OLLAMA_MODEL}). Ctrl-C to quit.\n")
    try:
        while True:
            q = input("task › ").strip()
            if q:
                run_and_print(q)
                print()
    except (KeyboardInterrupt, EOFError):
        print("\nbye")


if __name__ == "__main__":
    main()
