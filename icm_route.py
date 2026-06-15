#!/usr/bin/env python3
"""
ICM-Test runner v2 — constrained routing (the action-path approach).

The v1 runner (icm_run.py) gave Mistral a read_entry *tool* and hoped it would
elect to call it. A 7B doesn't — it answers from training instead. This version
borrows the pattern from prooflayer/mistral-action-path:

  1. ROUTE (constrained):  one /api/generate call whose `format` is a JSON Schema
     with an enum of valid entry ids. Grammar-constrained decode → the model
     CANNOT skip the choice and CANNOT name an entry that doesn't exist.
  2. CODE READS the chosen entries (AI proposes, code executes — no tool to skip).
  3. ANSWER (grounded): a second call that sees ONLY the chosen entry text and is
     told to answer from it. Empty choice → deterministic "not covered" decline.

A deterministic pre-filter (keyword match) skips the model entirely when the
request unambiguously names one entry — don't burn a model call on the obvious.

No dependencies. Talks to Ollama's HTTP API. WSL→Windows gateway auto-resolved.
Override with OLLAMA_URL / OLLAMA_MODEL.

    python3 icm_route.py "how do I make cold brew?"
    python3 icm_route.py                      # interactive
"""

import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request

ICM_DIR = pathlib.Path(os.environ.get("ICM_DIR") or pathlib.Path(__file__).parent).resolve()
MANIFEST = json.loads((ICM_DIR / "manifest.json").read_text())
SYSTEM_MD = (ICM_DIR / "SYSTEM.md").read_text()
ENTRIES = MANIFEST["entries"]
ENTRY_PATHS = {e["id"]: e["path"] for e in ENTRIES}
NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))   # default 2048 truncates


# --- Ollama endpoint (ported from server/src/ai/ollama.ts) --------------------
def _is_wsl():
    try:
        return bool(re.search(r"microsoft|wsl", pathlib.Path("/proc/version").read_text(), re.I))
    except OSError:
        return False


def _wsl_gateway_ip():
    try:
        for line in pathlib.Path("/proc/net/route").read_text().splitlines()[1:]:
            f = line.split()
            if len(f) > 2 and f[1] == "00000000" and f[2] != "00000000":
                h = f[2]
                octets = [int(h[i:i + 2], 16) for i in (6, 4, 2, 0)]
                if all(0 <= o <= 255 for o in octets):
                    return ".".join(map(str, octets))
    except OSError:
        pass
    return None


def _resolve_base_url():
    if os.environ.get("OLLAMA_URL"):
        return os.environ["OLLAMA_URL"].rstrip("/")
    if _is_wsl():
        gw = _wsl_gateway_ip()
        if gw:
            return f"http://{gw}:11434"
    return "http://localhost:11434"


OLLAMA_URL = _resolve_base_url()
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")


def generate(prompt, schema=None, timeout=120):
    """One /api/generate call. With a schema, output is grammar-constrained to it."""
    body = {"model": os.environ.get("OLLAMA_MODEL", "mistral"), "prompt": prompt, "stream": False,
            "options": {"num_ctx": NUM_CTX}}
    if schema is not None:
        body["format"] = schema
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())["response"].strip()
    return json.loads(resp) if schema is not None else resp


def health(timeout=3):
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=timeout) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", [])]
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


# --- step 1: route (deterministic pre-filter, then constrained model) ---------
def pre_filter(question):
    """Cheap exact-ish match: an entry whose id/title words appear in the question."""
    q = question.lower()
    hits = [e["id"] for e in ENTRIES
            if e["id"].replace("-", " ") in q or e["title"].lower() in q]
    return hits


def embed_prefilter(question, pf, verbose):
    """Opt-in (ICM_PREFILTER=embed): narrow the candidate set to the top-K entries
    by embedding similarity, so the constrained pick sees a handful, not the whole
    catalogue. Off by default — at a handful of entries you don't need it and the
    model should see them all. Set ICM_PREFILTER_MIN to only kick in above N entries.

    SAFETY: top-K is unioned with the deterministic keyword hits (`pf`), so a keyword
    match can rescue an entry the embedder ranks low. Even so, a hard pre-filter can
    drop the right entry — measure recall@K first (`icm_embed.py --eval`). On any
    failure we fall back to ALL entries rather than silently narrowing wrong."""
    if os.environ.get("ICM_PREFILTER") != "embed":
        return ENTRIES
    if len(ENTRIES) <= int(os.environ.get("ICM_PREFILTER_MIN", "0")):
        return ENTRIES
    try:
        import icm_embed
        k = int(os.environ.get("ICM_PREFILTER_K", "5"))
        ranked = icm_embed.top_k(question, k)
        keep = {cid for cid, _ in ranked} | set(pf)
        candidates = [e for e in ENTRIES if e["id"] in keep]
        if verbose:
            print(f"  · pre-filter(embed): top-{k} {[c for c, _ in ranked]} "
                  f"+ keywords {pf} → {len(candidates)}/{len(ENTRIES)} candidates",
                  file=sys.stderr)
        return candidates or ENTRIES
    except Exception as ex:                       # never let the pre-filter break routing
        if verbose:
            print(f"  · pre-filter(embed) failed ({ex}); using all entries", file=sys.stderr)
        return ENTRIES


def route(question, verbose):
    pf = pre_filter(question)
    if len(pf) == 1:
        if verbose:
            print(f"  · route: pre-filter → {pf} (no model call)", file=sys.stderr)
        return pf
    candidates = embed_prefilter(question, pf, verbose)
    cand_ids = [e["id"] for e in candidates]
    index = "\n".join(f"- id: {e['id']} | {e['title']} — {e['summary']}" for e in candidates)
    prompt = (
        "You are routing a question to a knowledge base. Pick ONLY the entry id(s) "
        "whose content can answer it — usually one, at most two for a comparison. "
        "If NOTHING in the index is relevant, return an empty list.\n\n"
        f"Index:\n{index}\n\nQuestion: {question}"
    )
    schema = {
        "type": "object",
        "properties": {
            "entry_ids": {
                "type": "array",
                "items": {"type": "string", "enum": cand_ids},
                "maxItems": 2,
            },
            "rationale": {"type": "string"},
        },
        "required": ["entry_ids", "rationale"],
    }
    decision = generate(prompt, schema)
    if verbose:
        print(f"  · route: model → {decision['entry_ids']} "
              f"({decision.get('rationale', '')[:60]})", file=sys.stderr)
    # de-dup, preserve order, drop anything bogus (schema already enforces enum)
    return list(dict.fromkeys(i for i in decision["entry_ids"] if i in ENTRY_PATHS))


# --- step 2: code reads the chosen entries ------------------------------------
def read_entry(entry_id):
    path = (ICM_DIR / ENTRY_PATHS[entry_id]).resolve()
    if not str(path).startswith(str(ICM_DIR)):
        raise ValueError("path escapes the ICM directory")
    return path.read_text()


# --- step 3: grounding check (constrained) — does the text actually answer it? -
def is_answerable(question, loaded, verbose):
    """A closed-set yes/no guard. Routing picks a *related* entry; this asks
    whether the entry text genuinely contains the answer. Schema-constrained so
    the model must commit to a boolean — its weak spot (declining) becomes a
    forced binary choice, not a free-text judgement it can wriggle out of."""
    prompt = (
        "Decide if the ENTRY TEXT below contains the information needed to answer "
        "the QUESTION. Being on a related topic is NOT enough — the specific facts "
        "asked for must be present in the text. If they are not, answerable=false.\n\n"
        f"--- ENTRY TEXT ---\n{loaded}\n--- END ---\n\n"
        f"QUESTION: {question}"
    )
    schema = {
        "type": "object",
        "properties": {
            "answerable": {"type": "boolean"},
            "missing": {"type": "string"},
        },
        "required": ["answerable", "missing"],
    }
    verdict = generate(prompt, schema)
    if verbose:
        tag = "answerable" if verdict["answerable"] else f"NOT answerable ({verdict['missing'][:50]})"
        print(f"  · ground-check: {tag}", file=sys.stderr)
    return verdict["answerable"]


# --- step 4: model answers from the read text ---------------------------------
def answer(question, verbose=True):
    chosen = route(question, verbose)
    if not chosen:                                    # deterministic decline
        return "That isn't covered in this knowledge base."
    loaded = "\n\n".join(f"### {cid}\n{read_entry(cid)}" for cid in chosen)
    if verbose:
        print(f"  · read: {chosen} → {len(loaded)} chars injected", file=sys.stderr)
    if not is_answerable(question, loaded, verbose):  # grounding gate
        return "That isn't fully covered in this knowledge base."
    prompt = (
        f"{SYSTEM_MD}\n\n"
        "Answer the question using ONLY the entry text below. Quote numbers "
        "(grind, ratio, time, temperature) exactly as written. If the text does "
        "not contain the answer, say so — do not use outside knowledge.\n\n"
        f"--- ENTRY TEXT ---\n{loaded}\n--- END ---\n\n"
        f"Question: {question}"
    )
    return generate(prompt).strip()


def main():
    models = health()
    if models is None:
        print(f"⚠ can't reach Ollama at {OLLAMA_URL} "
              f"(set OLLAMA_URL=http://<host>:11434)", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) > 1:
        print(answer(" ".join(sys.argv[1:])))
        return
    print(f"ICM-Test (constrained routing) → {OLLAMA_URL} ({OLLAMA_MODEL}). Ctrl-C to quit.\n")
    try:
        while True:
            q = input("you › ").strip()
            if q:
                print(f"\n{answer(q)}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nbye")


if __name__ == "__main__":
    main()
