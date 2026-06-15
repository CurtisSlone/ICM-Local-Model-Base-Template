#!/usr/bin/env python3
"""
ICM-Rust embedding pre-filter (next-step #2) — top-K by similarity, then pick.

The constrained router (icm_route.py) sends the model an enum of ALL entry ids. That
routes crisply at ~8 entries, but the enum grows with the KB and a 7B's pick-rate drops
once it's long (30-50+). The fix: a cheap, deterministic **pre-filter** ranks entries by
embedding similarity to the query and forwards only the top-K to the constrained pick. The
model then chooses among a handful, not the whole catalogue.

A pre-filter is only safe if it does NOT drop the entry the model should have picked. The
metric that matters is therefore **recall@K** — is the correct entry inside the top-K? Run
`--eval` to measure it against eval/routing-eval.json before trusting the narrowing.

Embeddings come from Ollama's /api/embed using the model already present (mistral). A
dedicated embedding model is better and faster — set OLLAMA_EMBED_MODEL=nomic-embed-text
(after `ollama pull nomic-embed-text`) to swap it in. Entry vectors are cached to
.emb_cache.json, keyed by a hash of the text, so only changed summaries re-embed.

No dependencies — cosine similarity is a dot product over two lists.

    python3 icm_embed.py "how do I propagate an error?"     # ranked entries + scores
    python3 icm_embed.py --k 3 "what is a trait?"
    python3 icm_embed.py --eval                              # recall@K on the labeled set
"""

import hashlib
import json
import math
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request

ICM_DIR = pathlib.Path(os.environ.get("ICM_DIR") or pathlib.Path(__file__).parent).resolve()
MANIFEST = json.loads((ICM_DIR / "manifest.json").read_text())
ENTRIES = MANIFEST["entries"]
CACHE_PATH = ICM_DIR / ".emb_cache.json"


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
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "mistral")


def embed(text, timeout=120):
    """One /api/embed call → a single embedding vector (list of floats)."""
    body = {"model": os.environ.get("OLLAMA_EMBED_MODEL", "mistral"), "input": text}
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embed",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["embeddings"][0]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# --- entry index, cached to disk ----------------------------------------------
def _entry_text(e):
    """What we embed for an entry: title + summary. The summary is what the model
    routes on, so it's the right thing to rank on too."""
    return f"{e['title']}. {e['summary']}"


def _load_cache():
    if CACHE_PATH.exists():
        try:
            c = json.loads(CACHE_PATH.read_text())
            if c.get("model") == EMBED_MODEL:
                return c
        except (json.JSONDecodeError, OSError):
            pass
    return {"model": os.environ.get("OLLAMA_EMBED_MODEL", "mistral"), "vecs": {}}


def build_index(verbose=False):
    """Return {entry_id: vector}, embedding only entries whose text changed."""
    cache = _load_cache()
    vecs = cache["vecs"]
    changed = False
    index = {}
    for e in ENTRIES:
        text = _entry_text(e)
        h = hashlib.sha256(text.encode()).hexdigest()
        hit = vecs.get(e["id"])
        if hit and hit.get("hash") == h:
            index[e["id"]] = hit["vec"]
        else:
            if verbose:
                print(f"  · embedding entry '{e['id']}' (cache miss)", file=sys.stderr)
            v = embed(text)
            vecs[e["id"]] = {"hash": h, "vec": v}
            index[e["id"]] = v
            changed = True
    if changed:
        CACHE_PATH.write_text(json.dumps({"model": os.environ.get("OLLAMA_EMBED_MODEL", "mistral"), "vecs": vecs}))
    return index


def top_k(query, k=5, verbose=False):
    """Rank entries by cosine similarity to the query; return [(id, score)] top-k."""
    index = build_index(verbose)
    qv = embed(query)
    scored = sorted(((cid, cosine(qv, v)) for cid, v in index.items()),
                    key=lambda t: t[1], reverse=True)
    return scored[:k]


# --- recall@K evaluation: is the correct entry inside the top-K? --------------
def run_eval(k):
    eval_path = ICM_DIR / "eval" / "routing-eval.json"
    cases = json.loads(eval_path.read_text())["cases"]
    build_index(verbose=True)
    hits = 0
    print(f"\nrecall@{k} over {len(cases)} labelled queries "
          f"(embed model: {EMBED_MODEL})\n", file=sys.stderr)
    for c in cases:
        ranked = top_k(c["query"], k)
        ids = [cid for cid, _ in ranked]
        ok = c["expected"] in ids
        hits += ok
        rank = ids.index(c["expected"]) + 1 if ok else None
        mark = f"✓ (rank {rank})" if ok else "✗ MISSED"
        print(f"  {mark:14} {c['expected']:22} ← {c['query']}", file=sys.stderr)
    pct = 100 * hits / len(cases)
    print(f"\nrecall@{k} = {hits}/{len(cases)} = {pct:.0f}%", file=sys.stderr)
    print("(a safe pre-filter keeps the correct entry in top-K; misses = entries the "
          "model would never get to see)", file=sys.stderr)
    return hits, len(cases)


def main():
    try:
        import icm_config; icm_config.load()   # respect icm.config.json model choices
    except Exception:
        pass
    argv = sys.argv[1:]
    k = 5
    do_eval = False
    rest = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--eval":
            do_eval = True
        elif a == "--k":
            i += 1
            k = int(argv[i])
        elif a.startswith("--k="):
            k = int(a.split("=", 1)[1])
        else:
            rest.append(a)
        i += 1

    if do_eval:
        run_eval(k)
        return

    if not rest:
        print("usage: python3 icm_embed.py [--k N] \"<query>\"   |   --eval", file=sys.stderr)
        sys.exit(1)

    query = " ".join(rest)
    for cid, score in top_k(query, k, verbose=True):
        print(f"  {score:.4f}  {cid}")


if __name__ == "__main__":
    main()
