#!/usr/bin/env python3
"""
ICM-Rust tier-2 reference search (the uncertainty fallback).

Tier 1 is the curated KB (icm_route.py). When its ground-check says the KB does NOT
cover a question, instead of just declining we search a larger reference corpus — the
official Rust std API docs, which ship as local HTML in the rustup toolchain. No network.

Two steps:
  1. BUILD (one-time) — convert the std rustdoc HTML into searchable chunks. rustdoc pages
     are regular: <title> gives "Item in path", <h1> the kind, the first
     <div class="docblock"> the item's documentation. One chunk per API item →
     refdocs/std.json. Pure stdlib parsing (regex + html.unescape).
  2. SEARCH (hybrid) — BM25-lite keyword ranking over the WHOLE corpus picks a cheap
     shortlist; then we embed-rerank just that shortlist (so we never embed all ~2.4k
     items — only ~20 per query, cached). Keyword finds the candidates, embeddings reorder
     by meaning. This is why hybrid beats either alone here: BM25 is exact but literal,
     mistral embeddings are semantic but weak (88% recall) and slow at scale.

    python3 icm_docs.py build              # convert std HTML -> refdocs/std.json (one-time)
    python3 icm_docs.py "grow an array and remove the last element"
    python3 icm_docs.py --k 5 "atomic reference counted pointer"
"""

import html as _html
import json
import math
import os
import pathlib
import re
import sys

ICM_DIR = pathlib.Path(os.environ.get("ICM_DIR") or pathlib.Path(__file__).parent).resolve()
REFDOCS = ICM_DIR / "refdocs"
CORPUS_PATH = REFDOCS / "std.json"
DOCEMB_CACHE = ICM_DIR / ".docemb_cache.json"
SHORTLIST = int(os.environ.get("ICM_DOCS_SHORTLIST", "20"))   # BM25 candidates to rerank
ITEM_RE = re.compile(r"^(struct|enum|trait|fn|macro|primitive|constant|type|union|keyword)\.")


def _doc_root():
    if os.environ.get("RUST_DOC_DIR"):
        return pathlib.Path(os.environ["RUST_DOC_DIR"])
    hits = sorted(pathlib.Path.home().glob(
        ".rustup/toolchains/*/share/doc/rust/html"))
    if not hits:
        raise SystemExit("can't find rustup html docs; set RUST_DOC_DIR")
    return hits[-1]


# --- BUILD: rustdoc HTML -> chunks --------------------------------------------
# internal/unstable modules that are noise for a user-facing API search
SKIP_PATHS = ("intrinsics", "/core_arch/", "simd")


def _clean(frag):
    return _html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", frag))).strip()


def _methods(h, owner):
    """Per-method chunks: each <section id="method.NAME"> → signature + its docblock.
    Methods with a real docblock are exactly what answers 'how do I X' API queries —
    indexing only the type summary misses them (Vec::pop lives inside the Vec page)."""
    out, seen = [], set()
    marks = [(m.group(1), m.start()) for m in re.finditer(r'id="method\.([a-z0-9_]+)"', h)]
    for i, (name, pos) in enumerate(marks):
        if name in seen:                       # same method across multiple impls
            continue
        end = marks[i + 1][1] if i + 1 < len(marks) else min(pos + 4000, len(h))
        blk = h[pos:end]
        db = re.search(r'<div class="docblock">(.*?)</div>', blk, re.S)
        if not db:
            continue
        doc = _clean(db.group(1))
        if len(doc) < 40:                      # skip trivial / boilerplate trait methods
            continue
        seen.add(name)
        sig = re.search(r'<h4 class="code-header">(.*?)</h4>', blk, re.S)
        sig = _clean(sig.group(1)) if sig else name
        out.append({"title": f"{owner}::{name}", "kind": "method", "anchor": f"method.{name}",
                    "text": (sig + " — " + doc)[:1200]})
    return out


def _extract(path, rel):
    h = path.read_text(errors="replace")
    title = re.search(r"<title>(.*?)</title>", h, re.S)
    title = re.sub(r"\s*-\s*Rust$", "", title.group(1).strip()) if title else path.stem
    owner = title.split(" in ")[0].strip()              # "Vec in std::vec" -> "Vec"
    h1 = re.search(r"<h1[^>]*>(.*?)</h1>", h, re.S)
    kind = (_clean(h1.group(1)).split() or [""])[0] if h1 else ""
    chunks = []
    db = re.search(r'<div class="docblock">(.*?)</div>', h, re.S)
    if db:
        text = _clean(db.group(1))
        if len(text) >= 20:
            chunks.append({"id": rel, "title": title, "kind": kind, "text": text[:1500]})
    for m in _methods(h, owner):                        # add method-level chunks
        m["id"] = f"{rel}#{m.pop('anchor')}"
        chunks.append(m)
    return chunks


def build():
    root = _doc_root()
    std = root / "std"
    REFDOCS.mkdir(exist_ok=True)
    files = [p for p in std.rglob("*.html")
             if ITEM_RE.match(p.name) and not any(s in str(p) for s in SKIP_PATHS)]
    print(f"  · scanning {len(files)} std item pages (intrinsics/simd excluded)", file=sys.stderr)
    chunks = []
    for p in files:
        chunks.extend(_extract(p, str(p.relative_to(root))))
    CORPUS_PATH.write_text(json.dumps(chunks))
    n_item = sum(1 for c in chunks if c["kind"] != "method")
    print(f"  · wrote {len(chunks)} chunks ({n_item} items + {len(chunks) - n_item} methods) "
          f"→ {CORPUS_PATH.relative_to(ICM_DIR)}", file=sys.stderr)
    return chunks


# --- BM25-lite (pure stdlib) --------------------------------------------------
_TOK = re.compile(r"[a-z0-9_]+")


def _tokens(s):
    return _TOK.findall(s.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.toks = [_tokens(d["title"] + " " + d["text"]) for d in docs]
        self.len = [len(t) for t in self.toks]
        self.avgdl = sum(self.len) / len(self.len) if self.len else 0
        self.df = {}
        for t in self.toks:
            for w in set(t):
                self.df[w] = self.df.get(w, 0) + 1
        self.N = len(docs)
        self.tf = [{} for _ in docs]
        for i, t in enumerate(self.toks):
            for w in t:
                self.tf[i][w] = self.tf[i].get(w, 0) + 1

    def idf(self, w):
        n = self.df.get(w, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def search(self, query, n):
        q = _tokens(query)
        scores = []
        for i in range(self.N):
            s = 0.0
            dl = self.len[i]
            for w in q:
                f = self.tf[i].get(w, 0)
                if not f:
                    continue
                s += self.idf(w) * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[:n]


# --- hybrid: BM25 shortlist -> embedding rerank -------------------------------
def _load_corpus():
    if not CORPUS_PATH.exists():
        raise SystemExit("corpus not built — run: python3 icm_docs.py build")
    return json.loads(CORPUS_PATH.read_text())


def _doc_emb_cache(model):
    """Model-keyed: chunk vectors are only valid for the model that produced them.
    Swapping embed_model (e.g. mistral→nomic, 4096-dim→768-dim) invalidates the cache."""
    if DOCEMB_CACHE.exists():
        try:
            c = json.loads(DOCEMB_CACHE.read_text())
            if c.get("model") == model:
                return c["vecs"]
        except (json.JSONDecodeError, OSError, KeyError):
            pass
    return {}


def search(query, k=5, verbose=False, rrf_k=60):
    """Hybrid via Reciprocal Rank Fusion: BM25 ranks the whole corpus → shortlist;
    embeddings re-rank that shortlist; the final order FUSES the two rankings instead
    of letting either win outright. RRF score = 1/(rrf_k+rank_bm) + 1/(rrf_k+rank_emb),
    so a doc ranked well by EITHER signal survives. This matters because mistral
    embeddings are weak (they alone push the right answer out of top-k) and BM25 alone
    is literal (term overlap outranks meaning) — fusing keeps the best of both."""
    import icm_embed                                   # reuse the embed client + cosine
    corpus = _load_corpus()
    bm = BM25(corpus)
    shortlist = bm.search(query, SHORTLIST)
    if not shortlist:
        return []
    if verbose:
        print(f"  · doc-search: BM25 shortlist {len(shortlist)} "
              f"(top: {corpus[shortlist[0][0]]['title']})", file=sys.stderr)
    cache = _doc_emb_cache(icm_embed.EMBED_MODEL)
    qv = icm_embed.embed(query)
    rows, dirty = [], False
    for bm_rank, (idx, bm_score) in enumerate(shortlist):
        c = corpus[idx]
        vec = cache.get(c["id"])
        if vec is None:
            vec = icm_embed.embed((c["title"] + ". " + c["text"])[:512])
            cache[c["id"]] = vec
            dirty = True
        rows.append({"idx": idx, "bm_rank": bm_rank, "bm25": bm_score,
                     "sim": icm_embed.cosine(qv, vec)})
    if dirty:
        DOCEMB_CACHE.write_text(json.dumps({"model": icm_embed.EMBED_MODEL, "vecs": cache}))
    # embedding rank within the shortlist
    for emb_rank, r in enumerate(sorted(rows, key=lambda r: r["sim"], reverse=True)):
        r["emb_rank"] = emb_rank
    for r in rows:
        r["rrf"] = 1.0 / (rrf_k + r["bm_rank"]) + 1.0 / (rrf_k + r["emb_rank"])
    rows.sort(key=lambda r: r["rrf"], reverse=True)
    out = []
    for r in rows[:k]:
        c = corpus[r["idx"]]
        out.append({"id": c["id"], "title": c["title"], "kind": c["kind"],
                    "text": c["text"], "sim": round(r["sim"], 4),
                    "bm25": round(r["bm25"], 3), "rrf": round(r["rrf"], 5)})
    return out


def main():
    try:
        import icm_config; icm_config.load()   # respect icm.config.json model choices
    except Exception:
        pass
    argv = sys.argv[1:]
    if argv and argv[0] == "build":
        build()
        return
    k = 5
    rest = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--k":
            i += 1
            k = int(argv[i])
        elif a.startswith("--k="):
            k = int(a.split("=", 1)[1])
        else:
            rest.append(a)
        i += 1
    if not rest:
        print("usage: python3 icm_docs.py build | [--k N] \"<query>\"", file=sys.stderr)
        sys.exit(1)
    for h in search(" ".join(rest), k, verbose=True):
        print(f"  sim={h['sim']:.3f} bm25={h['bm25']:.2f}  {h['kind']} {h['title']}")
        print(f"      {h['text'][:140]}")


if __name__ == "__main__":
    main()
