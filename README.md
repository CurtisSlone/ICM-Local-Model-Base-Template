# ICM-Local-Model-Base — the reusable ICM engine

A local, dependency-free engine for running **ICM (Interpretable Context Methodology)** on a
**small local model**. The *code* lives here; the *knowledge and workflows* live in an ICM
directory you point it at. Same engine, any domain — swap the ICM dir to change domain, swap
one config field to change model. Pure Python stdlib — no pip install.

> **Worked example:** a complete local Rust coding assistant built on this engine —
> **[ICM-Local-Model-Rust-Example]([https://github.com/CurtisSlone/ICM-Local-Model-Rust-Example](https://github.com/CurtisSlone/ICM-Local-Model-Rust-Coding-Assistant))**.
> Start there if you want to *see it run*; come here to *build your own*.
>
> Built on **ICM** by Jake Van Clief & David McDermott (paper included as `2603.16021v2.pdf`;
> [their repo](https://github.com/RinDig/Interpretable-Context-Methodology-ICM-)). This is an
> independent **local-model adaptation**: canonical ICM has a capable agent *roam* the
> folders; we *inject* scoped context into a weak local model and verify each step with a
> deterministic oracle.

## Quick start
**Prereqs:** [Ollama](https://ollama.com) running · Python 3 · (a build toolchain if your
ICM's oracle needs one, e.g. Rust's `cargo`/`clippy`).

```bash
python3 setup.py                       # preflight: checks Ollama + toolchain
cp -r icms/skeleton my-icm             # start a new ICM (or: VS Code → Run Task → "New ICM")
python3 icm_mcp.py --icm my-icm        # serve it over MCP
python3 test_mcp.py icms/skeleton      # protocol smoke test (no model needed)
```
Or open the folder in **VS Code** → **Run Task** (Setup / MCP test / New ICM / Run server).

## The split: code here, data in the ICM
```
ICM-Local-Model-Base/          ← ENGINE (code), parameterised by ICM_DIR
  icm_config.py   config loader (applies model/embed env from the ICM's config)
  icm_verify.py   generic oracle — runs the ICM's configured verify command
  icm_route.py    constrained routing over the ICM's manifest
  icm_docs.py     tier-2 doc search (BM25 + embed RRF) over the ICM's corpus
  icm_embed.py    embedding pre-filter + recall@K eval
  icm_flow.py     authored stateless flow engine (route → ground → …)
  icm_repair.py   generate → verify → bounded repair
  icm_mcp.py      MCP server (stdio) — the doorway; --icm picks the data dir
  icm_mcp_client.py / test_mcp.py
  icms/skeleton/  ← copy this to start a new ICM
  setup.py        ← preflight / quick-setup

<an ICM dir>/                  ← DATA only
  icm.config.json   model, embed_model, tools, the verify oracle
  manifest.json     the routing index (id/title/summary/path)
  SYSTEM.md         operating rules
  kb/*.md           the knowledge, one topic per file
  flows/*.json      authored workflow graphs
  refdocs/          (optional) converted reference corpus for tier-2 search
  eval/             (optional) labelled queries for recall@K
```
The server sets `ICM_DIR` so the engine reads that ICM's data, but imports the engine modules
from here — the content dir is never on `sys.path`, so an ICM's files can't shadow the code.

## The swap-points (why it's a template)
- **Swap model** → edit `model` / `embed_model` in the ICM's `icm.config.json` (env still overrides).
- **Swap domain** → `--icm <other-dir>`; that dir brings its own manifest, kb, flows, oracle.
- **Swap the oracle** → edit the `verify` block (project files + commands). The engine doesn't
  know or care whether it's `cargo`, `pytest`, a schema validator, …

## Connecting to Ollama
The engine finds Ollama automatically for the common cases; here are the **explicit manual
steps**. Test with `python3 setup.py` (it prints the URL it resolved).

**Two different env vars — don't confuse them:**
- `OLLAMA_HOST` — on the machine running **Ollama**; what it *listens on* (default `127.0.0.1`,
  local only). Set to `0.0.0.0` to accept connections from WSL/other hosts.
- `OLLAMA_URL` — on the machine running **this engine**; tells *our code* where Ollama is.

- **Same Linux/macOS machine** — nothing to do; defaults to `http://localhost:11434`
  (just ensure Ollama is running).
- **Engine in WSL2, Ollama on Windows** *(the usual failure — `localhost` won't reach it)*:
  on **Windows** run `setx OLLAMA_HOST "0.0.0.0"`, fully restart Ollama (tray → quit → reopen),
  allow it through **Windows Firewall**. WSL auto-detects the Windows host IP. Fallback:
  `ip route | grep default` → `export OLLAMA_URL=http://<that-ip>:11434`.
- **Remote host** — run that Ollama with `OLLAMA_HOST=0.0.0.0` + open port `11434`; here
  `export OLLAMA_URL=http://<remote-ip>:11434`.

Persist it in `~/.bashrc`; custom port → include it in `OLLAMA_URL`; manual test →
`curl http://<host>:11434/api/tags`. Override models per-run with `OLLAMA_MODEL` /
`OLLAMA_EMBED_MODEL`.

## Who it's for — and the one question that decides fit
Not enterprise-only. The dividing line isn't *who you are*, it's **"is the task narrow and
checkable?"** — that's what lets the structure + oracle carry the reliability a small model can't.

- **Enterprise:** privacy / regulated / air-gapped work with a checkable result — compliance
  evidence, code, data extraction. (Nothing leaves your network.)
- **Personal / maker:** cost-free, always-on, private home automation on a homelab or old PC,
  offline. Your own scripts and tools, file/data ETL, config generation, tagging/triage.

**Good-fit tasks (have an oracle → free, local, *and reliable*):** code & scripts
(compiler/tests), text → structured data (schema), generate-and-validate config (a regex that
must match cases, a SQL query that runs, a shell script that passes a linter, YAML/JSON that
validates), classify / sort / triage.

**Weak-fit tasks (no oracle):** open-ended creative generation (prose, brainstorming) — you
lose the verification guarantee. ICM's structure still helps, but for a *one-off* creative ask
a free frontier chat is usually easier. The edge is the **repeated, automated, private, or
structured** work.

## When to use this (local-model ICM) vs ICM / context-engineering on Claude
Be honest: **a frontier model (Claude) is more capable.** Context-engineering on Claude makes
a *strong* model more focused; this makes a *weak, local* model **trustworthy** via structure
+ an oracle. Different goals — pick by your constraints and your task shape.

**Reach for local-model ICM when you have a constraint that rules out a frontier API:**
- **Privacy / regulated / air-gapped data** — code, evidence, IP, or sensitive data never
  leaves your machine or network (compliance, classified, disconnected environments).
- **Cost** — no per-token bills; high-volume or always-on work on hardware you own.
- **Offline** — no internet dependency.
- **Control & reproducibility** — pin the model; no rate limits, ToS limits, or surprise
  deprecations. Reliability lives in the *structure + oracle*, so behavior is auditable and
  the model is swappable.
- **Understanding** — the whole machine is visible; nothing hidden behind an API.

…**and** the task is **narrow enough + has a checkable oracle** (a compiler, tests, a schema)
so the structure can carry the reliability the small model can't.

**Reach for Claude (frontier) instead when:** the task is broad, novel, or open-ended and the
model must reason/adapt across the workspace; there's **no cheap oracle** (open-ended
judgment/prose); or you simply want maximum capability and have no privacy/cost/offline
constraint. A 7B can't roam a large context and decide reliably — Claude can.

They also **compose**: the MCP server here is consumable by Claude Code *or* a local client,
so you can keep sensitive/bounded steps local and hand broad reasoning to a frontier model.

## Tools exposed (kinds are fixed; the ICM config names/picks them)
| kind | backed by | deterministic? |
|---|---|---|
| `verify` | `icm_verify` (configured oracle) | yes |
| `docs_search` | `icm_docs` (BM25 + embed RRF) | embed only |
| `kb_answer` | flow `answer-with-fallback` | model + oracles |
| `generate_verify` | flow `code-from-kb` | model + oracle |

The reliability comes from the oracle-backed tools and the authored flow, not the model — the
whole thesis: a small local model is a bounded proposer; structure carries the rest.

## License
MIT — see [LICENSE](LICENSE). Built on the MIT-licensed ICM methodology (credit above).
