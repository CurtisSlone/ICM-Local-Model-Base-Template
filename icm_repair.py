#!/usr/bin/env python3
"""
ICM-Test runner v3 — generate → verify → repair (the generative-axis twin).

icm_route.py wrapped the model's *decision* axis in a verifier: it proposes a
choice, a JSON-Schema grammar constrains it, code executes it. This wraps the
model's *generation* axis in the same shape — only the oracle changes:

  1. GENERATE (constrained): one /api/generate call whose `format` is a JSON
     Schema {code, notes}. The model proposes a full Rust program; the schema
     guarantees we get clean code back, not a markdown-fenced guess to scrape.
  2. VERIFY (deterministic oracle): code writes the program to a throwaway cargo
     crate and runs TWO gates — `cargo clippy -- -D warnings` (well-formedness)
     then `cargo run` with an expected-output assertion (behaviour). The
     compiler is a free, exact oracle — it doesn't have opinions, it has a
     return code. The run-gate matters: clippy is perfectly happy with an empty
     `fn main(){}`, so form alone is a misleading PASS. Only running the program
     proves it does the task. (For arbitrary user tasks we can't know the
     expected output, so they get the weaker form+runs guarantee — unless the
     code carries `#[test]`s, in which case `cargo test` becomes the oracle.)
  3. REPAIR (bounded): on failure, feed the EXACT compiler output back and ask
     for one corrected program. Repeat up to --max-repairs (default 4). This is
     a *bounded* loop, not an open ReAct chase — it converges or it reports the
     errors it couldn't fix. No infinite "let me try again."

Why this works on a 7B: the model is a strong proposal engine and a confident
liar. The traffic-light demo ships broken from training (uses `{}` on an enum
with no Display impl → E0277; `loop`-via-`while true` → clippy::while_true). The
model can't see that. The compiler can, every time. Reliability lives in the
oracle, not the model.

No dependencies. Talks to Ollama's HTTP API. WSL→Windows gateway auto-resolved.
Override with OLLAMA_URL / OLLAMA_MODEL / OLLAMA_NUM_CTX / ICM_MAX_REPAIRS.

    python3 icm_repair.py                       # built-in traffic-light demo (generate→repair)
    python3 icm_repair.py --seed-broken         # start from a known-broken program, repair only
    python3 icm_repair.py "write a Rust fn that reverses a linked list"
    python3 icm_repair.py --max-repairs 5 "..."
    python3 icm_repair.py --show-code "..."     # print every intermediate program
"""

import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))   # default 2048 truncates
MAX_REPAIRS = int(os.environ.get("ICM_MAX_REPAIRS", "4"))

# The default demo task. Phrased to naturally elicit the two classic mistakes a
# 7B reproduces from training: printing an enum with `{}` (needs Display) and an
# infinite `while true` (clippy flags it). The point is NOT the prompt — it's
# that whatever the model emits, the oracle judges it.
DEMO_TASK = (
    "Write a complete, runnable Rust program in a single file. Define an enum "
    "TrafficLight with variants Red, Green, and Yellow. Add a method "
    "`fn next(self) -> TrafficLight` that returns the next light in the cycle "
    "(Red -> Green -> Yellow -> Red). In main(), start at Red and print each "
    "light as the signal cycles, for exactly 6 steps, using println!. The "
    "program MUST terminate on its own — no infinite loops, no sleeping. "
    "Keep it simple."
)
# Ground truth for the demo: a correct program must actually print these. This
# is what makes the demo's PASS mean something — an empty main() can't satisfy it.
DEMO_EXPECT = ["Red", "Green", "Yellow"]

# The canonical near-miss a model emits for this task from training: prints an
# enum with `{}` (no Display impl → E0277) inside `while true` (clippy::while_true).
# `--seed-broken` skips generation and drops this straight into the repair loop,
# isolating the capability the demo is really about: fix-from-compiler-error. A
# 7B can't reliably *generate* a clean program here, but it CAN repair this.
BROKEN_SEED = """enum TrafficLight { Red, Green, Yellow }
impl TrafficLight {
    fn next(self) -> TrafficLight {
        match self {
            TrafficLight::Red => TrafficLight::Green,
            TrafficLight::Green => TrafficLight::Yellow,
            TrafficLight::Yellow => TrafficLight::Red,
        }
    }
}
fn main() {
    let mut light = TrafficLight::Red;
    while true {
        println!("{}", light);
        light = light.next();
    }
}
"""

CODE_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": ["code"],
}


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


def generate(prompt, schema=None, timeout=300, temperature=0.1):
    """One /api/generate call. With a schema, output is grammar-constrained to it.
    Low temperature by default: code gen wants the boring, correct token, not a
    creative one — a 7B at high temp invents external crates and rambling imports."""
    body = {"model": os.environ.get("OLLAMA_MODEL", "mistral"), "prompt": prompt, "stream": False,
            "options": {"num_ctx": NUM_CTX, "temperature": temperature}}
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


# --- the deterministic oracle: now config-driven (icm_verify + icm.config.json) -
# The cargo clippy+run logic moved into icm_verify.verify_code, parameterised by the
# `verify` block of icm.config.json. This file no longer knows anything Rust-specific;
# swap the config to verify another language. verify() is a thin compatibility wrapper.
import icm_config
import icm_verify

_VCFG = icm_config.load()["verify"]


def verify(code, workdir, expect):
    """Run the configured oracle against `code`, reusing `workdir` for incremental
    builds. Returns (ok, diagnostics)."""
    return icm_verify.verify_code(code, _VCFG, expect=expect, workdir=workdir)


# --- the loop: propose, verify, feed exact errors back, bounded retry ---------
def generate_verify_repair(task, max_repairs, verbose, show_code, expect, seed=None,
                           verify_fn=None):
    """verify_fn(code) -> (ok, diagnostics) lets a caller route the ORACLE through MCP
    (Phase 2). Default: the in-process config-driven oracle, reusing the temp crate."""
    if seed is not None:
        code = seed
        if verbose:
            print(f"  · seed: starting from a known-broken program "
                  f"({len(code)} chars) — repair only", file=sys.stderr)
    else:
        prompt = (
            "You are a careful Rust engineer. Write ONE complete, self-contained "
            "program that compiles cleanly under `cargo clippy -- -D warnings` (no "
            "warnings). Rules: standard library ONLY (no external crates / no `use` "
            "of anything outside std); no unused imports; no threads, channels, or "
            "sleep; include a real `fn main()` that exercises the code with a simple "
            "`for` loop and prints results. Return the FULL program in `code`.\n\n"
            f"Task: {task}"
        )
        code = generate(prompt, CODE_SCHEMA)["code"]
        if verbose:
            print(f"  · generate: {len(code)} chars proposed", file=sys.stderr)

    with tempfile.TemporaryDirectory(prefix="icm_repair_") as tmp:
        workdir = pathlib.Path(tmp)

        for attempt in range(max_repairs + 1):
            if show_code:
                label = "initial" if attempt == 0 else f"repair {attempt}"
                print(f"\n----- {label} -----\n{code}\n", file=sys.stderr)
            ok, diag = verify_fn(code) if verify_fn else verify(code, workdir, expect)
            if ok:
                if verbose:
                    tag = "clean on first try" if attempt == 0 else f"clean after {attempt} repair(s)"
                    print(f"  · verify: → PASS ({tag})", file=sys.stderr)
                return True, code, attempt

            if verbose:
                print(f"  · verify: FAIL — {icm_verify.first_problem(diag)[:90]}", file=sys.stderr)

            if attempt == max_repairs:
                return False, code, attempt   # bounded: stop chasing

            if verbose:
                print(f"  · repair {attempt + 1}/{max_repairs}: feeding exact errors back",
                      file=sys.stderr)
            repair_prompt = (
                "Your Rust program failed verification. Fix EVERY problem below "
                "while still accomplishing the original task. Keep the program "
                "complete and on-task — do NOT delete the logic to silence an "
                "error (an empty main() is not a fix), and prefer the SIMPLEST "
                "fix the diagnostics call for. To print an enum, implement "
                "`std::fmt::Display` by matching each variant to its name as a "
                "string literal (e.g. `TrafficLight::Red => write!(f, \"Red\")`). "
                "Use the standard library only. The loop must terminate. Make "
                "sure every brace `{` has a matching `}`. Do not write comments "
                "or explanations — return ONLY the complete, corrected program "
                "in the `code` field.\n\n"
                f"--- ORIGINAL TASK ---\n{task}\n--- END ---\n\n"
                f"--- YOUR PROGRAM ---\n{code}\n--- END ---\n\n"
                f"--- COMPILER DIAGNOSTICS ---\n{diag}\n--- END ---"
            )
            # nudge temperature up a touch on repair: greedy decoding gets stuck
            # re-emitting the same broken token (e.g. a dropped brace); a little
            # variation lets it escape the basin instead of looping on it.
            code = generate(repair_prompt, CODE_SCHEMA, temperature=0.3)["code"]


def main():
    argv = sys.argv[1:]
    show_code = False
    seed_broken = False
    max_repairs = MAX_REPAIRS
    rest = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--show-code":
            show_code = True
        elif a == "--seed-broken":
            seed_broken = True
        elif a == "--max-repairs":
            i += 1
            max_repairs = int(argv[i])
        elif a.startswith("--max-repairs="):
            max_repairs = int(a.split("=", 1)[1])
        else:
            rest.append(a)
        i += 1
    task = " ".join(rest) if rest else DEMO_TASK
    # ground truth (expected output) only applies to the built-in traffic-light demo
    is_demo = not rest
    expect = DEMO_EXPECT if is_demo else []
    seed = BROKEN_SEED if (seed_broken and is_demo) else None

    models = health()
    if models is None:
        print(f"⚠ can't reach Ollama at {OLLAMA_URL} "
              f"(set OLLAMA_URL=http://<host>:11434)", file=sys.stderr)
        sys.exit(1)

    if is_demo:
        mode = "seed-broken → repair" if seed else "generate → verify → repair"
        print(f"(no task given — running the built-in traffic-light demo: {mode})\n",
              file=sys.stderr)

    ok, code, repairs = generate_verify_repair(
        task, max_repairs, True, show_code, expect, seed)

    gate = "clippy + run" if expect else ("clippy + tests" if "#[test]" in code else "clippy + run")
    print("\n" + "=" * 60)
    if ok:
        print(f"✓ VERIFIED ({gate}), {repairs} repair(s)")
    else:
        print(f"✗ STILL FAILING after {repairs} repair(s) — "
              f"the oracle won, the model didn't converge")
    print("=" * 60 + "\n")
    print(code)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
