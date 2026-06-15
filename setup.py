#!/usr/bin/env python3
"""
Quick-setup / preflight for this ICM workspace. Pure stdlib — no pip install.

Checks everything you need to run with Ollama and reports a readiness summary:
  • Ollama reachable (WSL→Windows-host gateway auto-resolved; override with OLLAMA_URL)
  • the required models present (from icm.config.json) — `--pull` downloads any missing
  • the Rust toolchain (cargo, clippy) — the verify oracle
  • the std-docs reference corpus built (for tier-2 search)

    python3 setup.py            # check + report (and build the docs corpus if missing)
    python3 setup.py --pull     # also pull any missing Ollama models
"""

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).parent.resolve()
OK, BAD, WARN = "✓", "✗", "!"


# --- Ollama endpoint (WSL gateway aware) --------------------------------------
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
                o = [int(h[i:i + 2], 16) for i in (6, 4, 2, 0)]
                if all(0 <= x <= 255 for x in o):
                    return ".".join(map(str, o))
    except OSError:
        pass
    return None


def ollama_url():
    if os.environ.get("OLLAMA_URL"):
        return os.environ["OLLAMA_URL"].rstrip("/")
    if _is_wsl():
        gw = _wsl_gateway_ip()
        if gw:
            return f"http://{gw}:11434"
    return "http://localhost:11434"


def tags(url):
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=5) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", [])]
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


def have(model, present):
    base = model.split(":")[0]
    return any(t == model or t.split(":")[0] == base for t in present)


def pull(url, model):
    print(f"   pulling {model} (this can take several minutes)...", flush=True)
    body = json.dumps({"model": model, "stream": False}).encode()
    req = urllib.request.Request(f"{url}/api/pull", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.loads(r.read()).get("status") == "success"
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        print(f"   pull failed: {e}")
        return False


def main():
    do_pull = "--pull" in sys.argv[1:]
    results = []

    # config (instance vs engine)
    cfg_path = HERE / "icm.config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else None

    # 1) Ollama
    url = ollama_url()
    present = tags(url)
    if present is None:
        print(f"{BAD} Ollama not reachable at {url}")
        print("   To fix:")
        if _is_wsl():
            print("   • You're in WSL. If Ollama runs on WINDOWS it must listen on all")
            print("     interfaces — on Windows (PowerShell):  setx OLLAMA_HOST \"0.0.0.0\"")
            print("     then fully restart Ollama (quit from the tray + reopen), and allow it")
            print("     through Windows Firewall. WSL auto-detects the Windows host IP.")
            print("   • Or point at it explicitly:  export OLLAMA_URL=http://<host-ip>:11434")
            print("     (find <host-ip> with:  ip route | grep default   → the IP after 'via')")
        else:
            print("   • Same machine: make sure Ollama is running (`ollama serve`).")
            print("   • Remote host: export OLLAMA_URL=http://<host-ip>:11434")
            print("     (the remote Ollama must run with OLLAMA_HOST=0.0.0.0 + firewall open)")
        print("   then re-run:  python3 setup.py")
        results.append(False)
    else:
        print(f"{OK} Ollama reachable at {url} ({len(present)} model(s))")
        results.append(True)

        # 2) required models
        required = []
        if cfg:
            for k in ("model", "embed_model"):
                if cfg.get(k):
                    required.append(cfg[k])
        for m in required:
            if have(m, present):
                print(f"{OK} model present: {m}")
            elif do_pull:
                ok = pull(url, m)
                print(f"{OK if ok else BAD} pulled {m}" if ok else f"{BAD} could not pull {m}")
                results.append(ok)
            else:
                print(f"{BAD} model MISSING: {m}   →  ollama pull {m}   (or re-run with --pull)")
                results.append(False)

    # 3) toolchain (only if a verify command needs it — check cargo/clippy when present)
    needs_cargo = bool(cfg) and "cargo" in json.dumps(cfg.get("verify", {}))
    if needs_cargo or shutil.which("cargo"):
        for tool in ("cargo", "clippy-driver"):
            if shutil.which(tool):
                print(f"{OK} toolchain: {tool}")
            else:
                print(f"{BAD} toolchain missing: {tool}  (install Rust: https://rustup.rs)")
                results.append(False)

    # 4) reference corpus (build it from the local toolchain if this instance uses one)
    docs = HERE / "icm_docs.py"
    corpus = HERE / "refdocs" / "std.json"
    if docs.exists() and cfg and not corpus.exists():
        print(f"{WARN} building std-docs corpus (one-time)...")
        rc = subprocess.run([sys.executable, str(docs), "build"], cwd=HERE).returncode
        print(f"{OK if rc == 0 else BAD} std-docs corpus "
              f"{'built' if rc == 0 else 'FAILED — check the Rust toolchain'}")
        results.append(rc == 0)
    elif corpus.exists():
        print(f"{OK} std-docs corpus present")

    print()
    if all(results):
        if cfg:                                         # an instance
            print("READY. Try:  python3 icm_repair.py --show-code "
                  "\"write a function that reverses a string\"")
        else:                                           # the engine/template
            print("ENGINE READY. Make an ICM:  cp -r icms/skeleton my-icm  "
                  "→  python3 icm_mcp.py --icm my-icm")
    else:
        print("Some checks failed — fix the items marked above, then re-run `python3 setup.py`.")
        sys.exit(1)


if __name__ == "__main__":
    main()
