#!/usr/bin/env python3
"""
ICM generic oracle — config-driven code verification (templating Phase 0).

The cargo clippy+run oracle was hardcoded in icm_repair.py. To make the ICM pattern
reusable across languages/domains, the oracle is now driven by a `verify` block in
`icm.config.json`: which files to materialise (with `{code}` substituted), a `check`
command (must exit 0), and an optional behaviour gate (`run` with a timeout, or `test`
detected by a marker). Nothing here knows about Rust — a Python or Go ICM ships its own
verify block and this same code runs it.

    from icm_verify import verify_code
    ok, diagnostics = verify_code(code, cfg["verify"])      # one-shot (own temp dir)
    ok, diagnostics = verify_code(code, vcfg, workdir=wd)   # reuse dir (incremental builds)

Pure stdlib.
"""

import os
import pathlib
import re
import subprocess
import sys
import tempfile


def _cap(text, max_lines=50):
    """Bounded full diagnostics — the verbose form carries hints (e.g. 'the nearest
    open delimiter is here') that short/grep'd output drops."""
    return "\n".join((text or "").strip().splitlines()[:max_lines])


def _run(command, workdir, env_extra, timeout):
    """Run a command; return CompletedProcess, or None if it ran past `timeout`
    (a hang — e.g. an infinite loop — is a verification failure, not tolerated)."""
    try:
        return subprocess.run(
            command, cwd=workdir, capture_output=True, text=True, timeout=timeout,
            start_new_session=True, env={**os.environ, **(env_extra or {})},
        )
    except subprocess.TimeoutExpired:
        return None


def _materialise(code, vcfg, workdir):
    """Write the configured project files into workdir, substituting {code}."""
    for rel, template in vcfg["project_files"].items():
        path = workdir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(template.replace("{code}", code))


def verify_code(code, vcfg, expect=None, workdir=None):
    """Run the oracle described by `vcfg`. Returns (ok: bool, diagnostics: str).

    vcfg keys:
      project_files: {relpath: template}   # template may contain {code}
      check:  {command: [...], timeout?}   # gate 1: must exit 0 (compile/lint)
      test?:  {command: [...], detect, timeout?}  # gate 2a: run if `detect` in code
      run?:   {command: [...], timeout?}   # gate 2b: must exit 0; behaviour gate
      env?:   {VAR: val}
    `expect`: tokens that must appear in run stdout (behaviour ground truth).
    """
    env = vcfg.get("env")
    own = workdir is None
    tmp = tempfile.mkdtemp(prefix="icm_verify_") if own else None
    wd = pathlib.Path(tmp) if own else pathlib.Path(workdir)
    try:
        _materialise(code, vcfg, wd)

        # ---- Tier 1: CORRECTNESS (must pass) — compile + behaviour ----
        # Compiling here catches hard errors (E0277, syntax); tests/run catch behaviour and
        # infinite loops. This is the "does it meet the spec" gate.
        test = vcfg.get("test")
        run = vcfg.get("run")
        if test and test.get("detect", "\0") in code:
            t = _run(test["command"], wd, env, test.get("timeout", 60))
            if t is None:
                return False, "the test run did not terminate (likely an infinite loop)."
            if t.returncode != 0:
                return False, "tests failed:\n" + _cap(t.stdout or t.stderr)
        elif run:
            r = _run(run["command"], wd, env, run.get("timeout", 15))
            if r is None:
                return False, ("the program did not terminate (infinite loop / sleep). It "
                               "must run to completion on its own.")
            if r.returncode != 0:
                return False, "program failed at runtime:\n" + _cap(r.stderr)
            missing = [tok for tok in (expect or []) if tok not in r.stdout]
            if missing:
                return False, ("compiles and runs but the output is wrong. Expected "
                               f"{missing} in stdout, but got:\n"
                               f"{r.stdout.strip() or '(empty — printed nothing)'}")
        else:
            # no behaviour gate configured → use the check command purely as a compile gate
            c = _run(vcfg["check"]["command"], wd, env, vcfg["check"].get("timeout", 60))
            if c is None or c.returncode != 0:
                return False, _cap((c.stderr or c.stdout) if c else "check did not terminate")
            return True, ""

        # ---- Tier 2: LINT (advisory) — report, don't block ----
        # A correct, spec-conformant program can still trip a style lint (e.g. an unused import
        # nothing can auto-fix). That's polish, not correctness — so we report it and ship.
        check = vcfg.get("check")
        if check:
            c = _run(check["command"], wd, env, check.get("timeout", 60))
            if c is not None and c.returncode != 0:
                n = len([l for l in (c.stderr or "").splitlines()
                         if " error" in l or " warning" in l])
                print(f"  · lint (advisory): {n or 'some'} issue(s) — meets spec, not blocking",
                      file=sys.stderr)
        return True, ""
    finally:
        if own:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


def first_problem(text):
    for ln in (text or "").splitlines():
        if re.match(r"\s*(error|warning|fail)", ln, re.I):
            return ln.strip()
    return ((text or "").splitlines() or ["(no output)"])[0].strip()
