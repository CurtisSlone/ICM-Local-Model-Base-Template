# <name> — spec

Goal doc for a spec-driven build. The build runs from this file; its **Acceptance** section
becomes the oracle, so put every must-have requirement there as a checkable example.

## Goal
One line: what this program does.

## Inputs
args / stdin / files; types and formats.

## Outputs
stdout / exit codes; exact format.

## Behavior
The requirements, as bullets.

## Acceptance (these become tests — the oracle)
Put the core logic in a pure, unit-testable function. List concrete input→output examples;
they will be encoded as `#[test]`s (or your language's tests) and MUST pass.

| input | expected |
|---|---|
| ... | ... |

## Constraints
- standard library only (no external crates) / single file, unless stated
- must terminate
- MUST include tests covering the Acceptance rows (so the test runner is the oracle)

## Out of scope
What this deliberately does NOT do.

<!--
Why this shape: the spec is both GOAL and ORACLE. The model follows the *tests*, not the
prose — so anything not in Acceptance is unverified. Encode requirements as acceptance rows.
For an I/O program, keep logic in a pure function + tests so the test runner (not a blocking
`run` with no stdin) is the gate. See ICM-Rust-Test/DEVLOG.md for the worked lessons.
-->
