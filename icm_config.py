#!/usr/bin/env python3
"""
ICM config loader (templating Phase 0).

Reads `icm.config.json` from the ICM directory and applies its `model`/`embed_model`
as env defaults, so the existing runners (which read OLLAMA_MODEL / OLLAMA_EMBED_MODEL)
pick the configured models up with no code change. Explicit env still wins (setdefault).

This is the seam that lets you SWAP MODELS by editing one JSON field, and SWAP ICMs by
pointing at a different directory (ICM_DIR env or the dir this file lives in).
"""

import json
import os
import pathlib


def icm_dir():
    return pathlib.Path(os.environ.get("ICM_DIR", pathlib.Path(__file__).parent)).resolve()


def load(apply_env=True):
    cfg = json.loads((icm_dir() / "icm.config.json").read_text())
    if apply_env:
        if cfg.get("model"):
            os.environ.setdefault("OLLAMA_MODEL", cfg["model"])
        if cfg.get("embed_model"):
            os.environ.setdefault("OLLAMA_EMBED_MODEL", cfg["embed_model"])
    return cfg
