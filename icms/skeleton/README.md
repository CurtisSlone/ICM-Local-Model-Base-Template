# ICM skeleton — copy me

A starter ICM (data only). Copy this directory, then fill it in:

1. **`icm.config.json`** — set `name`, `domain`, `model`/`embed_model`, and the **`verify`**
   oracle for your language (project files + check/run/test commands; `{code}` is substituted).
   Trim the `tools` list to what your domain needs.
2. **`manifest.json`** — your routing index. Sharp, *discriminating*, non-overlapping summaries.
3. **`kb/*.md`** — your knowledge, one topic per file. Verify any checkable examples.
4. **`SYSTEM.md`** — operating rules + what's deliberately out of scope.
5. **`flows/`** — the two standard workflows are here; edit or add your own.
6. **`refdocs/`** + **`eval/`** *(optional)* — a tier-2 reference corpus and recall@K queries.

Then run it with the engine:
```bash
python3 /home/local/ollama/icm-core/icm_mcp.py --icm /path/to/this-icm
python3 /home/local/ollama/icm-core/test_mcp.py /path/to/this-icm   # protocol smoke test
```
No code goes in here — the engine lives in `icm-core/`. This dir is pure content.
