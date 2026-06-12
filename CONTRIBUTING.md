# Contributing to agentmem

Thanks for your interest! This is a small, focused codebase — the notes below
should get you productive quickly.

## Dev setup

```bash
git clone https://github.com/chenhaodev/agentmem
cd agentmem
pip install -e ".[dev]"          # core + pytest (offline development)
# pip install -e ".[all]"        # also pulls every backend's deps
cp .env.example .env             # add DEEPSEEK_API_KEY for anything that calls the LLM
```

Python 3.10+ is required. On older macOS the pinned CPU stack in
`requirements-local-cpu.txt` documents a working set of versions.

## Running tests

```bash
python tests/test_smoke.py       # offline suite — no network, no API key, no servers
pytest -q tests/test_smoke.py    # same suite via pytest
```

The offline suite is what CI runs (Python 3.10/3.11/3.12). **Keep it green and
offline** — it must never require network, an API key, or a backend server.

Live, opt-in tests exercise the real backends and self-skip unless `RUN_LIVE=1`:

```bash
set -a && . ./.env && set +a
export RUN_LIVE=1 TOKENIZERS_PARALLELISM=false
python tests/test_live.py        # each backend runs in its own subprocess
```

These need a DeepSeek key and, per backend, the relevant service (e.g. a Letta
server). See the README for backend-specific setup (mem0, LightRAG, Letta, Redis).

## Adding a long-term backend

The whole extension surface is the `LongTermBackend` Protocol in
`src/agentmem/backends/base.py`:

1. Implement `add` / `search` / `get_all` / `delete` over `MemoryItem`
   (`{text, metadata, score}`). Keep to this lowest-common-denominator —
   backend-specific extras go in `MemoryItem.metadata`.
2. Import the third-party lib lazily and raise a clear `ImportError` with the
   `pip install` line if it's missing (see the existing adapters).
3. Register it in `src/agentmem/backends/__init__.py::_build_single`.
4. Add the optional dependency as an extra in `pyproject.toml`.
5. Add an offline test (factory wiring / missing-dep message) to
   `tests/test_smoke.py`, and an opt-in live test to `tests/test_live.py`.

A router (`LONG_TERM_BACKEND="a+b"`) and `MemoryManager` work over the Protocol,
so a conforming backend needs no other changes.

## Style & conventions

- Match the surrounding code: type hints, `from __future__ import annotations`,
  small focused functions, comments that explain *why* not *what*.
- Don't put an LLM call on the short-term hot path.
- Run `python -m compileall src tests` and the offline suite before pushing.

## Pull requests

- Branch off `main`; keep commits small and focused with a clear message.
- Ensure the offline suite passes (CI will check it on 3.10–3.12).
- Note any new env vars in `.env.example` and the README config table.

By contributing you agree your contributions are licensed under the project's
[MIT License](LICENSE).
