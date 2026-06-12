<!-- Thanks for contributing! Keep PRs small and focused. -->

**What & why**
What does this change and what problem does it solve? Link any related issue
(e.g. `Closes #123`).

**How it was tested**
<!-- which backends / offline vs live -->

**Checklist**
- [ ] `python tests/test_smoke.py` passes (offline suite stays green and offline)
- [ ] `python -m compileall src tests` is clean
- [ ] New env vars documented in `.env.example` and the README config table
- [ ] New long-term backend? implements the `LongTermBackend` Protocol, registered
      in `_build_single`, deps added as a `pyproject.toml` extra, tests added
- [ ] Live behavior verified with `RUN_LIVE=1` (if it touches a real backend)
