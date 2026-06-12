---
name: Feature request
about: Suggest an idea or improvement
title: "[feat] "
labels: enhancement
---

**Problem / motivation**
What are you trying to do that's hard or impossible today?

**Proposed solution**
What you'd like to see. If it's a new long-term backend, note that the extension
point is the `LongTermBackend` Protocol (`src/agentmem/backends/base.py`) — see
[CONTRIBUTING.md](../../CONTRIBUTING.md).

**Alternatives considered**
Other approaches or workarounds you've tried.

**Scope**
- [ ] Fits the lowest-common-denominator interface (`add`/`search`/`get_all`/`delete`)
- [ ] Backend-specific behavior can ride in `MemoryItem.metadata`
