---
name: Bug report
about: Something isn't working as expected
title: "[bug] "
labels: bug
---

**What happened**
A clear description of the bug, and what you expected instead.

**Repro**
Minimal steps or code:

```python
from agentmem import MemoryManager, Message
# ...
```

**Config**
- Long-term backend: <!-- vector | mem0 | lightrag | letta | a+b -->
- Short-term store: <!-- memory | redis -->
- Consolidation: <!-- sync | async -->

**Environment**
- agentmem version / commit:
- Python version:
- OS:
- Backend lib version (if mem0/lightrag/letta):

**Logs / traceback**
```
paste here
```

**Checklist**
- [ ] Reproduces with the offline `vector` backend (helps isolate backend vs core), or noted why not
- [ ] `python tests/test_smoke.py` passes on my machine
