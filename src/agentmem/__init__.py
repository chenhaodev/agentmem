"""agentmem — a switchable memory middleware for AI agents.

Short-term (shared, lightweight) and long-term (pluggable: mem0 / vector / letta)
memory behind one interface, powered by DeepSeek-V4-Flash.

    from agentmem import MemoryManager, Message

    mm = MemoryManager()                       # backend chosen by LONG_TERM_BACKEND
    mm.add_turn("sess1", "alice", Message("user", "I love hiking in the Alps"))
    ctx = mm.build_context("sess1", "alice", query="outdoor plans")
    print(ctx.as_prompt_block())
"""

from .config import Config
from .manager import MemoryContext, MemoryManager
from .types import MemoryItem, Message

__all__ = ["MemoryManager", "MemoryContext", "Config", "Message", "MemoryItem"]
__version__ = "0.1.0"
