"""End-to-end demo of the agentmem middleware.

Runs fully offline by default (LONG_TERM_BACKEND=vector + hashing embedder +
LLM extraction that falls back to raw turns if DeepSeek is unreachable). With a
live DEEPSEEK_API_KEY it will additionally do real LLM fact extraction.

    pip install -r requirements.txt
    python demo.py                       # vector backend (no extra deps)
    LONG_TERM_BACKEND=mem0 python demo.py  # after: pip install mem0ai
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from agentmem import MemoryManager, Message  # noqa: E402

SESSION, USER = "sess-1", "alice"

CONVERSATION = [
    ("user", "Hi! I'm Alice, a backend engineer based in Berlin."),
    ("assistant", "Nice to meet you, Alice!"),
    ("user", "I love hiking in the Alps and I'm vegetarian."),
    ("assistant", "Great — I'll keep that in mind."),
    ("user", "I'm planning a team offsite next quarter."),
    ("assistant", "Sounds fun. Where are you thinking?"),
    ("user", "Somewhere with good trails and veggie food."),
]


def main() -> None:
    mm = MemoryManager()
    print(f"== agentmem demo ==  long-term backend: {mm.long_term.name}\n")

    for role, content in CONVERSATION:
        mm.add_turn(SESSION, USER, Message(role, content))
        print(f"  {role:>9}: {content}")

    print("\n-- end session (flush short-term -> long-term) --")
    stored = mm.end_session(SESSION, USER)
    print(f"  consolidated {len(stored)} long-term memories")

    print("\n-- recall: 'what food does the user eat?' --")
    for m in mm.recall(USER, "what food does the user eat?", k=3):
        print(f"  {m}")

    print("\n-- recall: 'where does she like outdoor activities?' --")
    for m in mm.recall(USER, "outdoor activities location", k=3):
        print(f"  {m}")

    print("\n-- assembled prompt context for a new turn --")
    mm.add_turn(SESSION, USER, Message("user", "Any offsite location ideas?"))
    ctx = mm.build_context(SESSION, USER, query="offsite location with trails and veggie food")
    print(ctx.as_prompt_block())


if __name__ == "__main__":
    main()
