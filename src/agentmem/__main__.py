"""`python -m agentmem` / `agentmem-demo` — a tiny end-to-end demo.

Uses whatever LONG_TERM_BACKEND is configured (defaults to the offline vector
backend). See demo.py at the repo root for a fuller walkthrough.
"""

from __future__ import annotations

from .manager import MemoryManager
from .types import Message


def main() -> None:
    mm = MemoryManager()
    print(f"agentmem demo — long-term backend: {mm.long_term.name}\n")

    convo = [
        ("user", "Hi, I'm Alex, a backend engineer in Berlin."),
        ("user", "I'm vegetarian and I love hiking in the Alps."),
        ("user", "I'm planning a team offsite next quarter."),
    ]
    for role, content in convo:
        mm.add_turn("demo", "alex", Message(role, content))
        print(f"  {role}: {content}")
    mm.end_session("demo", "alex")

    print("\nrecall 'what food does the user eat?':")
    for m in mm.recall("alex", "what food does the user eat?", k=3):
        print(f"  {m}")
    mm.close()


if __name__ == "__main__":
    main()
