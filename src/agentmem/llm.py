"""DeepSeek-V4-Flash client via the OpenAI-compatible endpoint.

DeepSeek V4 speaks the OpenAI ChatCompletions API at https://api.deepseek.com,
so we reuse the official `openai` SDK and just repoint base_url. One shared
instance is injected into any backend that needs an LLM for fact extraction.
"""

from __future__ import annotations

import json
from typing import Any

from .config import Config
from .types import Message


class DeepSeekLLM:
    def __init__(self, config: Config):
        self.config = config
        self.model = config.deepseek_model
        self._client = None  # lazy: don't import openai until first use

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.config.deepseek_api_key,
                base_url=self.config.deepseek_base_url,
                timeout=self.config.deepseek_timeout,
                max_retries=self.config.deepseek_max_retries,
            )
        return self._client

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )
        return resp.choices[0].message.content or ""

    # -- memory-specific helper -------------------------------------------------
    _EXTRACT_SYS = (
        "You extract durable, atomic facts worth remembering long-term about the "
        "user from a conversation snippet (preferences, identity, goals, decisions, "
        "stable context). Ignore small talk and transient details. "
        'Respond ONLY with JSON: {"facts": ["fact 1", "fact 2", ...]}. '
        "Empty list if nothing is worth remembering."
    )

    def extract_facts(self, messages: list[Message]) -> list[str]:
        """LLM-driven extraction. Degrades to raw user turns if the call fails,
        so the pipeline (and the demo) still works offline."""
        convo = "\n".join(f"{m.role}: {m.content}" for m in messages)
        try:
            out = self.chat(
                [
                    {"role": "system", "content": self._EXTRACT_SYS},
                    {"role": "user", "content": convo},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            facts = json.loads(out).get("facts", [])
            return [f.strip() for f in facts if isinstance(f, str) and f.strip()]
        except Exception:
            # Offline / no-key fallback: keep raw user utterances as facts.
            return [m.content.strip() for m in messages if m.role == "user" and m.content.strip()]
