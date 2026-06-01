"""Deterministic TA-compatible agent used by the CLI integration test."""
from __future__ import annotations


class FakeTAAgent:
    def ingest(self, conversation: dict) -> None:
        self.conversation = conversation

    def answer(self, question: str) -> str:
        return "a cat"
