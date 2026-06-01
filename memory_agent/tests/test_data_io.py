from __future__ import annotations

from pathlib import Path

from memory_agent.agent.data_io import load_dialogues, load_questions

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_official_locomo_layout():
    dialogues = load_dialogues(FIXTURES / "official")
    questions = load_questions(FIXTURES / "official")

    assert dialogues[0]["id"] == "conv-test"
    assert dialogues[0]["sessions"][0]["timestamp"] == "1 Jan 2026"
    assert dialogues[0]["sessions"][0]["turns"][0]["id"] == "D1:1"
    assert questions[0]["answer"] == "a cat"
    assert questions[1]["answer"] == "Not mentioned"


def test_load_simplified_layout_accepts_gold():
    dialogues = load_dialogues(FIXTURES / "simplified")
    questions = load_questions(FIXTURES / "simplified")

    assert dialogues[0]["sessions"][0]["turns"][0]["text"] == "I own a cat."
    assert questions[0]["answer"] == "No"
