from __future__ import annotations

from pathlib import Path

from memory_agent.eval.run_eval import summarize_run, token_f1_score

FIXTURES = Path(__file__).parent / "fixtures"


def test_token_f1_handles_overlap_and_unanswerable_questions():
    assert token_f1_score("blue bicycle", "blue bike", "1") == 0.5
    assert token_f1_score("I don't know.", "Not mentioned", "5") == 1.0


def test_summarize_run_reports_per_type_and_costs():
    result = summarize_run(FIXTURES / "run")

    assert result["system"] == "test"
    assert result["n_questions"] == 1
    assert result["by_type"]["1"]["avg_score"] > 0
    assert result["avg_llm_calls_per_question"] == 2.0
