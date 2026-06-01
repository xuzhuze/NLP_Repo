"""Aggregate experiment traces with an offline scorer or an optional LLM judge."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import yaml

from ..agent.llm_client import LLMClient, load_judge

ScoreFn = Callable[[str, str, str], float]


def _normalized_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def exact_match_score(pred: str, gold: str, _: str = "") -> float:
    return float(_normalized_text(pred) == _normalized_text(gold))


def token_f1_score(pred: str, gold: str, question_type: str = "") -> float:
    """Lightweight LoCoMo-style scorer for local iteration.

    The TA's final eval kit remains authoritative. This scorer intentionally has
    no heavyweight NLP dependency and is useful before API credentials exist.
    """
    pred_tokens = _normalized_text(pred).split()
    gold_tokens = _normalized_text(gold).split()
    if question_type == "5":
        unknown_markers = {
            "don t know",
            "do not know",
            "not mentioned",
            "no information available",
            "unknown",
        }
        normalized_pred = " ".join(pred_tokens)
        return float(any(marker in normalized_pred for marker in unknown_markers))
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _parse_judge_result(raw: str) -> tuple[float, str]:
    match = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
    if not match:
        return 0.0, "judge_parse_fail"
    try:
        result = json.loads(match.group(0))
    except json.JSONDecodeError:
        return 0.0, "judge_parse_fail"
    correct = result.get("correct", False)
    if isinstance(correct, str):
        correct = correct.strip().lower() in {"true", "yes", "1"}
    return float(bool(correct)), str(result.get("reason", ""))


def _judge_record(
    record: dict[str, Any],
    judge: LLMClient,
    prompt_template: str,
) -> tuple[float, str]:
    prompt = prompt_template.format(
        question=record.get("question", ""),
        gold=record.get("gold", ""),
        prediction=record.get("prediction", ""),
    )
    raw = judge.chat(prompt, temperature=0.0, max_tokens=128)
    return _parse_judge_result(raw)


def summarize_run(
    run_dir: Path,
    scorer_name: str = "token_f1",
    judge: LLMClient | None = None,
    judge_prompt: str = "",
) -> dict[str, Any]:
    traces = run_dir / "traces.jsonl"
    metrics_path = run_dir / "metrics.json"
    if not traces.exists():
        return {"run": run_dir.name, "error": "no traces.jsonl"}

    score_fn: ScoreFn
    if scorer_name == "exact_match":
        score_fn = exact_match_score
    elif scorer_name == "token_f1":
        score_fn = token_f1_score
    elif scorer_name == "judge" and judge is not None and judge_prompt:
        score_fn = token_f1_score  # unused, but keeps the branch explicit
    else:
        raise ValueError(f"Unsupported scorer: {scorer_name}")

    by_type_score: dict[str, float] = defaultdict(float)
    by_type_total: dict[str, int] = defaultdict(int)
    latencies = []
    judgments = []
    judge_calls_before = judge.call_count if judge is not None else 0
    with traces.open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            question_type = str(record.get("type", "unknown"))
            if scorer_name == "judge":
                score, reason = _judge_record(record, judge, judge_prompt)
                judgments.append({
                    "question_id": record.get("question_id", ""),
                    "score": score,
                    "reason": reason,
                })
            else:
                score = score_fn(
                    str(record.get("prediction", "")),
                    str(record.get("gold", "")),
                    question_type,
                )
            by_type_total[question_type] += 1
            by_type_score[question_type] += score
            latencies.append(float(record.get("latency_s", 0.0)))

    if judgments:
        (run_dir / "judgments.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in judgments),
            encoding="utf-8",
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    n_questions = sum(by_type_total.values())
    llm_calls = int(metrics.get("llm_calls", 0))
    embed_calls = int(metrics.get("embed_calls", 0))
    result = {
        "run": run_dir.name,
        "system": metrics.get("system", "?"),
        "scorer": scorer_name,
        "n_questions": n_questions,
        "by_type": {
            question_type: {
                "score_sum": round(by_type_score[question_type], 4),
                "total": by_type_total[question_type],
                "avg_score": round(by_type_score[question_type] / by_type_total[question_type], 4),
            }
            for question_type in sorted(by_type_total)
        },
        "overall_score": round(sum(by_type_score.values()) / n_questions, 4) if n_questions else 0.0,
        "avg_latency_s": round(sum(latencies) / max(1, len(latencies)), 3),
        "llm_calls": llm_calls,
        "embed_calls": embed_calls,
        "avg_llm_calls_per_question": round(llm_calls / max(1, n_questions), 3),
        "avg_embed_calls_per_question": round(embed_calls / max(1, n_questions), 3),
        "elapsed_s": metrics.get("elapsed_s", 0.0),
    }
    if scorer_name == "judge" and judge is not None:
        result["judge_calls"] = judge.call_count - judge_calls_before
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="parent dir of run subdirs")
    parser.add_argument("--out", default="", help="default: sibling results/summary.json")
    parser.add_argument(
        "--scorer",
        choices=["token_f1", "exact_match", "judge"],
        default="token_f1",
        help="Use judge for final LLM-as-Judge scoring; token_f1 is offline-only.",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs)
    judge = None
    judge_prompt = ""
    if args.scorer == "judge":
        judge = load_judge()
        cfg_dir = Path(__file__).resolve().parents[1] / "configs"
        prompts = yaml.safe_load((cfg_dir / "prompts.yaml").read_text(encoding="utf-8"))
        judge_prompt = prompts["judge"]

    summaries = [
        summarize_run(run_dir, scorer_name=args.scorer, judge=judge, judge_prompt=judge_prompt)
        for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir())
    ]

    out_path = Path(args.out) if args.out else runs_dir.parent / "results" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
