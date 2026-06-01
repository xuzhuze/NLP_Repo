"""B1: stuff the entire dialogue history into the prompt (truncated)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml
from tqdm import tqdm

from ..agent.data_io import flatten_history, load_dialogues, load_questions
from ..agent.llm_client import load_llm
from ..agent.tracer import Tracer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg_dir = Path(__file__).resolve().parents[1] / "configs"
    prompts = yaml.safe_load((cfg_dir / "prompts.yaml").read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load((cfg_dir / "model.yaml").read_text(encoding="utf-8"))
    max_chars = model_cfg["full_context"]["max_context_chars"]

    llm = load_llm()
    tracer = Tracer(args.out)

    dialogues = {d["id"]: d for d in load_dialogues(args.data)}
    histories = {did: flatten_history(d) for did, d in dialogues.items()}

    questions = load_questions(args.data)
    if args.limit:
        questions = questions[: args.limit]

    for q in tqdm(questions, desc="B1 full_context"):
        t0 = time.time()
        history = histories.get(q["dialogue_id"], "")
        if len(history) > max_chars:
            history = history[-max_chars:]  # keep most recent
        prompt = prompts["qa_full_context"].format(history=history, question=q["question"])
        answer = llm.chat(
            prompt,
            temperature=model_cfg["generation"]["temperature"],
            max_tokens=model_cfg["generation"]["max_tokens"],
        )
        tracer.log({
            "question_id": q["id"],
            "dialogue_id": q["dialogue_id"],
            "type": q["type"],
            "question": q["question"],
            "gold": q["answer"],
            "prediction": answer,
            "prompt": prompt,
            "retrieved": [],
            "history_chars": len(history),
            "latency_s": round(time.time() - t0, 3),
        })
    tracer.close({"llm_calls": llm.call_count, "system": "full_context"})


if __name__ == "__main__":
    main()
