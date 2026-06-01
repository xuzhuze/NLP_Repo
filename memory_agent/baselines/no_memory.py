"""B0: answer with the question only, no dialogue context."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml
from tqdm import tqdm

from ..agent.data_io import load_questions
from ..agent.llm_client import load_llm
from ..agent.tracer import Tracer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    cfg_dir = Path(__file__).resolve().parents[1] / "configs"
    prompts = yaml.safe_load((cfg_dir / "prompts.yaml").read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load((cfg_dir / "model.yaml").read_text(encoding="utf-8"))

    llm = load_llm()
    tracer = Tracer(args.out)

    questions = load_questions(args.data)
    if args.limit:
        questions = questions[: args.limit]

    for q in tqdm(questions, desc="B0 no_memory"):
        t0 = time.time()
        prompt = prompts["qa_no_memory"].format(question=q["question"])
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
            "latency_s": round(time.time() - t0, 3),
        })
    tracer.close({"llm_calls": llm.call_count, "system": "no_memory"})


if __name__ == "__main__":
    main()
