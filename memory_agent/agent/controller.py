"""Main agent: write memories per session, then retrieve+answer per question.

Supports both update modes (S1: append_only, S2: conflict_aware).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from tqdm import tqdm

from .data_io import load_dialogues, load_questions
from .llm_client import load_embed, load_llm
from .tracer import Tracer
from ..memory.retriever import DenseRetriever
from ..memory.store import MemoryStore
from ..memory.updater import MemoryUpdater
from ..memory.writer import MemoryWriter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--mode",
        choices=["append_only", "conflict_aware"],
        default="append_only",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg_dir = Path(__file__).resolve().parents[1] / "configs"
    prompts = yaml.safe_load((cfg_dir / "prompts.yaml").read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load((cfg_dir / "model.yaml").read_text(encoding="utf-8"))
    top_k = model_cfg["retrieval"]["top_k"]
    embed_dim = model_cfg["retrieval"]["embed_dim"]

    llm = load_llm()
    embed = load_embed()
    tracer = Tracer(args.out)

    dialogues = {d["id"]: d for d in load_dialogues(args.data)}
    questions = load_questions(args.data)
    if args.limit:
        questions = questions[: args.limit]
        # keep only dialogues referenced by limited questions
        needed = {q["dialogue_id"] for q in questions}
        dialogues = {k: v for k, v in dialogues.items() if k in needed}
    unknown_dialogues = sorted({q["dialogue_id"] for q in questions} - dialogues.keys())
    if unknown_dialogues:
        raise ValueError(f"Questions reference missing dialogues: {unknown_dialogues}")

    questions_by_dialogue: dict[str, list[dict]] = defaultdict(list)
    for question in questions:
        questions_by_dialogue[question["dialogue_id"]].append(question)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_dialogues.json").write_text(
        json.dumps(list(dialogues.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Per-dialogue stores. Build all writes BEFORE QA so memory is complete.
    writer = MemoryWriter(llm, prompts["writer"], max_memories=model_cfg["writer"]["max_memories_per_session"])
    stores: dict[str, MemoryStore] = {}
    write_summary: dict[str, list[dict]] = {}

    for did, dlg in tqdm(dialogues.items(), desc=f"writing ({args.mode})"):
        store = MemoryStore(dim=embed_dim)
        updater = MemoryUpdater(
            store=store,
            embed_client=embed,
            llm=llm,
            prompt_template=prompts["conflict"],
            mode=args.mode,
            similarity_threshold=model_cfg["conflict"]["similarity_threshold"],
            top_k_neighbors=model_cfg["conflict"]["top_k_neighbors"],
        )
        actions_all = []
        for sess in dlg["sessions"]:
            units = writer.extract(
                session_id=sess["id"],
                turns=sess["turns"],
                session_timestamp=sess.get("timestamp", ""),
            )
            actions = updater.write(units)
            actions_all.extend(actions)
        stores[did] = store
        write_summary[did] = actions_all
        # dump store for inspection
        store.dump(out_dir / "stores" / f"{did}.json")

    # QA pass
    for did in dialogues:
        retriever = DenseRetriever(stores[did], embed, top_k=top_k)
        for q in questions_by_dialogue[did]:
            t0 = time.time()
            hits = retriever.retrieve(q["question"])
            mem_text = "\n".join(f"- {m.text}" for m, _ in hits) if hits else "(no memories)"
            prompt = prompts["qa"].format(memories=mem_text, question=q["question"])
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
                "evidence": q.get("evidence", []),
                "prediction": answer,
                "prompt": prompt,
                "retrieved": [{"text": m.text, "score": s, "id": m.id} for m, s in hits],
                "n_active_memories": len(stores[did].all_active()),
                "latency_s": round(time.time() - t0, 3),
            })

    tracer.close({
        "llm_calls": llm.call_count,
        "embed_calls": embed.call_count,
        "system": f"ours_{args.mode}",
        "n_dialogues": len(dialogues),
        "write_actions_summary": {
            did: dict(Counter(action["action"] for action in actions))
            for did, actions in write_summary.items()
        },
        "vector_backends": {did: store.backend for did, store in stores.items()},
    })


if __name__ == "__main__":
    main()
