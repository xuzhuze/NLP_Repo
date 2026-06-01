"""B2: embed raw dialogue turns, retrieve top-k, answer."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from ..agent.data_io import load_dialogues, load_questions
from ..agent.llm_client import load_embed, load_llm
from ..agent.tracer import Tracer
from ..memory.store import MemoryStore, MemoryUnit


def _chunk_dialogue(dialogue: dict) -> list[tuple[str, str, str]]:
    """Return list of (chunk_id, text, session_id). One chunk per turn."""
    chunks = []
    for s in dialogue["sessions"]:
        for i, t in enumerate(s["turns"]):
            cid = f"{s['id']}_t{i}"
            text = f"{t['speaker']}: {t['text']}"
            chunks.append((cid, text, s["id"]))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
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

    # Build one store per dialogue, lazily on first question for that dialogue.
    dialogues = {d["id"]: d for d in load_dialogues(args.data)}
    stores: dict[str, MemoryStore] = {}

    def _ensure_store(did: str) -> MemoryStore:
        if did in stores:
            return stores[did]
        store = MemoryStore(dim=embed_dim)
        chunks = _chunk_dialogue(dialogues[did])
        if chunks:
            texts = [c[1] for c in chunks]
            # batch embed in chunks of 32 to be polite to the API
            vecs: list[np.ndarray] = []
            for i in range(0, len(texts), 32):
                vecs.append(embed.embed(texts[i : i + 32]))
            all_vecs = np.vstack(vecs)
            for (cid, text, sid), v in zip(chunks, all_vecs):
                unit = MemoryUnit.new(text=text, source_session_id=sid, source_turn_ids=[cid])
                store.add(unit, v)
        stores[did] = store
        return store

    questions = load_questions(args.data)
    if args.limit:
        questions = questions[: args.limit]

    for q in tqdm(questions, desc="B2 vanilla_rag"):
        t0 = time.time()
        store = _ensure_store(q["dialogue_id"])
        q_emb = embed.embed([q["question"]])
        hits = store.search(q_emb, k=top_k)
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
            "prediction": answer,
            "prompt": prompt,
            "retrieved": [{"text": m.text, "score": s} for m, s in hits],
            "latency_s": round(time.time() - t0, 3),
        })
    tracer.close({
        "llm_calls": llm.call_count,
        "embed_calls": embed.call_count,
        "system": "vanilla_rag",
    })


if __name__ == "__main__":
    main()
