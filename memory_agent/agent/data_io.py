"""LoCoMo data loader.

Supported layouts:

1. Simplified TA layout:
    data/locomo_simplified/dialogues.json
    data/locomo_simplified/questions.json

2. Official LoCoMo release:
    data/locomo_official/locomo10.json

Both layouts are normalized to:

    Dialogue = {"id": str, "sessions": [{"id": str, "timestamp": str,
                "turns": [{"id": str, "speaker": str, "text": str}]}]}
    Question = {"id": str, "dialogue_id": str, "type": str,
                "question": str, "answer": str, "evidence": list[str]}
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _coerce_turns(raw_turns: Any) -> list[dict[str, str]]:
    out = []
    if not isinstance(raw_turns, list):
        return out
    for idx, t in enumerate(raw_turns):
        if isinstance(t, dict):
            out.append({
                "id": str(t.get("id") or t.get("turn_id") or t.get("dia_id") or f"turn_{idx}"),
                "speaker": str(t.get("speaker") or t.get("role") or t.get("from") or "user"),
                "text": str(t.get("text") or t.get("content") or t.get("utterance") or ""),
            })
        elif isinstance(t, str):
            out.append({"id": f"turn_{idx}", "speaker": "user", "text": t})
    return out


def _coerce_dialogue(raw: dict[str, Any], idx: int) -> dict[str, Any]:
    did = str(raw.get("id") or raw.get("dialogue_id") or f"dlg_{idx}")
    raw_sessions = raw.get("sessions") or raw.get("conversations") or []
    if not raw_sessions and "turns" in raw:
        raw_sessions = [{"id": f"{did}_s0", "turns": raw["turns"]}]
    sessions = []
    for j, s in enumerate(raw_sessions):
        if not isinstance(s, dict):
            continue
        sessions.append({
            "id": str(s.get("id") or f"{did}_s{j}"),
            "timestamp": str(s.get("timestamp") or s.get("date") or ""),
            "turns": _coerce_turns(s.get("turns") or s.get("utterances") or []),
        })
    return {"id": did, "sessions": sessions}


def _resolve_json(path: str | Path, candidates: list[str]) -> Path:
    p = Path(path)
    if p.is_dir():
        resolved = next((p / name for name in candidates if (p / name).exists()), None)
        if resolved is None:
            raise FileNotFoundError(
                f"No supported JSON file found in {p}. Tried: {', '.join(candidates)}"
            )
        return resolved
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def _read_json(path: str | Path, candidates: list[str]) -> Any:
    return json.loads(_resolve_json(path, candidates).read_text(encoding="utf-8"))


def _is_official_locomo(raw: Any) -> bool:
    return (
        isinstance(raw, list)
        and bool(raw)
        and isinstance(raw[0], dict)
        and "sample_id" in raw[0]
        and "conversation" in raw[0]
        and "qa" in raw[0]
    )


def _coerce_official_dialogue(raw: dict[str, Any], idx: int) -> dict[str, Any]:
    did = str(raw.get("sample_id") or f"dlg_{idx}")
    conversation = raw.get("conversation") or {}
    numbered_sessions = []
    for key, turns in conversation.items():
        match = re.fullmatch(r"session_(\d+)", key)
        if match and isinstance(turns, list):
            numbered_sessions.append((int(match.group(1)), key, turns))
    sessions = [
        {
            "id": f"{did}_{key}",
            "timestamp": str(conversation.get(f"{key}_date_time") or ""),
            "turns": _coerce_turns(turns),
        }
        for _, key, turns in sorted(numbered_sessions)
    ]
    return {"id": did, "sessions": sessions}


def _answer_text(raw: dict[str, Any]) -> str:
    if raw.get("answer") is not None:
        return str(raw["answer"])
    if raw.get("gold") is not None:
        return str(raw["gold"])
    if str(raw.get("category")) == "5":
        # Official LoCoMo adversarial questions intentionally have no answer.
        return "Not mentioned"
    return ""


def _coerce_question(
    raw: dict[str, Any],
    idx: int,
    dialogue_id: str = "",
) -> dict[str, Any]:
    evidence = raw.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = [evidence]
    return {
        "id": str(raw.get("id") or f"{dialogue_id or 'q'}_q{idx}"),
        "dialogue_id": str(raw.get("dialogue_id") or raw.get("conv_id") or dialogue_id),
        "type": str(raw.get("type") or raw.get("category") or "unknown"),
        "question": str(raw.get("question") or raw.get("query") or ""),
        "answer": _answer_text(raw),
        "evidence": [str(item) for item in evidence],
        "adversarial_answer": str(raw.get("adversarial_answer") or ""),
    }


def load_dialogues(path: str | Path) -> list[dict[str, Any]]:
    raw = _read_json(
        path,
        ["dialogues.json", "conversations.json", "locomo.json", "locomo10.json"],
    )
    if _is_official_locomo(raw):
        return [_coerce_official_dialogue(item, idx) for idx, item in enumerate(raw)]
    items = raw if isinstance(raw, list) else raw.get("dialogues") or raw.get("data") or []
    return [_coerce_dialogue(r, i) for i, r in enumerate(items)]


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    raw = _read_json(path, ["questions.json", "qa.json", "test.json", "locomo10.json"])
    if _is_official_locomo(raw):
        out = []
        for sample in raw:
            did = str(sample.get("sample_id") or "")
            out.extend(
                _coerce_question(question, idx, dialogue_id=did)
                for idx, question in enumerate(sample.get("qa") or [])
            )
        return out
    items = raw if isinstance(raw, list) else raw.get("questions") or raw.get("data") or []
    return [_coerce_question(question, idx) for idx, question in enumerate(items)]


def flatten_history(dialogue: dict[str, Any]) -> str:
    """Concatenate every turn across sessions; for full_context baseline."""
    parts = []
    for s in dialogue["sessions"]:
        if s.get("timestamp"):
            parts.append(f"[{s['timestamp']}]")
        for t in s["turns"]:
            parts.append(f"{t['speaker']}: {t['text']}")
    return "\n".join(parts)
