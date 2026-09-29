"""LoCoMo: question answering over very long multi-turn conversations.

The data is a json array; each entry = one conversation spanning several sessions + QA about it.
The public version looks roughly like this (field names vary between versions; the parsing below handles the common variants)::

    [{"sample_id": "conv-26",
      "conversation": {
        "speaker_a": "Alice", "speaker_b": "Bob",
        "session_1_date_time": "2023-05-08 10:00",
        "session_1": [{"speaker": "Alice", "text": "...", "dia_id": "D1:1"}, ...],
        "session_2_date_time": ..., "session_2": [...]},
      "qa": [{"question": "...", "answer": "...", "category": 1}, ...]}]

**Before running, use --inspect to confirm parsing is right** (how many conversations, turns, whether timestamps were read);
if fields don't match, fix those few lines in load() -- better than finding out it was all wrong after a two-hour run.
"""
from __future__ import annotations

import json
import re

#: Display name (the CLI key is lowercase "locomo")
NAME = "LoCoMo"

from evaluation.datasets import Conversation, Question, Score, Turn

#: LoCoMo question-type ids -> readable names, for per-category scores
CATEGORIES = {
    1: "multi_hop", 2: "temporal", 3: "open_domain",
    4: "single_hop", 5: "adversarial",
}


def load(path: str) -> list[Conversation]:
    raw = json.loads(open(path, encoding="utf-8").read())
    if isinstance(raw, dict):                     # some versions are {"conv-26": {...}}
        raw = [{**v, "sample_id": k} for k, v in raw.items()]

    out: list[Conversation] = []
    for i, sample in enumerate(raw):
        cid = str(sample.get("sample_id") or sample.get("id") or f"conv_{i}")
        out.append(Conversation(id=cid,
                                turns=_turns(sample.get("conversation") or sample),
                                questions=_questions(sample.get("qa") or sample.get("questions") or [])))
    return out


def _turns(conv: dict) -> list[Turn]:
    """Flatten session_1 / session_2 ... in numeric order into one sequence of turns.

    Each session has its own date (session_N_date_time) and it must be kept -- LoCoMo has a whole class of
    temporal questions ("did this happen before which event"), which score zero if the dates are lost.
    """
    sessions: list[tuple[int, list, str]] = []
    for key, val in conv.items():
        m = re.fullmatch(r"session_(\d+)", str(key))
        if m and isinstance(val, list):
            when = conv.get(f"session_{m.group(1)}_date_time") or conv.get(f"{key}_date_time") or ""
            sessions.append((int(m.group(1)), val, str(when)))
    sessions.sort()

    turns: list[Turn] = []
    for _, msgs, when in sessions:
        for m in msgs:
            text = (m.get("text") or m.get("clean_text") or m.get("utterance") or "").strip()
            if not text:
                continue
            # some samples keep image captions in a separate field; feed them in too, or questions like "the cat in the photo" have no basis
            if m.get("blip_caption"):
                text = f"{text} (image: {m['blip_caption']})"
            turns.append(Turn(speaker=str(m.get("speaker") or "user"),
                              text=text, observed_at=_date(when)))
    return turns


def _date(when: str) -> str:
    """"2023-05-08 10:00" / "8 May, 2023" -> ISO date; empty if unrecognised."""
    if not when:
        return ""
    if m := re.search(r"(\d{4})-(\d{2})-(\d{2})", when):
        return m.group(0)
    if m := re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", when):
        months = {m_: f"{i:02d}" for i, m_ in enumerate(
            ["jan", "feb", "mar", "apr", "may", "jun",
             "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
        mm = months.get(m.group(2)[:3].lower())
        if mm:
            return f"{m.group(3)}-{mm}-{int(m.group(1)):02d}"
    return ""


def _questions(qa: list) -> list[Question]:
    out = []
    for j, q in enumerate(qa):
        text = (q.get("question") or "").strip()
        if not text:
            continue
        ans = q.get("answer", q.get("adversarial_answer", ""))
        out.append(Question(
            id=str(q.get("question_id") or f"q{j}"), text=text,
            answer="" if ans is None else str(ans),
            category=CATEGORIES.get(q.get("category"), str(q.get("category", ""))),
            meta={"evidence": q.get("evidence", [])},
        ))
    return out


JUDGE = """Decide whether the "model answer" and the "gold answer" say the same thing.

Gold answer: {gold}
Model answer: {pred}

Be lenient: if the meaning is right it counts as correct; differences in wording, detail or format are not penalised; when the
gold answer is a date/number, the right value counts as correct. Only mark it wrong if the model answer is clearly different
information, or says it doesn't know.

Reply with one word only: Yes / No"""


def score(q: Question, answer: str, judge) -> Score:
    """Compare with the gold answer. LoCoMo answers are mostly short phrases; literal matching misses too many, so a judge model decides."""
    gold = (q.answer or "").strip()
    if not gold:                       # questions without a gold answer are excluded from the total so they neither lower nor inflate it
        return Score(correct=0.0, total=0.0, note="no gold answer, skipped")

    pred = (answer or "").strip()
    if pred and gold.lower() in pred.lower():      # an obvious substring match needs no paid judge call
        return Score(correct=1.0, note="literal match")

    verdict = judge("You are an evaluation judge. Reply only \"Yes\" or \"No\".",
                    JUDGE.format(gold=gold, pred=pred or "(empty)"))
    ok = verdict.strip().startswith(("Y", "y", "T", "t"))
    return Score(correct=1.0 if ok else 0.0, note=verdict.strip()[:40])
