"""Extract emotion-graph query terms from ASR and left-brain context."""

from __future__ import annotations


def build_query_terms(
    asr_text: str | None,
    left_context_summary: str | None,
    *,
    max_terms: int = 12,
) -> list[str]:
    terms: list[str] = []
    for chunk in (asr_text or "", left_context_summary or ""):
        for token in chunk.replace("\uff0c", " ").replace("\u3002", " ").replace(",", " ").split():
            token = token.strip()
            if len(token) >= 2:
                terms.append(token)
    return terms[:max_terms]
