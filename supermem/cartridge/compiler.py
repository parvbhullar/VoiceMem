"""Context Compiler: SuperMem memory -> canonical, versioned cartridges.

SuperMem answers *what should the agent know*. The compiler turns that into
text blocks that stay byte-identical until the memory behind them changes, so
the engine can keep their KV. Canonicalisation is the whole job: the same
memories must always render to the same tokens (dedupe, stable sort, no
timestamps of compilation, no "now"-relative wording).
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from supermem.cartridge.contract import Cartridge, sha


class TokenCounter:
    """Counts tokens with the model's own tokenizer when it can be loaded,
    otherwise estimates (~4 chars/token) and says so on the cartridge."""

    def __init__(self, model: str, tokenizer: str | None = None, load: bool = True) -> None:
        """``load=False`` skips the tokenizer download: for hosted models (e.g. an
        OpenAI model name) there is no HF tokenizer, and the engine's own
        ``usage.prompt_tokens`` is the exact count anyway."""
        self.model = model
        self.tokenizer_id = tokenizer or model
        self._tok = None
        if not load:
            return
        try:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.tokenizer_id)
        except Exception:  # noqa: BLE001 -- offline / not installed: fall back to an estimate
            self._tok = None

    @property
    def exact(self) -> bool:
        return self._tok is not None

    def count(self, text: str) -> int:
        if self._tok is not None:
            return len(self._tok.encode(text, add_special_tokens=False))
        return max(1, len(text) // 4)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().rstrip("."))


@dataclass(frozen=True)
class Fact:
    slot: str
    content: str
    observed_at: str = ""         # YYYY-MM-DD the fact was said, if known


class ContextCompiler:
    def __init__(self, model: str, tenant: str, counter: TokenCounter | None = None,
                 codec: str = "fp16") -> None:
        self.model = model
        self.tenant = tenant
        self.counter = counter or TokenCounter(model)
        self.codec = codec

    def _make(self, kind: str, scope: dict, body: str, version: int | None = None) -> Cartridge:
        return Cartridge(
            kind=kind, tenant=self.tenant, scope=scope, body=body,
            # With no explicit version the content hash stands in for it:
            # it changes exactly when the rendered memory changes.
            version=version if version is not None else int(sha(body, 8), 16) % 100000,
            model=self.model, tokenizer=self.counter.tokenizer_id,
            tokens=self.counter.count(body), tokens_estimated=not self.counter.exact,
            codec=self.codec,
        )

    # ── organisation (A) ──────────────────────────────────────────────────────

    def compile_org(self, org_id: str, sections: list[tuple[str, str]],
                    version: int | None = None) -> Cartridge:
        """``sections``: (title, text) pairs. Order is kept as given: policy
        documents have an authored order and sorting would scramble them."""
        body = "\n\n".join(f"## {title}\n{text.strip()}" for title, text in sections)
        return self._make("org", {"org": org_id}, body, version)

    # ── caller (B) ────────────────────────────────────────────────────────────

    def compile_user(self, user_id: str, facts: list[Fact], traits: list[str] = (),
                     display_name: str = "", version: int | None = None) -> Cartridge:
        # Sort before deduping so the surviving copy of a repeated fact (the
        # earliest one) never depends on the order the memories arrived in.
        seen: set[str] = set()
        uniq: list[Fact] = []
        for f in sorted(facts, key=lambda f: (f.slot, f.observed_at, _norm(f.content), f.content)):
            k = _norm(f.content)
            if k and k not in seen:
                seen.add(k)
                uniq.append(f)

        lines = [f"Caller: {display_name or user_id} (id {user_id})", "", "Known facts:"]
        slot = None
        for f in uniq:
            if f.slot != slot:
                slot = f.slot
                lines.append(f"[{slot}]")
            lines.append(f"- [{f.observed_at}] {f.content}" if f.observed_at else f"- {f.content}")

        trait_lines = sorted({t.strip() for t in traits if t.strip()})
        if trait_lines:
            lines += ["", "How to talk to this caller (never say these back to them):"]
            lines += [f"- {t}" for t in trait_lines]
        return self._make("user", {"user": user_id}, "\n".join(lines), version)

    # ── caller x organisation (A x B) ─────────────────────────────────────────

    def compile_rel(self, org_id: str, user_id: str, records: dict[str, str],
                    version: int | None = None) -> Cartridge:
        body = "\n".join(f"- {k}: {records[k]}" for k in sorted(records))
        return self._make("rel", {"org": org_id, "user": user_id}, body, version)


def facts_from_space(space_dir: str | Path, user_id: str | None = None,
                     limit: int = 400, max_chars: int = 600) -> tuple[list[Fact], list[str]]:
    """Read a SuperMem memory space (``supermem_memoryspace/<space>/``) straight
    from its sqlite: left-brain facts and right-brain traits for one user.

    Read-only; the live SuperMem process keeps writing the same file. Rows
    longer than ``max_chars`` are skipped: a memory is a sentence or two, and a
    runaway row would otherwise swallow the whole cartridge budget."""
    space_dir = Path(space_dir)
    try:
        from supermem.utils.common import space as _space
        db = Path(_space.db(space_dir))
    except Exception:  # noqa: BLE001 -- keep the compiler usable without the full package
        db = space_dir / f"{space_dir.name}.sqlite"
    if not db.exists():
        raise FileNotFoundError(f"no memory space database at {db}")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        where, args = ("WHERE user_id = ? AND", (user_id,)) if user_id else ("WHERE", ())
        where += f" length(content) <= {int(max_chars)}"
        facts = [
            Fact(slot=slot or "general", content=content, observed_at=(created or "")[:10])
            for slot, content, created in con.execute(
                f"SELECT slot, content, created_at FROM memories {where} "
                f"ORDER BY created_at LIMIT ?", (*args, limit))
        ]
        traits = [
            content for (content,) in con.execute(
                f"SELECT content FROM right_brain_memories {where} "
                f"ORDER BY priority DESC, created_at LIMIT 40", args)
        ]
    finally:
        con.close()
    return facts, traits
