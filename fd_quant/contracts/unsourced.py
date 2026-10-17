"""Unsourced-number detection: find numeric claims that lack an evidence link.

Consumers (packs, audit operators, the decision ledger) reuse this instead of
re-implementing the "a number needs a source" rule.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional


def unsourced_fields(instance: Any, kind: Optional[str] = None) -> list[str]:
    """Return the field paths of numeric claims with no resolvable evidence.

    Currently: `price_target` when set but `target_evidence_ids` is empty, and
    any `target_evidence_ids` entry that does not resolve inside `evidence`.
    Deterministic and local.
    """
    if not isinstance(instance, Mapping):
        return []
    if kind is None:
        kind = "rating" if ("side" in instance or "subject" in instance) else "target_weights"
    out: list[str] = []
    if kind == "rating":
        pt = instance.get("price_target")
        refs = instance.get("target_evidence_ids") or []
        ids = {
            e.get("evidence_id")
            for e in (instance.get("evidence") or [])
            if isinstance(e, Mapping)
        }
        if pt is not None:
            if not refs:
                out.append("price_target")
            else:
                for r in refs:
                    if r not in ids:
                        out.append(f"target_evidence_ids[{r}]")
    return out
