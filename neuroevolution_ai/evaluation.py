"""Deterministic benchmark scoring without test-set-based model selection."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Candidate:
    name: str
    validation_score: float
    test_score: float
    cost: float


def select_candidate(candidates: list[Candidate]) -> Candidate:
    """Select on validation only; test scores are reported, never ranked."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate.name, str) or not candidate.name.strip() or candidate.name in seen:
            raise ValueError("candidate names must be nonempty, unique strings")
        seen.add(candidate.name)
        for field in (candidate.validation_score, candidate.test_score, candidate.cost):
            if isinstance(field, bool) or not isinstance(field, (int, float)):
                raise ValueError("scores and costs must be numbers")
            try:
                finite = math.isfinite(field)
            except OverflowError as exc:
                raise ValueError("scores and costs must be finite and nonnegative") from exc
            if not finite or field < 0:
                raise ValueError("scores and costs must be finite and nonnegative")
    return min(candidates, key=lambda c: (-c.validation_score, c.name))
