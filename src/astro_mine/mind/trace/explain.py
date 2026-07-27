"""Plan explanation derived from the decision trace (RM-P1-MIND-07; LUNAR-UX-003).

Turns a :class:`~astro_mine.mind.trace.model.DecisionTrace` into an operator-facing explanation —
the "why this assignment / why dig here" an operator supervising under latency needs (LUNAR-UX-003),
built from the same records the determinism gate compares. For each tick it narrates which tiers
replanned and why (the replan trigger), where Guard intervened (with the invoked ``SafetySpec``
clauses), and where the swarm degraded (act-while-stale under comms loss, reconcile on recovery,
fallback / coord activations). Pure and deterministic: a given trace always explains the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

from astro_mine.mind.trace.model import DecisionTrace, TickRecord

__all__ = ["PlanExplanation", "TickExplanation", "explain"]

_NOTE_PHRASES = {
    "comms_stale_hold": "held cached intent through comms loss (act-while-stale)",
    "comms_recovered": "reconciled on comms recovery",
    "coord_yield": "yielded a task to a neighbor (decentralized conflict resolution)",
}


@dataclass(frozen=True, slots=True)
class TickExplanation:
    """One tick's narrative: its time, comms state, and a human-readable ``summary``."""

    tick: int
    sim_time_s: float
    comms_denied: bool
    summary: str


@dataclass(frozen=True, slots=True)
class PlanExplanation:
    """A trace's operator-facing explanation: the stack, its provenance, and per-tick narrative."""

    stack_id: str
    seed: int | None
    plugin_versions: dict[str, str]
    input_hashes: dict[str, str]
    ticks: tuple[TickExplanation, ...]

    def to_text(self) -> str:
        """Render the explanation as plain text (one line per tick)."""
        header = f"Plan explanation for stack {self.stack_id!r} (seed={self.seed})"
        lines = [f"  t={t.sim_time_s:g}s: {t.summary}" for t in self.ticks]
        return "\n".join([header, *lines])


def explain(trace: DecisionTrace) -> PlanExplanation:
    """Derive a :class:`PlanExplanation` from ``trace``."""
    return PlanExplanation(
        stack_id=trace.stack_id,
        seed=trace.provenance.seed,
        plugin_versions=dict(trace.provenance.plugin_versions),
        input_hashes=dict(trace.provenance.input_hashes),
        ticks=tuple(_explain_tick(tick) for tick in trace.ticks),
    )


def _explain_tick(tick: TickRecord) -> TickExplanation:
    clauses: list[str] = []
    for record in tick.tiers:
        if record.replanned and record.trigger not in (None, "initial", "reactive", "bt_tick"):
            clauses.append(f"{record.role} replanned ({record.trigger})")
        elif record.replanned and record.trigger == "initial":
            clauses.append(f"{record.role} planned")
        if record.note is not None:
            phrase = _NOTE_PHRASES.get(record.note, record.note)
            clauses.append(f"{record.role} {phrase}")
        if record.fallback_used:
            clauses.append(f"{record.role} degraded to its fallback")
    if tick.shield.intervened:
        detail = f" [{', '.join(tick.shield.clauses)}]" if tick.shield.clauses else ""
        clauses.append(f"guard intervened ({tick.shield.kind or 'edit'}){detail}")
    if not clauses:
        clauses.append("acted on valid cached plans")
    prefix = "comms denied — " if tick.comms_denied else ""
    return TickExplanation(
        tick=tick.tick,
        sim_time_s=tick.sim_time_s,
        comms_denied=tick.comms_denied,
        summary=prefix + "; ".join(clauses),
    )
