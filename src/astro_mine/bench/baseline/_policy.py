"""The reference baseline policy (RM-P0-BENCH-05; bench.md §7, §12).

A :class:`BaselinePolicy` is the platform's floor for the anchor scenario: the simplest
thing that honours the [Core](core.md) Policy API, so a leaderboard has a published,
reproducible baseline to beat and the local scoring path has a real policy to run. It is a
*replaceable example* (conventions.md §1.3), not a good policy — it neither plans nor
allocates; it keeps every observed agent in a single working mode.

It implements Core's ``Policy`` structurally: a single
``decide(observations, context) -> ActionBatch`` (``astro_mine.core.policy``). Determinism
is the contract (not statelessness): the same observations + ``context`` (incl. ``seed``)
yield a byte-identical batch, so Bench's determinism gate holds over the scoring path.

**This is the floor for the fixture path, not for a physics-backed one** (astro-mine-sim#61).
A single global mode cannot serve a heterogeneous roster: the anchor's six assets declare
disjoint mode vocabularies, and the default ``"prospect"`` is declared by exactly *one* of
them — so on the other five it names a mode the asset never publishes a power draw for, and
an engine-backed run prices a tick the asset cannot be in. Mode strings are free-form (no
enum in ``sadf.schema.json``, and Fleet's lint checks only ``power_w >= 0``), so nothing
rejects the mismatch: it fails silently, in whatever consumes the mode.

The concrete instance of that in Sim: extraction is gated on the mode string, and
``"prospect"`` is not in Sim's ``DEFAULT_EXTRACTION_MODES``, so a ``BaselinePolicy``-driven
run stores no water. **That is not a defect in either package** — the two vocabularies are
coupled by convention alone, with no validator on either side. A runner that resolves real
content supplies its own baseline through the optional
:class:`~astro_mine.bench.baseline.DefaultPolicyProvider` seam instead; Sim's is a
capability-aware mode policy built from each asset's own SADF.

Backlog: RM-P0-BENCH-05 — astro-mine-bench#5
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from astro_mine.core.messages import Action, ActionBatch, ModeCommand, Observation
from astro_mine.core.messages.enums import ActionKind
from astro_mine.core.policy import AgentId, DecisionContext

__all__ = ["BaselinePolicy"]


@dataclass(frozen=True, slots=True)
class BaselinePolicy:
    """A deterministic reference baseline: hold every observed agent in one working mode.

    Emits one ``MODE`` :class:`~astro_mine.core.messages.Action` per observed agent naming
    :attr:`mode`, **the same mode for every agent** — ``"prospect"`` by default. That is the
    simplification to be aware of: the mode is *not* checked against any agent's declared
    SADF ``loads_by_mode``, so on a heterogeneous roster most agents receive a mode they do
    not declare (see the module docstring). Agents are decided in sorted id order so the
    :class:`~astro_mine.core.messages.ActionBatch` — and any hash of it — is independent of
    observation iteration order. Structurally satisfies
    :class:`astro_mine.core.policy.Policy` (a plain class with ``decide``; no base needed).
    """

    mode: str = "prospect"

    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        """Keep each observed agent in :attr:`mode` — the reference floor behaviour."""
        actions = [
            Action(agent_id=agent_id, kind=ActionKind.MODE, mode=ModeCommand(mode=self.mode))
            for agent_id in sorted(observations)
        ]
        return ActionBatch(actions=actions)
