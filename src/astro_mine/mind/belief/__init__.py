# SPDX-License-Identifier: Apache-2.0
"""Belief-state assembly — the partial-observability-aware input each tier consumes.

The executive refreshes a :class:`~astro_mine.mind.belief.view.BeliefView` from the
Environment API observation each tick and threads it to the tiers through Core's
sanctioned open channel, ``DecisionContext.extras[BELIEF_EXTRAS_KEY]`` (belief/info-gain
handles ride in ``extras``, never new Core schema; policy/model.py). A tier backend reads
it there — e.g. the PDDL mission backend (RM-P1-MIND-03) generates its problem file from
the belief view, and the comms-regime strategy switch (RM-P1-MIND-06) reads its comms
masks. v0.1 is a thin state+comms view; explicit uncertainty tagging grows here later
(mind.md §3, principle 6).
"""

from __future__ import annotations

from astro_mine.mind.belief.view import BELIEF_EXTRAS_KEY, BeliefView, assemble_belief

__all__ = ["BELIEF_EXTRAS_KEY", "BeliefView", "assemble_belief"]
