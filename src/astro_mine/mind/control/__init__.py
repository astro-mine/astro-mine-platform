"""The reactive control tier — closed-loop controllers (mind.md §3, §4).

Pluggable per asset class behind Core's :class:`~astro_mine.core.policy.protocol.Controller`
sub-interface (RM-P1-MIND-03): classical MPC/PID baselines that always work
(:mod:`~astro_mine.mind.control.reference`) plus learned ONNX controllers where they win
(:mod:`~astro_mine.mind.control.policy`) — all behind one contract, all Guard-wrapped. The
reference controllers are stateless pure functions of the observation + upstream target, so a
composed stack reproduces across runs (principle 9); a learned controller drops in through the
Core ``OnnxPolicy`` contract with no framework change.
"""

from __future__ import annotations
