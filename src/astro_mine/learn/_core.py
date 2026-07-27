"""Learn's binding to the Core narrow waist.

The single place that declares which Core interface versions Learn builds against, plus
a thin compatibility check over them. Learn *consumes* the waist — it presents the world
through the Core Environment API and emits policies against the Policy API, but never
re-defines a Core contract (CONTRIBUTING.md: "consume the waist, never widen it"). This
module only records the contract Learn builds against so the env wrappers, the policy
export path, and the contract test all cite one source of truth.
"""

from __future__ import annotations

from astro_mine.core import compat

__all__ = ["CORE_INTERFACES", "assert_core_compatible"]

#: Core interface versions Learn is built against — the input to Core's version
#: negotiation. Learn wraps the world through the Environment API (``env``), and the
#: SwarmEnv adapter (RM-P1-LEARN-01) genuinely consumes the message catalog (``messages``:
#: Observation/Action encode/decode) and SADF (``sadf``: capability-keyed spaces) to do
#: it; trained policies export against the Policy API (``policy``) — learn.md §2/§3.
#: Append as Learn grows to consume more of the waist.
CORE_INTERFACES: dict[str, str] = {
    "env": "0.1.0",
    "messages": "0.1.0",
    "sadf": "0.1.0",
    "policy": "0.1.0",
}


def assert_core_compatible() -> None:
    """Assert the installed Core satisfies the interface versions Learn is built against.

    Delegates to :func:`astro_mine.core.compat.assert_core_compatible`. This is the
    consumer-driven contract test Learn runs in its own CI (RM-P0-CORE-07); raises
    :class:`~astro_mine.core.compat.IncompatibleCoreInterface` on any mismatch.
    """
    compat.assert_core_compatible(CORE_INTERFACES)
