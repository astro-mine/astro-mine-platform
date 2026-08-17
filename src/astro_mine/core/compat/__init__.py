# SPDX-License-Identifier: Apache-2.0
"""Interface version negotiation and contract-test utilities (RM-P0-CORE-07).

Core publishes a small set of independently-versioned interfaces — SADF, the message
catalog, the objective contract, and the Environment / Policy / registry APIs
(core.md §3, §11; conventions.md §13). Every consumer or plugin declares which Core
interface versions it is built against; at load or test time it negotiates against
what *this* Core provides, refusing incompatible combinations with a clear error.

This module is the shared primitive:

- :data:`CORE_INTERFACE_VERSIONS` — the interface versions this Core publishes.
- :func:`check_compatible` — the per-interface SemVer compatibility rule.
- :func:`assert_core_compatible` — a **contract-test helper** a downstream component
  (Fleet, Sim, …) calls in its own CI to prove it honors the Core interface versions
  it claims. The registry's load-time gate (RM-P0-CORE-05) builds on this same rule.

Per-interface versions are decoupled from the package version (conventions.md §13).
The Phase-0 interfaces are pre-1.0 (``0.y``); by SemVer a ``0.y`` line may break on
any minor bump, so ``0.y`` compatibility requires an *exact* minor, while ``>=1.0``
follows the usual same-major / provided-minor-at-least-required rule.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = [
    "CORE_INTERFACE_VERSIONS",
    "IncompatibleCoreInterface",
    "assert_core_compatible",
    "check_compatible",
    "parse_version",
]

#: The Core interface versions this build publishes (per-interface SemVer, decoupled
#: from the ``astro-mine-core`` package version). Append-only; bump on interface change.
#:
#: Deliberately *excluded*: ``units`` and ``compat`` themselves. ``units`` (frames/time/SI
#: primitives) and ``compat`` (this negotiation machinery) are shared *primitives/tooling*,
#: not independently version-negotiated wire contracts a plugin builds against and the
#: registry gates on — they move with the package, so giving them an entry here would imply
#: a negotiation that does not (and should not) happen. Only the swappable-edge interfaces
#: a consumer declares a built-against version for are listed.
CORE_INTERFACE_VERSIONS: dict[str, str] = {
    "sadf": "0.1.0",
    "messages": "0.1.0",
    "objective": "0.1.0",
    "env": "0.1.0",
    "policy": "0.1.0",
    "registry": "0.1.0",
    "resource_field": "0.1.0",
    "world_provider": "0.1.0",
    # RFC-0001 reserved mission-architecture schema (RM-P1-CORE-04) — additive, no
    # mechanism; consumers of the MissionSpec schema declare a built-against version.
    "mission": "0.1.0",
}


class IncompatibleCoreInterface(Exception):
    """Raised when a declared Core interface version is not satisfied by this Core."""


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse a ``MAJOR.MINOR.PATCH`` string into a tuple, failing loudly otherwise."""
    parts = version.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"not a MAJOR.MINOR.PATCH semantic version: {version!r}")
    major, minor, patch = (int(p) for p in parts)
    return major, minor, patch


def check_compatible(required: str, provided: str) -> bool:
    """Return whether a ``provided`` interface version satisfies a ``required`` one.

    Same major is mandatory. For ``0.y`` (pre-1.0) the minor must match exactly, since
    a ``0.y`` minor bump is allowed to break. For ``>=1.0`` the provided minor must be
    at least the required minor (additive evolution). Patch is ignored.
    """
    r_major, r_minor, _ = parse_version(required)
    p_major, p_minor, _ = parse_version(provided)
    if r_major != p_major:
        return False
    if r_major == 0:
        return p_minor == r_minor
    return p_minor >= r_minor


def assert_core_compatible(
    claimed: Mapping[str, str],
    *,
    provided: Mapping[str, str] | None = None,
) -> None:
    """Assert every claimed Core interface version is satisfied by this Core.

    ``claimed`` maps interface name → the version the consumer is built against (e.g.
    ``{"sadf": "0.1.0", "env": "0.1.0"}``). Raises :class:`IncompatibleCoreInterface`
    listing *all* problems (unknown interface or version mismatch); returns ``None`` on
    success. This is the consumer-driven contract test a downstream component runs in
    its own CI (RM-P0-CORE-07 acceptance).
    """
    available = dict(provided or CORE_INTERFACE_VERSIONS)
    problems: list[str] = []
    for interface, required in claimed.items():
        if interface not in available:
            problems.append(f"unknown Core interface {interface!r} (known: {sorted(available)})")
        elif not check_compatible(required, available[interface]):
            problems.append(
                f"{interface}: consumer requires {required}, Core provides {available[interface]}"
            )
    if problems:
        raise IncompatibleCoreInterface(
            "incompatible Core interface version(s): " + "; ".join(problems)
        )
