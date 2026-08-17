# SPDX-License-Identifier: Apache-2.0
"""Ground-truth/belief isolation — a leak is a security-class defect (prospect.md §9).

The sealed ground-truth field is **access-gated** and **structurally unreachable** through the
agent-facing Environment API. Two complementary layers enforce that here, mirroring how
[Core](../../../../docs/architecture/core.md) gates its own capabilities:

- **Capability gate** (:func:`require_ground_truth_access`) — reading the sealed field requires the
  Core ``GROUND_TRUTH_ACCESS`` capability. Privileged consumers (Sim's sensor model, the
  calibration harness) present it; agent/policy code holds no capability and is refused loudly. The
  Prospect-side mirror of Core's gated-tag check in ``registry``/``sadf`` loaders.
- **Reachability check** (:func:`assert_isolated`) — a bounded, cycle-safe walk of an
  agent-facing object's whole attribute graph that fails if any reachable object is a sealed
  ground-truth field. This is the contract-test core: an information leak from ground truth into a
  belief would silently invalidate every active-perception result, so it must fail CI.

A leak is an error, never a warning. :class:`IsolationError` is the security-class signal.

Backlog: RM-P0-PROSPECT-05 — astro-mine-prospect#5
"""

from __future__ import annotations

import types
from collections.abc import Iterable, Iterator, Mapping

import numpy as np

from astro_mine.core.sadf.enums import GATED_CAPABILITY_TAGS, CapabilityTag

__all__ = [
    "GROUND_TRUTH_ACCESS",
    "SEALED_MARKER",
    "IsolationError",
    "assert_agent_safe_capabilities",
    "assert_isolated",
    "require_ground_truth_access",
]

#: The Core capability tag that gates reading the sealed ground-truth field. Re-exported from
#: :mod:`astro_mine.core.sadf.enums` so Prospect speaks Core's single capability vocabulary — it is
#: one of Core's :data:`~astro_mine.core.sadf.enums.GATED_CAPABILITY_TAGS`, so an open-commons asset
#: or plugin that declares it is rejected by Core's loaders (conventions.md §12).
GROUND_TRUTH_ACCESS = CapabilityTag.GROUND_TRUTH_ACCESS

#: The class attribute by which a sealed type advertises itself to :func:`assert_isolated`. A type
#: carrying it (value: its required :class:`~astro_mine.core.sadf.enums.CapabilityTag`) is treated
#: as ground truth and must never be reachable from an agent view —
#: :class:`~astro_mine.prospect.belief.ground_truth.GroundTruthField` sets it. Keying off a marker
#: (not an import) keeps this module decoupled from the field implementations it guards and lets a
#: future sealed type opt in without editing the checker.
SEALED_MARKER = "REQUIRED_CAPABILITY"

#: Depth bound for the reachability walk — deep enough for any realistic agent view (an
#: ``Observation`` nests ~4 levels), shallow enough that a pathological graph cannot stall the gate.
_MAX_DEPTH = 8

#: Types the walk stops at — they hold no field-bearing children worth recursing into: scalars and
#: byte strings; ``numpy`` arrays/scalars (which cannot *hold* a sealed field, only be one — caught
#: by its marker); and modules/callables/classes (a reference to a type, not an instance, is safe).
_STOP_TYPES: tuple[type, ...] = (
    str,
    bytes,
    bytearray,
    bool,
    int,
    float,
    complex,
    type(None),
    np.ndarray,
    np.generic,
    types.ModuleType,
    types.FunctionType,
    types.MethodType,
    type,
)


class IsolationError(RuntimeError):
    """A ground-truth isolation breach — security-class.

    Raised when sealed ground truth is read without the gating capability, when an agent-facing
    artifact declares a gated capability, or when a sealed field is found reachable from an
    agent-facing view. Never a warning: a silent leak breaks every downstream uncertainty result.
    """


def require_ground_truth_access(capabilities: Iterable[CapabilityTag]) -> None:
    """Gate a sealed ground-truth read on the ``GROUND_TRUTH_ACCESS`` capability (AC2).

    The Prospect-side mirror of Core's gated-capability check: privileged, non-agent consumers —
    Sim's sensor model, the #7 calibration harness — present ``[GROUND_TRUTH_ACCESS]``; agent and
    policy code holds no such capability. Raises :class:`IsolationError` unless the tag is present.
    """
    tags = frozenset(capabilities)
    if GROUND_TRUTH_ACCESS not in tags:
        held = ", ".join(sorted(t.value for t in tags)) or "none"
        raise IsolationError(
            f"reading the sealed ground-truth field requires the {GROUND_TRUTH_ACCESS.value!r} "
            f"capability; caller presented: {held} (prospect.md §9 ground-truth isolation)"
        )


def assert_agent_safe_capabilities(capabilities: Iterable[CapabilityTag]) -> None:
    """Assert an agent-facing artifact declares no reserved/gated capability.

    Mirrors Core's loader rejection (``registry/loader.py``, ``sadf/loader.py``): an artifact handed
    to the swarm MUST NOT carry ``GROUND_TRUTH_ACCESS`` (or any other gated tag). Raises
    :class:`IsolationError` naming the offending tags.
    """
    gated = sorted(t.value for t in capabilities if t in GATED_CAPABILITY_TAGS)
    if gated:
        raise IsolationError(
            f"agent-facing artifact declares reserved/gated capability tag(s): {', '.join(gated)} "
            "— these are privileged and must not reach the swarm (conventions.md §12)"
        )


def assert_isolated(agent_view: object, *, max_depth: int = _MAX_DEPTH) -> None:
    """Assert no sealed ground-truth field is reachable from an agent-facing view (AC1).

    A bounded, cycle-safe walk of ``agent_view``'s attribute graph (instance ``__dict__`` and
    ``__slots__``, plus container contents — *including* private attributes, so a stashed handle is
    caught) raises :class:`IsolationError` if any reachable object is a sealed ground-truth field,
    i.e. carries the :data:`SEALED_MARKER` class attribute. Reusable for anything handed to the
    swarm — a :class:`~astro_mine.prospect.belief.field.BeliefField`, a Core
    :class:`~astro_mine.core.messages.model.Observation`, an info-gain map (#6) — so an isolation
    regression fails loudly in CI.

    Scope: it detects sealed *field objects*; a bare leaked array is indistinguishable from any
    other array and is instead protected by the read gate (:func:`require_ground_truth_access`).
    """
    seen: set[int] = set()

    def visit(obj: object, path: str, depth: int) -> None:
        if getattr(type(obj), SEALED_MARKER, None) is not None:
            raise IsolationError(
                f"sealed ground truth is reachable from the agent view at {path}: "
                f"{type(obj).__module__}.{type(obj).__qualname__} — a leak is a security-class "
                "defect (prospect.md §9)"
            )
        if depth >= max_depth or isinstance(obj, _STOP_TYPES):
            return
        marker = id(obj)
        if marker in seen:
            return
        seen.add(marker)
        for child, child_path in _children(obj, path):
            visit(child, child_path, depth + 1)

    visit(agent_view, "<root>", 0)


def _children(obj: object, path: str) -> Iterator[tuple[object, str]]:
    """Yield ``(child, path)`` for the field-bearing references held directly by ``obj``."""
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            yield key, f"{path}[<key>]"
            yield value, f"{path}[{key!r}]"
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for index, value in enumerate(obj):
            yield value, f"{path}[{index}]"
        return
    instance_dict = getattr(obj, "__dict__", None)
    if instance_dict:
        for name, value in instance_dict.items():
            yield value, f"{path}.{name}"
    for name in getattr(type(obj), "__slots__", ()):
        try:
            yield getattr(obj, name), f"{path}.{name}"
        except AttributeError:
            continue
