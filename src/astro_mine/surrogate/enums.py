"""Closed vocabularies for the surrogate contracts (RM-P1-SURR-01).

Small, append-only StrEnums — the platform idiom for a closed vocabulary
(:class:`astro_mine.core.registry.PluginKind`, the SADF enums). They grow only by
adding a member; members are never removed or repurposed, so the ``string`` wire form
(``surrogate.proto``) and every persisted :class:`~astro_mine.surrogate.report.ErrorReport`
stay valid across versions.

The domains are deliberately generic. The contract's first instance is the
granular/excavation step surrogate (RM-P1-SURR-02), but the *same* three types must
carry a learned **illumination field** surrogate (RM-P1-WORLDS-10) and, later, a
microgravity contact surrogate (RFC-0001) — so ``PhysicsDomain`` spans both the
dynamical-step and field-query families, and :class:`ChannelKind` lets a single
``ErrorReport`` mix a continuous channel (excavation next-state, solar flux) with a
categorical one (illumination ``lit|penumbra|shadow``).
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["ChannelKind", "PhysicsDomain", "ServedBackend"]


class PhysicsDomain(StrEnum):
    """The physics a surrogate approximates — the ``domain`` a manifest declares.

    Two families share one contract (surrogate.md §1, §3; the domains are additive —
    "declare the state/action space and a ``SamplingPolicy``; the rest is reused"):

    - **dynamical-step** surrogates behind Sim's physics-step tier (``regime_engine``):
      granular/excavation contact (the MVP, ``LUNAR-TR-002``), wheel/soil
      terramechanics, manipulation contact, and microgravity contact (RFC-0001, P3);
    - **field-query** surrogates behind a Worlds field model (``field_model``): a
      learned illumination field (RM-P1-WORLDS-10) and, later, a thermal field.
    """

    GRANULAR_EXCAVATION = "granular_excavation"
    WHEEL_SOIL = "wheel_soil"
    MANIPULATION_CONTACT = "manipulation_contact"
    MICROGRAVITY_CONTACT = "microgravity_contact"
    ILLUMINATION_FIELD = "illumination_field"
    THERMAL_FIELD = "thermal_field"


class ChannelKind(StrEnum):
    """Whether an output channel is continuous or categorical.

    A single :class:`~astro_mine.surrogate.report.ErrorReport` mixes both: an
    excavation surrogate's next-state channels are all ``continuous`` (RMSE + interval
    coverage), while an illumination surrogate pairs a ``continuous`` solar-flux channel
    with a ``categorical`` visibility channel (``lit|penumbra|shadow``: class accuracy +
    confidence reliability). The error model is therefore per-channel-*typed*, not a
    uniform RMSE vector.
    """

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


class ServedBackend(StrEnum):
    """How the trained surrogate is served behind the Core contract (surrogate.md §4, §11).

    ``onnx`` is the served form (content-addressed, cross-runtime) and the only one
    implemented: :mod:`astro_mine.surrogate.serve` exports a self-contained ONNX graph and
    serves it through ONNX Runtime (RM-P1-SURR-04).

    ``native_graph`` is the *declarable but unimplemented* fallback for the in-sim path when
    ONNX cannot express an op. The manifest records the choice — so a loader knows what it is
    admitting, and :func:`~astro_mine.surrogate.manifest.build_surrogate_manifest` will
    faithfully carry it — but **no native-graph serving runtime exists**: the loaders
    (``load_served_surrogate``/``resolve_and_load``) handle only the ONNX bundle layer. The
    excavation graph exports cleanly, so an export failure raises ``OnnxExportError`` rather
    than falling back. Implementing the fallback is deferred past Phase 1.
    """

    ONNX = "onnx"
    NATIVE_GRAPH = "native_graph"
