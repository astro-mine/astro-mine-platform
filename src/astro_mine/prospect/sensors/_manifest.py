"""The Core ``observation_model`` plugin manifest for a sensor likelihood — the registry seam.

prospect.md §3 requires a sensor likelihood to be **selectable through a Core plugin manifest**, the
way a field backend is (:mod:`astro_mine.prospect.publish._manifest`). Prospect consumes Core's
manifest schema rather than inventing one: a likelihood maps onto a
:class:`~astro_mine.core.registry.PluginManifest` of kind
:attr:`~astro_mine.core.registry.PluginKind.OBSERVATION_MODEL`, whose ``attributes`` carry the
instrument's Core ``SensorKind`` / ``CapabilityTag`` and its full frozen model as canonical JSON. So
:func:`likelihood_from_manifest` reconstructs **the exact instrument** a run used, and a Bench
scenario pins it by ``provenance.digest`` — a content address over the likelihood itself.

An observation model is **open-commons**: it declares no gated capability tag (a likelihood only
ever reads a values array *handed to it* by privileged code — it is never a path to the sealed
field), and :func:`build_likelihood_manifest` asserts exactly that through Prospect's own
:func:`~astro_mine.prospect.isolation.assert_agent_safe_capabilities` gate before emitting.

Backlog: prospect.md §3 — https://github.com/astro-mine/astro-mine-prospect/issues/31
"""

from __future__ import annotations

import hashlib
import json

from astro_mine.core.registry import PluginKind, PluginManifest, Provenance
from astro_mine.prospect import __version__ as _PROSPECT_VERSION
from astro_mine.prospect.isolation import assert_agent_safe_capabilities
from astro_mine.prospect.sensors._likelihood import SensorLikelihood

__all__ = [
    "LIKELIHOOD_ATTRIBUTE",
    "OBSERVATION_MODEL_ENTRY_POINT",
    "OBSERVATION_MODEL_INTERFACE",
    "OBSERVATION_MODEL_INTERFACE_VERSION",
    "build_likelihood_manifest",
    "likelihood_content_hash",
    "likelihood_from_manifest",
]

#: The Core interface an observation-model plugin implements, and the version it is built against.
OBSERVATION_MODEL_INTERFACE = "observation_model"
OBSERVATION_MODEL_INTERFACE_VERSION = "0.1.0"

#: The ``astro_mine.providers`` entry-point name :func:`likelihood_from_manifest` registers under,
#: so a consumer (Sim's sensor simulation) resolves a likelihood by manifest without importing
#: Prospect by name — the narrow-waist decoupling (conventions.md §1.1).
OBSERVATION_MODEL_ENTRY_POINT = "observation_model"

#: The manifest attribute the frozen likelihood model is serialized into (canonical JSON).
LIKELIHOOD_ATTRIBUTE = "likelihood"


def _canonical(likelihood: SensorLikelihood) -> str:
    """The likelihood's canonical JSON — sorted keys, no whitespace, so the bytes are stable."""
    return json.dumps(likelihood.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def likelihood_content_hash(likelihood: SensorLikelihood) -> str:
    """A stable ``sha256`` content address over the likelihood's frozen model (canonical JSON)."""
    return hashlib.sha256(_canonical(likelihood).encode("utf-8")).hexdigest()


def build_likelihood_manifest(
    likelihood: SensorLikelihood, *, version: str = "0.1.0"
) -> PluginManifest:
    """Build the ``kind=observation_model`` manifest for ``likelihood`` (unsigned; caller signs).

    The manifest is self-contained: ``attributes[LIKELIHOOD_ATTRIBUTE]`` carries the whole frozen
    model, so :func:`likelihood_from_manifest` rebuilds it with no access to this package's catalog.
    Raises :class:`~astro_mine.prospect.isolation.IsolationError` if the instrument declares a
    reserved/gated capability tag — an observation model is agent-facing and must never carry one.
    """
    assert_agent_safe_capabilities([likelihood.capability])
    return PluginManifest(
        name=likelihood.name,
        version=version,
        kind=PluginKind.OBSERVATION_MODEL,
        core_interfaces={OBSERVATION_MODEL_INTERFACE: OBSERVATION_MODEL_INTERFACE_VERSION},
        license="Apache-2.0",
        description=likelihood.description or f"Sensor likelihood {likelihood.name!r}.",
        capability_tags=[likelihood.capability],
        provenance=Provenance(
            digest=f"sha256:{likelihood_content_hash(likelihood)}",
            code_version=version,
            toolchain_version=f"astro-mine-prospect {_PROSPECT_VERSION}",
        ),
        attributes={
            "sensor_kind": likelihood.kind.value,
            "capability": likelihood.capability.value,
            "footprint_sigma_m": repr(likelihood.footprint_sigma_m),
            "depth_gain": repr(likelihood.gain),
            "provider_entry_point": OBSERVATION_MODEL_ENTRY_POINT,
            LIKELIHOOD_ATTRIBUTE: _canonical(likelihood),
        },
    )


def likelihood_from_manifest(manifest: PluginManifest) -> SensorLikelihood:
    """Rebuild a :class:`SensorLikelihood` from its Core manifest — the entry-point factory.

    The inverse of :func:`build_likelihood_manifest`, and the ``astro_mine.providers`` ->
    ``observation_model`` entry point: Sim resolves the instrument model a scenario pins and
    synthesizes its returns with **the same object** Prospect conditions its belief with
    (prospect.md §3, §6). Raises :class:`ValueError` if the manifest is not an observation model, or
    carries no serialized likelihood.
    """
    if manifest.kind is not PluginKind.OBSERVATION_MODEL:
        raise ValueError(
            f"manifest {manifest.name!r} is kind {manifest.kind.value!r}, not "
            f"{PluginKind.OBSERVATION_MODEL.value!r} — it is not a sensor likelihood"
        )
    payload = manifest.attributes.get(LIKELIHOOD_ATTRIBUTE)
    if not payload:
        raise ValueError(
            f"observation-model manifest {manifest.name!r} carries no "
            f"{LIKELIHOOD_ATTRIBUTE!r} attribute; it cannot be rebuilt"
        )
    return SensorLikelihood.model_validate_json(payload)
