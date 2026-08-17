# SPDX-License-Identifier: Apache-2.0
"""``SurrogateManifest`` — the Core plugin manifest a surrogate publishes itself as.

Surrogate **consumes** Core's manifest schema, it does not invent one (surrogate.md §3;
core.md §9; the Prospect precedent ``build_field_manifest``). There is no
``SurrogateManifest`` subclass: Core's :class:`~astro_mine.core.registry.PluginManifest`
is ``extra="forbid"`` with a closed JSON Schema, so surrogate-specific facets ride in its
open ``attributes`` map (:class:`SurrogateAttributes`) while identity, the Core interface
it implements, provenance, and signature use the manifest's own typed fields. The manifest
is what a signed, verified registry load gates on before Sim would ever instantiate the
tier (surrogate.md §9).

The kind and Core interface are chosen **by domain**, which is what lets one contract
serve two consumers: a dynamical-step surrogate publishes as a ``regime_engine`` behind
Sim's ``env`` interface (RM-P1-SURR-02); a field surrogate publishes as a ``field_model``
behind Worlds' ``world_provider`` interface (RM-P1-WORLDS-10). Both reuse **existing**
Core ``PluginKind`` members — no new kind and no Core change (surrogate.md §3: domains are
additive).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.core.hashing import content_hash_json
from astro_mine.core.registry import PluginKind, PluginManifest, Provenance
from astro_mine.surrogate import __version__ as _SURROGATE_VERSION
from astro_mine.surrogate.enums import PhysicsDomain, ServedBackend
from astro_mine.surrogate.report import ErrorReport, TrustRegion

__all__ = [
    "ENV_INTERFACE",
    "ERROR_REPORT_MEDIA_TYPE_JSON",
    "ERROR_REPORT_MEDIA_TYPE_PROTO",
    "INTERFACE_VERSION",
    "WORLD_PROVIDER_INTERFACE",
    "SurrogateAttributes",
    "build_surrogate_manifest",
]

#: Core interface a dynamical-step surrogate tier implements (Sim's Environment step).
ENV_INTERFACE = "env"
#: Core interface a field surrogate implements (Worlds' WorldProvider field query).
WORLD_PROVIDER_INTERFACE = "world_provider"
#: The Core interface version held through Phase 1 (frozen at 0.1.0; VERSIONING.md §4).
INTERFACE_VERSION = "0.1.0"

#: Media types for the ``ErrorReport`` in its two forms — the canonical Protobuf wire
#: form Sim's scheduler consumes cross-language, and the JSON projection.
ERROR_REPORT_MEDIA_TYPE_PROTO = "application/vnd.astro-mine.surrogate.error-report.v1+protobuf"
ERROR_REPORT_MEDIA_TYPE_JSON = "application/vnd.astro-mine.surrogate.error-report.v1+json"

# Which Core PluginKind / interface a domain publishes as. Dynamical-step domains are
# Sim fidelity-tier engines; field domains are Worlds field models. Reuses existing
# closed-vocabulary members — adding a `surrogate_model` kind would need a Core RFC and
# buys nothing here (surrogate.md §3, additive domains).
_FIELD_DOMAINS = frozenset({PhysicsDomain.ILLUMINATION_FIELD, PhysicsDomain.THERMAL_FIELD})


def _kind_for_domain(domain: PhysicsDomain) -> PluginKind:
    return PluginKind.FIELD_MODEL if domain in _FIELD_DOMAINS else PluginKind.REGIME_ENGINE


def _interface_for_domain(domain: PhysicsDomain) -> str:
    return WORLD_PROVIDER_INTERFACE if domain in _FIELD_DOMAINS else ENV_INTERFACE


class SurrogateAttributes(BaseModel):
    """The surrogate-specific facets carried in ``PluginManifest.attributes``.

    Typed here (frozen, ``extra="forbid"``) and folded into the manifest's open
    ``attributes`` object via :meth:`model_dump`, so the facets are schema-checked and
    JSON-Schema-exportable without widening Core's manifest. ``error_report_digest``
    references the :class:`ErrorReport` by content hash — the signed manifest thereby
    commits to the exact bound (surrogate.md §9).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    domain: PhysicsDomain
    input_channels: list[str] = Field(min_length=1)
    output_channels: list[str] = Field(min_length=1)
    trust_region: TrustRegion
    #: The surrogate's recommended per-output-channel admission budget (from the
    #: :class:`~astro_mine.surrogate.report.SubstitutionPolicy`), projected into the manifest so a
    #: consumer's scheduler — Sim's multi-fidelity scheduler (RM-P1-SIM-03) *or* Worlds'
    #: illumination-model selection (RM-P1-WORLDS-10) — can decide admission from the **Core
    #: manifest alone**, without fetching and parsing the full ``ErrorReport`` (surrogate.md §6;
    #: the narrow-waist "consume across Core contracts" path). The full report stays available by
    #: ``error_report_digest`` for the detailed per-channel distribution/coverage.
    recommended_error_budget: dict[str, float] = Field(min_length=1)
    #: The rollout horizon (steps) the ``recommended_error_budget`` holds at — projected into the
    #: manifest alongside the budget so a scheduler can decide admission *and* the cadence at which
    #: it must re-validate from the Core manifest alone (surrogate#23). A consumer that grades the
    #: tier over a longer rollout than this is checking a bound the producer never declared.
    budget_horizon_steps: int = Field(default=1, ge=1)
    served_backend: ServedBackend
    native_graph_fallback: bool
    error_report_digest: str
    error_report_media_type: str = ERROR_REPORT_MEDIA_TYPE_PROTO


def build_surrogate_manifest(
    *,
    name: str,
    version: str,
    report: ErrorReport,
    artifact_digest: str,
    served_backend: ServedBackend = ServedBackend.ONNX,
    native_graph_fallback: bool = False,
    code_version: str | None = None,
    toolchain_version: str | None = None,
    seed: int | None = None,
    env_lockfile: str | None = None,
    hyperparameters: Mapping[str, Any] | None = None,
    train_dataset_hash: str | None = None,
    sampling_policy_hash: str | None = None,
) -> PluginManifest:
    """Build the Core ``PluginManifest`` for a surrogate (unsigned; caller attaches).

    Identity and the served bound come from *report* and *artifact_digest*: the manifest
    declares the Core interface its ``domain`` implements, carries the surrogate's own
    ``sha256:<hex>`` content address as ``provenance.digest`` (the tier identity Sim pins
    and a signature binds to), records the validation-dataset hash in ``input_hashes``,
    and folds the :class:`SurrogateAttributes` — including the ``ErrorReport`` digest — into
    ``attributes``. It declares **no capability tags**: a surrogate is open-commons, and no
    gated tag applies. The caller attaches a signature (``sign_digest`` over
    ``provenance.digest``) before a signature-requiring registry will load it.

    The optional ``seed`` / ``env_lockfile`` / ``hyperparameters`` / ``train_dataset_hash`` /
    ``sampling_policy_hash`` record the **full retrain provenance** (RM-P1-SURR-03; surrogate.md
    §5) in the Core :class:`~astro_mine.core.registry.Provenance` — ``input_hashes`` becomes
    ``[train_dataset_hash, validation_dataset_hash]`` (train prepended; it falls back to just the
    report's ``validation_dataset_hash`` when unset), and the hyperparameter/sampling-policy
    content hashes ride in ``source_content_hashes``. All are keyword-optional, so existing
    RM-P1-SURR-04 callers (``publish_served_surrogate``) are unchanged.
    """
    domain = report.domain
    attrs = SurrogateAttributes(
        domain=domain,
        input_channels=list(report.trust_region.bounds),
        output_channels=[c.channel for c in report.channels],
        trust_region=report.trust_region,
        recommended_error_budget=dict(report.substitution_policy.recommended_error_budget),
        budget_horizon_steps=report.substitution_policy.budget_horizon_steps,
        served_backend=served_backend,
        native_graph_fallback=native_graph_fallback,
        error_report_digest=report.content_hash(),
    )
    input_hashes = (
        [train_dataset_hash, report.validation_dataset_hash]
        if train_dataset_hash is not None
        else [report.validation_dataset_hash]
    )
    source_content_hashes: dict[str, str] = {}
    if hyperparameters is not None:
        source_content_hashes["hyperparameters"] = content_hash_json(dict(hyperparameters))
    if sampling_policy_hash is not None:
        source_content_hashes["sampling_policy"] = sampling_policy_hash
    return PluginManifest(
        name=name,
        version=version,
        kind=_kind_for_domain(domain),
        core_interfaces={_interface_for_domain(domain): INTERFACE_VERSION},
        license="Apache-2.0",
        description=(
            f"Learned {domain} surrogate {name!r} — a calibrated fidelity tier carrying a "
            "bounded ErrorReport (surrogate.md §3)."
        ),
        provenance=Provenance(
            digest=artifact_digest,
            code_version=code_version or version,
            toolchain_version=toolchain_version or f"astro-mine-surrogate {_SURROGATE_VERSION}",
            input_hashes=input_hashes,
            env_lockfile=env_lockfile,
            seed=seed,
            source_content_hashes=source_content_hashes,
        ),
        attributes=attrs.model_dump(mode="json"),
    )
