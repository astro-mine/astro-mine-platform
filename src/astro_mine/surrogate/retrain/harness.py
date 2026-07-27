"""The offline-retrain + gated-promotion harness (RM-P1-SURR-03; surrogate.md §5, §10, §11).

The offline **build** loop's terminal step: retrain on a (possibly resampled) dataset, then admit
the model into Sim **only** through the automated validation gate. Concretely
:func:`retrain_surrogate`:

1. trains a new surrogate on the dataset via the injected-dataset seam
   (:func:`~astro_mine.surrogate.models.excavation.build_excavation_surrogate`);
2. assigns a **new SemVer** bumped from ``prior_version`` (minor on new data, patch on a
   hyperparameter-only re-fit) — the prior artifact is untouched and stays reproducible
   (surrogate.md §5: "a retrain produces a new version ... the prior remains reproducible");
3. evaluates the coverage/error-budget **promotion gate**
   (:func:`~astro_mine.surrogate.eval.gate.evaluate_promotion`); a gate failure returns
   ``promoted=False`` with **no** bundle — continual data collection is allowed, but weights never
   silently enter Sim (surrogate.md §11, principle 7);
4. on a pass, exports the served :class:`~astro_mine.surrogate.serve.bundle.OnnxBundle` and records
   **full provenance** (train/validation dataset hashes, seed, lockfile, hyperparameters,
   sampling-policy hash) in the Core :class:`~astro_mine.core.registry.Provenance` — sufficient to
   reproduce both the model and its ``ErrorReport`` exactly.

Imports **only** Core + the Surrogate library (train/eval/store) — never ``astro_mine.sim``; the
high-fidelity Sim data reached this harness as a content-addressed dataset through the ``datagen``
:class:`~astro_mine.surrogate.datagen.oracle.RolloutOracle` seam, not an import. The ONNX export
(``[serve]``) is imported lazily, only on a gate pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from astro_mine.core.hashing import content_hash_json
from astro_mine.core.registry import Provenance
from astro_mine.surrogate import __version__ as _SURROGATE_VERSION
from astro_mine.surrogate.datagen.store import DatasetRef, read_dataset, split_dataset
from astro_mine.surrogate.eval.gate import GateResult, PromotionCriteria, evaluate_promotion
from astro_mine.surrogate.manifest import build_surrogate_manifest
from astro_mine.surrogate.models.dataset import DemDataset
from astro_mine.surrogate.models.excavation import build_excavation_surrogate
from astro_mine.surrogate.models.train import TrainConfig

if TYPE_CHECKING:
    from astro_mine.surrogate.datagen.policy import SamplingPolicy
    from astro_mine.surrogate.serve.bundle import OnnxBundle

__all__ = ["BumpKind", "RetrainResult", "retrain_surrogate"]


class BumpKind(StrEnum):
    """How a retrain bumps the prior SemVer (surrogate.md §5; §11 offline-retrain).

    ``MINOR`` — the default — marks a retrain on **new data** (a new dataset / resample);
    ``PATCH`` marks a **hyperparameter-only re-fit** on the same data. Either way the prior version
    is never overwritten.
    """

    MINOR = "minor"
    PATCH = "patch"


def _bump_version(prior: str, kind: BumpKind) -> str:
    """Bump a ``MAJOR.MINOR.PATCH`` string per ``kind`` (minor → ``.N+1.0``; patch → ``..N+1``)."""
    try:
        major, minor, patch = (int(part) for part in prior.split("."))
    except ValueError as exc:
        raise ValueError(f"prior_version {prior!r} is not a MAJOR.MINOR.PATCH SemVer") from exc
    if kind is BumpKind.MINOR:
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


@dataclass(frozen=True)
class RetrainResult:
    """The outcome of a retrain: the new version, the gate result, and (if promoted) the bundle.

    ``promoted`` is ``True`` only when ``gate.passed``; then ``bundle`` is the served
    :class:`OnnxBundle` and ``provenance`` carries the full reproduction lineage. On a gate failure
    ``promoted`` is ``False`` and ``bundle`` is ``None`` (the model is not admitted into Sim), while
    ``provenance`` still records what was trained — an auditable record of the rejected attempt.
    """

    new_version: str
    gate: GateResult
    promoted: bool
    provenance: Provenance
    bundle: OnnxBundle | None = None


def _retrain_provenance(
    *,
    report_validation_hash: str,
    train_dataset_hash: str,
    seed: int,
    env_lockfile_hash: str,
    hyperparameters: dict[str, object],
    sampling_policy_hash: str | None,
    code_version: str,
    artifact_digest: str | None,
) -> Provenance:
    """The full retrain :class:`Provenance` — mirrors ``build_surrogate_manifest`` on the fail path.

    ``artifact_digest`` is ``None`` when the gate failed (no promoted artifact), otherwise the
    served bundle's content hash.
    """
    source_content_hashes: dict[str, str] = {"hyperparameters": content_hash_json(hyperparameters)}
    if sampling_policy_hash is not None:
        source_content_hashes["sampling_policy"] = sampling_policy_hash
    return Provenance(
        digest=artifact_digest,
        code_version=code_version,
        toolchain_version=f"astro-mine-surrogate {_SURROGATE_VERSION}",
        input_hashes=[train_dataset_hash, report_validation_hash],
        env_lockfile=env_lockfile_hash,
        seed=seed,
        source_content_hashes=source_content_hashes,
    )


def retrain_surrogate(
    *,
    dataset: DemDataset | DatasetRef,
    hyperparameters: TrainConfig,
    seed: int,
    prior_version: str,
    criteria: PromotionCriteria,
    code_version: str,
    env_lockfile_hash: str,
    sampling_policy: SamplingPolicy | None = None,
    bump: BumpKind = BumpKind.MINOR,
    name: str = "excavation-gns",
) -> RetrainResult:
    """Retrain, gate, and — only on a pass — promote a surrogate to a new SemVer version.

    ``dataset`` is either an in-memory :class:`DemDataset` or an immutable
    :class:`~astro_mine.surrogate.datagen.store.DatasetRef` (read + hash-verified). The new version
    is ``prior_version`` bumped by ``bump`` (minor on new data, patch on a hyperparameter-only
    re-fit). Promotion is gated by ``criteria``; a failure returns ``promoted=False`` with no
    bundle. On a pass the served bundle is exported and the returned ``provenance`` records the
    full reproduction lineage (train/validation hashes, seed, lockfile, hyperparameters, policy).
    """
    new_version = _bump_version(prior_version, bump)

    if isinstance(dataset, DatasetRef):
        resolved = read_dataset(dataset)
        train_hash = dataset.train_split_hash
    else:
        resolved = dataset
        train_split, _validation_split = split_dataset(dataset)
        train_hash = train_split.content_hash()

    surrogate = build_excavation_surrogate(
        dataset=resolved, config=hyperparameters, seed=seed, name=name, version=new_version
    )
    report = surrogate.error_report
    gate = evaluate_promotion(report, criteria)

    hyperparameter_map: dict[str, object] = asdict(hyperparameters)
    sampling_policy_hash = sampling_policy.content_hash() if sampling_policy is not None else None

    if not gate.passed:
        # Gate-fail: the retrained weights never enter Sim (surrogate.md §11, principle 7). Record
        # the attempt's provenance for audit; no bundle, no manifest.
        provenance = _retrain_provenance(
            report_validation_hash=report.validation_dataset_hash,
            train_dataset_hash=train_hash,
            seed=seed,
            env_lockfile_hash=env_lockfile_hash,
            hyperparameters=hyperparameter_map,
            sampling_policy_hash=sampling_policy_hash,
            code_version=code_version,
            artifact_digest=None,
        )
        return RetrainResult(
            new_version=new_version, gate=gate, promoted=False, provenance=provenance, bundle=None
        )

    # Gate-pass: export the served tier and build the manifest with full provenance.
    from astro_mine.surrogate.serve.export import export_excavation_surrogate

    bundle = export_excavation_surrogate(surrogate)
    manifest = build_surrogate_manifest(
        name=name,
        version=new_version,
        report=report,
        artifact_digest=bundle.content_hash(),
        code_version=code_version,
        seed=seed,
        env_lockfile=env_lockfile_hash,
        hyperparameters=hyperparameter_map,
        train_dataset_hash=train_hash,
        sampling_policy_hash=sampling_policy_hash,
    )
    assert manifest.provenance is not None  # build_surrogate_manifest always sets it
    return RetrainResult(
        new_version=new_version,
        gate=gate,
        promoted=True,
        provenance=manifest.provenance,
        bundle=bundle,
    )
