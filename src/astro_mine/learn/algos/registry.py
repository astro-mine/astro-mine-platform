"""Learn-internal algorithm registry + Core ``POLICY`` manifest emission (RM-P1-LEARN-03).

Two registries, kept strictly separate to honour the narrow waist (learn.md §2):

1. **Learn-internal** :class:`AlgorithmRegistry` — the *trainer* side. Algorithms register
   here (built-ins + Python entry points, group ``astro_mine.learn.algorithms``) and are
   discovered by :attr:`~astro_mine.learn.algos._contract.AlgorithmSpec.capability_tag`.
   This adds **nothing** to Core; there is deliberately no ``ALGORITHM``/``TRAINER``
   :class:`~astro_mine.core.registry.enums.PluginKind` (adding one would need an RFC).
2. **Core** :class:`~astro_mine.core.registry.PluginRegistry` — the *policy* side. The only
   thing that crosses the waist is a **produced policy**, registered as a
   :class:`~astro_mine.core.registry.PluginManifest` of kind ``POLICY`` via
   :func:`policy_manifest` / :func:`manifest_from_export`.

Built-in :class:`~astro_mine.learn.algos._contract.Algorithm` classes are **lazy-loaded** by
dotted path so listing/resolving specs and emitting manifests never imports Torch — the
no-``[rllib]``-extra CI job exercises this whole module.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Iterator
from importlib.metadata import entry_points

from astro_mine.core.registry.enums import CapabilityTag, DeterminismClass, PluginKind
from astro_mine.core.registry.model import PluginManifest, Provenance
from astro_mine.learn.algos._contract import (
    POLICY_MANIFEST_INTERFACES,
    Algorithm,
    AlgorithmSpec,
    PolicyExport,
)
from astro_mine.learn.algos._specs import COMMS_PPO_SPEC, IPPO_SPEC, MAPPO_SPEC, QMIX_SPEC

__all__ = [
    "ALGORITHM_ENTRY_POINT_GROUP",
    "AlgorithmRegistry",
    "comms_learning_specs",
    "default_registry",
    "manifest_from_export",
    "policy_manifest",
]

#: The Python entry-point group a third-party algorithm plugin advertises itself under.
ALGORITHM_ENTRY_POINT_GROUP = "astro_mine.learn.algorithms"

#: Built-in baselines: capability tag → (spec, "module:attr" for the lazy Algorithm class).
#: IPPO is the independent control, MAPPO/QMIX the CTDE defaults, and ``comms_ppo`` the
#: comms-learning research track — all four are *replaceable examples*, discovered exactly like
#: a third-party plugin (charter §10.2; learn.md §11).
_BUILTINS: dict[str, tuple[AlgorithmSpec, str]] = {
    IPPO_SPEC.capability_tag: (IPPO_SPEC, "astro_mine.learn.algos.ippo:IppoAlgorithm"),
    MAPPO_SPEC.capability_tag: (MAPPO_SPEC, "astro_mine.learn.algos.mappo:MappoAlgorithm"),
    QMIX_SPEC.capability_tag: (QMIX_SPEC, "astro_mine.learn.algos.qmix:QmixAlgorithm"),
    COMMS_PPO_SPEC.capability_tag: (
        COMMS_PPO_SPEC,
        "astro_mine.learn.algos.comms_ppo:CommsPpoAlgorithm",
    ),
}


def _load(dotted: str) -> Algorithm:
    module_name, _, attr = dotted.partition(":")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    algorithm: Algorithm = factory()
    return algorithm


class AlgorithmRegistry:
    """An in-process registry of MARL algorithms discovered by capability tag.

    Seeded with the three built-in baselines; :meth:`discover_entry_points` adds any
    third-party algorithm advertised under :data:`ALGORITHM_ENTRY_POINT_GROUP`. Specs are
    available without instantiating (Torch-free); :meth:`get` lazy-builds the concrete
    :class:`~astro_mine.learn.algos._contract.Algorithm`."""

    def __init__(self, *, builtins: bool = True) -> None:
        self._specs: dict[str, AlgorithmSpec] = {}
        self._loaders: dict[str, str] = {}
        self._instances: dict[str, Algorithm] = {}
        if builtins:
            for tag, (spec, dotted) in _BUILTINS.items():
                self._specs[tag] = spec
                self._loaders[tag] = dotted

    # --- registration --------------------------------------------------------------

    def register(self, algorithm: Algorithm) -> None:
        """Register an already-constructed algorithm instance (e.g. a test double)."""
        tag = algorithm.spec.capability_tag
        self._specs[tag] = algorithm.spec
        self._instances[tag] = algorithm

    def discover_entry_points(self) -> list[str]:
        """Add every algorithm advertised under the entry-point group; return their tags."""
        added: list[str] = []
        for ep in entry_points(group=ALGORITHM_ENTRY_POINT_GROUP):
            algorithm: Algorithm = ep.load()()
            tag = algorithm.spec.capability_tag
            self._specs[tag] = algorithm.spec
            self._instances[tag] = algorithm
            added.append(tag)
        return added

    # --- resolution ----------------------------------------------------------------

    def specs(self) -> list[AlgorithmSpec]:
        """Every registered algorithm's spec (no Torch import)."""
        return list(self._specs.values())

    def spec(self, tag: str) -> AlgorithmSpec:
        """The spec for one capability tag (or plain name); raise ``KeyError`` if absent."""
        return self._specs[self._resolve_tag(tag)]

    def get(self, tag: str) -> Algorithm:
        """Resolve and (lazily) build the :class:`Algorithm` for a tag or plain name."""
        key = self._resolve_tag(tag)
        if key not in self._instances:
            self._instances[key] = _load(self._loaders[key])
        return self._instances[key]

    def __contains__(self, tag: object) -> bool:
        return isinstance(tag, str) and self._try_resolve(tag) is not None

    def __iter__(self) -> Iterator[AlgorithmSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)

    def _resolve_tag(self, tag: str) -> str:
        key = self._try_resolve(tag)
        if key is None:
            raise KeyError(f"no algorithm registered for {tag!r}")
        return key

    def _try_resolve(self, tag: str) -> str | None:
        if tag in self._specs:
            return tag
        for candidate, spec in self._specs.items():
            if spec.name == tag:
                return candidate
        return None


def default_registry() -> AlgorithmRegistry:
    """A registry seeded with the built-ins and third-party entry points."""
    registry = AlgorithmRegistry()
    registry.discover_entry_points()
    return registry


def comms_learning_specs(registry: AlgorithmRegistry | None = None) -> list[AlgorithmSpec]:
    """Every registered algorithm that **learns messages** over the CommsModel channel.

    The discovery surface for the comms-learning research track (learn.md §11): a Bench
    comparison sorts a leaderboard into comms-blind vs comms-learning by this flag, and a
    third-party comms-learning plugin appears here the moment it declares
    ``AlgorithmSpec.comms_learning``."""
    reg = registry if registry is not None else default_registry()
    return [spec for spec in reg.specs() if spec.comms_learning]


# --- Core POLICY manifest emission -------------------------------------------------


def policy_manifest(
    name: str,
    version: str,
    *,
    provenance: Provenance | None = None,
    capability_tags: Iterable[CapabilityTag] = (),
    description: str | None = None,
    attributes: dict[str, object] | None = None,
) -> PluginManifest:
    """Build the Core ``POLICY`` :class:`~astro_mine.core.registry.PluginManifest` for a
    policy produced by a Learn baseline.

    ``kind`` is always :attr:`~astro_mine.core.registry.enums.PluginKind.POLICY`; the
    manifest declares the Core interfaces the policy is built against
    (:data:`~astro_mine.learn.algos._contract.POLICY_MANIFEST_INTERFACES`) and that it
    consumes ``Observation`` and produces ``ActionBatch`` — so a
    :class:`~astro_mine.core.registry.PluginRegistry` (``require_signature=False`` locally)
    negotiates and registers it."""
    return PluginManifest(
        name=name,
        version=version,
        kind=PluginKind.POLICY,
        core_interfaces=dict(POLICY_MANIFEST_INTERFACES),
        inputs=["Observation"],
        outputs=["ActionBatch"],
        capability_tags=list(capability_tags),
        determinism_class=DeterminismClass.BIT_EXACT,
        description=description,
        provenance=provenance,
        attributes=attributes or {},
    )


def manifest_from_export(export: PolicyExport, *, name: str, version: str) -> PluginManifest:
    """Emit a ``POLICY`` manifest for a :class:`PolicyExport`, folding its declared comms
    assumptions and reported metrics into the manifest ``attributes`` (the honest metadata
    Guard reads; learn.md §9)."""
    attributes: dict[str, object] = {
        "algorithm": export.algorithm,
        "backend": export.backend,
        "partial_observability": export.assumptions.partial_observability,
        "metrics": dict(export.metrics),
    }
    if export.assumptions.comms_observability is not None:
        attributes["comms_observability"] = dict(export.assumptions.comms_observability)
    if export.assumptions.surrogate_fidelity_caveats:
        attributes["surrogate_fidelity_caveats"] = list(
            export.assumptions.surrogate_fidelity_caveats
        )
    return policy_manifest(
        name,
        version,
        provenance=export.provenance,
        capability_tags=export.capability_tags(),
        description=f"{export.algorithm} policy trained by astro-mine-learn (RM-P1-LEARN-03)",
        attributes=attributes,
    )
