"""Illumination field-model selection + the Core ``field_model`` manifest (RM-P1-WORLDS-10).

The backend a world uses for Sun visibility is a **plugin choice**, not a code path: worlds.md §3
lists "Field models — alternative illumination … implementations registered against the same
abstract interface **and selected in ``WorldSpec``**". :func:`build_illumination_model` is that
selection — a ``backend`` selector (``horizon`` | ``raycast_cpu`` | ``raycast_gpu`` |
``surrogate:<name>`` | *any advertised plugin id*) → the matching
:class:`~astro_mine.worlds.illumination._backend.SunVisibilityModel` — with the GPU path degrading
to the portable CPU ray-cast when CuPy/CUDA is absent (worlds.md §11).

A backend reaches that selection one of two ways, and :func:`build_illumination_model` treats them
identically (issue #52):

* **Built-in** — the horizon map, the CPU/GPU ray-casts, and the surrogate adapter, seeded in
  :data:`_BUILTINS` so selection works from a raw checkout with no installed metadata (CX-LOCAL);
* **Plugin** — any distribution advertising :data:`FIELD_MODEL_ENTRY_POINT_GROUP`, discovered
  through ``importlib.metadata`` (conventions.md §7: in-process plugins use Python entry points).

Until #52 this module *declared* the group in ``pyproject.toml`` and never read it, so a third-party
field model installed cleanly and was never discoverable — the same closed extension point
astro-mine-allocate#31 fixed for solver backends. ``field_model`` is a Core ``PluginKind``, so this
is a platform contract, not a Worlds-private notion.

**Laziness is a feature.** :func:`known_backends` reads entry-point *names* from installed metadata
and never calls ``load()``, so listing backends imports neither a plugin nor an optional dependency
— CuPy and ONNX Runtime are imported only when their backend is actually resolved *and queried*.

**Worlds advertises its own built-ins** under the group (they are the reference implementations
conventions.md §1.3 calls replaceable examples). Those self-advertised entry points resolve to the
in-code built-in rather than being ``load()``-ed: the entry-point targets below are thin adapters
over the same factories, so loading them here would recurse. A **foreign** distribution claiming a
built-in id is therefore unambiguous — and is a hard error, see :func:`_resolve_factory`.

:func:`build_illumination_field_manifest` emits the Core :class:`PluginManifest` that registers a
Worlds-native illumination backend as a ``FIELD_MODEL`` (``provenance.digest`` = the model's
``illumination_hash``), mirroring
:func:`~astro_mine.worlds.spec._publish.build_world_manifest`. A **learned** surrogate carries its
own signed manifest from [Surrogate](surrogate.md); this builder is for the horizon / ray-cast
backends Worlds itself produces.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from importlib.metadata import EntryPoint, entry_points
from typing import Any, cast

from astro_mine.core.registry import PluginKind, PluginManifest, Provenance
from astro_mine.worlds.illumination import IlluminationError, IlluminationModel
from astro_mine.worlds.illumination._backend import DEFAULT_BACKEND, SunVisibilityModel
from astro_mine.worlds.illumination._raycast import RAYCAST_CPU_BACKEND, RayCastIlluminationModel
from astro_mine.worlds.illumination._raycast_gpu import (
    RAYCAST_GPU_BACKEND,
    RayCastGpuIlluminationModel,
)
from astro_mine.worlds.illumination._surrogate import SurrogateIlluminationModel

__all__ = [
    "FIELD_MODEL_ENTRY_POINT_GROUP",
    "FIELD_MODEL_INTERFACE",
    "FIELD_MODEL_INTERFACE_VERSION",
    "SURROGATE_BACKEND",
    "available_backends",
    "build_illumination_field_manifest",
    "build_illumination_model",
    "horizon_field_model",
    "known_backends",
    "raycast_cpu_field_model",
    "raycast_gpu_field_model",
    "surrogate_field_model",
]

#: The Core interface an illumination field model implements (the world-provider field query).
FIELD_MODEL_INTERFACE = "world_provider"
FIELD_MODEL_INTERFACE_VERSION = "0.1.0"

#: The entry-point group a third-party illumination field model advertises itself under. The entry
#: point's **name** is the backend id — the string a ``WorldSpec``'s ``layers.illumination_backend``
#: carries and :func:`build_illumination_model` selects on; its value resolves to a
#: :data:`FieldModelFactory`, the same shape the built-ins have, so one call handles both.
#:
#: .. code-block:: toml
#:
#:     [project.entry-points."astro_mine.field_models"]
#:     acme-illum = "acme_illum.backend:build"
FIELD_MODEL_ENTRY_POINT_GROUP = "astro_mine.field_models"

#: The learned-surrogate backend id. Selected bare, or parameterized as ``surrogate:<name>``.
SURROGATE_BACKEND = "surrogate"

_SURROGATE_PREFIX = f"{SURROGATE_BACKEND}:"

#: The distribution that provides the built-in backends — see the module docstring on why a
#: self-advertised entry point resolves to the in-code built-in instead of being loaded.
_SELF_DISTRIBUTION = "astro-mine-platform"

#: Fallback self-recognition for an install whose entry point carries no resolvable distribution
#: (an editable/raw checkout): the built-ins are the only field models under this module path.
_SELF_VALUE_PREFIX = "astro_mine.worlds."

#: A field-model factory: builds a Sun-visibility backend over a terrain product.
FieldModelFactory = Callable[..., SunVisibilityModel]


def build_illumination_model(
    terrain: Any,
    *,
    backend: str = DEFAULT_BACKEND,
    surrogate: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> SunVisibilityModel:
    """Build the illumination model for a ``backend`` selector (a ``WorldSpec`` field-model choice).

    ``horizon`` (the default) is the precomputed map; ``raycast_cpu`` / ``raycast_gpu`` are the fine
    on-demand path (the GPU selector falls back to CPU when CuPy is unavailable, with the *same*
    numbers and backend label); ``surrogate:<name>`` needs the published artifacts (``onnx_model`` /
    ``manifest_attributes`` / ``error_report``) passed in ``surrogate`` — a consumer resolves those
    from [Hub] by content hash. **Any other id** is resolved against
    :data:`FIELD_MODEL_ENTRY_POINT_GROUP`, so a third-party backend is selectable with no change to
    Worlds (issue #52). Extra ``kwargs`` (``n_azimuth``, ``max_radius_m`` …) flow to the model
    unchanged, so the default path is byte-identical to constructing an :class:`IlluminationModel`
    directly.

    Raises :class:`IlluminationError` for an unknown id — naming what *is* known — or for an id
    claimed by both a built-in and a foreign plugin.
    """
    if backend == SURROGATE_BACKEND or backend.startswith(_SURROGATE_PREFIX):
        # The surrogate id is parameterized (``surrogate:<name>``) and needs the published
        # artifacts, so it resolves under its bare id and is handed the extra context.
        factory = _resolve_factory(SURROGATE_BACKEND)
        return factory(terrain, backend=backend, surrogate=surrogate, **kwargs)
    return _resolve_factory(backend)(terrain, **kwargs)


# --- the registry: built-ins + the entry-point group -------------------------------------------


def _horizon_backend(terrain: Any, **kwargs: Any) -> SunVisibilityModel:
    """The precomputed per-azimuth horizon map — the Phase-0 default."""
    return IlluminationModel(terrain, **kwargs)


def _raycast_cpu_backend(terrain: Any, **kwargs: Any) -> SunVisibilityModel:
    """The portable CPU ray-cast."""
    return RayCastIlluminationModel(terrain, backend=RAYCAST_CPU_BACKEND, **kwargs)


def _raycast_gpu_backend(terrain: Any, **kwargs: Any) -> SunVisibilityModel:
    """The GPU ray-cast model, degrading to the portable CPU fallback when CuPy/CUDA is absent."""
    try:
        return RayCastGpuIlluminationModel(terrain, backend=RAYCAST_GPU_BACKEND, **kwargs)
    except ImportError:
        return RayCastIlluminationModel(terrain, backend=RAYCAST_GPU_BACKEND, **kwargs)


def _surrogate_backend(
    terrain: Any,
    *,
    backend: str = SURROGATE_BACKEND,
    surrogate: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> SunVisibilityModel:
    """The learned surrogate, loaded from the artifacts a consumer resolved from Hub."""
    if surrogate is None:
        raise IlluminationError(
            f"the {backend!r} backend needs the published surrogate artifacts; resolve them from "
            "Hub by content hash and pass surrogate={'onnx_model':..., 'manifest_attributes':..., "
            "'error_report':...}"
        )
    # ``backend`` is either the bare id or ``surrogate:<name>``; the artifacts' own name wins.
    name = str(surrogate.get("name") or backend.partition(":")[2])
    return SurrogateIlluminationModel(
        terrain,
        onnx_model=surrogate["onnx_model"],
        manifest_attributes=surrogate["manifest_attributes"],
        error_report=surrogate["error_report"],
        name=name,
        backend=backend,
        **kwargs,
    )


#: The backends Worlds ships, seeded in code so selection works from a raw checkout with no
#: installed metadata (CX-LOCAL). Mirrors Bench's runner registry and Allocate's solver registry.
_BUILTINS: dict[str, FieldModelFactory] = {
    DEFAULT_BACKEND: _horizon_backend,
    RAYCAST_CPU_BACKEND: _raycast_cpu_backend,
    RAYCAST_GPU_BACKEND: _raycast_gpu_backend,
    SURROGATE_BACKEND: _surrogate_backend,
}

#: The illumination backend names this package provides itself, as names only.
#:
#: Public for the scaffold's sake and no other reason: `astro-mine plugin new field-model`
#: warns when the chosen name would shadow a built-in, and should not import a backend -- or
#: reach into `_BUILTINS` -- to learn that (astro-mine-cli#12).
BUILTIN_FIELD_MODELS: frozenset[str] = frozenset(_BUILTINS)



def _advertised() -> dict[str, EntryPoint]:
    """Field models advertised in this environment, by backend id — **nothing is loaded**.

    Reading installed metadata is what keeps :func:`known_backends` cheap and import-free: a machine
    with ten field-model plugins installed pays ten dictionary entries, not ten imports."""
    return {ep.name: ep for ep in entry_points(group=FIELD_MODEL_ENTRY_POINT_GROUP)}


def _describe(entry: EntryPoint) -> str:
    """Name a plugin's provider precisely enough to act on — distribution *and* target."""
    dist = getattr(entry.dist, "name", None)
    return f"{entry.value!r} (from {dist!r})" if dist else repr(entry.value)


def _is_self(entry: EntryPoint) -> bool:
    """Is ``entry`` one of Worlds' own declarations of its built-ins?

    Distribution name is the reliable signal (normalized per PEP 503, since installed metadata may
    spell it ``astro_mine_worlds``). An entry point with no resolvable distribution falls back to
    its target module path — the built-ins are the only field models under ``astro_mine.worlds``."""
    dist = getattr(entry.dist, "name", None)
    if dist is not None:
        return re.sub(r"[-_.]+", "-", dist).lower() == _SELF_DISTRIBUTION
    return entry.value.startswith(_SELF_VALUE_PREFIX)


def _resolve_factory(name: str) -> FieldModelFactory:
    """Resolve one backend id to its factory — built-in or plugin, one path.

    A **foreign** distribution advertising a built-in id is a hard error naming both claimants,
    rather than a silent precedence rule: the backend id is folded into ``illumination_hash`` and
    stamped into the ``field_model`` :class:`PluginManifest` as provenance (see
    :func:`build_illumination_field_manifest`), so an ambiguous id would mis-attribute which model
    produced a published illumination product. This follows Allocate's solver registry, whose
    backend id is provenance for the same reason; Bench's runner registry lets the built-in win
    silently, which is right for a *runner* selection that is never signed and published.
    """
    builtin = _BUILTINS.get(name)
    advertised = _advertised().get(name)
    if builtin is not None:
        if advertised is not None and not _is_self(advertised):
            raise IlluminationError(
                f"illumination backend id {name!r} is claimed by both the built-in backend and "
                f"the plugin {_describe(advertised)}; rename the plugin's entry point"
            )
        return builtin
    if advertised is None:
        raise IlluminationError(
            f"unknown illumination backend {name!r}; known backends: {known_backends()}"
        )
    factory = advertised.load()
    if not callable(factory):
        raise IlluminationError(
            f"illumination backend {name!r} entry point {_describe(advertised)} is not callable; "
            f"it must resolve to a factory building a SunVisibilityModel"
        )
    return cast("FieldModelFactory", factory)


def known_backends() -> tuple[str, ...]:
    """Every backend id the registry knows — built-ins plus advertised plugins — whether or not its
    optional dependency is installed.

    Never imports a backend, so a broken or heavyweight plugin cannot make listing fail or slow."""
    return tuple(sorted({*_BUILTINS, *_advertised()}))


def available_backends() -> tuple[str, ...]:
    """The backend ids that actually resolve in this environment (a subset of
    :func:`known_backends`).

    This one *does* import, since importability is the question being asked. Any failure — a missing
    dependency, a plugin that raises on load, an id collision — excludes that backend rather than
    propagating, so one broken plugin cannot deny the list to every other backend.
    """
    available: list[str] = []
    for name in known_backends():
        try:
            _resolve_factory(name)
        except Exception:  # a probe: any failure means "not available", never a crash
            continue
        available.append(name)
    return tuple(available)


# --- ``astro_mine.field_models`` entry points -------------------------------------------------
#
# Worlds' own declarations of its built-ins. These are thin adapters over the factories above and
# deliberately do NOT route back through `build_illumination_model`: the registry resolves a
# self-advertised entry point to the in-code built-in precisely so that this pair cannot recurse.


def horizon_field_model(terrain: Any, **kwargs: Any) -> SunVisibilityModel:
    """Entry point: the precomputed horizon-map field model (the default)."""
    return _horizon_backend(terrain, **kwargs)


def raycast_cpu_field_model(terrain: Any, **kwargs: Any) -> SunVisibilityModel:
    """Entry point: the portable CPU ray-cast field model."""
    return _raycast_cpu_backend(terrain, **kwargs)


def raycast_gpu_field_model(terrain: Any, **kwargs: Any) -> SunVisibilityModel:
    """Entry point: the GPU ray-cast field model (CPU fallback when CuPy is absent)."""
    return _raycast_gpu_backend(terrain, **kwargs)


def surrogate_field_model(
    terrain: Any, *, surrogate: Mapping[str, Any], **kwargs: Any
) -> SunVisibilityModel:
    """Entry point: a learned illumination-field surrogate loaded from its published artifacts."""
    name = str(surrogate.get("name", ""))
    return _surrogate_backend(
        terrain, backend=f"{_SURROGATE_PREFIX}{name}", surrogate=surrogate, **kwargs
    )


def build_illumination_field_manifest(
    model: IlluminationModel, *, name: str, version: str
) -> PluginManifest:
    """The Core ``field_model`` :class:`PluginManifest` for a Worlds-native illumination backend.

    Registers the backend as a ``FIELD_MODEL`` behind the ``world_provider`` interface — no Core
    change, a second illumination backend ships purely as a plugin (issue #26 acceptance). The
    ``provenance.digest`` is the model's ``illumination_hash`` (which folds in the active backend),
    so the manifest pins exactly which backend produced the illumination.
    """
    manifest = model.to_manifest()
    params = manifest["params"]
    backend = str(params.get("backend", DEFAULT_BACKEND))
    toolchain = manifest.get("toolchain", {})
    toolchain_version = toolchain.get("astro_mine_worlds") if isinstance(toolchain, dict) else None
    return PluginManifest(
        name=name,
        version=version,
        kind=PluginKind.FIELD_MODEL,
        core_interfaces={FIELD_MODEL_INTERFACE: FIELD_MODEL_INTERFACE_VERSION},
        license="Apache-2.0",
        description=f"Astro-Mine illumination field model (backend={backend})",
        provenance=Provenance(
            digest=model.illumination_hash,
            source_content_hashes={"terrain": model.terrain_hash} if model.terrain_hash else {},
            toolchain_version=str(toolchain_version) if toolchain_version else None,
        ),
        attributes={"backend": backend, "illumination_schema": str(manifest["schema"])},
    )
