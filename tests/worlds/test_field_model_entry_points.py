"""The ``astro_mine.field_models`` entry-point group — an open extension point (issue #52).

Worlds declared four entry points under ``astro_mine.field_models`` and **nothing read the group**:
illumination backends were selected by a hardcoded string switch, so a third-party field model
installed cleanly and was never discoverable — the closed extension point
astro-mine-allocate#31 fixed for solver backends. ``field_model`` is a Core ``PluginKind``, so this
is a platform contract, not a Worlds-private notion (conventions.md §1.3, §7; worlds.md §11).

The suite is deliberately split:

* :func:`test_a_plugin_is_discovered_from_a_really_installed_distribution` and its neighbours use a
  **genuinely installed** distribution — a ``.dist-info`` written to ``tmp_path`` and put on
  ``sys.path`` — because a monkeypatched ``entry_points`` can prove the registry reads *something*
  but can never prove ``importlib.metadata`` would have found a real plugin (issue #52 AC1).
* The rest monkeypatch the module-scope ``entry_points`` symbol, the mechanism Allocate's and
  Learn's registry tests use, to reach the error branches cheaply.

The registry never touches ``terrain`` — it resolves a factory and hands the object through — so
these tests pass a sentinel rather than paying for a DEM ingest. The backends' *behaviour* is
covered by ``test_illumination_raycast.py`` / ``test_illumination_surrogate.py``.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from astro_mine.worlds.illumination import (
    DEFAULT_BACKEND,
    FIELD_MODEL_ENTRY_POINT_GROUP,
    RAYCAST_CPU_BACKEND,
    RAYCAST_GPU_BACKEND,
    SURROGATE_BACKEND,
    IlluminationError,
    available_backends,
    build_illumination_model,
    known_backends,
)
from astro_mine.worlds.illumination import _registry as registry_mod

BUILTINS = {DEFAULT_BACKEND, RAYCAST_CPU_BACKEND, RAYCAST_GPU_BACKEND, SURROGATE_BACKEND}

PLUGIN_BACKEND = "acme-illum"

#: A field model that shares no base class with Worlds' own — the point being that the entry-point
#: contract is structural (:class:`SunVisibilityModel`), not "subclass ``IlluminationModel``".
_PLUGIN_MODULE = '''
"""A third-party illumination field model, reachable only through the entry point."""


class AcmeIlluminationModel:
    def __init__(self, terrain, **kwargs):
        self.terrain = terrain
        self.kwargs = kwargs


def build(terrain, **kwargs):
    return AcmeIlluminationModel(terrain, **kwargs)


NOT_CALLABLE = "not-a-factory"
'''


# --- a genuinely installed distribution ---------------------------------------------------


def _install_distribution(
    root: Path, *, dist: str, version: str, module: str, entries: dict[str, str]
) -> None:
    """Write a real importable distribution under ``root`` — module + ``.dist-info`` metadata.

    This is what makes the discovery claim meaningful: ``importlib.metadata`` scans ``sys.path``
    entries for ``*.dist-info`` directories, so a plugin planted this way is found by exactly the
    machinery a ``pip install`` would exercise. Nothing here is patched.
    """
    (root / f"{module}.py").write_text(_PLUGIN_MODULE)
    info = root / f"{dist.replace('-', '_')}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        textwrap.dedent(f"""\
        Metadata-Version: 2.1
        Name: {dist}
        Version: {version}
        """)
    )
    rendered = "\n".join(f"{name} = {target}" for name, target in entries.items())
    (info / "entry_points.txt").write_text(f"[{FIELD_MODEL_ENTRY_POINT_GROUP}]\n{rendered}\n")


@pytest.fixture
def installed_plugin(tmp_path: Path) -> Iterator[Path]:
    """Really install a field-model plugin for the duration of one test."""
    site = tmp_path / "site-packages"
    site.mkdir()
    _install_distribution(
        site,
        dist="acme-illum",
        version="1.2.3",
        module="acme_illum",
        entries={PLUGIN_BACKEND: "acme_illum:build"},
    )
    sys.path.insert(0, str(site))
    importlib.invalidate_caches()  # so importlib.metadata re-scans sys.path
    try:
        yield site
    finally:
        sys.path.remove(str(site))
        sys.modules.pop("acme_illum", None)
        importlib.invalidate_caches()


def test_a_plugin_is_discovered_from_a_really_installed_distribution(installed_plugin) -> None:
    """AC1: selectable through ``build_illumination_model`` with **no change to Worlds**.

    No monkeypatch anywhere in this test: the plugin is found because it is installed."""
    assert PLUGIN_BACKEND in known_backends()
    model = build_illumination_model(_TERRAIN, backend=PLUGIN_BACKEND, n_azimuth=8)
    assert type(model).__name__ == "AcmeIlluminationModel"
    assert model.terrain is _TERRAIN  # the terrain is handed through untouched
    assert model.kwargs == {"n_azimuth": 8}  # and so are the model kwargs


def test_the_package_really_advertises_its_builtins(installed_plugin) -> None:
    """The four ``pyproject.toml`` declarations are live metadata, not a stale comment.

    This is the half of #52 that made the group *look* open: guard it so a rename of a built-in
    entry point cannot silently re-close it."""
    advertised = {ep.name for ep in registry_mod.entry_points(group=FIELD_MODEL_ENTRY_POINT_GROUP)}
    assert advertised >= BUILTINS
    assert PLUGIN_BACKEND in advertised


def test_the_self_advertised_builtins_are_not_collisions() -> None:
    """The trap this design had to avoid.

    Worlds advertises its own four backends under the group. Read naively, every built-in would
    look like a built-in/plugin id collision and every selector would fail in a *real installed*
    environment — which is exactly the environment this test runs in.
    """
    assert set(available_backends()) >= BUILTINS


# --- the cheap error branches, via a patched ``entry_points`` ------------------------------

_TERRAIN = object()  # the registry never dereferences it


@dataclass
class _FakeDist:
    name: str


class _FakeEntryPoint:
    """The ``importlib.metadata`` surface the registry reads: name, value, dist, and ``load()``."""

    def __init__(
        self,
        name: str,
        *,
        loads: Any = None,
        raises: Exception | None = None,
        dist: _FakeDist | None = None,
        value: str = "acme_illum:build",
    ) -> None:
        self.name = name
        self.value = value
        self.dist = dist
        self._loads = loads if loads is not None else _plugin_factory
        self._raises = raises

    def load(self) -> Any:
        if self._raises is not None:
            raise self._raises
        return self._loads


class _FakePluginModel:
    def __init__(self, terrain: Any, **kwargs: Any) -> None:
        self.terrain = terrain
        self.kwargs = kwargs


def _plugin_factory(terrain: Any, **kwargs: Any) -> _FakePluginModel:
    return _FakePluginModel(terrain, **kwargs)


def _advertise(monkeypatch: pytest.MonkeyPatch, *entries: _FakeEntryPoint) -> None:
    """Advertise ``entries`` under the field-model group, as an installed distribution would."""
    monkeypatch.setattr(registry_mod, "entry_points", lambda *, group: list(entries))


def test_a_plugin_backend_is_listed_alongside_the_builtins(monkeypatch) -> None:
    _advertise(monkeypatch, _FakeEntryPoint(PLUGIN_BACKEND))
    assert set(known_backends()) == BUILTINS | {PLUGIN_BACKEND}


def test_listing_backends_never_loads_one(monkeypatch) -> None:
    """AC2: a plugin whose ``load()`` explodes is still *listed* — listing reads names only."""
    _advertise(monkeypatch, _FakeEntryPoint(PLUGIN_BACKEND, raises=RuntimeError("must not load")))
    assert PLUGIN_BACKEND in known_backends()


def test_listing_backends_imports_no_optional_dependency(monkeypatch) -> None:
    """AC2: CuPy / ONNX Runtime are imported only when their backend is resolved and queried.

    Compares against the modules already imported rather than asserting absence outright, so the
    claim stays honest whichever order the suite runs in.
    """
    _advertise(monkeypatch, _FakeEntryPoint(PLUGIN_BACKEND))
    before = set(sys.modules)
    assert set(known_backends()) >= BUILTINS
    newly_imported = set(sys.modules) - before
    assert not {name for name in newly_imported if name.split(".")[0] in {"cupy", "onnxruntime"}}


def test_available_backends_reports_only_what_resolves(monkeypatch) -> None:
    _advertise(
        monkeypatch,
        _FakeEntryPoint(PLUGIN_BACKEND),
        _FakeEntryPoint("broken", raises=ImportError("no acme_illum")),
    )
    available = set(available_backends())
    assert PLUGIN_BACKEND in available
    assert "broken" not in available


def test_a_broken_plugin_does_not_break_selection_for_the_others(monkeypatch) -> None:
    """AC3: one hostile plugin must not deny listing or selection to every other backend."""
    _advertise(
        monkeypatch,
        _FakeEntryPoint("hostile", raises=RuntimeError("boom")),
        _FakeEntryPoint(PLUGIN_BACKEND),
    )
    assert "hostile" in known_backends()  # advertised, so listed
    assert "hostile" not in available_backends()  # but not usable
    assert set(available_backends()) >= BUILTINS
    assert isinstance(build_illumination_model(_TERRAIN, backend=PLUGIN_BACKEND), _FakePluginModel)


def test_a_foreign_plugin_may_not_hijack_a_builtin_id(monkeypatch) -> None:
    """AC4: an id claimed by both a built-in and a *foreign* distribution fails loudly.

    Allocate's semantics, for Allocate's reason: the backend id is folded into
    ``illumination_hash`` and stamped into the published ``field_model`` manifest as provenance, so
    an ambiguous id would mis-attribute which model produced an illumination product. Bench lets
    the built-in win silently, which is right for a runner selection that is never signed.
    """
    _advertise(monkeypatch, _FakeEntryPoint(DEFAULT_BACKEND, dist=_FakeDist("acme-illum")))
    with pytest.raises(IlluminationError, match="claimed by both") as excinfo:
        build_illumination_model(_TERRAIN, backend=DEFAULT_BACKEND)
    message = str(excinfo.value)
    assert "built-in" in message
    assert "acme-illum" in message  # the other claimant is named, so the error is actionable
    assert DEFAULT_BACKEND not in available_backends()  # and the ambiguity is not papered over


@pytest.mark.parametrize(
    "entry",
    [
        # The built-ins now ship from the consolidated platform distribution.
        pytest.param(_FakeDist("astro-mine-platform"), id="dist-name"),
        pytest.param(_FakeDist("astro_mine_platform"), id="dist-name-unnormalized"),
        pytest.param(None, id="no-dist-falls-back-to-the-module-path"),
    ],
)
def test_worlds_own_declaration_of_a_builtin_is_not_a_collision(monkeypatch, entry) -> None:
    """Self-recognition, both signals: the normalized distribution name and the value prefix."""
    _advertise(
        monkeypatch,
        _FakeEntryPoint(
            DEFAULT_BACKEND,
            dist=entry,
            value="astro_mine.worlds.illumination._registry:horizon_field_model",
            raises=AssertionError("a self-advertised built-in must never be load()-ed"),
        ),
    )
    assert DEFAULT_BACKEND in available_backends()


def test_a_non_callable_entry_point_is_rejected(monkeypatch) -> None:
    _advertise(monkeypatch, _FakeEntryPoint(PLUGIN_BACKEND, loads="not-a-factory"))
    with pytest.raises(IlluminationError, match="not callable"):
        build_illumination_model(_TERRAIN, backend=PLUGIN_BACKEND)


def test_unknown_backend_error_still_says_what_is_known(monkeypatch) -> None:
    """The pre-existing error is preserved — and now names advertised plugins too."""
    _advertise(monkeypatch, _FakeEntryPoint(PLUGIN_BACKEND))
    with pytest.raises(IlluminationError, match="unknown illumination backend") as excinfo:
        build_illumination_model(_TERRAIN, backend="bogus")
    message = str(excinfo.value)
    assert DEFAULT_BACKEND in message
    assert PLUGIN_BACKEND in message


# --- AC5: existing selectors resolve exactly as before -------------------------------------


def test_the_surrogate_selectors_keep_their_actionable_error(monkeypatch) -> None:
    """``surrogate:<name>`` still explains how to get the artifacts, with a plugin installed.

    The bare ``surrogate`` id used to fall through to "unknown illumination backend"; it now
    reaches the same actionable message as the parameterized form, which is strictly better and
    still an error.
    """
    _advertise(monkeypatch, _FakeEntryPoint(PLUGIN_BACKEND))
    for selector in ("surrogate:acme", SURROGATE_BACKEND):
        with pytest.raises(IlluminationError, match="needs the published surrogate artifacts"):
            build_illumination_model(_TERRAIN, backend=selector)


def test_the_builtin_ids_still_resolve_with_a_plugin_installed(monkeypatch) -> None:
    """A third-party plugin on the machine must not perturb the built-in selectors.

    That the built-ins still *build* the same models is covered by ``test_illumination_raycast.py``
    (``test_factory_horizon_default_is_hash_stable`` pins the default path to byte-identical)."""
    _advertise(monkeypatch, _FakeEntryPoint(PLUGIN_BACKEND))
    assert set(available_backends()) >= BUILTINS
