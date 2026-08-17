# SPDX-License-Identifier: Apache-2.0
"""Curriculum plugin discovery (learn.md §3 "Extension / plugin points"; charter §10.2).

"New **algorithms**, **curricula**, **scenario generators**, and **model architectures** register
through the plugin registry … In-process plugins use Python entry points; the trainer discovers
them by capability tag. Reference baselines ship as *replaceable examples*, not privileged
internals" (learn.md §3).

This is the curriculum half of that surface, deliberately built as the **exact twin** of
:mod:`astro_mine.learn.algos.registry`'s :class:`AlgorithmRegistry`: built-ins plus Python entry
points under a group, lazy-loaded by name. Like the algorithm registry, it adds **nothing** to
the Core waist — a curriculum produces training *configs*, never a Core schema, so there is no
``CURRICULUM`` :class:`~astro_mine.core.registry.enums.PluginKind` (adding one would need an
RFC).

**The automatic-curriculum seam.** learn.md §11 defers automatic curricula (PLR, teacher-student,
regret-based) to research/Phase 2 but requires the *interface* now. It is here and it is the
whole interface: a plugin registers a factory returning any
:class:`~astro_mine.learn.curriculum.staged.Curriculum` — the hand-authored
:class:`StagedCurriculum` walks a fixed ladder, an automatic one picks its next stage from the
metrics handed to :meth:`~Curriculum.update`. Nothing else in Learn changes.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from importlib.metadata import entry_points

from astro_mine.learn.algos.config import TrainConfig
from astro_mine.learn.curriculum.spec import CurriculumSpec
from astro_mine.learn.curriculum.staged import Curriculum, StagedCurriculum

__all__ = [
    "CURRICULUM_ENTRY_POINT_GROUP",
    "CurriculumFactory",
    "CurriculumRegistry",
    "default_curriculum_registry",
]

#: The Python entry-point group a third-party curriculum plugin advertises itself under — the
#: automatic-curriculum research seam (learn.md §11).
CURRICULUM_ENTRY_POINT_GROUP = "astro_mine.learn.curricula"

#: A registered curriculum plugin: ``(seed, base_config) -> Curriculum``. A hand-authored ladder
#: closes over a :class:`CurriculumSpec`; an automatic curriculum builds whatever state it needs.
CurriculumFactory = Callable[[int, TrainConfig], Curriculum]

#: Built-in curricula: name → "module:attr" of a zero-arg :class:`CurriculumSpec` builder.
#: Lazy-loaded by dotted path, exactly like the algorithm registry's baselines.
_BUILTINS: dict[str, str] = {
    "comms_ladder": "astro_mine.learn.curriculum.library:comms_ladder",
    "randomized_comms": "astro_mine.learn.curriculum.library:randomized_comms",
}


def _load_spec(dotted: str) -> CurriculumSpec:
    module_name, _, attr = dotted.partition(":")
    builder = getattr(importlib.import_module(module_name), attr)
    spec: CurriculumSpec = builder()
    return spec


def _staged_factory(spec: CurriculumSpec) -> CurriculumFactory:
    def factory(seed: int, base: TrainConfig) -> Curriculum:
        return StagedCurriculum(spec, seed=seed, base=base)

    return factory


class CurriculumRegistry:
    """An in-process registry of curricula discovered by name.

    Seeded with the built-in :mod:`~astro_mine.learn.curriculum.library` curricula;
    :meth:`discover_entry_points` adds any third-party curriculum (including an **automatic**
    one) advertised under :data:`CURRICULUM_ENTRY_POINT_GROUP`. Specs are listable without
    building a curriculum; :meth:`build` constructs one bound to a run's seed + base config."""

    def __init__(self, *, builtins: bool = True) -> None:
        self._factories: dict[str, CurriculumFactory] = {}
        self._specs: dict[str, CurriculumSpec] = {}
        self._lazy: dict[str, str] = dict(_BUILTINS) if builtins else {}

    # --- registration ---------------------------------------------------------------

    def register(self, name: str, factory: CurriculumFactory) -> None:
        """Register a curriculum factory directly (a test double, or an automatic curriculum)."""
        self._factories[name] = factory

    def register_spec(self, spec: CurriculumSpec) -> None:
        """Register a hand-authored :class:`CurriculumSpec` (wrapped in a
        :class:`StagedCurriculum`) — the path a contributor's JSON/YAML curriculum takes."""
        self._specs[spec.name] = spec
        self._factories[spec.name] = _staged_factory(spec)

    def discover_entry_points(self) -> list[str]:
        """Add every curriculum advertised under the entry-point group; return their names.

        A plugin's entry point resolves to either a :class:`CurriculumSpec` builder (a
        hand-authored ladder) or a :data:`CurriculumFactory` (an automatic curriculum) — both
        are accepted, so a research curriculum needs no adapter."""
        added: list[str] = []
        for ep in entry_points(group=CURRICULUM_ENTRY_POINT_GROUP):
            loaded = ep.load()()
            if isinstance(loaded, CurriculumSpec):
                self.register_spec(loaded)
                added.append(loaded.name)
            else:
                self.register(ep.name, loaded)
                added.append(ep.name)
        return added

    # --- resolution -----------------------------------------------------------------

    def names(self) -> list[str]:
        """Every registered curriculum name (built-ins are listed without being loaded)."""
        return sorted({*self._factories, *self._lazy})

    def spec(self, name: str) -> CurriculumSpec:
        """The declarative spec for a *staged* curriculum (raises ``KeyError`` if unknown; a
        purely programmatic automatic curriculum has no spec and raises too)."""
        self._ensure(name)
        if name not in self._specs:
            raise KeyError(f"curriculum {name!r} has no declarative CurriculumSpec")
        return self._specs[name]

    def build(self, name: str, *, seed: int = 0, base: TrainConfig | None = None) -> Curriculum:
        """Construct the named curriculum bound to this run's seed + base :class:`TrainConfig`."""
        self._ensure(name)
        factory = self._factories[name]
        return factory(seed, base if base is not None else TrainConfig(seed=seed))

    def _ensure(self, name: str) -> None:
        if name in self._factories:
            return
        if name not in self._lazy:
            raise KeyError(f"no curriculum registered for {name!r}")
        self.register_spec(_load_spec(self._lazy[name]))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and (name in self._factories or name in self._lazy)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self.names())


def default_curriculum_registry() -> CurriculumRegistry:
    """A registry seeded with the built-in curricula and any third-party entry points."""
    registry = CurriculumRegistry()
    registry.discover_entry_points()
    return registry
