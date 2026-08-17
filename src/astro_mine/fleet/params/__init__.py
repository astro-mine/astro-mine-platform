# SPDX-License-Identifier: Apache-2.0
"""Parametric asset families — the parameter-resolution engine (RM-P1-FLEET-10).

``fleet.md`` §3 splits parametric modelling into ``templates/`` (the family definitions:
base skeleton + parameter ranges + derived-quantity rules) and ``params/`` (this engine:
bind values → emit a concrete, validated SADF doc). "Parametric over copy-paste"
(``fleet.md`` §2.3): a "10-500 kg rover" is one :class:`Family`, not fifty files.

The engine is deliberately tiny and dependency-free: a :class:`ParamSpec` is a typed,
range-checked scalar; a :class:`Family` binds a set of them (defaults + overrides),
runs the family's derived-quantity builder, and returns a **Core-validated**
:class:`~astro_mine.core.sadf.SadfDocument`. Resolution is **deterministic** — the
builder is a pure function of the bound parameters, so the same inputs always yield the
same canonical SADF bytes (``fleet.md`` §10 golden/determinism gate; the acceptance
criterion for this issue). Fleet only *applies* Core's SADF/capability vocabulary here;
the resolved document is validated through Core's loader, so a family can never drift
from the schema.

Backlog: RM-P1-FLEET-10 -- astro-mine-fleet#21
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from astro_mine.core.sadf import SadfDocument, load_sadf

__all__ = ["AssetBuilder", "Family", "ParamError", "ParamSpec"]


class ParamError(ValueError):
    """A parameter binding names an unknown parameter, or is out of range/malformed."""


@dataclass(frozen=True)
class ParamSpec:
    """A single typed, range-checked family parameter (SI, closed ``[minimum, maximum]``)."""

    name: str
    minimum: float
    maximum: float
    default: float
    unit: str
    description: str
    integer: bool = False

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise ParamError(
                f"parameter {self.name!r}: minimum {self.minimum} exceeds maximum {self.maximum}"
            )
        if not (self.minimum <= self.default <= self.maximum):
            raise ParamError(
                f"parameter {self.name!r}: default {self.default} is outside "
                f"[{self.minimum}, {self.maximum}]"
            )

    def coerce(self, value: float) -> float:
        """Validate *value* against this spec and return it (as a float; int-snapped if integer).

        Raises :class:`ParamError` for a non-numeric value, a non-integer where an integer is
        required, or a value outside ``[minimum, maximum]``.
        """
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ParamError(f"parameter {self.name!r}: {value!r} is not a number") from exc
        if self.integer:
            if number != int(number):
                raise ParamError(f"parameter {self.name!r}: {value!r} must be an integer")
            number = float(int(number))
        if not (self.minimum <= number <= self.maximum):
            raise ParamError(
                f"parameter {self.name!r}={number} {self.unit} is outside its validated range "
                f"[{self.minimum}, {self.maximum}] {self.unit}"
            )
        return number


#: A family's derived-quantity builder: ``(variant, version, params) -> asset mapping``.
#: The returned ``dict`` is the SADF ``asset`` object; the engine wraps it in the document
#: envelope and validates it through Core's loader.
AssetBuilder = Callable[[str, str, Mapping[str, float]], dict[str, Any]]


@dataclass(frozen=True)
class Family:
    """A parametric asset family: a typed parameter set + a derived-quantity builder."""

    name: str  # family handle, e.g. "surface-rover"
    kind: str  # SADF identity.kind, e.g. "rover"
    summary: str
    params: tuple[ParamSpec, ...]
    build_asset: AssetBuilder

    def spec(self, name: str) -> ParamSpec:
        """The :class:`ParamSpec` named *name*; raises :class:`ParamError` if unknown."""
        for spec in self.params:
            if spec.name == name:
                return spec
        known = ", ".join(spec.name for spec in self.params)
        raise ParamError(f"unknown parameter {name!r} for family {self.name!r}; known: {known}")

    def defaults(self) -> dict[str, float]:
        """The default binding — the family's anchor point in parameter space."""
        return {spec.name: spec.default for spec in self.params}

    def bind(self, overrides: Mapping[str, float] | None = None) -> dict[str, float]:
        """Resolve *overrides* over the defaults into a fully validated parameter binding.

        Every override is range-checked against its :class:`ParamSpec`; an unknown key or an
        out-of-range value raises :class:`ParamError`.
        """
        values = self.defaults()
        for key, value in (overrides or {}).items():
            values[key] = self.spec(key).coerce(value)
        return values

    def resolve(
        self,
        overrides: Mapping[str, float] | None = None,
        *,
        variant: str = "custom",
        version: str = "0.1.0",
    ) -> SadfDocument:
        """Bind parameters, build the asset, and return a Core-validated SADF document.

        Deterministic: identical *overrides*/*variant*/*version* always produce byte-identical
        canonical SADF (:func:`astro_mine.core.sadf.to_wire`). Raises :class:`ParamError` on a
        bad binding and :class:`~astro_mine.core.sadf.SadfValidationError` if the builder ever
        emits a document Core rejects (a family defect — covered by the family tests).
        """
        params = self.bind(overrides)
        asset = self.build_asset(variant, version, params)
        document = {"sadf_version": "0.1", "asset": asset}
        return load_sadf(json.dumps(document, sort_keys=True))
