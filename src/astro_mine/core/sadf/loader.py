"""SADF loading, validation, and the semantic (dual-use) gate.

Pipeline (``load_sadf``):

1. parse YAML/JSON (YAML is a JSON superset, so one parser handles both);
2. **structural** validation against the canonical JSON Schema (the authoritative
   gate — rejects unknown/typo'd fields, bad enum values, missing required, etc.);
3. build the typed :class:`~astro_mine.core.sadf.model.SadfDocument`;
4. **semantic** checks that exceed JSON Schema's expressiveness — the reserved/gated
   capability-tag gate, ``root_frame`` resolution, and well-formedness of each declared
   ``core_interface_versions`` entry (mirrors the registry's manifest check).

Keeping steps 2 and 3 behaviourally identical (the semantic checks in step 4 are
applied to *both* ``load_sadf`` and ``validate_sadf``) is what lets the consistency
test treat JSON Schema and Pydantic as one structural contract.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from astro_mine.core.compat import parse_version
from astro_mine.core.sadf.enums import GATED_CAPABILITY_TAGS
from astro_mine.core.sadf.model import Asset, SadfDocument

__all__ = [
    "SadfError",
    "SadfValidationError",
    "load_sadf",
    "load_schema",
    "validate_sadf",
]

_SCHEMA_RESOURCE = "schema/sadf.schema.json"


class SadfError(Exception):
    """Base class for SADF errors."""


class SadfValidationError(SadfError):
    """Raised when a SADF document fails structural or semantic validation."""


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    """Return the canonical SADF JSON Schema (shipped inside the package)."""
    text = (
        resources.files("astro_mine.core.sadf")
        .joinpath(_SCHEMA_RESOURCE)
        .read_text(encoding="utf-8")
    )
    schema: dict[str, Any] = json.loads(text)
    return schema


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    return Draft202012Validator(load_schema())


def _parse(source: str | bytes) -> Any:
    if isinstance(source, bytes):
        source = source.decode("utf-8")
    return yaml.safe_load(source)


def _check_structural(data: Any) -> None:
    if not isinstance(data, dict):
        raise SadfValidationError("SADF document must be a YAML/JSON mapping")
    errors = sorted(_validator().iter_errors(data), key=lambda e: list(e.path))
    if errors:
        rendered = "; ".join(
            f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors[:10]
        )
        raise SadfValidationError(f"SADF failed schema validation: {rendered}")


def _check_semantics(doc: SadfDocument) -> None:
    asset = doc.asset
    gated = [c for c in asset.capabilities if c in GATED_CAPABILITY_TAGS]
    if gated:
        names = ", ".join(sorted(str(c) for c in gated))
        raise SadfValidationError(
            f"asset declares reserved/gated capability tag(s) not permitted in the open "
            f"commons: {names} (RFC-0001 §6; conventions.md §12)"
        )
    if asset.frames:
        frame_names = {f.name for f in asset.frames}
        if asset.root_frame not in frame_names:
            raise SadfValidationError(
                f"root_frame {asset.root_frame!r} is not among declared frames "
                f"{sorted(frame_names)}"
            )
    for interface, version in asset.core_interface_versions.items():
        try:
            parse_version(version)
        except ValueError as exc:
            raise SadfValidationError(
                f"core_interface_versions[{interface!r}] is not a MAJOR.MINOR.PATCH version: {exc}"
            ) from exc
    _check_referential_closure(asset)


def _check_referential_closure(asset: Asset) -> None:
    """Verify every intra-document cross-reference in the kinematic graph resolves to a
    declared entity (RM-P1-CORE-05; core.md §2.7 "fail validation early and loudly").

    JSON Schema and Pydantic validate each field's *shape* but cannot express that a
    joint's ``child_body`` names a body that is actually declared. That gap let an asset
    ship joints referencing bodies it never declared — schema-valid and lint-clean, yet
    unrealizable at the ``articulated`` tier it advertised (companion to
    ``astro-mine-fleet#15``). Checking closure here catches the whole class at author
    time, once, at the waist, for every asset and consumer. All dangling references are
    collected and reported together so the author sees every problem at once.
    """
    frame_names = {f.name for f in asset.frames}
    body_names = {b.name for b in asset.bodies}
    joint_names = {j.name for j in asset.joints}
    problems: list[str] = []

    def _frame_ref(where: str, ref: str | None) -> None:
        if ref is not None and ref not in frame_names:
            problems.append(f"{where} references undeclared frame {ref!r}")

    for joint in asset.joints:
        if joint.parent_body not in body_names:
            problems.append(
                f"joint {joint.name!r} parent_body references undeclared body {joint.parent_body!r}"
            )
        if joint.child_body not in body_names:
            problems.append(
                f"joint {joint.name!r} child_body references undeclared body {joint.child_body!r}"
            )
    for actuator in asset.actuators:
        if actuator.target_joint is not None and actuator.target_joint not in joint_names:
            problems.append(
                f"actuator {actuator.name!r} target_joint references undeclared joint "
                f"{actuator.target_joint!r}"
            )
    for frame in asset.frames:
        _frame_ref(f"frame {frame.name!r} parent", frame.parent)
    for body in asset.bodies:
        _frame_ref(f"body {body.name!r}", body.frame)
    for geometry in asset.geometry:
        _frame_ref(f"geometry {geometry.uri!r}", geometry.frame)
    for sensor in asset.sensors:
        _frame_ref(f"sensor {sensor.name!r}", sensor.frame)
    for radio in asset.comms:
        if radio.antenna is not None:
            _frame_ref(
                f"comms {radio.name!r} antenna boresight_frame", radio.antenna.boresight_frame
            )
    if asset.payload is not None:
        for slot in asset.payload.slots:
            _frame_ref(f"payload slot {slot.name!r}", slot.frame)
    for sub in asset.subassemblies:
        _frame_ref(f"subassembly {sub.ref!r} mount_frame", sub.mount_frame)

    if problems:
        raise SadfValidationError(
            "SADF kinematic graph is not referentially closed: " + "; ".join(sorted(problems))
        )


def load_sadf(source: str | bytes) -> SadfDocument:
    """Parse, validate, and return a typed SADF document.

    Raises :class:`SadfValidationError` on any structural or semantic failure.
    """
    data = _parse(source)
    _check_structural(data)
    try:
        doc = SadfDocument.model_validate(data)
    except ValidationError as exc:
        raise SadfValidationError(f"SADF failed model validation: {exc}") from exc
    _check_semantics(doc)
    return doc


def validate_sadf(document: Any) -> None:
    """Validate a SADF document without returning it.

    Accepts raw YAML/JSON text/bytes, a parsed mapping, or a
    :class:`~astro_mine.core.sadf.model.SadfDocument`. Raises
    :class:`SadfValidationError` on failure.
    """
    if isinstance(document, SadfDocument):
        _check_structural(document.model_dump(by_alias=True, mode="json"))
        _check_semantics(document)
        return
    if isinstance(document, str | bytes):
        load_sadf(document)
        return
    if isinstance(document, dict):
        _check_structural(document)
        try:
            doc = SadfDocument.model_validate(document)
        except ValidationError as exc:
            raise SadfValidationError(f"SADF failed model validation: {exc}") from exc
        _check_semantics(doc)
        return
    raise SadfValidationError(f"cannot validate object of type {type(document).__name__}")
