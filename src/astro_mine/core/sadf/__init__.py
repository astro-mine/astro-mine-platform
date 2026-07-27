"""SADF — the Swarm Asset Description Format (v0.1, RM-P0-CORE-01).

Engine-neutral, declarative description of an asset: identity, USD/glTF geometry
references, kinematics/dynamics, power/thermal budgets, sensor suite, comms
capabilities, declared autonomy capability tags, propulsion/staging/return
capabilities (RFC-0001), multi-fidelity profiles, and composability via
sub-assemblies and payload slots. Authored as YAML/JSON, validated by JSON Schema,
with a canonical Protobuf wire form. MUST stay engine-neutral.

The canonical schema is ``schema/sadf.schema.json`` (shipped in-package); the typed
models live in :mod:`astro_mine.core.sadf.model`, the closed vocabularies in
:mod:`astro_mine.core.sadf.enums`.

Public API:

- :func:`load_sadf` / :func:`validate_sadf` — parse + validate (structural + the
  dual-use semantic gate);
- :func:`to_wire` / :func:`from_wire` — byte-stable Protobuf round-trip;
- :func:`to_proto` / :func:`from_proto` — message-level conversion;
- :func:`load_schema` — the canonical JSON Schema as a dict.

Backlog: RM-P0-CORE-01 — https://github.com/astro-mine/astro-mine-core/issues/1
"""

from __future__ import annotations

from astro_mine.core.sadf import enums, model
from astro_mine.core.sadf.enums import CapabilityTag
from astro_mine.core.sadf.loader import (
    SadfError,
    SadfValidationError,
    load_sadf,
    load_schema,
    validate_sadf,
)
from astro_mine.core.sadf.model import Asset, SadfDocument
from astro_mine.core.sadf.wire import from_proto, from_wire, to_proto, to_wire

__all__ = [
    "Asset",
    "CapabilityTag",
    "SadfDocument",
    "SadfError",
    "SadfValidationError",
    "enums",
    "from_proto",
    "from_wire",
    "load_sadf",
    "load_schema",
    "model",
    "to_proto",
    "to_wire",
    "validate_sadf",
]
