#!/usr/bin/env python
"""Smoke check that Core satisfies the downstream consumer contract (RFC-0009).

**This used to be a false green.** It asserted ``core.__version__ == "0.1.0"`` against a Core
resolved from git tag ``v0.1.0`` — so it never exercised the code under review, could not fail
for *any* change to Core, and stayed green through three PRs that rewrote Core's schemas. The
``consumer-smoke`` job now installs **the checkout under review** and runs this.

It therefore asserts the **contract**, not a version string: that a downstream package can
``$ref`` a Core schema by its ``$id`` and deep-validate a document through it. That is exactly
what astro-mine-core#54 broke for three repos while every check stayed green.
"""

from __future__ import annotations

from jsonschema import Draft202012Validator

import astro_mine.core as core
from astro_mine.core import SCHEMA_DIGEST, schema_registry
from astro_mine.core.compat import CORE_INTERFACE_VERSIONS
from astro_mine.core.objective.model import ObjectiveDocument
from astro_mine.core.registry.model import ManifestDocument
from astro_mine.core.sadf.model import SadfDocument

UNITS_ID = "https://schemas.astro-mine.org/core/units/v0.1/units.schema.json"

#: A consumer schema in *its own* namespace, ``$ref``-ing Core's units vocabulary by absolute
#: ``$id`` — the shape RFC-0009 requires of every downstream package.
CONSUMER_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://schemas.astro-mine.org/downstream-consumer/v0.1/probe.schema.json",
    "$ref": f"{UNITS_ID}#/$defs/Epoch",
}


def main() -> int:
    # The wheel installs with its typed models.
    assert CORE_INTERFACE_VERSIONS["sadf"] == "0.1.0"
    for model in (SadfDocument, ObjectiveDocument, ManifestDocument):
        assert hasattr(model, "model_fields"), f"{model.__name__} is not a typed model"

    # Core can name the exact schema set it carries (astro-mine-core#55).
    assert SCHEMA_DIGEST.startswith("sha256:"), f"bad SCHEMA_DIGEST: {SCHEMA_DIGEST!r}"

    # The contract: `$ref` Core by `$id`, and *descend into* the referenced `$def`. Resolution
    # is lazy — a document that never reaches the ref validates fine even when the ref is
    # broken, which is how this class of bug ships unnoticed.
    validator = Draft202012Validator(CONSUMER_SCHEMA, registry=schema_registry(CONSUMER_SCHEMA))
    assert not list(validator.iter_errors({"tdb_seconds": 0.0, "scale": "tdb"})), (
        "a valid Epoch was rejected"
    )
    assert list(validator.iter_errors({"tdb_seconds": 0.0, "scale": "utc"})), (
        "an invalid TimeScale was accepted — the cross-file $ref resolved to nothing, or to the "
        "wrong document. A downstream package cannot validate against this Core."
    )

    print(
        f"OK: astro-mine-core {core.__version__} — typed models import; schema digest "
        f"{SCHEMA_DIGEST[:19]}…; a consumer resolves a Core $ref by $id and deep-validates."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
