#!/usr/bin/env python
"""Write the canonical allocation JSON Schema files from the Pydantic models (RM-P1-ALLOC-01).

The models are the source of truth; this materializes their JSON Schema into
``src/astro_mine/allocate/schema/`` so external / cross-language consumers have a
first-class, checked-in schema. ``tests/test_schema.py`` regenerates-and-compares against
:func:`astro_mine.allocate._schema.json_schemas` as the drift guard.

    uv run python scripts/export_schemas.py
"""

from __future__ import annotations

import json
from pathlib import Path

from astro_mine.allocate._schema import json_schemas

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "src" / "astro_mine" / "allocate" / "schema"


def main() -> None:
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, schema in json_schemas().items():
        path = SCHEMA_DIR / filename
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
