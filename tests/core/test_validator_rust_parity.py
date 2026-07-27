"""The Rust fast path stays in lockstep with the Python reference (RM-P0-CORE-07, RM-P1-CORE-08).

``validator/rust`` is the optional Rust fast path (core.md §8, §11); ``astro_mine.core`` is the
authoritative reference. Their *verdicts* are compared by running both over the same checked-in
corpora — Python here (``test_plan.py``, ``test_policy.py``, the ``*_consistency`` tests), Rust in
``cargo test``. What that arrangement cannot catch is **coverage drift**: a new Core schema, unit,
frame class, or example that lands on the Python side and is never taught to the Rust binding —
exactly how the fast path fell four schema families behind before this gate existed.

So this module reads the Rust sources as text (no toolchain needed, so the gate runs in the normal
pytest job, not only in the conda codegen job) and asserts the two bindings cover the same surface:
every canonical JSON Schema is compiled, every rooted document family is registered under a logical
name, every example file is in the Rust parity corpus, and the closed units vocabularies agree
token-for-token.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from astro_mine.core.schemas import core_schema
from astro_mine.core.units.enums import DIMENSIONLESS_UNITS, SI_UNITS, FrameClass, TimeScale

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
EXAMPLES = ROOT / "examples"
CRATE = ROOT / "validator" / "rust" / "src"
LIB_RS = (CRATE / "lib.rs").read_text(encoding="utf-8")
UNITS_RS = (CRATE / "units.rs").read_text(encoding="utf-8")

#: ``include_str!("../../../<relpath>")`` — every file the crate bundles at compile time.
_INCLUDE_RE = re.compile(r'include_str!\(\s*"\.\./\.\./\.\./([^"]+)"\s*\)')

#: ``by_name.insert("<logical name>".to_string(), …)`` — the schemas ``CoreValidators`` exposes.
_REGISTERED_RE = re.compile(r'by_name\.insert\(\s*"([a-z_]+)"\.to_string\(\)')

#: ``pub const NAME: [&str; N] = ["a", "b"];`` — the crate's closed vocabularies.
_CONST_LIST_RE = re.compile(r"pub const (\w+): \[&str; \d+\] = \[([^\]]*)\];", re.DOTALL)

#: Logical name → schema file, for the schemas ``CoreValidators`` validates documents against.
#: ``units.schema.json`` is deliberately absent: it is a cross-file ``$ref`` *target* (RFC-0007),
#: not a document root — it is compiled into the mission/messages validators, never on its own.
#: ``messages.schema.json`` is a ``$defs``-only catalog registered as ``messages.<Root>``.
_ROOTED_SCHEMAS = {
    "sadf": "sadf/schema/sadf.schema.json",
    "objective": "objective/schema/objective.schema.json",
    "manifest": "registry/schema/manifest.schema.json",
    "mission": "mission/schema/mission.schema.json",
    "policy_package": "policy/schema/policy_package.schema.json",
    "run_provenance": "provenance/schema/run_provenance.schema.json",
    "plan": "plan/schema/plan.schema.json",
}


def _includes(source: str) -> set[str]:
    return set(_INCLUDE_RE.findall(source))


def _const_list(source: str, name: str) -> list[str]:
    for const, body in _CONST_LIST_RE.findall(source):
        if const == name:
            return re.findall(r'"([^"]+)"', body)
    raise AssertionError(f"`pub const {name}` not found in the Rust binding")


def test_rust_and_python_agree_on_the_units_ref_uri() -> None:
    """Both bindings must resolve the cross-file units ``$ref`` at the *same* URI (#53).

    messages/mission ``$ref`` the units vocabulary by its absolute ``$id``; each binding
    registers the vocabulary under that URI. Nothing used to guard that the two agreed, so
    a Rust/Python mismatch would sail through CI — and the symptom (Rust silently accepting
    a document Python rejects, because a validator compiled with the resource registered at
    the wrong URI never enforces the units types) is exactly the kind of divergence this
    parity suite exists to catch.
    """
    units_id = str(core_schema("astro_mine.core.units", "units.schema.json")["$id"])

    match = re.search(r'const UNITS_REF_URI: &str = "([^"]+)";', LIB_RS)
    assert match, "`const UNITS_REF_URI` not found in the Rust binding"
    assert match.group(1) == units_id, (
        "Rust and Python disagree on the units $ref URI — the cross-file ref will resolve "
        f"in one binding and not the other: Rust {match.group(1)!r} vs "
        f"Python {units_id!r}"
    )


def test_every_core_json_schema_is_compiled_by_the_rust_validator() -> None:
    # The gap this issue closed: mission/policy_package/run_provenance/plan existed in Python and
    # were never compiled in Rust. A new Core schema now fails here until the fast path covers it.
    on_disk = {
        p.relative_to(ROOT).as_posix() for p in SRC.glob("astro_mine/core/*/schema/*.schema.json")
    }
    assert on_disk <= _includes(LIB_RS), (
        "a canonical Core JSON Schema is not compiled by the Rust fast path: "
        f"{sorted(on_disk - _includes(LIB_RS))}"
    )


@pytest.mark.parametrize("name", sorted(_ROOTED_SCHEMAS), ids=sorted(_ROOTED_SCHEMAS))
def test_every_rooted_document_family_is_registered(name: str) -> None:
    assert (SRC / "astro_mine" / "core" / _ROOTED_SCHEMAS[name]).is_file()
    assert name in set(_REGISTERED_RE.findall(LIB_RS)), (
        f"CoreValidators does not register {name!r} — the Rust fast path cannot validate it"
    )


def test_every_example_document_is_in_the_rust_parity_corpus() -> None:
    # The examples are the shared parity fixture: whatever the Python loaders accept here, the Rust
    # validator must accept in `cargo test`. An example added on one side only is drift.
    on_disk = {
        p.relative_to(ROOT).as_posix()
        for p in EXAMPLES.glob("*/*.yaml")
        if p.parent.name != "downstream-consumer"
    }
    assert on_disk, "the example corpus is missing"
    assert on_disk <= _includes(LIB_RS), (
        "an example document is not in the Rust parity corpus: "
        f"{sorted(on_disk - _includes(LIB_RS))}"
    )


def test_rust_binding_runs_the_shared_conformance_vectors() -> None:
    # RM-P1-CORE-08 exit criterion: "every Core binding … passes the shared units conformance
    # vectors". The Rust binding discharges it by including *this* file — the same one the Python
    # reference runs in tests/test_units_conformance.py — and asserting each verdict in cargo test.
    vectors = "src/astro_mine/core/units/schema/conformance.json"
    assert (ROOT / vectors).is_file()
    assert vectors in _includes(UNITS_RS)
    for guard in ("require_frame", "require_crs", "require_epoch", "require_epoch_window"):
        assert f"pub fn {guard}(" in UNITS_RS, f"the Rust binding is missing {guard}"
    assert "pub fn scales_equivalent(" in UNITS_RS
    assert "pub fn require_si_unit(" in UNITS_RS


def test_closed_vocabularies_agree() -> None:
    # The units vocabularies are Core-owned and append-only by RFC (units/enums.py). The Rust guards
    # hard-code them, so a token added in Python must be added there too — or the fast path would
    # reject a document the reference accepts.
    assert _const_list(UNITS_RS, "TIME_SCALES") == [s.value for s in TimeScale]
    assert _const_list(UNITS_RS, "FRAME_CLASSES") == [c.value for c in FrameClass]
    assert set(_const_list(UNITS_RS, "SI_UNITS")) == set(SI_UNITS)
    assert set(_const_list(UNITS_RS, "DIMENSIONLESS_UNITS")) == set(DIMENSIONLESS_UNITS)
