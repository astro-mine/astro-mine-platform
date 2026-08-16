"""The republish guard in ``scripts/surrogate/publish_surrogate.py`` (astro-mine-platform#42).

The guard refuses to mint a new artifact under a ``name:version`` the tree has already published.
Its defect was that it asked the registry whether the name was taken — so a *pruned* name looked
identical to a never-published one, and the publish proceeded. Because ``--version`` is a label
rather than a checkout, the result was today's model under an old tier's name.

Per conventions.md §11 each check is tested twice: once that the legitimate path still works, and
once against a synthetic violation proving the refusal fires. A guard that silently stopped working
reads as a passing suite forever — which is precisely how this one got shipped.

The tests drive :func:`_check_republish` against a fake registry and a patched inventory rather than
training a surrogate: the guard is a decision procedure over a committed record, and a CPU-bound
train/calibrate/export cycle would prove nothing extra about it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "surrogate" / "publish_surrogate.py"

#: A stand-in for the tier CROSSOVER.md pins. The real value is in `registry-inventory.json`; using
#: a distinct literal keeps the test honest about *behaviour* rather than about that one digest.
RECORDED = "sha256:" + "aa" * 32
OTHER = "sha256:" + "bb" * 32


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("publish_surrogate", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeRegistry:
    """Resolves only the references it was given — the one behaviour the guard reads."""

    def __init__(self, held: set[str]) -> None:
        self._held = held

    def resolve(self, reference: str) -> object:
        from astro_mine.hub.registry import ArtifactNotFound

        if reference not in self._held:
            raise ArtifactNotFound(reference)
        return object()


@pytest.fixture
def guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The script module with its inventory pointed at a synthetic record."""
    module = _load()
    inventory = {
        "artifacts": {
            # Published and still held: the ordinary idempotent-rerun case.
            "excavation-gns:0.6.0": {
                "disposition": "published",
                "bundle_digest": RECORDED,
            },
            # Recorded, pruned, digest known: the case the guard exists for.
            "excavation-gns:0.4.0": {
                "disposition": "lost",
                "bundle_digest": RECORDED,
            },
            # Recorded, pruned, digest unknown: nothing can verify a rebuild.
            "excavation-gns:0.3.0": {
                "disposition": "lost",
                "bundle_digest": None,
            },
        }
    }
    path = tmp_path / "registry-inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    monkeypatch.setattr(module, "INVENTORY_PATH", path)
    return module


# --- the legitimate paths still work --------------------------------------------------------


def test_a_genuinely_new_version_publishes(guard: Any) -> None:
    """The common case: a name:version the tree has never published needs no ceremony."""
    registry = _FakeRegistry(held=set())
    assert guard._check_republish(registry, "excavation-gns", "0.7.0", None, rebuilt=OTHER) is None


def test_an_identical_rebuild_of_a_held_artifact_is_allowed(guard: Any) -> None:
    """The script is deterministic, so a re-run legitimately reproduces the published tier.

    No ``--expect-digest`` is required here: the registry still holds the artifact, so the
    immutability check has something to compare against and does not need the operator's word.
    """
    registry = _FakeRegistry(held={"excavation-gns:0.6.0"})
    assert (
        guard._check_republish(registry, "excavation-gns", "0.6.0", None, rebuilt=RECORDED) is None
    )


def test_a_pruned_name_republishes_when_the_claim_is_stated_and_true(guard: Any) -> None:
    """The escape hatch is real: state the digest, reproduce it, and the publish proceeds."""
    registry = _FakeRegistry(held=set())
    assert (
        guard._check_republish(registry, "excavation-gns", "0.4.0", RECORDED, rebuilt=RECORDED)
        is None
    )


# --- the synthetic violations: prove each refusal fires -------------------------------------


def test_a_pruned_name_is_refused_without_an_expected_digest(guard: Any) -> None:
    """**The defect.** Absent bytes must not read as an empty slot.

    This is the exact shape of the 2026-08-08 incident: `excavation-gns:0.4.0` pruned, the operator
    reaching for the obvious rebuild command, and the old guard silent because the registry had
    nothing to compare against.
    """
    registry = _FakeRegistry(held=set())
    refusal = guard._check_republish(registry, "excavation-gns", "0.4.0", None, rebuilt=None)
    assert refusal is not None
    assert "--expect-digest" in refusal
    assert RECORDED in refusal, "the refusal must name the digest the operator should state"
    assert "label, not a checkout" in refusal


def test_a_pruned_name_with_no_recorded_digest_is_refused_outright(guard: Any) -> None:
    """No evidence exists, so no flag can be offered — the refusal has no escape hatch."""
    registry = _FakeRegistry(held=set())
    refusal = guard._check_republish(
        registry, "excavation-gns", "0.3.0", RECORDED, rebuilt=RECORDED
    )
    assert refusal is not None
    assert "nothing to verify a rebuild against" in refusal
    assert "--expect-digest" not in refusal, "a flag that cannot help is worse than a refusal"


def test_a_claim_that_disagrees_with_the_record_is_refused(guard: Any) -> None:
    """``--expect-digest`` is checked against the committed record, not trusted."""
    registry = _FakeRegistry(held=set())
    refusal = guard._check_republish(registry, "excavation-gns", "0.4.0", OTHER, rebuilt=OTHER)
    assert refusal is not None
    assert "committed record says" in refusal


def test_a_rebuild_that_does_not_reproduce_the_record_is_refused(guard: Any) -> None:
    """The claim is true about intent and false about bytes — which is the counterfeit case."""
    registry = _FakeRegistry(held=set())
    refusal = guard._check_republish(registry, "excavation-gns", "0.4.0", RECORDED, rebuilt=OTHER)
    assert refusal is not None
    assert "is not the one on record" in refusal


def test_a_different_tier_claiming_a_held_version_is_refused(guard: Any) -> None:
    """The original guard's job, preserved: different bytes may not take a taken version."""
    registry = _FakeRegistry(held={"excavation-gns:0.6.0"})
    refusal = guard._check_republish(registry, "excavation-gns", "0.6.0", None, rebuilt=OTHER)
    assert refusal is not None
    assert "immutable" in refusal


def test_a_missing_inventory_fails_closed(guard: Any, tmp_path: Path) -> None:
    """An absent record must not read as "nothing was ever published"."""
    guard.INVENTORY_PATH = tmp_path / "does-not-exist.json"
    with pytest.raises(FileNotFoundError, match="committed record"):
        guard._check_republish(_FakeRegistry(held=set()), "excavation-gns", "0.4.0", None, None)


def test_an_unrecorded_name_still_checks_a_stated_digest(guard: Any) -> None:
    """If the operator makes a claim about a name the tree does not know, check it anyway."""
    registry = _FakeRegistry(held=set())
    refusal = guard._check_republish(registry, "some-other-tier", "1.0.0", RECORDED, rebuilt=OTHER)
    assert refusal is not None
    assert "nothing corroborates either value" in refusal
