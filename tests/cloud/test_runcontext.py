"""RunContext provenance envelope: fields, strictness, hashing, store round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from astro_mine.cloud.artifacts.runcontext import (
    EnvironmentFingerprint,
    RunContext,
    code_version,
)
from astro_mine.cloud.artifacts.store import FilesystemArtifactStore


def test_minimum_provenance_fields() -> None:
    ctx = RunContext(
        source_content_hashes={"scenario": "sha256:" + "a" * 64},
        env_lockfile="sha256:" + "b" * 64,
        seed=7,
    )
    assert ctx.schema_version == "0.1"
    assert ctx.seed == 7
    assert ctx.source_content_hashes["scenario"].startswith("sha256:")


def test_reserved_fields_default_empty() -> None:
    ctx = RunContext()
    assert ctx.run_id is None
    assert ctx.image_digest is None
    assert ctx.core_interface_version is None
    assert ctx.outputs == {}


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        RunContext(unexpected="nope")  # type: ignore[call-arg]


def test_content_address_is_environment_independent() -> None:
    common = {"seed": 1, "source_content_hashes": {"a": "sha256:" + "c" * 64}}
    a = RunContext(**common, environment=EnvironmentFingerprint(python="3.12.1", platform="linux"))
    b = RunContext(**common, environment=EnvironmentFingerprint(python="9.9.9", platform="win32"))
    assert a.content_address() == b.content_address()


def test_content_address_tracks_deterministic_fields() -> None:
    assert RunContext(seed=1).content_address() != RunContext(seed=2).content_address()


def test_content_address_ignores_mlflow_run_id() -> None:
    # run_id is bookkeeping assigned from the pin, so it must not change the pin.
    common = {"seed": 1, "image_digest": "sha256:" + "a" * 64}
    assert (
        RunContext(**common, run_id="run-abc").content_address()
        == RunContext(**common, run_id="run-xyz").content_address()
    )


def test_json_round_trip() -> None:
    ctx = RunContext(seed=3, outputs={"trace": "sha256:" + "d" * 64})
    restored = RunContext.from_json(ctx.to_json())
    assert restored == ctx


def test_store_and_load(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    ctx = RunContext(seed=5, env_lockfile="sha256:" + "e" * 64)
    address = ctx.store(store)
    assert RunContext.load(store, address) == ctx


def test_core_interface_version_admits_compatible() -> None:
    assert RunContext(core_interface_version="0.1.0").core_interface_version == "0.1.0"


@pytest.mark.parametrize("bad", ["0.2.0", "0.0.0", "1.0.0", "0.1", "nope"])
def test_core_interface_version_rejects_incompatible(bad: str) -> None:
    with pytest.raises(ValidationError):
        RunContext(core_interface_version=bad)


def test_code_version_falls_back_when_absent() -> None:
    assert code_version("no-such-distribution-xyz") == "0.0.0"


def test_code_version_reads_installed_distribution() -> None:
    assert code_version("astro-mine-cloud")  # non-empty string
