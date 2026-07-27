"""Publishing to a relative store, and never leaving half a PolicyPackage behind (#33).

``--export ./policies`` — the path anyone types first — used to write ``model.onnx`` and then die
in :meth:`Path.as_uri`, which refuses a relative path, *between* the two writes. What survived was
a digest directory holding a graph with no sidecar: it still resolves by hash, and what a consumer
gets is a model with no IO signature, no assumptions and no provenance.

So these tests pin two properties, not one: a relative store works, **and** no failure path leaves
a partial entry. Needs the [export] extra + Torch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from astro_mine.core.policy.loader import load_policy_package
from astro_mine.learn import TrainConfig, default_registry, make_swarm_env
from astro_mine.learn.export import export_policy_package, publish
from tests.learn.fakes import FakeSwarmWorld, build_assets

_CFG = TrainConfig(seed=1, iterations=1, rollout_steps=8, hidden_sizes=(16, 16))


@pytest.fixture(scope="module")
def exported():
    """One trained-and-exported policy, reused across the cases (export is the slow part)."""
    env = make_swarm_env(FakeSwarmWorld(), build_assets())
    trainer = default_registry().get("ippo").make_trainer(env, _CFG)
    trainer.train_iteration()
    return export_policy_package(trainer.export(), "rover", version="0.1.0")


def _entry(store: Path, digest: str) -> Path:
    return store / digest.split(":", 1)[-1]


# --- the reported bug ---------------------------------------------------------------------------


def test_publish_accepts_a_relative_store(exported, tmp_path, monkeypatch) -> None:
    """The exact failure from #33: a relative `--export` target."""
    monkeypatch.chdir(tmp_path)

    published = publish(exported, Path("policies"))

    assert published.onnx_path.is_absolute()
    assert published.onnx_path.exists()
    assert published.sidecar_path.exists()
    # Both halves, which is what the crash prevented.
    assert published.onnx_path.read_bytes() == exported.onnx_bytes
    assert json.loads(published.sidecar_path.read_text())["policy_package_version"] == "0.1"


def test_relative_store_records_an_absolute_uri(exported, tmp_path, monkeypatch) -> None:
    """`as_uri()` is why this failed; the sidecar has to end up with a usable one."""
    monkeypatch.chdir(tmp_path)

    published = publish(exported, "policies")

    reloaded = load_policy_package(published.sidecar_path.read_text())
    uri = reloaded.policy_package.onnx_model.uri
    assert uri.startswith("file:///")
    assert Path(uri.removeprefix("file://")).exists()


def test_dot_prefixed_relative_store_works(exported, tmp_path, monkeypatch) -> None:
    """`./policies` is what a user actually types."""
    monkeypatch.chdir(tmp_path)

    published = publish(exported, "./policies")

    assert published.onnx_path.exists() and published.sidecar_path.exists()
    assert (tmp_path / "policies").is_dir()


# --- no half-entries ----------------------------------------------------------------------------


def test_a_failed_write_leaves_no_entry(exported, tmp_path, monkeypatch) -> None:
    """The property the bug violated, asserted directly.

    With the sidecar write forced to fail, the digest directory must not exist — a graph with no
    sidecar beside it is not a PolicyPackage, and it would still resolve by hash.
    """
    real_write = Path.write_bytes

    def _fail_on_sidecar(self: Path, data: bytes) -> int:
        if self.name == "policy_package.json":
            raise OSError("no space left on device")
        return real_write(self, data)

    monkeypatch.setattr(Path, "write_bytes", _fail_on_sidecar)

    with pytest.raises(OSError, match="no space left"):
        publish(exported, tmp_path)

    entry = _entry(tmp_path, exported.digest)
    assert not entry.exists()
    # And no staging directory is left lying around either.
    assert list(tmp_path.iterdir()) == []


def test_republishing_the_same_digest_is_idempotent(exported, tmp_path) -> None:
    """Content-addressed: a second publish of one digest is a no-op, not an error."""
    first = publish(exported, tmp_path)
    second = publish(exported, tmp_path)

    assert first.onnx_path == second.onnx_path
    assert second.onnx_path.read_bytes() == exported.onnx_bytes
    assert second.sidecar_path.exists()
    # One entry, not two.
    assert [p.name for p in tmp_path.iterdir()] == [exported.digest.split(":", 1)[-1]]


# --- the CLI flag -------------------------------------------------------------------------------


def test_cli_resolves_a_relative_export_path(tmp_path, monkeypatch) -> None:
    """`--export` normalizes at parse time, so everything downstream sees one unambiguous place."""
    from astro_mine.learn.train.run import _parser

    monkeypatch.chdir(tmp_path)
    args = _parser().parse_args(
        ["--env-factory", "tests.learn.fakes:FakeSwarmWorld", "--export", "./policies"]
    )

    assert Path(args.export).is_absolute()
    assert Path(args.export) == tmp_path.resolve() / "policies"
