"""RM-P0-CLOUD-01 — Sim's digest-pinned workload image, via Cloud's BuildSpec/ImageRef discipline.

The gap this closes: RM-P0-CLOUD-01 requires container-first, digest-pinned packaging of every P0
workload and names Sim explicitly, but the repo had no ``Dockerfile`` and no reference to
``astro_mine.cloud`` anywhere. Cloud's machinery already exists — this is Sim *adopting* it, not
building new Cloud plumbing.

The load-bearing test is
:func:`test_the_workload_stage_is_clouds_rendered_output_verbatim`: it asserts the committed
Dockerfile contains ``render_dockerfile(spec)`` **byte-for-byte**, so the packaging discipline is
provably Cloud's rather than a hand-rolled lookalike that merely resembles it.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from astro_mine.cloud.packaging import BuildSpec, ImageRef, build_image, render_dockerfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from render_dockerfile import (
    BASE_IMAGE,
    DOCKERFILE,
    REPOSITORY,
    build_spec,
    main,
    render,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


class _FakeBuilder:
    """An injected ``ImageBuilder`` — the seam Cloud provides so the recipe is testable with no
    Docker daemon (mirroring how Cloud tests its own builder)."""

    def __init__(self, digest: str) -> None:
        self.digest = digest
        self.calls: list[dict[str, object]] = []

    def build(
        self,
        *,
        context: str,
        dockerfile: str,
        tags: list[str],
        build_args: dict[str, str],
    ) -> str:
        self.calls.append(
            {"context": context, "dockerfile": dockerfile, "tags": tags, "args": build_args}
        )
        return self.digest


# --- the recipe is Cloud's, not a lookalike --------------------------------------


def test_sim_has_a_dockerfile_produced_via_clouds_build_spec() -> None:
    # Acceptance criterion: "Sim has a Dockerfile (or equivalent) produced via Cloud's
    # BuildSpec/render_dockerfile".
    assert DOCKERFILE.exists()
    assert isinstance(build_spec(), BuildSpec)


def test_the_workload_stage_is_clouds_rendered_output_verbatim(dockerfile: str) -> None:
    # THE test: the workload stage is `render_dockerfile(spec)` byte-for-byte, so Sim genuinely
    # *adopts* Cloud's packaging discipline rather than re-implementing something that looks like
    # it.
    assert render_dockerfile(build_spec()) in dockerfile


def test_the_committed_dockerfile_is_not_stale(dockerfile: str) -> None:
    # The generated file is committed (so the image builds with no codegen step); this gates it
    # against drift from the BuildSpec — the same check CI runs.
    assert dockerfile == render()
    assert main(["--check"]) == 0


def test_the_base_image_is_pinned_by_digest_not_a_floating_tag(dockerfile: str) -> None:
    # Acceptance criterion: "the base image is pinned by digest, not a floating tag". A tag is
    # mutable — `python:3.12-slim` is different bytes next month — so a build on a tag is not
    # reproducible, whatever else the Dockerfile says.
    base = ImageRef.parse(BASE_IMAGE)
    assert base.digest.startswith("sha256:") and len(base.digest) == len("sha256:") + 64

    for line in dockerfile.splitlines():
        if line.startswith("FROM "):
            reference = line.removeprefix("FROM ").split(" AS ")[0].strip()
            # EVERY stage — the builder as well as the workload — is digest-pinned. A floating base
            # in
            # the builder would make the resolved dependency set irreproducible just as surely.
            assert "@sha256:" in reference, f"unpinned FROM: {line}"
            ImageRef.parse(reference)  # raises on anything not digest-pinned

    # And no FROM anywhere carries a *tag-only* reference (one with no `@sha256:` digest).
    assert "python:3.12-slim" not in dockerfile
    tag_only = re.findall(r"^FROM\s+[^@\s]+:[\w.-]+(?:\s+AS\s+\w+)?\s*$", dockerfile, re.M)
    assert tag_only == [], f"tag-pinned FROM lines: {tag_only}"


def test_the_image_runs_as_non_root(dockerfile: str) -> None:
    # Acceptance criterion: "the built image runs as non-root". Cloud's template bakes this in
    # (`USER 65532:65532`, the distroless nonroot uid) and offers no way to opt out.
    assert "USER 65532:65532" in dockerfile
    assert "USER root" not in dockerfile
    # The payload is owned by that uid, so the workload needs no privilege to read its own venv.
    assert "--chown=65532:65532" in dockerfile


def test_the_build_is_reproducible_by_construction(dockerfile: str) -> None:
    # A fixed SOURCE_DATE_EPOCH (so image metadata is byte-stable) and a locked dependency
    # resolution
    # (`uv sync --locked` fails rather than re-resolving) — cloud.md §4 principle 4; conventions.md
    # §7.
    assert "ARG SOURCE_DATE_EPOCH=0" in dockerfile
    assert "uv sync --locked" in dockerfile
    assert build_spec().source_date_epoch == 0


def test_the_entrypoint_runs_a_scenario() -> None:
    # A Cloud job runs a scenario, so that is what the workload's entrypoint takes.
    assert build_spec().entrypoint == ["python", "-m", "astro_mine.sim"]


# --- the image resolves by an ImageRef content hash -------------------------------


def test_building_yields_a_digest_pinned_image_ref() -> None:
    # Acceptance criterion: "the image is published and resolvable via an ImageRef content hash".
    # `build_image` returns the digest-pinned ref a Bench/Cloud JobSpec resolves the image by — a
    # JobSpec's `image` field is typed `ImageRef`, so an unpinned image cannot enter one at all.
    digest = "sha256:" + "dd" * 32
    builder = _FakeBuilder(digest)
    spec = build_spec(version="0.1.0", revision="c0ffee")

    ref = build_image(spec, context=".", tags=[f"{REPOSITORY}:0.1.0"], builder=builder)

    assert isinstance(ref, ImageRef)
    assert ref.repository == REPOSITORY and ref.digest == digest
    assert ref.reference == f"{REPOSITORY}@{digest}"  # the pinned pull reference
    # The deterministic build clock is passed through as a build arg.
    (call,) = builder.calls
    assert call["args"] == {"SOURCE_DATE_EPOCH": "0"}


def test_an_unpinned_reference_is_rejected_outright() -> None:
    # The boundary Cloud enforces: an image referenced by tag can never enter a job spec.
    with pytest.raises(ValueError, match="unpinned image reference"):
        ImageRef.parse(f"{REPOSITORY}:0.1.0")


def test_the_revision_is_not_baked_into_the_committed_file() -> None:
    # The commit sha would change on every commit and the golden test would never be green, so CI
    # passes it at build time instead. The spec still *supports* it.
    assert build_spec(revision="c0ffee").revision == "c0ffee"
    assert "c0ffee" not in render()


def test_render_check_detects_a_stale_dockerfile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import render_dockerfile as module

    stale = tmp_path / "Dockerfile"
    stale.write_text("FROM scratch\n", encoding="utf-8")
    original = module.DOCKERFILE
    module.DOCKERFILE = stale
    try:
        assert main(["--check"]) == 1
        assert "stale" in capsys.readouterr().err
        assert main([]) == 0  # ... and regenerating fixes it
        assert stale.read_text(encoding="utf-8") == render()
    finally:
        module.DOCKERFILE = original


# --- the entrypoint actually runs -------------------------------------------------


def test_the_entrypoint_runs_a_scenario_end_to_end(tmp_path: Path) -> None:
    # The image is only useful if its ENTRYPOINT works. Exercise it directly (the same code the
    # container runs) rather than trusting the Dockerfile's ENTRYPOINT line.
    from astro_mine.sim.__main__ import main as sim_main

    scenario = tmp_path / "scenario.json"
    scenario.write_text(
        '{"name": "smoke", "horizon_steps": 3, "agents": [{"agent_id": "a"}]}', encoding="utf-8"
    )
    out = tmp_path / "run.mcap"
    assert sim_main(["--scenario", str(scenario), "--seed", "5", "--out", str(out)]) == 0
    assert out.exists() and out.stat().st_size > 0

    from astro_mine.sim.recording import read_recording

    recording = read_recording(out)
    assert len(recording.frames) == 4  # the reset frame plus one per step
    assert recording.content_hash  # the determinism key the container prints


# --- the live build (opt-in: it needs a Docker daemon) -----------------------------


@pytest.mark.docker
def test_the_image_builds_and_runs_non_root() -> None:  # pragma: no cover  (needs a daemon)
    # The real build, deselected by default: `-m docker` on a host with a daemon, and the `image` CI
    # job. Everything the recipe *claims* (digest-pinned base, non-root, reproducible clock) is
    # gated
    # above without a daemon; this proves the claims survive an actual build.
    subprocess.run(
        ["docker", "build", "-f", str(DOCKERFILE), "-t", "astro-mine-sim:test", str(_REPO_ROOT)],
        check=True,
    )
    whoami = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "id", "astro-mine-sim:test", "-u"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert whoami.stdout.strip() == "65532"
