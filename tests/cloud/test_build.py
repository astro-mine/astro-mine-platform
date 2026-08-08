"""Reproducible packaging: Dockerfile rendering, the golden template, build_image."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import astro_mine.cloud.packaging.build as build_mod
from astro_mine.cloud.packaging import (
    BuildSpec,
    ImageRef,
    build_image,
    render_dockerfile,
)

TEMPLATE_PATH = Path(build_mod.__file__).parent / "templates" / "workload.Dockerfile"


def _reference_spec() -> BuildSpec:
    """The spec the committed reference Dockerfile is generated from (golden fixture)."""
    return BuildSpec(
        base=ImageRef.parse("docker.io/library/python@sha256:" + "cafe" * 16),
        repository="ghcr.io/astro-mine/astro-mine-sim",
        version="0.1.0",
        entrypoint=["python", "-m", "astro_mine.sim"],
        source_date_epoch=0,
        revision="cafe" * 10,
        # An OCI `image.source` is defined as a *URL*, so this is one of the few places the
        # link sweep must not shorten to a bare name -- and it points at the repository that
        # ships the code now, not the archived one it was extracted from.
        labels={
            "org.opencontainers.image.source": "https://github.com/astro-mine/astro-mine-platform"
        },
    )


def test_render_reproduces_the_committed_reference_dockerfile() -> None:
    assert render_dockerfile(_reference_spec()) == TEMPLATE_PATH.read_text()


def test_render_pins_base_epoch_entrypoint_and_nonroot() -> None:
    spec = _reference_spec()
    rendered = render_dockerfile(spec)
    assert f"FROM {spec.base.reference}" in rendered
    assert "ARG SOURCE_DATE_EPOCH=0" in rendered
    assert 'ENTRYPOINT ["python", "-m", "astro_mine.sim"]' in rendered
    assert "USER 65532:65532" in rendered


def test_render_labels_are_sorted_and_caller_labels_win() -> None:
    spec = BuildSpec(
        base=ImageRef.parse("base@sha256:" + "11" * 32),
        repository="repo",
        version="9",
        entrypoint=["run"],
        title="custom-title",
        labels={"org.opencontainers.image.title": "overridden", "zzz.custom": "x"},
    )
    rendered = render_dockerfile(spec)
    # caller labels override the derived defaults (title here), and appear sorted last.
    assert 'org.opencontainers.image.title="overridden"' in rendered
    assert 'zzz.custom="x"' in rendered
    labels_only = [ln for ln in rendered.splitlines() if "opencontainers" in ln or "zzz" in ln]
    keys = [ln.split("=")[0].strip().removeprefix("LABEL ") for ln in labels_only]
    assert keys == sorted(keys)


class _FakeBuilder:
    """Records what it was asked to build and returns a fixed digest."""

    def __init__(self, digest: str) -> None:
        self.digest = digest
        self.calls: list[dict[str, object]] = []

    def build(
        self,
        *,
        context: str,
        dockerfile: str,
        tags: Sequence[str],
        build_args: Mapping[str, str],
    ) -> str:
        self.calls.append(
            {
                "context": context,
                "dockerfile": dockerfile,
                "tags": list(tags),
                "args": dict(build_args),
            }
        )
        return self.digest


def test_build_image_returns_pinned_ref_and_passes_source_date_epoch() -> None:
    builder = _FakeBuilder("sha256:" + "dd" * 32)
    spec = _reference_spec()
    ref = build_image(
        spec, context="ctx", tags=["ghcr.io/astro-mine/astro-mine-sim:0.1.0"], builder=builder
    )

    assert ref.repository == spec.repository
    assert ref.digest == "sha256:" + "dd" * 32
    assert ref.tag == spec.version

    (call,) = builder.calls
    assert call["context"] == "ctx"
    assert call["args"] == {"SOURCE_DATE_EPOCH": "0"}
    assert call["dockerfile"] == render_dockerfile(spec)


def test_entrypoint_is_json_encoded() -> None:
    spec = BuildSpec(
        base=ImageRef.parse("b@sha256:" + "22" * 32),
        repository="r",
        version="1",
        entrypoint=["a", "b c", "d"],
    )
    rendered = render_dockerfile(spec)
    assert f"ENTRYPOINT {json.dumps(['a', 'b c', 'd'])}" in rendered
