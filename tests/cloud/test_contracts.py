"""Contract wiring: each Cloud subpackage exists and its public surface is live."""

from __future__ import annotations

from astro_mine.cloud import artifacts, packaging, submission


def test_subpackages_import() -> None:
    for module in (packaging, submission, artifacts):
        assert module is not None


def test_public_entry_points_are_wired() -> None:
    assert callable(submission.submit)
    assert callable(packaging.build_image)
    assert callable(packaging.render_dockerfile)
    # both Phase-0 backends register on import
    assert {"local", "docker"} <= set(submission.registered_backends())
