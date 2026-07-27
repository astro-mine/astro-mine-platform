"""Contract stubs: each Bench subpackage exists and its placeholder is wired.

Pins the public surface and the not-yet-implemented entry points so the backlog has a
concrete starting point. Replace the ``NotImplementedError`` checks with real behavior
as each RM-P0-BENCH-* item lands.
"""

from __future__ import annotations

from pathlib import Path

import astro_mine.bench.eval as bench_eval
from astro_mine.bench import baseline, harness, leaderboard, metrics, recording, scenario


def test_subpackages_import() -> None:
    for module in (scenario, metrics, harness, baseline, leaderboard, recording, bench_eval):
        assert module is not None


def test_local_scoring_path_and_baseline_are_public() -> None:
    # RM-P0-BENCH-05: the scoring path + baseline policy are wired (no longer stubbed).
    assert callable(baseline.run)
    assert hasattr(baseline.BaselinePolicy(), "decide")


def test_bench_never_imports_sim() -> None:
    # bench.md §2.2 / RM-P1-BENCH-10: Bench composes, it does not own engines. Importing the full
    # leaderboard surface (including Hub-digest intake) must not pull Sim into the process.
    # Run in a fresh interpreter: in astro-mine-platform Sim is co-installed, so the parent
    # pytest process may legitimately have it loaded from *other* components' tests.
    import subprocess
    import sys

    probe = (
        "import sys\n"
        "import astro_mine.bench.leaderboard\n"
        "assert not any(n == 'astro_mine.sim' or n.startswith('astro_mine.sim.')"
        " for n in sys.modules)\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_hub_client_is_the_only_new_cross_component_dependency() -> None:
    # RM-P1-BENCH-10: community submissions resolve from Hub by digest through the Hub *client* —
    # never a private Hub schema, never Sim (bench.md §2.2, §6).
    assert callable(leaderboard.open_registry)  # lazily opens astro_mine.hub.registry.Registry
    assert callable(leaderboard.resolve_submission)


def test_the_hub_seam_offers_no_unverified_byte_route() -> None:
    # bench#44: Bench's structural Hub seam must not declare a raw blob read. Such a call returns
    # bytes no manifest vouched for, and Bench hands the registry to a deployment-injected loader
    # that *does* consume bytes — an ONNX policy, a payload-layer metric. The one door those loaders
    # get is `pull_verified_layer`, which re-hashes the layer against the verified manifest and
    # refuses a digest that manifest does not commit to (hub.md §2.3; conventions.md §9).
    from astro_mine.bench.leaderboard import HubRegistry as PolicyIntakeRegistry
    from astro_mine.bench.metrics import HubRegistry as MetricIntakeRegistry

    for protocol in (PolicyIntakeRegistry, MetricIntakeRegistry):
        assert not hasattr(protocol, "pull_blob"), (
            f"{protocol.__module__}.HubRegistry must not advertise an unverified byte route"
        )
    assert callable(leaderboard.pull_verified_layer)
    assert callable(metrics.pull_verified_layer)


def test_the_sandbox_is_dependency_clean() -> None:
    # bench#30: the seccomp filter is hand-built through ctypes and the container backend shells
    # out to the runtime, so the submission sandbox pulls in **no** sandboxing library — the base
    # package stays core + pydantic and the offline local tier is untouched (CX-LOCAL).
    import sys

    import astro_mine.bench.sandbox  # noqa: F401

    for forbidden in ("seccomp", "docker", "wasmtime", "gvisor"):
        assert not any(name.split(".")[0] == forbidden for name in sys.modules), (
            f"the sandbox must not import {forbidden!r}: the base package is core + pydantic"
        )


def test_telemetry_is_optional_not_required() -> None:
    # bench#32: OTel and prometheus_client ride the [observability] extra.
    # `astro_mine.bench.telemetry` probes for both and degrades to no-op shims, so instrumented code
    # needs no feature flags and the local tier runs with neither (conventions.md §7, tier 1).
    from astro_mine.bench import telemetry

    assert callable(telemetry.otel_available)
    assert callable(telemetry.prometheus_available)
    # The instrumentation surface is importable regardless of whether the SDKs are.
    assert callable(telemetry.span)
    assert callable(telemetry.metrics_exposition)


def test_bench_ships_no_crypto_of_its_own() -> None:
    # bench#29 / RFC-0005: Seal is the platform's single home for `cryptography`. Bench reuses its
    # cosign/SLSA/SBOM verify primitives through the Hub client rather than re-implementing signing,
    # so a signature bug can only ever be fixed in one place.
    from astro_mine.bench.leaderboard import _supply_chain

    source = Path(_supply_chain.__file__).read_text(encoding="utf-8")
    assert "from astro_mine.hub.supply_chain import" in source
    assert "import cryptography" not in source
    assert _supply_chain.REQUIRED_EVIDENCE == ("signature", "slsa", "sbom")


def test_the_zoo_catalog_default_needs_no_database() -> None:
    # bench#33 AC3: the filesystem scan stays the tier-1 offline default; pgvector is opt-in.
    from astro_mine.bench.zoo import FilesystemCatalog, default_catalog

    assert isinstance(default_catalog({}), FilesystemCatalog)
