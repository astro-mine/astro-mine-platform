"""The Bench command line — clone, run, and score a baseline (RM-P0-BENCH-05).

Drives :func:`astro_mine.bench.cli.main` with injected stdout/stderr buffers, covering the
``score`` (default anchor, ``--seeds``, ``--json``) and ``list`` commands and the unknown-scenario
and missing-command error paths.
"""

from __future__ import annotations

import io
import json
import pathlib

import pytest

from astro_mine.bench.baseline import (
    REFERENCE_EPISODE_RUNNER_ID,
    ScoringRefused,
    reference_episode_runner,
)
from astro_mine.bench.cli import main
from astro_mine.bench.content import FetchError
from astro_mine.bench.zoo import ANCHOR_SCENARIO_ID, list_scenarios


def _run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_score_defaults_to_the_anchor() -> None:
    code, out, _ = _run("score")
    assert code == 0
    assert ANCHOR_SCENARIO_ID in out
    assert "water_mass" in out and "scorecard:" in out


def test_score_emits_json() -> None:
    code, out, _ = _run("score", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["scenario_id"] == ANCHOR_SCENARIO_ID
    assert {m["metric"] for m in payload["metrics"]} >= {"water_mass", "discovery_latency"}


def test_score_honours_seed_selection() -> None:
    code, out, _ = _run("score", "--seeds", "1001")
    assert code == 0
    assert ANCHOR_SCENARIO_ID in out


def test_list_prints_the_zoo() -> None:
    code, out, _ = _run("list")
    assert code == 0
    # The zoo lists every registered scenario (sorted), including the anchor and the newer tasks.
    printed = out.strip().splitlines()
    assert printed == list(list_scenarios())
    assert ANCHOR_SCENARIO_ID in printed


def test_unknown_scenario_is_a_clean_error() -> None:
    code, out, err = _run("score", "no-such-scenario")
    assert code == 2
    assert "error" in err and "no-such-scenario" in err
    assert out == ""


# --- runner selection + provenance (G1.1 / G1.8) ------------------------------------------------


def test_score_default_runner_is_the_labelled_fixture() -> None:
    """The default is the fixture, and the output says so unmistakably — not by implication."""
    code, out, _ = _run("score")
    assert code == 0
    assert f"runner:    {REFERENCE_EPISODE_RUNNER_ID}" in out
    assert "deterministic trace fixture, not a physics engine" in out


def test_score_json_names_the_runner() -> None:
    """The machine-readable path — the one that feeds leaderboards and papers — is honest too."""
    code, out, _ = _run("score", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["runner"] == REFERENCE_EPISODE_RUNNER_ID


@pytest.mark.skip(
    reason="sibling-absent state unreachable in astro-mine-platform: Sim ships in the same "
    "distribution, so the 'sim runner not installed' install-hint path cannot occur"
)
def test_score_runner_sim_without_sim_is_a_clean_error() -> None:
    """`--runner sim` with no Sim installed fails with an install hint, not a traceback."""
    code, out, err = _run("score", "--runner", "sim")
    assert code == 2
    assert out == ""
    assert "astro-mine-sim[bench]" in err
    assert "Traceback" not in err


def test_score_unknown_runner_lists_the_registered_ones() -> None:
    code, _, err = _run("score", "--runner", "nope")
    assert code == 2
    assert "unknown runner 'nope'" in err
    assert "fixture" in err  # the built-in is named so the user can recover


def test_score_runner_that_fails_to_start_is_a_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A *registered* runner that can't construct (e.g. the `sim` runner with no content store)
    surfaces the provider's actionable message, not a traceback."""

    class _NeedsAStore:
        runner_id = "sim/0.1.0"

        def episode_runner(self, store: object | None = None) -> object:
            raise RuntimeError(
                "the 'sim' runner needs a content store: set $ASTRO_MINE_HUB_REGISTRY"
            )

        def harness_runner(self, store: object | None = None) -> object:  # pragma: no cover
            raise RuntimeError

    monkeypatch.setattr("astro_mine.bench.cli.load_runner_provider", lambda _name: _NeedsAStore())
    code, out, err = _run("score", "--runner", "sim")
    assert code == 2
    assert out == ""
    assert "needs a content store" in err
    assert "Traceback" not in err


def test_missing_command_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code != 0


# --- the hosted-catalog operator commands (bench#33) ---------------------------------------------


def test_zoo_sync_seeds_the_catalog_from_the_packaged_zoo(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC4: the migration/seed utility that populates the catalog from the zoo's scenario.json."""
    from astro_mine.bench.zoo import CATALOG_DSN_ENV, open_sql_catalog

    dsn = f"sqlite:///{tmp_path / 'zoo.db'}"  # type: ignore[operator]
    monkeypatch.delenv(CATALOG_DSN_ENV, raising=False)

    code, out, _ = _run("zoo-sync", "--dsn", dsn)
    assert code == 0
    assert ANCHOR_SCENARIO_ID in out
    assert "scenario(s) indexed" in out

    catalog = open_sql_catalog(dsn)
    assert catalog.list_scenarios() == list_scenarios()
    catalog.dispose()


def test_zoo_search_ranks_the_catalog(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    from astro_mine.bench.zoo import CATALOG_DSN_ENV

    dsn = f"sqlite:///{tmp_path / 'zoo.db'}"  # type: ignore[operator]
    monkeypatch.setenv(CATALOG_DSN_ENV, dsn)  # the DSN may also come from the environment
    _run("zoo-sync")

    code, out, _ = _run("zoo-search", "lunar", "polar", "ice", "prospecting")
    assert code == 0
    assert "lunar-polar-ice" in out
    assert "distance=" in out


def test_zoo_search_on_an_empty_catalog_says_so(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    from astro_mine.bench.zoo import CATALOG_DSN_ENV

    monkeypatch.setenv(CATALOG_DSN_ENV, f"sqlite:///{tmp_path / 'empty.db'}")  # type: ignore[operator]
    code, out, _ = _run("zoo-search", "ice")
    assert code == 0
    assert "zoo-sync" in out  # tells the operator what to do next


def test_the_catalog_commands_need_a_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    from astro_mine.bench.zoo import CATALOG_DSN_ENV

    monkeypatch.delenv(CATALOG_DSN_ENV, raising=False)
    for command in (("zoo-sync",), ("zoo-search", "ice")):
        code, _, err = _run(*command)
        assert code == 2
        assert CATALOG_DSN_ENV in err


# --- fetch (G1.2; bench#56) --------------------------------------------------------------


def test_fetch_reports_the_store_path_and_the_signer_posture(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The posture is printed, not implied: a reader must not infer signer pinning that did not
    happen (bench#56 D6)."""
    calls: dict[str, object] = {}

    def fake_fetch(spec: object, **kwargs: object) -> tuple[object, ...]:
        calls.update(kwargs)
        return ()

    monkeypatch.setattr("astro_mine.bench.cli.fetch_scenario_content", fake_fetch)
    code, out, _ = _run("fetch", "--registry", str(tmp_path / "store"))

    assert code == 0
    assert str(tmp_path / "store") in out
    assert "signer not pinned" in out
    assert "--runner sim" in out  # the next command is copy-pasteable
    assert calls["source"] == "ghcr.io/astro-mine"


def test_fetch_names_the_pinned_signer_when_a_key_is_given(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = tmp_path / "anchor.pub"
    key.write_bytes(b"-----BEGIN PUBLIC KEY-----\nx\n-----END PUBLIC KEY-----\n")
    monkeypatch.setattr("astro_mine.bench.cli.fetch_scenario_content", lambda spec, **kw: ())

    code, out, _ = _run("fetch", "--registry", str(tmp_path / "s"), "--trusted-key", str(key))
    assert code == 0
    assert "signer pinned" in out and "signer not pinned" not in out


def test_fetch_unreadable_trusted_key_is_a_clean_error(tmp_path: pathlib.Path) -> None:
    code, _, err = _run("fetch", "--trusted-key", str(tmp_path / "missing.pub"))
    assert code == 2
    assert "--trusted-key" in err


def test_fetch_unknown_scenario_is_a_clean_error() -> None:
    code, _, err = _run("fetch", "no-such-scenario")
    assert code == 2
    assert "error:" in err


def test_fetch_failure_is_a_clean_error_not_a_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Including the missing-[fetch]-extra path, which surfaces as a FetchError install hint."""

    def boom(spec: object, **kwargs: object) -> tuple[object, ...]:
        raise FetchError("`fetch` needs the Hub client: uv sync --extra fetch")

    monkeypatch.setattr("astro_mine.bench.cli.fetch_scenario_content", boom)
    code, _, err = _run("fetch")
    assert code == 2
    assert "--extra fetch" in err


def test_score_registry_flag_is_passed_to_the_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--registry` reaches the provider as a plain path — Bench never names an engine type."""
    seen: dict[str, object] = {}

    class _Provider:
        runner_id = REFERENCE_EPISODE_RUNNER_ID

        def episode_runner(self, store: object | None = None) -> object:
            seen["store"] = store
            return reference_episode_runner

    monkeypatch.setattr("astro_mine.bench.cli.load_runner_provider", lambda name: _Provider())
    code, _, _ = _run("score", "--registry", "/tmp/some-store")
    assert code == 0
    assert seen["store"] == "/tmp/some-store"


def test_score_refusal_is_a_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A runner that refuses mid-run reports it as an error, not a traceback (#79).

    `sim` refuses a scenario whose pinned providers did not rebuild — the scorecard would be a
    claim about content it never modelled. The refusal is a designed outcome carrying a message a
    user can act on, so it must read as one.
    """

    class _Refuses:
        runner_id = "sim/0.1.0"

        def episode_runner(self, store: object | None = None) -> object:
            def _runner(resolved: object, policy: object, seed: int) -> object:
                raise ScoringRefused(
                    "refusing to score this scenario: 1 pinned input resolved by digest but "
                    "rebuilt no provider — install astro-mine-worlds"
                )

            return _runner

        def harness_runner(self, store: object | None = None) -> object:  # pragma: no cover
            raise RuntimeError

    monkeypatch.setattr("astro_mine.bench.cli.load_runner_provider", lambda _name: _Refuses())
    code, out, err = _run("score", "--runner", "sim", "--seeds", "1001")

    assert code == 2
    assert out == ""  # no half-printed scorecard
    assert "refusing to score this scenario" in err
    assert "astro-mine-worlds" in err  # the package that fixes it survives into the message
    assert "Traceback" not in err


def test_score_does_not_swallow_a_genuine_runner_bug(monkeypatch: pytest.MonkeyPatch) -> None:
    """The other half of #79: only a refusal is presented cleanly.

    Catching RuntimeError around the run would convert an engine bug into `error: ...` and throw
    away the traceback that makes it debuggable. A bug must still crash.
    """

    class _Buggy:
        runner_id = "sim/0.1.0"

        def episode_runner(self, store: object | None = None) -> object:
            def _runner(resolved: object, policy: object, seed: int) -> object:
                raise RuntimeError("dictionary changed size during iteration")

            return _runner

        def harness_runner(self, store: object | None = None) -> object:  # pragma: no cover
            raise RuntimeError

    monkeypatch.setattr("astro_mine.bench.cli.load_runner_provider", lambda _name: _Buggy())
    with pytest.raises(RuntimeError, match="dictionary changed size"):
        _run("score", "--runner", "sim", "--seeds", "1001")


def test_scoring_refused_is_public_api() -> None:
    """Runners import it to raise it, so it is part of the seam, not an internal detail."""
    from astro_mine.bench import baseline

    assert "ScoringRefused" in baseline.__all__
    assert issubclass(baseline.ScoringRefused, RuntimeError)
