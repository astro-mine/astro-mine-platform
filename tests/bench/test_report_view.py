"""View data + MCAP replay export — the View-handoff surface (RM-P1-BENCH-12; bench.md §6).

Bench provides the data and MCAP replays; View renders them. These tests cover Bench's half: the
full per-metric leaderboard dataset (:func:`export_leaderboard`), the decoded replay manifest over a
golden Sim MCAP (:func:`replay_manifest`), and the FastAPI edge that serves both plus the raw replay
bytes. View itself is a separate, not-yet-existent repo (a co-dependency with RM-P1-STUDIO-06), so
only the Bench-provided surface is exercised here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from astro_mine.bench.leaderboard import (
    InMemoryObjectStore,
    LeaderboardService,
    MetricScore,
    OidcTokenVerifier,
    Submission,
    create_app,
)
from astro_mine.bench.leaderboard._objects import blob_digest
from astro_mine.bench.report import (
    ViewLeaderboard,
    ViewReplay,
    export_leaderboard,
    replay_manifest,
)
from astro_mine.bench.sandbox import SandboxScorer
from astro_mine.bench.zoo import ANCHOR_SCENARIO_ID
from tests.bench._factories import InProcessSandbox, make_idp

_GOLDEN = Path(__file__).parent / "data" / "anchor-lunar-polar-ice-prospecting-v1.mcap"
BASELINE_ENTRYPOINT = "tests.bench._factories:BASELINE_INSTANCE"


def _submission(
    submission_id: str, primary_value: float, *, runner: str = "fixture/0.1.0"
) -> Submission:
    """A submission with a full two-metric scorecard (primary higher-better + a lower-better)."""
    return Submission(
        submission_id=submission_id,
        scenario_id=ANCHOR_SCENARIO_ID,
        policy_ref="mod:policy",
        method="method",
        author="author",
        scorecard_hash="sha256:" + "0" * 64,
        runner=runner,
        integrity="verified",
        scores=(
            MetricScore(
                metric="water_mass",
                unit="kg",
                direction="higher_better",
                aggregation="mean",
                value=primary_value,
                dispersion=1.5,
                n=5,
            ),
            MetricScore(
                metric="energy_per_kg",
                unit="J/kg",
                direction="lower_better",
                aggregation="mean",
                value=100.0,
                dispersion=None,
                n=5,
            ),
        ),
    )


# --- export_leaderboard: the full-metric dataset ------------------------------------------------


def test_export_leaderboard_carries_full_per_metric_rows() -> None:
    board = export_leaderboard(ANCHOR_SCENARIO_ID, [_submission("a", 10.0), _submission("b", 20.0)])
    assert isinstance(board, ViewLeaderboard)
    assert board.primary_metric == "water_mass"
    # higher_better primary ⇒ b (20) ranks above a (10).
    assert [row.submission_id for row in board.rows] == ["b", "a"]
    assert [row.rank for row in board.rows] == [1, 2]
    # Each row carries EVERY metric (not just the primary), with uncertainty preserved.
    top = board.rows[0]
    assert [s.metric for s in top.scores] == ["water_mass", "energy_per_kg"]
    assert top.scores[0].dispersion == 1.5  # the uncertainty bound View surfaces


def test_export_leaderboard_carries_runner_provenance() -> None:
    """Each row surfaces its runner faithfully — so View can label a fixture score (G1.1/G1.8)."""
    board = export_leaderboard(
        ANCHOR_SCENARIO_ID,
        [
            _submission("a", 10.0, runner="fixture/0.1.0"),
            _submission("b", 20.0, runner="sim/1.2.0"),
        ],
    )
    by_id = {row.submission_id: row.runner for row in board.rows}
    assert by_id == {"a": "fixture/0.1.0", "b": "sim/1.2.0"}


def test_export_leaderboard_empty() -> None:
    board = export_leaderboard(ANCHOR_SCENARIO_ID, [])
    assert board.primary_metric is None
    assert board.rows == ()


# --- replay_manifest: decode a golden Sim MCAP --------------------------------------------------


def test_replay_manifest_summarizes_a_golden_recording() -> None:
    mcap = _GOLDEN.read_bytes()
    manifest = replay_manifest(mcap, scenario_id=ANCHOR_SCENARIO_ID, submission_id="sub-1")
    assert isinstance(manifest, ViewReplay)
    assert manifest.submission_id == "sub-1"
    assert manifest.mcap_digest == blob_digest(mcap)
    assert manifest.size_bytes == len(mcap)
    assert manifest.agents == ("relay", "rover")  # the golden fixture's two agents
    assert manifest.seed == 1001
    assert manifest.observation_count > 0
    assert manifest.frame_count > 0
    assert manifest.sim_time_start_s is not None


# --- the FastAPI View endpoints -----------------------------------------------------------------


#: A throwaway IdP for the authenticated write path (bench#29). The View *read* endpoints below
#: stay account-free — the token is only needed to put a submission on the board in the first place.
_IDP = make_idp()


def _service_and_client() -> tuple[LeaderboardService, TestClient]:
    service = LeaderboardService(
        object_store=InMemoryObjectStore(),
        authn=OidcTokenVerifier(issuer=_IDP.issuer, audience=_IDP.audience, jwks=_IDP.jwks),
        scorer=SandboxScorer(InProcessSandbox()),
    )
    return service, TestClient(create_app(service=service))


def _submit_baseline(client: TestClient) -> str:
    response = client.post(
        "/submissions",
        json={"scenario_id": ANCHOR_SCENARIO_ID, "policy_ref": BASELINE_ENTRYPOINT},
        headers=_IDP.header(),
    )
    assert response.status_code == 200, response.text
    return response.json()["submission_id"]


@pytest.mark.skip(
    reason="REST surface not migrated: astro-mine-platform ships the leaderboard "
    "library but not the FastAPI route module (astro_mine.bench.leaderboard._app)"
)
def test_scorecards_endpoint_returns_full_metric_rows() -> None:
    _, client = _service_and_client()
    submission_id = _submit_baseline(client)
    response = client.get(f"/leaderboard/{ANCHOR_SCENARIO_ID}/scorecards")
    assert response.status_code == 200
    board = ViewLeaderboard.model_validate(response.json())
    assert [row.submission_id for row in board.rows] == [submission_id]
    assert len(board.rows[0].scores) == len(load_anchor_metric_count())


def load_anchor_metric_count() -> tuple[str, ...]:
    from astro_mine.bench.zoo import load_scenario

    return tuple(ref.name for ref in load_scenario(ANCHOR_SCENARIO_ID).metrics)


@pytest.mark.skip(
    reason="REST surface not migrated: astro-mine-platform ships the leaderboard "
    "library but not the FastAPI route module (astro_mine.bench.leaderboard._app)"
)
def test_replay_endpoints_serve_attached_mcap() -> None:
    service, client = _service_and_client()
    submission_id = _submit_baseline(client)
    mcap = _GOLDEN.read_bytes()
    service.attach_replay(submission_id, mcap)

    raw = client.get(f"/submissions/{submission_id}/replay")
    assert raw.status_code == 200
    assert raw.headers["content-type"] == "application/octet-stream"
    assert raw.content == mcap

    manifest = client.get(f"/submissions/{submission_id}/replay/manifest")
    assert manifest.status_code == 200
    assert ViewReplay.model_validate(manifest.json()).seed == 1001

    # The full-metric row now advertises the attached replay by digest.
    board = ViewLeaderboard.model_validate(
        client.get(f"/leaderboard/{ANCHOR_SCENARIO_ID}/scorecards").json()
    )
    assert board.rows[0].trace_hash == blob_digest(mcap)


@pytest.mark.skip(
    reason="REST surface not migrated: astro-mine-platform ships the leaderboard "
    "library but not the FastAPI route module (astro_mine.bench.leaderboard._app)"
)
def test_replay_missing_returns_404() -> None:
    _, client = _service_and_client()
    submission_id = _submit_baseline(client)
    # No replay attached yet.
    assert client.get(f"/submissions/{submission_id}/replay").status_code == 404
    assert client.get(f"/submissions/{submission_id}/replay/manifest").status_code == 404
    # Unknown submission.
    assert client.get("/submissions/nope/replay").status_code == 404
