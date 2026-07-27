"""Characterization tests for the leaderboard submission client (``astro_mine.bench.submit``).

The original suite (``test_submit.py``) drove the client against the real FastAPI app through
Starlette's TestClient; astro-mine-platform did not migrate the REST route module, so these pin the
client's **current** behavior against an ``httpx.MockTransport`` stub instead: the exact request
shapes it emits (method, path, headers including the bearer token, JSON body) and how it handles
every response class, including the error paths. Nothing here touches a network.

The properties that matter most are the security ones: identity comes from the verified token and
from nothing a caller can set, and the token rides only the Authorization header of writes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from astro_mine.bench.leaderboard import JobRecord, SubmissionStatus
from astro_mine.bench.submit import (
    DEFAULT_POLL_INTERVAL_S,
    DEFAULT_WAIT_TIMEOUT_S,
    TOKEN_ENV,
    SubmitError,
    await_job,
    get_job,
    get_submission,
    is_rejected,
    poll_job,
    rank_of,
    read_token,
    submit_hub,
    submit_policy,
)
from tests.bench._factories import BASELINE_REF

_SCENARIO = "lunar-polar-ice-prospecting-v1"
_TOKEN = "a-perfectly-ordinary-bearer-token"
_DIGEST = "sha256:" + "a" * 64


def _transport(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://board")


def _capture(
    status: int = 200, body: Any = None
) -> tuple[httpx.Client, list[httpx.Request]]:
    """A stub client that records every request and answers with one canned response."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=body if body is not None else {})

    return _transport(handler), seen


def _scored_payload() -> dict[str, Any]:
    """A valid Submission wire document, as the service would return it."""
    return {
        "submission_id": "sha256:" + "c" * 64,
        "scenario_id": _SCENARIO,
        "policy_ref": BASELINE_REF,
        "method": None,
        "author": None,
        "scorecard_hash": "sha256:" + "d" * 64,
        "runner": "fixture/0.1.0",
        "integrity": "verified",
        "scores": [
            {
                "metric": "water_mass",
                "unit": "kg",
                "direction": "higher_better",
                "aggregation": "mean",
                "value": 4.0,
                "dispersion": 0.0,
                "n": 1,
            }
        ],
    }


def _row(rank: int, submission_id: str, value: float) -> dict[str, Any]:
    return {
        "rank": rank,
        "submission_id": submission_id,
        "method": None,
        "author": None,
        "integrity": "verified",
        "primary_metric": "water_mass",
        "primary_value": value,
        "primary_unit": "kg",
        "source": None,
        "provenance_hash": None,
    }


# --- token handling ------------------------------------------------------------------------------


def test_a_token_file_is_read_and_stripped(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("  a-token\n", encoding="utf-8")
    assert read_token(token_file) == "a-token"


def test_an_empty_token_file_is_refused(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("   \n", encoding="utf-8")
    with pytest.raises(SubmitError, match="empty"):
        read_token(token_file)


def test_an_unreadable_token_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(SubmitError, match="cannot read token file"):
        read_token(tmp_path / "does-not-exist")


def test_the_environment_supplies_the_token_when_no_file_is_given() -> None:
    assert read_token(None, {TOKEN_ENV: " env-token "}) == "env-token"


def test_a_missing_token_is_an_actionable_message() -> None:
    """The failure happens before any request goes out, and names where a token belongs."""
    with pytest.raises(SubmitError) as caught:
        read_token(None, {})
    assert TOKEN_ENV in str(caught.value)
    assert "score" in str(caught.value)  # points out that reading needs no account


def test_a_token_file_wins_over_the_environment(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("file-token", encoding="utf-8")
    assert read_token(token_file, {TOKEN_ENV: "env-token"}) == "file-token"


# --- request shapes ------------------------------------------------------------------------------


def test_submit_policy_request_shape(tmp_path: Path) -> None:
    """POST /submissions with exactly the display metadata and the token on the auth header."""
    http, seen = _capture(200, _scored_payload())
    submission = submit_policy(
        "http://board",
        scenario_id=_SCENARIO,
        policy_ref=BASELINE_REF,
        token=_TOKEN,
        method="MAPPO",
        author="lab-1",
        http=http,
    )
    (request,) = seen
    assert request.method == "POST"
    assert str(request.url) == "http://board/submissions"  # injected client owns the base_url
    assert request.headers["authorization"] == f"Bearer {_TOKEN}"
    body = json.loads(request.content)
    assert body == {
        "scenario_id": _SCENARIO,
        "policy_ref": BASELINE_REF,
        "method": "MAPPO",
        "author": "lab-1",
    }
    assert "identity" not in body  # identity is the token's, and only the token's
    assert submission.submission_id == "sha256:" + "c" * 64
    assert submission.integrity == "verified"


def test_submit_policy_omits_unset_display_metadata() -> None:
    http, seen = _capture(200, _scored_payload())
    submit_policy(
        "http://board", scenario_id=_SCENARIO, policy_ref=BASELINE_REF, token=_TOKEN, http=http
    )
    assert set(json.loads(seen[0].content)) == {"scenario_id", "policy_ref"}


def test_submit_hub_request_shape() -> None:
    """POST /submissions/hub carrying the Hub reference; the response parses to a JobRecord."""
    http, seen = _capture(200, {"job_id": "j1", "status": "queued"})
    job = submit_hub(
        "http://board", scenario_id=_SCENARIO, hub_ref=_DIGEST, token=_TOKEN, http=http
    )
    (request,) = seen
    assert request.method == "POST"
    assert str(request.url) == "http://board/submissions/hub"
    assert request.headers["authorization"] == f"Bearer {_TOKEN}"
    assert json.loads(request.content) == {"scenario_id": _SCENARIO, "hub_ref": _DIGEST}
    assert job.job_id == "j1"
    assert job.status is SubmissionStatus.QUEUED


def test_submit_hub_carries_optional_metadata() -> None:
    http, seen = _capture(200, {"job_id": "j1", "status": "queued"})
    submit_hub(
        "http://board",
        scenario_id=_SCENARIO,
        hub_ref=_DIGEST,
        token=_TOKEN,
        method="acme-v1",
        author="lab-1",
        http=http,
    )
    body = json.loads(seen[0].content)
    assert body["method"] == "acme-v1" and body["author"] == "lab-1"


def test_read_paths_carry_no_token() -> None:
    """Leaderboard and job reads are account-free — the client sends no Authorization header."""
    http, seen = _capture(200, {"job_id": "j1", "status": "queued"})
    get_job("http://board", "j1", http=http)
    (request,) = seen
    assert request.method == "GET"
    assert str(request.url) == "http://board/jobs/j1"
    assert "authorization" not in request.headers


def test_get_submission_request_shape() -> None:
    http, seen = _capture(200, _scored_payload())
    submission = get_submission("http://board", "sha256:" + "c" * 64, http=http)
    assert str(seen[0].url) == "http://board/submissions/sha256:" + "c" * 64
    assert "authorization" not in seen[0].headers
    assert submission.scenario_id == _SCENARIO


# --- failure surfacing ---------------------------------------------------------------------------


def test_a_bad_token_is_reported_as_a_token_problem() -> None:
    http, _ = _capture(401, {"detail": "invalid token"})
    with pytest.raises(SubmitError, match=TOKEN_ENV):
        submit_policy(
            "http://board", scenario_id=_SCENARIO, policy_ref=BASELINE_REF, token="stale",
            http=http,
        )


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (403, {"detail": "namespace not permitted"}, "namespace not permitted"),
        (404, {"detail": "no such scenario"}, "not found"),
        (404, {}, "unknown scenario or job"),  # the 404 fallback wording
        (429, {"detail": "quota exhausted"}, "rate limited"),
        (500, {"detail": "boom"}, r"failed \(500\)"),
        (503, {"detail": "down"}, r"failed \(503\)"),  # a 503 off the hub path is just an error
    ],
)
def test_error_statuses_are_explained_not_echoed(
    status: int, body: dict[str, str], expected: str
) -> None:
    http, _ = _capture(status, body)
    with pytest.raises(SubmitError, match=expected):
        submit_policy(
            "http://board", scenario_id=_SCENARIO, policy_ref=BASELINE_REF, token=_TOKEN,
            http=http,
        )


def test_a_503_on_the_hub_intake_says_what_is_actually_wrong() -> None:
    """A 503 on /submissions/hub means "no Hub registry wired" — retrying will not change that."""
    http, _ = _capture(503, {"detail": "Hub-digest intake is not configured"})
    with pytest.raises(SubmitError, match="not configured on this deployment") as caught:
        submit_hub(
            "http://board", scenario_id=_SCENARIO, hub_ref=_DIGEST, token=_TOKEN, http=http
        )
    assert "--policy-ref" in str(caught.value)  # names the alternative


def test_a_non_json_error_body_is_still_reported() -> None:
    """A proxy or gateway error is not JSON; the caller still sees what came back."""
    http = _transport(lambda _r: httpx.Response(502, text="upstream is down"))
    with pytest.raises(SubmitError, match="upstream is down"):
        submit_policy(
            "http://board", scenario_id=_SCENARIO, policy_ref=BASELINE_REF, token=_TOKEN,
            http=http,
        )


def test_an_unreachable_board_is_not_a_traceback() -> None:
    def refuse(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(SubmitError, match="could not reach the leaderboard"):
        submit_policy(
            "http://board", scenario_id=_SCENARIO, policy_ref=BASELINE_REF, token=_TOKEN,
            http=_transport(refuse),
        )


def test_an_unreachable_board_on_a_read_is_not_a_traceback() -> None:
    def refuse(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(SubmitError, match="could not reach the leaderboard"):
        rank_of("http://board", _SCENARIO, "sha256:x", http=_transport(refuse))


def test_a_read_path_404_is_explained() -> None:
    """Reads carry no token, so a 404 means the job or board is genuinely absent."""
    http, _ = _capture(404, {"detail": "no job"})
    with pytest.raises(SubmitError, match="not found"):
        await_job("http://board", "missing", http=http, sleep=lambda _s: None)


# --- following a job to a verdict ----------------------------------------------------------------


def test_poll_job_yields_each_state_and_sleeps_between_polls() -> None:
    states = iter(
        [
            {"job_id": "j1", "status": "queued"},
            {"job_id": "j1", "status": "running"},
            {"job_id": "j1", "status": "ranked", "result_id": "sha256:abc"},
        ]
    )
    sleeps: list[float] = []

    def serve(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(states))

    jobs = list(
        poll_job("http://board", "j1", http=_transport(serve), sleep=sleeps.append)
    )
    assert [job.status.value for job in jobs] == ["queued", "running", "ranked"]
    assert sleeps == [DEFAULT_POLL_INTERVAL_S, DEFAULT_POLL_INTERVAL_S]  # none after terminal


def test_await_job_returns_the_terminal_record() -> None:
    states = iter(
        [
            {"job_id": "j1", "status": "queued"},
            {"job_id": "j1", "status": "ranked", "result_id": "sha256:abc"},
        ]
    )
    job = await_job(
        "http://board",
        "j1",
        http=_transport(lambda _r: httpx.Response(200, json=next(states))),
        sleep=lambda _s: None,
    )
    assert job.status is SubmissionStatus.RANKED
    assert job.result_id == "sha256:abc"


def test_a_job_that_never_finishes_says_it_is_not_lost() -> None:
    """The timeout message tells the user how to resume, rather than implying the job died."""
    http, _ = _capture(200, {"job_id": "j1", "status": "queued"})
    ticks = iter([0.0, 0.0, 999.0, 999.0])
    with pytest.raises(SubmitError, match="not lost") as caught:
        await_job(
            "http://board",
            "j1",
            http=http,
            sleep=lambda _s: None,
            now=lambda: next(ticks),
            timeout_s=1.0,
        )
    assert "--job j1" in str(caught.value)  # the resume command, spelled out


def test_rank_of_finds_the_submission_on_the_board() -> None:
    rows = [_row(1, "sha256:other", 9.0), _row(2, "sha256:mine", 4.0)]
    http, seen = _capture(200, rows)
    entry = rank_of("http://board", _SCENARIO, "sha256:mine", http=http)
    assert entry is not None and entry.rank == 2
    assert str(seen[0].url) == f"http://board/leaderboard/{_SCENARIO}"
    assert rank_of("http://board", _SCENARIO, "sha256:absent", http=http) is None


def test_rank_of_tolerates_a_non_list_board_payload() -> None:
    """A dict payload (e.g. an error shape that slipped through) yields None, not a crash."""
    http, _ = _capture(200, {"unexpected": "shape"})
    assert rank_of("http://board", _SCENARIO, "sha256:mine", http=http) is None


def test_is_rejected_maps_terminal_statuses() -> None:
    """`rejected` and `flagged` both mean the submission never made the board; `ranked` did."""
    assert is_rejected(JobRecord(job_id="j", status=SubmissionStatus.REJECTED))
    assert is_rejected(JobRecord(job_id="j", status=SubmissionStatus.FLAGGED))
    assert not is_rejected(JobRecord(job_id="j", status=SubmissionStatus.RANKED))
    assert not is_rejected(JobRecord(job_id="j", status=SubmissionStatus.QUEUED))


def test_poll_defaults_are_pinned() -> None:
    """The CLI's --wait cadence and ceiling ride these module constants."""
    assert DEFAULT_POLL_INTERVAL_S == 2.0
    assert DEFAULT_WAIT_TIMEOUT_S == 3600.0
