"""The leaderboard submission client — the CLI's write path (G2.14; RM-P1-BENCH-10).

The leaderboard service ships complete: intake, sandboxed execution, OIDC auth, ranking. What it
had no client for was *submitting*, so the last step of the Phase-1 flywheel — "put my policy on
the board" — was ``curl`` with a hand-assembled JSON body and a bearer token. This is that client.

**Two intakes, deliberately not equals.**

- :func:`submit_hub` posts a **Hub reference** and is the path a community submission should take.
  ``bench.md`` §6 states the contract: *"a leaderboard submission references Hub artifacts by
  digest; Bench resolves and verifies them through Hub"*. The artifact is authenticated by content
  hash and signature, so the entry stays reproducible.
- :func:`submit_policy` posts an importable ``module:attribute`` reference. It ships and it is the
  offline/dev path, but **nothing pins what it resolves to** — a re-run can import different code
  under the same name. It appears in no architecture document (`bench.md` §6, `system.md` §7, and
  `hub.md` §6 all describe the intake as digest-referenced), so it is documented here as *not
  leaderboard-grade* rather than as a peer of the Hub path.

**Identity is never a parameter.** It comes from the verified OIDC bearer token and nothing else;
rate limits, quotas, job tickets, and audit records are all keyed on it (bench#29). Neither request
model has an ``identity`` field — the pre-bench#29 model did, and a submitter could reset their own
quota by editing a JSON field. No function here accepts one, and the token is never logged, echoed,
or persisted.

``httpx`` lives in the optional ``[submit]`` extra: the base package stays install-light, and
``score``/``list`` keep working offline with no account and no new dependency (CX-LOCAL).
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astro_mine.bench.leaderboard._jobs import TERMINAL_STATUSES, JobRecord, SubmissionStatus
from astro_mine.bench.leaderboard._models import LeaderboardEntry, Submission

if TYPE_CHECKING:  # `httpx` is the [submit] extra; annotations must not require it at import
    import httpx

_INSTALL_HINT = (
    "`submit` needs an HTTP client, which is not in the base package: install it with "
    "`uv sync --extra submit` (or `pip install 'astro-mine-platform[bench-submit]'`)"
)


def _httpx() -> Any:
    """The HTTP client module, or an actionable install hint (CX-LOCAL) — the `[fetch]` shape.

    Imported here rather than at module scope so this module — and therefore the CLI's `submit`
    parser and its `$ASTRO_MINE_BENCH_TOKEN` constant — imports on a base install.
    """
    try:
        import httpx
    except ModuleNotFoundError as exc:  # pragma: no cover - trivial re-raise wrapper
        raise SubmitError(_INSTALL_HINT) from exc
    return httpx


__all__ = [
    "DEFAULT_POLL_INTERVAL_S",
    "DEFAULT_WAIT_TIMEOUT_S",
    "TOKEN_ENV",
    "SubmitError",
    "await_job",
    "rank_of",
    "read_token",
    "submit_hub",
    "submit_policy",
]

#: Where the bearer token is read from. An environment variable rather than a flag: a token on the
#: command line lands in shell history and in `ps` output for every process on the box.
TOKEN_ENV = "ASTRO_MINE_BENCH_TOKEN"

#: Polling cadence and ceiling for ``--wait``. A hosted evaluation is minutes-to-low-hours
#: (bench.md §8), so the default ceiling is generous and the caller can raise it.
DEFAULT_POLL_INTERVAL_S = 2.0
DEFAULT_WAIT_TIMEOUT_S = 3600.0


class SubmitError(Exception):
    """A submission could not be made, or could not be followed to a verdict.

    Always carries something the caller can act on — never a bare status code.
    """


def read_token(
    token_file: str | Path | None = None, environ: Mapping[str, str] | None = None
) -> str:
    """Resolve the bearer token from ``token_file`` else the environment, fail-closed.

    Raises :class:`SubmitError` naming where to put a token rather than letting the request go out
    unauthenticated and returning a 401 traceback.
    """
    if token_file is not None:
        try:
            token = Path(token_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise SubmitError(
                f"cannot read token file {token_file}: {exc.strerror or exc}"
            ) from exc
        if not token:
            raise SubmitError(f"token file {token_file} is empty")
        return token

    env = os.environ if environ is None else environ
    token = (env.get(TOKEN_ENV) or "").strip()
    if not token:
        raise SubmitError(
            f"no bearer token: set ${TOKEN_ENV} (or pass --token-file). Submitting is the one "
            f"leaderboard action that needs an account; `score` and `list` do not"
        )
    return token


def _client(http: httpx.Client | None, base_url: str) -> tuple[Any, bool]:
    """The caller's client, or one we own — the seam that keeps tests off the network."""
    if http is not None:
        return http, False
    return _httpx().Client(base_url=base_url, timeout=30.0), True


def _headers(token: str) -> dict[str, str]:
    """The one place the token touches a request. Never logged, never echoed."""
    return {"Authorization": f"Bearer {token}"}


def _explain(response: Any, *, endpoint: str) -> str:
    """Turn a failed response into something the caller can act on."""
    try:
        detail = response.json().get("detail")
    except Exception:  # a non-JSON error body is still worth reporting verbatim
        detail = response.text.strip() or None

    if response.status_code == 401:
        return (
            f"the leaderboard rejected the bearer token (401). Check ${TOKEN_ENV} is a current "
            f"token for this deployment"
        )
    if response.status_code == 403:
        return f"the leaderboard refused this submission (403): {detail or 'not permitted'}"
    if response.status_code == 404:
        return f"{endpoint} not found (404): {detail or 'unknown scenario or job'}"
    if response.status_code == 429:
        return f"rate limited (429): {detail or 'too many submissions'}"
    if response.status_code == 503 and endpoint.endswith("/submissions/hub"):
        # The deployment has no Hub registry wired; retrying will not change that.
        return (
            "Hub-digest intake is not configured on this deployment (503). Submit against a "
            "deployment with a Hub registry, or use --policy-ref for a local/dev submission"
        )
    return f"{endpoint} failed ({response.status_code}): {detail or response.reason_phrase}"


def _post(
    base_url: str, path: str, payload: dict[str, Any], *, token: str, http: httpx.Client | None
) -> dict[str, Any]:
    client, owned = _client(http, base_url)
    try:
        try:
            response = client.post(
                _url(base_url, path, http), json=payload, headers=_headers(token)
            )
        except _httpx().HTTPError as exc:  # unreachable host, TLS failure, timeout
            raise SubmitError(f"could not reach the leaderboard at {base_url}: {exc}") from exc
        if response.status_code >= 400:
            raise SubmitError(_explain(response, endpoint=path))
        return dict(response.json())
    finally:
        if owned:
            client.close()


def _get(base_url: str, path: str, *, http: httpx.Client | None) -> dict[str, Any] | list[Any]:
    """Read paths carry no token — leaderboard and job reads are account-free (bench#29 AC5)."""
    client, owned = _client(http, base_url)
    try:
        try:
            response = client.get(_url(base_url, path, http))
        except _httpx().HTTPError as exc:
            raise SubmitError(f"could not reach the leaderboard at {base_url}: {exc}") from exc
        if response.status_code >= 400:
            raise SubmitError(_explain(response, endpoint=path))
        payload: dict[str, Any] | list[Any] = response.json()
        return payload
    finally:
        if owned:
            client.close()


def _url(base_url: str, path: str, http: httpx.Client | None) -> str:
    """An injected client may carry its own base_url (the ASGI test transport does)."""
    return path if http is not None else f"{base_url.rstrip('/')}{path}"


def submit_hub(
    base_url: str,
    *,
    scenario_id: str,
    hub_ref: str,
    token: str,
    method: str | None = None,
    author: str | None = None,
    http: httpx.Client | None = None,
) -> JobRecord:
    """Submit a community artifact **by Hub reference** — the recommended path.

    ``hub_ref`` is a ``name:version`` tag or a ``sha256:`` image-manifest digest. Bench resolves it
    from Hub and verifies it fail-closed (content address → cosign signature → SLSA provenance →
    SBOM) before running it sandboxed. Returns the :class:`JobRecord` ticket to poll.
    """
    payload: dict[str, Any] = {"scenario_id": scenario_id, "hub_ref": hub_ref}
    if method is not None:
        payload["method"] = method
    if author is not None:
        payload["author"] = author
    return JobRecord.model_validate(
        _post(base_url, "/submissions/hub", payload, token=token, http=http)
    )


def submit_policy(
    base_url: str,
    *,
    scenario_id: str,
    policy_ref: str,
    token: str,
    method: str | None = None,
    author: str | None = None,
    http: httpx.Client | None = None,
) -> Submission:
    """Submit an importable ``module:attribute`` policy — the **local/dev** path.

    Runs under submit-policy-we-run in a sandbox, exactly as the Hub path does. What it does *not*
    give you is a reproducible entry: nothing pins what the reference resolves to, so prefer
    :func:`submit_hub` for anything that should stand as a published result.
    """
    payload: dict[str, Any] = {"scenario_id": scenario_id, "policy_ref": policy_ref}
    if method is not None:
        payload["method"] = method
    if author is not None:
        payload["author"] = author
    return Submission.model_validate(
        _post(base_url, "/submissions", payload, token=token, http=http)
    )


def get_job(base_url: str, job_id: str, *, http: httpx.Client | None = None) -> JobRecord:
    """The current state of a submission job."""
    return JobRecord.model_validate(_get(base_url, f"/jobs/{job_id}", http=http))


def get_submission(
    base_url: str, submission_id: str, *, http: httpx.Client | None = None
) -> Submission:
    """The scored submission a finished job produced."""
    return Submission.model_validate(_get(base_url, f"/submissions/{submission_id}", http=http))


def poll_job(
    base_url: str,
    job_id: str,
    *,
    http: httpx.Client | None = None,
    interval_s: float = DEFAULT_POLL_INTERVAL_S,
    timeout_s: float = DEFAULT_WAIT_TIMEOUT_S,
    sleep: Any = time.sleep,
    now: Any = time.monotonic,
) -> Iterator[JobRecord]:
    """Yield the job's state until it reaches a terminal status, or the deadline passes.

    ``sleep``/``now`` are injected so a test can drive the loop without wall-clock time.
    """
    deadline = now() + timeout_s
    while True:
        job = get_job(base_url, job_id, http=http)
        yield job
        if job.status in TERMINAL_STATUSES:
            return
        if now() >= deadline:
            raise SubmitError(
                f"job {job_id} was still {job.status.value!r} after {timeout_s:.0f}s; it is not "
                f"lost — poll it later with `astro-mine bench submit --job {job_id} --wait`"
            )
        sleep(interval_s)


def await_job(base_url: str, job_id: str, **kwargs: Any) -> JobRecord:
    """Poll until terminal and return the final :class:`JobRecord`."""
    final = None
    for job in poll_job(base_url, job_id, **kwargs):
        final = job
    assert final is not None  # poll_job always yields at least once before returning
    return final


def rank_of(
    base_url: str, scenario_id: str, submission_id: str, *, http: httpx.Client | None = None
) -> LeaderboardEntry | None:
    """This submission's place on the board, or ``None`` if it is not ranked (yet)."""
    rows = _get(base_url, f"/leaderboard/{scenario_id}", http=http)
    for row in rows if isinstance(rows, list) else []:
        entry = LeaderboardEntry.model_validate(row)
        if entry.submission_id == submission_id:
            return entry
    return None


def is_rejected(job: JobRecord) -> bool:
    """Whether a terminal job means the submission never made the board."""
    return job.status in {SubmissionStatus.REJECTED, SubmissionStatus.FLAGGED}
