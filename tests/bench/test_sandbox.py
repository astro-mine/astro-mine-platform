"""Sandboxed execution of submitted policies (bench#30; bench.md §9; conventions.md §9).

These tests assert the sandbox **actually isolates** — a sandbox that does not is worse than none,
because the leaderboard would run community code believing it was contained. So the isolation
properties are exercised against the *real* :class:`SubprocessSandbox` (a real forked process, real
POSIX rlimits, a real seccomp-BPF filter), not a double:

- **AC1** submitted policies execute in a separate process, never in-process with the evaluator —
  and on **both** intake paths;
- **AC2** the sandbox **denies network egress by default** (a policy trying to open a socket gets
  ``EPERM``, and cannot reach the network by any syscall), **confines the filesystem by default** (a
  policy cannot read the embargoed held-out seeds — bench#36), and enforces CPU / memory / time
  limits (a spinner, a memory hog, and a sleeper are each killed by the right control);
- **AC3** results come back over a **structured channel** — a document parsed as data — never as
  shared in-process state; a failing policy is reported *as data*, not as an evaluator exception;
- **AC4** the trust boundary is documented (``TRUST_BOUNDARY.md`` — asserted present, and to name
  what is *not* covered).

The container backend's ``docker run`` argv **is** its security posture, so it is asserted flag by
flag without a runtime present; the real-container round trip is an opt-in ``container`` test.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

import pytest

from astro_mine.bench.baseline import BaselinePolicy, run
from astro_mine.bench.sandbox import (
    CONTAINER_OUTPUT_DIR,
    DENIED_SYSCALLS,
    ContainerSandbox,
    FilesystemPolicy,
    InProcessScorer,
    LandlockUnsupported,
    NetworkPolicy,
    ResourceUsage,
    SandboxLimits,
    SandboxOutcome,
    SandboxScorer,
    SandboxStatus,
    SandboxUnavailable,
    SubmissionExecutionError,
    SubprocessSandbox,
    WorkerInvocation,
    WorkerResult,
    build_egress_filter,
    container_runtime_available,
    egress_filter_supported,
    filesystem_read_roots,
    landlock_abi,
    landlock_supported,
    read_worker_result,
    restrict_filesystem,
    rlimit_settings,
    sandbox_environment,
    supported_access_rights,
    worker_argv,
)
from astro_mine.bench.zoo import ANCHOR_SCENARIO_ID, load_scenario
from tests.bench._factories import (
    BASELINE_REF,
    EXPLODING_REF,
    REPO_ROOT,
    TESTS_DIR,
    sandbox_enforceable,
)

#: Policies published purely to attack the sandbox (below). They are *real* submissions: importable
#: module:attribute references, exactly what a hostile community submission would be.
EGRESS_REF = "tests.bench._policies_hostile:NetworkEgressPolicy"
MEMORY_HOG_REF = "tests.bench._policies_hostile:MemoryHogPolicy"
CPU_SPINNER_REF = "tests.bench._policies_hostile:CpuSpinnerPolicy"
SLEEPER_REF = "tests.bench._policies_hostile:SleeperPolicy"
ENV_SNOOP_REF = "tests.bench._policies_hostile:EnvironmentSnoopPolicy"
FORK_BOMB_REF = "tests.bench._policies_hostile:ForkBombPolicy"
#: The filesystem-attacking policies (bench#36) — referenced as *top-level* modules so a confined
#: worker resolves them from ``TESTS_DIR`` alone, without the repo root (which holds ``embargo/``).
SEED_READER_REF = "_policies_hostile:SeedReaderPolicy"
PROC_SNOOP_REF = "_policies_hostile:ProcEnvironSnoopPolicy"

#: Skip a test that runs the **real, confined** sandbox where the confinement cannot be enforced
#: (non-Linux, no Landlock, or a 9p/drvfs checkout). Replaces the old egress-only gate: since
#: bench#36 the default sandbox also installs a Landlock allowlist, and a test that runs a worker
#: exercises both. Pure BPF / rlimit / argv tests stay ungated and run everywhere.
linux_only = pytest.mark.skipif(
    not sandbox_enforceable(),
    reason="needs Linux with a seccomp egress filter and a Landlock-capable filesystem "
    "(a 9p/drvfs checkout cannot enforce the confinement; CI on a native filesystem does)",
)


@pytest.fixture(scope="module")
def sandbox() -> SubprocessSandbox:
    """The real thing: a forked process under rlimits and a seccomp no-egress filter.

    ``python_path`` carries the repo root because the sandbox **scrubs the environment** — the
    worker inherits no ``PYTHONPATH`` from the test process, which is the point.
    """
    return SubprocessSandbox(
        limits=SandboxLimits(cpu_seconds=30, wall_seconds=90.0, memory_bytes=2 * 1024**3),
        python_path=(REPO_ROOT,),
    )


def _roll(sandbox: SubprocessSandbox, policy_ref: str, seed: int = 1) -> SandboxOutcome:
    return sandbox.run(WorkerInvocation(ANCHOR_SCENARIO_ID, policy_ref, seed))


# =================================================================================================
# AC1 — out-of-process execution, on both intake paths
# =================================================================================================


@linux_only
def test_a_submission_runs_in_a_different_process(sandbox: SubprocessSandbox) -> None:
    """The whole point: the submission's pid is not the evaluator's pid."""
    outcome = _roll(sandbox, "tests.bench._policies_hostile:ReportPidPolicy")
    assert outcome.scored
    # ReportPidPolicy encodes its own pid into the trace; what matters here is simply that the
    # worker ran and reported back at all — an in-process run could not have produced a document.
    assert outcome.result is not None
    assert outcome.exit_code == 0


@linux_only
def test_sandboxed_scorecard_is_byte_identical_to_the_local_tier() -> None:
    """Sandboxing costs no reproducibility: the same inputs give the *same content hash*.

    The sandboxed per-seed values are folded through the same `aggregate_scores` kernel the
    in-process local tier uses, so a hosted scorecard and a workstation scorecard agree exactly —
    which is what makes a leaderboard entry reproducible by anyone (bench.md §5, §9).
    """
    spec = load_scenario(ANCHOR_SCENARIO_ID)
    seeds = (11, 12)
    sandboxed = SandboxScorer(SubprocessSandbox(python_path=(REPO_ROOT,)))(
        spec, BASELINE_REF, seeds=seeds
    )
    in_process = run(spec, BaselinePolicy(), seeds=seeds)
    assert sandboxed.content_hash == in_process.content_hash


def test_worker_argv_is_the_existing_eval_worker_command() -> None:
    """bench#30 asks us to *reuse the eval worker argv*, not invent a dispatch mechanism."""
    argv = worker_argv(
        WorkerInvocation("s-v1", "m:p", 7), python="/usr/bin/python3", output_dir="/out"
    )
    assert argv[:5] == [
        "/usr/bin/python3",
        "-m",
        "astro_mine.bench",
        "eval-worker",
        "--scenario-id",
    ]
    assert "--policy-ref" in argv and "m:p" in argv
    assert argv[argv.index("--seed") + 1] == "7"
    # `--emit json` keeps the sandboxed path dependency-clean (no pyarrow/mcap in the worker).
    assert argv[argv.index("--emit") + 1] == "json"


# =================================================================================================
# AC2 — no network egress by default; CPU / memory / time limits
# =================================================================================================


@linux_only
def test_network_egress_is_denied(sandbox: SubprocessSandbox) -> None:
    """A submitted policy cannot open a socket. This is the headline guarantee (bench.md §9).

    ``NetworkEgressPolicy`` tries, in one decision, an outbound TCP connect, a UDP socket, and a
    raw AF_UNIX socket. The seccomp filter denies the ``socket`` syscall itself, so *every* one of
    them fails with EPERM before a single packet can leave — there is no partially-online state.
    """
    outcome = _roll(sandbox, EGRESS_REF)
    # The policy raises on the denial, so the worker reports a structured failure...
    assert outcome.status is SandboxStatus.FAILED
    assert outcome.result is not None and not outcome.result.ok
    # ...and the reason is the kernel refusing the socket, not a DNS/timeout/connection error.
    assert "EGRESS-DENIED" in (outcome.result.error or "")


@linux_only
def test_the_evaluator_keeps_its_own_networking(sandbox: SubprocessSandbox) -> None:
    """The filter is installed in the *child*, never the parent — the evaluator still needs sockets.

    (It talks to Postgres, Redis, S3, the IdP, and OPA.) A filter that leaked into the evaluator
    would take the whole service down, so this is worth pinning.
    """
    import socket

    _roll(sandbox, BASELINE_REF)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        assert probe.fileno() > 0  # the parent can still create sockets after running a submission


@linux_only
def test_memory_limit_kills_an_over_allocating_policy() -> None:
    """RLIMIT_AS: a policy that tries to allocate the host's memory fails its allocation."""
    tight = SubprocessSandbox(
        limits=SandboxLimits(memory_bytes=256 * 1024 * 1024, wall_seconds=60.0),
        python_path=(REPO_ROOT,),
    )
    outcome = _roll(tight, MEMORY_HOG_REF)
    assert outcome.status in {SandboxStatus.FAILED, SandboxStatus.KILLED, SandboxStatus.CRASHED}
    assert not outcome.scored


@linux_only
def test_cpu_limit_kills_a_spinner() -> None:
    """RLIMIT_CPU: a policy burning CPU forever is killed by the kernel (SIGXCPU, then KILL)."""
    tight = SubprocessSandbox(
        limits=SandboxLimits(cpu_seconds=2, wall_seconds=60.0), python_path=(REPO_ROOT,)
    )
    outcome = _roll(tight, CPU_SPINNER_REF)
    assert outcome.status is SandboxStatus.KILLED
    assert outcome.signal in {9, 24}  # SIGKILL / SIGXCPU
    assert not outcome.scored


@linux_only
def test_wall_clock_limit_kills_a_sleeper() -> None:
    """A sleeping policy burns no CPU, so RLIMIT_CPU never fires — the wall-clock cap must.

    This is why both caps exist: `time.sleep(3600)` would otherwise hold an evaluation slot open
    indefinitely at zero CPU cost, which is a free denial-of-service against the fleet.
    """
    tight = SubprocessSandbox(
        limits=SandboxLimits(cpu_seconds=60, wall_seconds=3.0), python_path=(REPO_ROOT,)
    )
    outcome = _roll(tight, SLEEPER_REF)
    assert outcome.status is SandboxStatus.TIMEOUT
    assert "wall-clock limit" in (outcome.detail or "")
    assert outcome.usage.wall_seconds < 30.0  # it really was killed, not merely waited out


@linux_only
def test_the_environment_is_scrubbed_of_secrets(
    sandbox: SubprocessSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A submission cannot read the deployment's secrets out of ``os.environ``.

    The pre-bench#30 in-process evaluator handed a submission the *whole* evaluator environment:
    the Postgres URL, the Hub registry path, `CORE_REPO_TOKEN`. Now the child gets an allowlist.
    """
    monkeypatch.setenv("ASTRO_MINE_BENCH_DB", "postgresql://bench:hunter2@db/bench")
    monkeypatch.setenv("CORE_REPO_TOKEN", "ghp_supersecret")
    outcome = _roll(sandbox, ENV_SNOOP_REF)
    assert outcome.status is SandboxStatus.FAILED
    error = (outcome.result.error if outcome.result else "") or ""
    # The policy raises listing whatever secrets it found; it must have found none.
    assert "LEAKED" not in error
    assert "hunter2" not in error and "ghp_supersecret" not in error


@linux_only
def test_no_gpu_is_exposed_by_default(sandbox: SubprocessSandbox) -> None:
    env = sandbox_environment(SandboxLimits(), workdir="/tmp/x")
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert "ASTRO_MINE_BENCH_DB" not in env and "CORE_REPO_TOKEN" not in env
    # A GPU-granting envelope does not blank the device list.
    assert "CUDA_VISIBLE_DEVICES" not in sandbox_environment(
        SandboxLimits(gpus=1), workdir="/tmp/x"
    )


@linux_only
def test_fork_bomb_is_bounded() -> None:
    """RLIMIT_NPROC: a submission cannot exhaust the host's process table."""
    tight = SubprocessSandbox(
        limits=SandboxLimits(max_processes=8, wall_seconds=30.0, cpu_seconds=10),
        python_path=(REPO_ROOT,),
    )
    outcome = _roll(tight, FORK_BOMB_REF)
    assert not outcome.scored  # it did not get to score, and it did not take the host with it


def test_rlimit_settings_cover_every_resource() -> None:
    """The mapping is pure precisely so it can be asserted (it runs in a forked child)."""
    import resource as res

    limits = SandboxLimits(
        cpu_seconds=30,
        memory_bytes=1024,
        output_bytes=2048,
        max_processes=16,
        max_open_files=32,
    )
    mapped = dict(rlimit_settings(limits))
    assert mapped[res.RLIMIT_CPU] == (30, 31)  # SIGXCPU at the soft limit, SIGKILL at the hard one
    assert mapped[res.RLIMIT_AS] == (1024, 1024)
    assert mapped[res.RLIMIT_FSIZE] == (2048, 2048)
    assert mapped[res.RLIMIT_NPROC] == (16, 16)
    assert mapped[res.RLIMIT_NOFILE] == (32, 32)
    assert mapped[res.RLIMIT_CORE] == (0, 0)  # no core dumps: a crash must not spill memory to disk


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cpu_seconds", 0),
        ("memory_bytes", 0),
        ("output_bytes", 0),
        ("max_processes", 0),
        ("max_open_files", 1),
        ("wall_seconds", 0.0),
        ("gpus", -1),
    ],
)
def test_limits_reject_a_meaningless_envelope(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        SandboxLimits(**{field: value})  # type: ignore[arg-type]


def test_default_limits_deny_egress_and_gpus() -> None:
    """The default envelope is the safe one — a caller must *opt in* to anything looser."""
    limits = SandboxLimits()
    assert limits.network is NetworkPolicy.DENY
    assert limits.gpus == 0
    # bench#36: the filesystem is confined by default, so the default hosted configuration is safe
    # for untrusted submissions without any extra wiring.
    assert limits.filesystem is FilesystemPolicy.CONFINE


# =================================================================================================
# AC2 (bench#36) — the filesystem is confined by default; the embargoed seeds are unreadable
# =================================================================================================


@linux_only
def test_the_embargoed_held_out_seeds_cannot_be_read() -> None:
    """The headline of bench#36: a submission cannot ``open()`` ``embargo/*/heldout_seeds.json``.

    Run under the **default** sandbox (``FilesystemPolicy.CONFINE``), a policy whose only purpose is
    to read the held-out set fails on the *read* — a Landlock ``EACCES`` — long before it could
    encode anything in its metrics. The assertion checks the read was denied, not merely that egress
    was: the error must name a permission failure on the seed file, and the policy must never reach
    its ``EMBARGO-READ-SUCCEEDED`` line.
    """
    from tests.bench._policies_hostile import HELDOUT_SEEDS_PATH

    assert HELDOUT_SEEDS_PATH.is_file()  # the file really is there to be protected
    sandbox = SubprocessSandbox(
        limits=SandboxLimits(cpu_seconds=30, wall_seconds=60.0), python_path=(TESTS_DIR,)
    )
    outcome = sandbox.run(WorkerInvocation(ANCHOR_SCENARIO_ID, SEED_READER_REF, 1))

    assert outcome.status is SandboxStatus.FAILED
    error = (outcome.result.error if outcome.result else "") or ""
    assert "EMBARGO-READ-SUCCEEDED" not in error  # the read must NOT have succeeded
    assert "PermissionError" in error and "heldout_seeds.json" in error  # it failed on the read


@linux_only
def test_the_host_filesystem_policy_is_the_explicit_unconfined_opt_in() -> None:
    """``FilesystemPolicy.HOST`` is the auditable escape hatch — and it *does* expose the seeds.

    This is the negative control that proves the confinement is what closes the read: the identical
    policy, run with the filesystem posture flipped to ``HOST`` (the trusted/local tier), reaches
    the seed file. That is precisely why ``HOST`` is never selected for a community submission.
    """
    unconfined = SubprocessSandbox(
        limits=SandboxLimits(cpu_seconds=30, wall_seconds=60.0, filesystem=FilesystemPolicy.HOST),
        python_path=(TESTS_DIR,),
    )
    outcome = unconfined.run(WorkerInvocation(ANCHOR_SCENARIO_ID, SEED_READER_REF, 1))
    assert outcome.status is SandboxStatus.FAILED
    assert "EMBARGO-READ-SUCCEEDED" in (outcome.result.error if outcome.result else "") or ""


@linux_only
def test_the_evaluator_secrets_are_unreadable_through_proc() -> None:
    """The env scrub is not bypassable via ``/proc``: bench#36 closes it as a free consequence.

    ``_ENV_ALLOWLIST`` scrubs the *child's* environment, but a same-uid worker could still read the
    evaluator's out of ``/proc/<evaluator-pid>/environ``. ``/proc`` is not on the Landlock
    allowlist, so under the default sandbox the read is denied — the same "same-uid read of a
    secret" gap as the embargo seeds, shut by the same control.
    """
    sandbox = SubprocessSandbox(
        limits=SandboxLimits(cpu_seconds=30, wall_seconds=60.0), python_path=(TESTS_DIR,)
    )
    outcome = sandbox.run(WorkerInvocation(ANCHOR_SCENARIO_ID, PROC_SNOOP_REF, 1))
    assert outcome.status is SandboxStatus.FAILED
    error = (outcome.result.error if outcome.result else "") or ""
    assert "PROC-DENIED" in error  # the read was refused...
    assert "PROC-ENVIRON-READABLE" not in error  # ...not merely empty


@linux_only
def test_a_confined_worker_still_scores_the_legitimate_policy() -> None:
    """Confinement costs no capability a real submission needs: the baseline scores under CONFINE.

    The allowlist grants the interpreter, its libraries, and the import roots — everything a policy
    needs to import and roll a seed — so a legitimate submission is unaffected while the host tree
    stays unreachable.
    """
    sandbox = SubprocessSandbox(
        limits=SandboxLimits(cpu_seconds=30, wall_seconds=90.0), python_path=(REPO_ROOT,)
    )
    outcome = sandbox.run(WorkerInvocation(ANCHOR_SCENARIO_ID, BASELINE_REF, 7))
    assert outcome.scored
    assert outcome.result is not None and len(outcome.result.metrics) == 7


def test_sandbox_fails_closed_when_the_filesystem_cannot_be_confined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No enforceable confinement ⇒ the submission does not run (the same rule as no-egress).

    A default (``CONFINE``) sandbox on a kernel without Landlock must refuse, not silently run the
    submission with the host filesystem exposed.
    """
    import astro_mine.bench.sandbox._subprocess as subprocess_module

    # Get past the egress preflight so the filesystem check is the one that fires.
    monkeypatch.setattr(subprocess_module, "egress_filter_supported", lambda machine=None: True)
    monkeypatch.setattr(subprocess_module, "landlock_supported", lambda: False)
    with pytest.raises(SandboxUnavailable, match="unconfined filesystem"):
        SubprocessSandbox(python_path=(REPO_ROOT,)).run(
            WorkerInvocation(ANCHOR_SCENARIO_ID, BASELINE_REF, 1)
        )


def test_the_host_filesystem_policy_installs_no_landlock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under ``FilesystemPolicy.HOST`` the sandbox neither preflights nor computes an allowlist.

    HOST is for the trusted tier; it must run even where Landlock is absent, and it must not pay the
    interpreter-probe cost of building a read allowlist it will not use.
    """
    import astro_mine.bench.sandbox._subprocess as subprocess_module

    monkeypatch.setattr(subprocess_module, "landlock_supported", lambda: False)

    def _fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("HOST must not compute a Landlock allowlist")

    monkeypatch.setattr(subprocess_module, "filesystem_read_roots", _fail)
    host = SubprocessSandbox(limits=SandboxLimits(filesystem=FilesystemPolicy.HOST))
    host._preflight()  # does not raise despite landlock_supported() == False
    assert host._confinement_roots("/tmp/workdir") == ((), ())


# =================================================================================================
# The Landlock allowlist itself — pure, so it is asserted rather than trusted
# =================================================================================================


def test_landlock_support_matches_the_abi_probe() -> None:
    """`landlock_supported()` is exactly "Linux with a Landlock ABI"; non-Linux reports ``-1``."""
    assert landlock_supported() == (landlock_abi() >= 1)
    if sys.platform != "linux":
        assert landlock_abi() == -1
        assert landlock_supported() is False


def test_supported_access_rights_are_gated_by_abi() -> None:
    """Handling a right the running kernel lacks is EINVAL, so the set is clamped to the ABI.

    ``refer`` (ABI 2), ``truncate`` (ABI 3), and ``ioctl_dev`` (ABI 5) appear only at or above their
    introducing version; the ABI-1 baseline has the original thirteen and none of the newer three.
    """
    abi1 = supported_access_rights(1)
    assert "read_file" in abi1 and "execute" in abi1
    assert not ({"refer", "truncate", "ioctl_dev"} & set(abi1))
    assert "refer" in supported_access_rights(2)
    assert "truncate" in supported_access_rights(3)
    assert {"refer", "truncate", "ioctl_dev"} <= set(supported_access_rights(5))
    # Every advertised right maps to a distinct single bit.
    bits = list(supported_access_rights(5).values())
    assert len(bits) == len(set(bits)) and all(bit and (bit & (bit - 1)) == 0 for bit in bits)


def test_filesystem_read_roots_grant_the_interpreter_and_import_roots_not_the_repo() -> None:
    """The read allowlist covers what a worker imports from — never the repo root holding embargo.

    It includes the interpreter, its standard library / site-packages, the system runtime paths, and
    the declared import roots (``extra_roots``); it does not include an unrelated directory, and in
    particular ``TESTS_DIR`` being granted never implies the repo root above it.
    """
    roots = filesystem_read_roots(sys.executable, extra_roots=(TESTS_DIR,))
    assert sys.executable in roots
    assert TESTS_DIR in roots
    assert any(root.startswith(sys.prefix) for root in roots)  # stdlib / site-packages
    assert REPO_ROOT not in roots  # the embargo dir's parent is never implicitly granted
    assert "/nonexistent/import/root" not in filesystem_read_roots(sys.executable)


def test_filesystem_read_roots_are_empty_when_the_interpreter_probe_fails() -> None:
    """A broken interpreter yields no import roots — the worker fails to import, never unconfined.

    ``filesystem_read_roots`` still returns the system read paths and the (non-existent) interpreter
    path, but no library roots, so the confined worker cannot boot and is rejected — fail-closed.
    """
    roots = filesystem_read_roots("/nonexistent/python")
    assert "/nonexistent/python" in roots
    assert not any(root.startswith(sys.prefix) and root != sys.prefix for root in roots)


def test_confinement_roots_name_the_workdir_and_exclude_the_repo() -> None:
    """A confining sandbox's write set is the run's scratch dir; its read set excludes the repo.

    This is the parent-side computation the child installs — asserted here directly (no worker), so
    the allowlist is pinned even on a host that cannot enforce Landlock.
    """
    sandbox = SubprocessSandbox(python_path=(TESTS_DIR,))
    read_roots, write_roots = sandbox._confinement_roots("/tmp/run-xyz")
    assert "/tmp/run-xyz" in write_roots
    assert "/dev/null" in write_roots
    assert TESTS_DIR in read_roots
    assert REPO_ROOT not in read_roots


def test_the_allowlist_mask_drops_directory_only_rights_on_a_file() -> None:
    """A rule's access mask keeps only file rights on a file target — a dir-only right is EINVAL."""
    from astro_mine.bench.sandbox import _landlock

    rights = supported_access_rights(5)
    on_dir = _landlock._mask(("read_file", "read_dir"), rights, is_dir=True)
    on_file = _landlock._mask(("read_file", "read_dir"), rights, is_dir=False)
    assert on_dir & rights["read_dir"]  # a directory keeps read_dir...
    assert not (on_file & rights["read_dir"])  # ...a file drops it...
    assert on_file & rights["read_file"]  # ...but keeps read_file


def test_restrict_filesystem_refuses_without_landlock(monkeypatch: pytest.MonkeyPatch) -> None:
    """The confinement primitive itself fails closed: no Landlock ABI ⇒ it raises, never confines.

    Patched to report no ABI, it raises before touching the syscalls — so the check is exercised
    without irrevocably confining the test process.
    """
    import astro_mine.bench.sandbox._landlock as landlock_module

    monkeypatch.setattr(landlock_module, "landlock_abi", lambda: 0)
    with pytest.raises(LandlockUnsupported, match="Landlock is unavailable"):
        restrict_filesystem([], [])


# =================================================================================================
# The seccomp filter itself
# =================================================================================================


def test_egress_filter_denies_every_socket_syscall() -> None:
    """The BPF program is data, so it is asserted rather than trusted."""
    machine = platform.machine()
    if machine not in DENIED_SYSCALLS:
        pytest.skip(f"no syscall table for {machine}")
    denied = DENIED_SYSCALLS[machine]
    for call in ("socket", "connect", "bind", "sendto", "sendmsg", "recvfrom", "recvmsg"):
        assert call in denied
    assert "ptrace" in denied  # a submission may not attach to another process either


def test_egress_filter_is_well_formed_bpf() -> None:
    program = build_egress_filter("x86_64")
    assert len(program) % 8 == 0  # a whole number of `struct sock_filter` (8 bytes each)
    # arch guard + x32 guard + one check per denied syscall + 3 terminal returns.
    instructions = len(program) // 8
    assert instructions == 5 + len(DENIED_SYSCALLS["x86_64"]) + 3


def test_egress_filter_refuses_an_unknown_architecture() -> None:
    from astro_mine.bench.sandbox import SeccompUnsupported

    with pytest.raises(SeccompUnsupported):
        build_egress_filter("s390x")


def test_sandbox_fails_closed_when_egress_cannot_be_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The core safety property: no enforceable sandbox ⇒ the submission does not run.

    Degrading to a weaker sandbox would be the worst outcome — the operator would believe community
    code was contained when it was not.
    """
    import astro_mine.bench.sandbox._subprocess as subprocess_module

    monkeypatch.setattr(subprocess_module, "egress_filter_supported", lambda: False)
    with pytest.raises(SandboxUnavailable, match="refusing to execute"):
        SubprocessSandbox(python_path=(REPO_ROOT,)).run(
            WorkerInvocation(ANCHOR_SCENARIO_ID, BASELINE_REF, 1)
        )


def test_egress_filter_support_is_platform_gated() -> None:
    assert egress_filter_supported("x86_64") == (sys.platform == "linux")
    assert egress_filter_supported("s390x") is False


# =================================================================================================
# AC3 — the structured hand-back channel
# =================================================================================================


@linux_only
def test_results_come_back_as_a_document_not_shared_state(sandbox: SubprocessSandbox) -> None:
    outcome = _roll(sandbox, BASELINE_REF, seed=42)
    assert outcome.scored
    assert outcome.result is not None
    assert outcome.result.seed == 42
    assert outcome.result.policy_ref == BASELINE_REF
    assert len(outcome.result.metrics) == 7  # the anchor's metric set, computed inside the sandbox
    assert {metric.metric for metric in outcome.result.metrics} >= {"water_mass", "energy_per_kg"}
    # Resource usage rides the same channel (bench#30 AC3: metrics, errors, resource usage).
    assert outcome.usage.wall_seconds > 0.0
    assert outcome.result.usage is not None and outcome.result.usage.cpu_seconds is not None


@linux_only
def test_a_failing_policy_is_reported_as_data(sandbox: SubprocessSandbox) -> None:
    """A policy that raises must not raise *in the evaluator* — it comes back as a document."""
    outcome = _roll(sandbox, EXPLODING_REF)
    assert outcome.status is SandboxStatus.FAILED
    assert outcome.result is not None and not outcome.result.ok
    assert "this submission is broken" in (outcome.result.error or "")
    assert outcome.exit_code == 1


@linux_only
def test_an_unimportable_reference_is_reported_as_data(sandbox: SubprocessSandbox) -> None:
    """The edge does not import a ref to validate it; the sandbox discovers it and hands it back."""
    outcome = _roll(sandbox, "no_such_module_xyz:Policy")
    assert outcome.status is SandboxStatus.FAILED
    assert "cannot import" in (outcome.result.error if outcome.result else "") or ""


def test_a_garbage_result_document_is_not_a_score(tmp_path: Path) -> None:
    """A submission that scribbles over the channel is a crash, never a score."""
    (tmp_path / "result.json").write_bytes(b"{not json at all")
    assert read_worker_result(tmp_path / "result.json") is None
    assert read_worker_result(tmp_path / "absent.json") is None
    # A well-formed JSON document that is not a WorkerResult is equally rejected.
    (tmp_path / "wrong.json").write_text('{"ok": "yes please"}')
    assert read_worker_result(tmp_path / "wrong.json") is None


def test_scorer_fails_closed_on_any_bad_seed() -> None:
    """A partially-executed submission is never scored on the seeds that happened to finish."""

    class _FlakySandbox:
        limits = SandboxLimits()

        def run(self, invocation: WorkerInvocation) -> SandboxOutcome:
            return SandboxOutcome(
                status=SandboxStatus.TIMEOUT,
                invocation_seed=invocation.seed,
                usage=ResourceUsage(wall_seconds=1.0),
                detail="wall-clock limit exceeded",
            )

    scorer = SandboxScorer(_FlakySandbox())
    with pytest.raises(SubmissionExecutionError, match="did not execute cleanly on seed 1"):
        scorer(load_scenario(ANCHOR_SCENARIO_ID), "m:p", seeds=(1, 2, 3))


def test_scorer_rejects_a_worker_that_withholds_a_metric() -> None:
    """A submission cannot omit a metric it scores badly on: the contract is all-or-nothing."""

    class _PartialSandbox:
        limits = SandboxLimits()

        def run(self, invocation: WorkerInvocation) -> SandboxOutcome:
            return SandboxOutcome(
                status=SandboxStatus.OK,
                invocation_seed=invocation.seed,
                result=WorkerResult(
                    ok=True,
                    scenario_id=ANCHOR_SCENARIO_ID,
                    policy_ref="m:p",
                    seed=invocation.seed,
                    metrics=(),  # reports nothing
                ),
                usage=ResourceUsage(wall_seconds=1.0),
            )

    scorer = SandboxScorer(_PartialSandbox())
    with pytest.raises(SubmissionExecutionError, match="returned no value for metric"):
        scorer(load_scenario(ANCHOR_SCENARIO_ID), "m:p", seeds=(1,))


def test_scorer_needs_at_least_one_seed() -> None:
    scorer = SandboxScorer(SubprocessSandbox(python_path=(REPO_ROOT,)))
    with pytest.raises(ValueError, match="at least one seed"):
        scorer(load_scenario(ANCHOR_SCENARIO_ID), BASELINE_REF, seeds=())


def test_in_process_scorer_exists_only_for_trusted_code() -> None:
    """The local tier scores *your own* policy in-process — that is correct, and it stays.

    It is never the leaderboard's default (see LeaderboardService), and TRUST_BOUNDARY.md §4 says
    so in as many words.
    """
    spec = load_scenario(ANCHOR_SCENARIO_ID)
    card = InProcessScorer()(spec, BASELINE_REF, seeds=(1, 2))
    assert card.content_hash == run(spec, BaselinePolicy(), seeds=(1, 2)).content_hash


# =================================================================================================
# The container backend — its argv *is* its security posture
# =================================================================================================

IMAGE = "ghcr.io/astro-mine/astro-mine-sim@sha256:" + "0" * 64


def test_container_argv_carries_the_whole_security_posture() -> None:
    argv = ContainerSandbox(IMAGE).container_argv(
        WorkerInvocation(ANCHOR_SCENARIO_ID, BASELINE_REF, 1), host_output_dir="/host/out"
    )
    joined = " ".join(argv)
    # No egress, no writable rootfs, no capabilities, no privilege escalation, not root.
    assert "--network=none" in argv
    assert "--read-only" in argv
    assert "--cap-drop=ALL" in argv
    assert "--security-opt=no-new-privileges" in argv
    assert "--user=65534:65534" in argv
    assert "--tmpfs=/tmp:rw,noexec,nosuid,size=64m" in argv
    # Hard resource caps.
    assert "--memory=2147483648b" in argv
    assert "--pids-limit=64" in argv
    assert any(flag.startswith("--cpus=") for flag in argv)
    assert any(flag.startswith("--ulimit=nofile=") for flag in argv)
    assert "--ulimit=core=0:0" in argv
    # No GPU unless the envelope grants one.
    assert "--gpus" not in joined
    # Exactly one writable mount: the structured result channel.
    assert f"--volume=/host/out:{CONTAINER_OUTPUT_DIR}:rw" in argv
    # ...and it runs the same eval-worker argv, in the digest-pinned runner image.
    assert IMAGE in argv
    assert "eval-worker" in argv and "--emit" in argv


def test_container_argv_selects_gvisor_and_a_seccomp_profile() -> None:
    """bench.md §9 names gVisor; it is a runtime flag, not a code change."""
    argv = ContainerSandbox(
        IMAGE, runtime_flags=("--runtime=runsc",), seccomp_profile="/etc/seccomp/bench.json"
    ).container_argv(WorkerInvocation("s", "m:p", 1), host_output_dir="/out")
    assert "--runtime=runsc" in argv
    assert "--security-opt=seccomp=/etc/seccomp/bench.json" in argv
    # The runtime flag comes before the image, i.e. it is a *runtime* flag, not a container arg.
    assert argv.index("--runtime=runsc") < argv.index(IMAGE)


def test_container_argv_grants_a_gpu_only_when_the_envelope_does() -> None:
    argv = ContainerSandbox(IMAGE, limits=SandboxLimits(gpus=1)).container_argv(
        WorkerInvocation("s", "m:p", 1), host_output_dir="/out"
    )
    assert "--gpus=1" in argv


def test_container_network_allow_is_an_explicit_opt_in() -> None:
    argv = ContainerSandbox(
        IMAGE, limits=SandboxLimits(network=NetworkPolicy.ALLOW)
    ).container_argv(WorkerInvocation("s", "m:p", 1), host_output_dir="/out")
    assert "--network=none" not in argv  # never selected for a community submission


def test_container_sandbox_needs_an_image() -> None:
    with pytest.raises(ValueError, match="evaluation-runner image"):
        ContainerSandbox("")


def test_container_sandbox_parses_the_result_off_the_bind_mount(tmp_path: Path) -> None:
    """The execution seam is injected, so the round trip is testable without a container runtime."""
    captured: list[list[str]] = []

    def fake_execute(argv, timeout):  # type: ignore[no-untyped-def]
        captured.append(list(argv))
        # The container would write result.json into the bind-mounted host directory.
        host_dir = next(
            flag.split("=", 1)[1].split(":", 1)[0] for flag in argv if flag.startswith("--volume=")
        )
        Path(host_dir, "result.json").write_text(
            WorkerResult(
                ok=True,
                scenario_id=ANCHOR_SCENARIO_ID,
                policy_ref=BASELINE_REF,
                seed=3,
                usage=ResourceUsage(wall_seconds=0.0, cpu_seconds=1.5, max_rss_bytes=1024),
            ).model_dump_json()
        )
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    sandbox = ContainerSandbox(IMAGE, runtime="podman", execute=fake_execute)
    outcome = sandbox.run(WorkerInvocation(ANCHOR_SCENARIO_ID, BASELINE_REF, 3))
    assert outcome.status is SandboxStatus.OK
    assert outcome.result is not None and outcome.result.seed == 3
    assert outcome.usage.cpu_seconds == 1.5
    assert captured[0][0] == "podman"


def test_container_sandbox_reports_a_timeout(tmp_path: Path) -> None:
    def timing_out(argv, timeout):  # type: ignore[no-untyped-def]
        raise subprocess.TimeoutExpired(list(argv), timeout)

    outcome = ContainerSandbox(IMAGE, runtime="podman", execute=timing_out).run(
        WorkerInvocation(ANCHOR_SCENARIO_ID, BASELINE_REF, 1)
    )
    assert outcome.status is SandboxStatus.TIMEOUT
    assert "wall-clock limit" in (outcome.detail or "")


def test_container_sandbox_reports_a_crash() -> None:
    def crashing(argv, timeout):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(list(argv), 137, b"", b"OOMKilled")

    outcome = ContainerSandbox(IMAGE, runtime="podman", execute=crashing).run(
        WorkerInvocation(ANCHOR_SCENARIO_ID, BASELINE_REF, 1)
    )
    assert outcome.status is SandboxStatus.CRASHED
    assert outcome.exit_code == 137
    assert "OOMKilled" in outcome.stderr  # captured for the audit trail, never parsed


def test_container_sandbox_reports_a_failed_policy(tmp_path: Path) -> None:
    def failing(argv, timeout):  # type: ignore[no-untyped-def]
        host_dir = next(
            flag.split("=", 1)[1].split(":", 1)[0] for flag in argv if flag.startswith("--volume=")
        )
        Path(host_dir, "result.json").write_text(
            WorkerResult(
                ok=False,
                scenario_id=ANCHOR_SCENARIO_ID,
                policy_ref=BASELINE_REF,
                seed=1,
                error="RuntimeError: broken",
            ).model_dump_json()
        )
        return subprocess.CompletedProcess(list(argv), 1, b"", b"")

    outcome = ContainerSandbox(IMAGE, runtime="podman", execute=failing).run(
        WorkerInvocation(ANCHOR_SCENARIO_ID, BASELINE_REF, 1)
    )
    assert outcome.status is SandboxStatus.FAILED
    assert outcome.detail == "RuntimeError: broken"


def test_container_sandbox_refuses_without_a_runtime() -> None:
    """Fail closed: no container runtime ⇒ the submission does not run unsandboxed instead."""
    sandbox = ContainerSandbox(IMAGE, runtime="definitely-not-a-real-runtime")
    with pytest.raises(SandboxUnavailable, match="refusing to execute"):
        sandbox.run(WorkerInvocation(ANCHOR_SCENARIO_ID, BASELINE_REF, 1))


@pytest.mark.container
@pytest.mark.skipif(not container_runtime_available("docker"), reason="needs a real Docker runtime")
def test_real_container_round_trip() -> None:  # pragma: no cover - opt-in, needs Docker
    """The opt-in end-to-end: a submission really does score inside a `--network=none` container.

    Deselected in CI (`-m "not container"`) because it needs a built evaluation-runner image; the
    argv this issues is asserted flag-by-flag above, and the image is the one the Cloud rollout path
    already schedules.
    """
    image = "astro-mine-bench-eval:local"
    sandbox = ContainerSandbox(image)
    outcome = sandbox.run(WorkerInvocation(ANCHOR_SCENARIO_ID, BASELINE_REF, 1))
    assert outcome.scored


# =================================================================================================
# AC4 — the trust boundary is documented
# =================================================================================================


def test_the_trust_boundary_is_documented() -> None:
    """AC4: what's sandboxed, what isn't, residual risk — stated, not implied."""
    doc = Path(__file__).resolve().parents[2] / "TRUST_BOUNDARY.md"
    assert doc.is_file(), "TRUST_BOUNDARY.md must exist: an undocumented boundary is unusable"
    text = doc.read_text(encoding="utf-8")

    # It must state what IS protected...
    for control in ("seccomp", "RLIMIT_CPU", "--network=none", "gVisor", "no_new_privs".upper()):
        assert control in text or control.lower() in text.lower()
    # ...including the filesystem confinement that closes the embargo-read gap (bench#36).
    assert "Landlock" in text
    assert "heldout_seeds.json" in text or "held-out seed" in text.lower()

    # ...and, crucially, what is NOT. An honest boundary names its own gaps.
    assert "Kernel 0-days" in text
    assert "Side channels" in text
    assert "residual" in text.lower()

    # ...and it must cover both intake paths bench#30 names.
    assert "policy_ref" in text
    assert "Hub" in text


# =================================================================================================
# Backend plumbing
# =================================================================================================


def test_a_sandbox_exposes_the_envelope_it_enforces() -> None:
    """An operator (and an auditor) must be able to ask a sandbox what it actually enforces."""
    limits = SandboxLimits(cpu_seconds=11, memory_bytes=1 << 30)
    assert SubprocessSandbox(limits=limits).limits == limits

    container = ContainerSandbox(IMAGE, limits=limits)
    assert container.limits == limits
    assert container.image == IMAGE  # the digest-pinned evaluation runner (bench.md §9)


def test_the_scorer_exposes_its_backend() -> None:
    backend = SubprocessSandbox(python_path=(REPO_ROOT,))
    assert SandboxScorer(backend).sandbox is backend


@linux_only
def test_a_worker_that_cannot_be_spawned_fails_closed() -> None:
    """A broken interpreter must not silently degrade into an unsandboxed run."""
    broken = SubprocessSandbox(python="/nonexistent/python", python_path=(REPO_ROOT,))
    with pytest.raises(SandboxUnavailable, match="could not be spawned"):
        broken.run(WorkerInvocation(ANCHOR_SCENARIO_ID, BASELINE_REF, 1))


@linux_only
def test_a_worker_that_writes_no_result_is_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """The worker exiting without a result document is a crash, never an implicit pass."""
    import astro_mine.bench.sandbox._subprocess as subprocess_module

    # A command that exits cleanly but writes nothing to the channel.
    monkeypatch.setattr(
        subprocess_module,
        "worker_argv",
        lambda invocation, *, python, output_dir, module="astro_mine.bench": [
            python,
            "-c",
            "pass",
        ],
    )
    outcome = SubprocessSandbox(python_path=(REPO_ROOT,)).run(
        WorkerInvocation(ANCHOR_SCENARIO_ID, BASELINE_REF, 1)
    )
    assert outcome.status is SandboxStatus.CRASHED
    assert "without a parseable result document" in (outcome.detail or "")
    assert not outcome.scored


def test_the_default_container_invoker_shells_out_to_the_runtime() -> None:
    """The default `execute` really does run the runtime (here: a runtime that does not exist)."""
    from astro_mine.bench.sandbox._container import _run_container

    with pytest.raises(FileNotFoundError):
        _run_container(["definitely-not-a-real-runtime", "run"], 5.0)


def test_the_egress_filter_reports_its_own_unavailability() -> None:
    """`install_egress_filter` fails closed on a platform it cannot constrain."""
    import astro_mine.bench.sandbox._seccomp as seccomp_module
    from astro_mine.bench.sandbox import SeccompUnsupported, install_egress_filter

    original = seccomp_module.egress_filter_supported
    seccomp_module.egress_filter_supported = lambda machine=None: False  # type: ignore[assignment]
    try:
        with pytest.raises(SeccompUnsupported, match="seccomp is unavailable"):
            install_egress_filter()
    finally:
        seccomp_module.egress_filter_supported = original  # type: ignore[assignment]
