"""Hostile submissions — the adversary the sandbox exists to contain (bench#30; bench.md §9).

These are *real* Core policies, published exactly the way a community submission is: an importable
``module:attribute`` reference the leaderboard is asked to run. Each one attacks a different control
in :mod:`astro_mine.bench.sandbox`, and ``tests/test_sandbox.py`` asserts the control holds.

They exist because bench.md §2 principle 5 says so: *"Adversarial by assumption. Public leaderboards
are gamed."* A sandbox tested only against well-behaved code is not tested.

Each policy signals its finding by **raising**, because the sandboxed worker reports a raised
exception back over the structured channel as ``ok=false`` + an ``error`` string — which is a
channel the test can read, and (deliberately) the *only* channel it has, since the sandbox denies
the policy every other way of talking to the outside world.

Not measured for coverage (coverage source is ``src/astro_mine``); kept as a private module so
pytest does not collect it as a test.
"""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path

from astro_mine.core.messages import Action, ActionBatch, ModeCommand
from astro_mine.core.messages.enums import ActionKind

#: The secrets a real deployment holds in its environment. A submission that can read *any* of these
#: has the database, the artifact registry, and the org's GitHub token.
SECRET_ENV_VARS = (
    "ASTRO_MINE_BENCH_DB",
    "CORE_REPO_TOKEN",
    "ASTRO_MINE_HUB_REGISTRY",
    "AWS_SECRET_ACCESS_KEY",
    "ASTRO_MINE_BENCH_OIDC_ISSUER",
)

#: The file a submission must not be able to read: a **stand-in**, not the real held-out set.
#:
#: bench#36's guarantee is about a *location* — nothing under the repo root is on the Landlock
#: allowlist — so the target only has to be a real file at a real repo-root path. It used to be the
#: embargoed seeds themselves, which was the strongest possible target right up until they left for
#: the private ``astro-mine/embargo`` store (astro-mine-platform#37) and these tests started failing
#: on a missing file.
#:
#: Two constraints pin where it can live, and both are load-bearing:
#:
#: * **not under ``tests/``** — that is the sandbox's ``python_path``, which *is* allowlisted
#:   (:func:`~astro_mine.bench.sandbox.filesystem_read_roots`), so a probe there would be readable
#:   and the test would quietly assert the opposite of what it means to;
#: * **not named ``heldout_seeds.json``** — ``test_zoo_anchor.py`` asserts no file of that name
#:   exists anywhere in the working tree.
PROTECTED_PROBE_PATH = Path(__file__).parent.parent.parent / "embargo" / "confinement-probe.json"


def _noop_batch(mode: str = "idle") -> ActionBatch:
    return ActionBatch(
        actions=[Action(agent_id="rover", kind=ActionKind.MODE, mode=ModeCommand(mode=mode))]
    )


class NetworkEgressPolicy:
    """Tries to reach the network three different ways. The seccomp filter must deny all of them.

    A submission with egress can exfiltrate the embargoed held-out seeds (which destroys the
    leaderboard's entire anti-overfitting story), call home, or turn the evaluation fleet into a
    botnet. So the filter denies the ``socket`` syscall itself rather than trying to block
    destinations: there is no allowlist to get wrong, and no partially-online state.
    """

    def decide(self, observations: object, context: object) -> ActionBatch:
        denied: list[str] = []
        # 1. An outbound TCP connection.
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(2.0)
                sock.connect(("1.1.1.1", 53))
        except PermissionError:
            denied.append("tcp")
        except OSError as exc:  # a network error is NOT a denial — the socket was created
            raise AssertionError(f"EGRESS-ALLOWED: a TCP socket was created ({exc})") from exc

        # 2. A UDP socket (no connect needed to send).
        try:
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except PermissionError:
            denied.append("udp")
        else:
            raise AssertionError("EGRESS-ALLOWED: a UDP socket was created")

        # 3. A unix-domain socket — the local escape hatch (a Docker socket, an agent, a sidecar).
        try:
            socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        except PermissionError:
            denied.append("unix")
        else:
            raise AssertionError("EGRESS-ALLOWED: an AF_UNIX socket was created")

        raise RuntimeError(f"EGRESS-DENIED: {'+'.join(denied)}")


class EnvironmentSnoopPolicy:
    """Reads the environment looking for the deployment's secrets.

    The pre-bench#30 in-process evaluator handed a submission the *whole* evaluator environment: the
    Postgres URL with its password, ``CORE_REPO_TOKEN``, the Hub registry path. Now the sandboxed
    child inherits an allowlist, so this finds nothing.
    """

    def decide(self, observations: object, context: object) -> ActionBatch:
        found = {name: os.environ[name] for name in SECRET_ENV_VARS if name in os.environ}
        if found:
            raise RuntimeError(f"LEAKED: {found}")
        raise RuntimeError("NO-SECRETS: the environment was scrubbed")


class MemoryHogPolicy:
    """Allocates until the host dies — unless ``RLIMIT_AS`` stops it first."""

    def decide(self, observations: object, context: object) -> ActionBatch:
        blocks: list[bytes] = []
        for _ in range(4096):
            blocks.append(b"\x00" * (64 * 1024 * 1024))  # 64 MiB a time, up to 256 GiB
        raise AssertionError(f"MEMORY-UNBOUNDED: allocated {len(blocks) * 64} MiB")


class CpuSpinnerPolicy:
    """Burns CPU forever — unless ``RLIMIT_CPU`` kills it. (A cryptominer, essentially.)"""

    def decide(self, observations: object, context: object) -> ActionBatch:
        total = 0
        while True:
            total = (total + 1) % 1_000_003


class SleeperPolicy:
    """Sleeps for an hour at **zero CPU cost**, so only the wall-clock cap can stop it.

    This is why the sandbox enforces both: ``RLIMIT_CPU`` never fires on a sleeping process, so a
    CPU-only cap would let this hold an evaluation slot open indefinitely — a free denial of service
    against the fleet.
    """

    def decide(self, observations: object, context: object) -> ActionBatch:
        time.sleep(3600)
        return _noop_batch()


class ForkBombPolicy:
    """Forks recursively — unless ``RLIMIT_NPROC`` bounds it."""

    def decide(self, observations: object, context: object) -> ActionBatch:
        for _ in range(10_000):
            try:
                if os.fork() == 0:
                    os._exit(0)
            except OSError:  # RLIMIT_NPROC: the kernel refused the fork
                raise RuntimeError("FORK-DENIED") from None
        raise AssertionError("FORK-UNBOUNDED")


class ReportPidPolicy:
    """A benign policy that simply proves it is running in a *different* process."""

    def decide(self, observations: object, context: object) -> ActionBatch:
        return _noop_batch(mode=f"pid-{os.getpid()}")


class SeedReaderPolicy:
    """Reads the embargoed held-out seeds off the filesystem — the attack bench#36 closes.

    seccomp denies the *socket* that would exfiltrate the seeds, but not the *read*: a policy that
    can ``open()`` a file under the repo root can read whatever an evaluator left there — the
    held-out set, when it lived here — and overfit to it or encode it in the metric floats the
    leaderboard publishes (``TRUST_BOUNDARY.md`` §4). Under
    :attr:`~astro_mine.bench.sandbox.FilesystemPolicy.CONFINE` the Landlock allowlist does not
    include the repo tree, so the ``open()`` fails with ``EACCES`` before a byte is read — which is
    the guarantee, and is exactly what the test asserts (the *read* fails, not merely egress).
    """

    def decide(self, observations: object, context: object) -> ActionBatch:
        contents = PROTECTED_PROBE_PATH.read_text(encoding="utf-8")
        # Reached only if the read was NOT denied — the sandbox failed. Surface the length so a
        # failing test shows the file really was reachable, never its contents.
        raise AssertionError(
            f"EMBARGO-READ-SUCCEEDED: {len(contents)} bytes read from under the repo root"
        )


class ProcEnvironSnoopPolicy:
    """Recovers the evaluator's secrets from ``/proc/<pid>/environ``, behind the env scrub.

    The environment allowlist (``_ENV_ALLOWLIST``) scrubs the *child's own* environment, but a
    same-uid worker can still read its parent's — the evaluator's — full environment out of
    ``/proc``, where the database URL and Hub token live. The Landlock confinement closes this for
    free: ``/proc`` is not on the allowlist, so the read is denied (bench#36). This is the same gap
    class as :class:`SeedReaderPolicy` — a same-uid filesystem read of a secret — and the same
    control shuts it.
    """

    def decide(self, observations: object, context: object) -> ActionBatch:
        environ_path = f"/proc/{os.getppid()}/environ"
        try:
            data = open(environ_path, "rb").read()  # noqa: SIM115 - the read itself is the probe
        except OSError as exc:
            raise RuntimeError(f"PROC-DENIED: {type(exc).__name__}") from exc
        raise AssertionError(f"PROC-ENVIRON-READABLE: {len(data)} bytes of the evaluator's environ")
