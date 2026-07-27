# Trust boundary — what the leaderboard protects, and what it does not

> **Scope.** This document is normative for anyone **deploying** the hosted Astro-Mine-Bench
> leaderboard. It states, precisely, what a submitted policy can and cannot do, so an operator can
> decide whether the posture is good enough for their exposure. It exists because
> [bench.md §9](https://github.com/astro-mine/docs/blob/main/architecture/bench.md) names submitted
> code as *"the central safety concern for Bench: the leaderboard runs arbitrary community code at
> scale"*, and because a sandbox whose limits are not written down is a sandbox nobody can rely on.
>
> Implements bench#29, bench#30, and bench#36. Traces: bench.md §9; conventions.md §9;
> `LUNAR-SR-001/-002/-003`.

## 1. The asset being protected

The leaderboard is a **commons**. What an attacker wants from it is:

1. **A rank they did not earn** — by fabricating results, reading the embargoed held-out seeds,
   overfitting to them, or tampering with a stored scorecard.
2. **The evaluator itself** — the machine that runs everyone's code, and the credentials it holds
   (database URLs, the Hub token, the object store, the signing trust root).
3. **Other submitters' work** — unpublished policies, private scenarios, another lab's traces.
4. **Free compute** — the evaluation fleet as a cryptominer or a botnet.

Everything below is organized around denying those four.

## 2. The trust model, stated plainly

| Party | Trusted? |
|---|---|
| The **scenario zoo**, metric plugins, and the reference runner | **Yes** — they ship in the image and are reviewed. |
| The **evaluator** process (`LeaderboardService`, FastAPI) | **Yes** — it holds the secrets. |
| The **IdP** that issues bearer tokens | **Yes** — it is the root of identity. |
| The **cosign signer** pinned by `ASTRO_MINE_BENCH_TRUSTED_KEY` | **Yes** — it is the root of artifact trust. |
| A **submitted policy** (local `policy_ref` *or* Hub digest) | **NO. It is hostile code.** |
| The **result document** a submission writes back | **NO.** Parsed as data, never executed. |
| A submission's **stderr** | **NO.** Captured for audit, never parsed. |

The one rule everything else follows from: **the evaluator never imports a submission.** It hands a
*reference string* to a sandboxed worker and reads a *document* back. Before bench#30 the evaluator
called `importlib.import_module()` on submitted references and ran them in-process, which meant a
submission had, from its first line, everything the evaluator had.

## 3. What each tier isolates

Two backends, both running the *same* eval-worker argv (the one Cloud already fans out per seed).

### Tier A — `SubprocessSandbox` (the default)

Out-of-process, and before the submission's first bytecode executes:

| Control | Mechanism | Denies |
|---|---|---|
| **Network egress** | **seccomp-BPF** filter: `socket`, `socketpair`, `connect`, `bind`, `listen`, `accept`, `accept4`, `sendto`, `sendmsg`, `recvfrom`, `recvmsg` → `EPERM` | Exfiltrating the held-out seeds; calling home; using the fleet as a botnet; reaching the database or Hub over the network |
| **Filesystem** | A **Landlock** allowlist grants read+exec only on the interpreter, its libraries, the submission's import roots, and the run's scratch directory; read+write on that scratch directory. Everything else — the repo tree, the operator's home, SSH keys — is unreachable (`EACCES`), inherited across `exec` and unrevocable | **Reading the embargoed held-out seeds** (`embargo/*/heldout_seeds.json`); reading config files, keys, and the rest of the host — closing the gap bench#30 left open (bench#36) |
| **Foreign-arch bypass** | Any syscall from a non-native arch, or with the x32 bit (`nr >= 0x40000000`), → `SECCOMP_RET_KILL_PROCESS` | The classic seccomp filter bypass |
| **Privilege escalation** | `PR_SET_NO_NEW_PRIVS` (also what makes the unprivileged filter legal; both are inherited across `exec` and by every descendant) | setuid binaries; dropping the filter; escaping via `exec` |
| **Debugging other processes** | `ptrace` → `EPERM` | Reading another submission's or the evaluator's memory |
| **CPU** | `RLIMIT_CPU` (SIGXCPU, then SIGKILL) | Cryptomining; compute-for-score |
| **Wall clock** | Parent-enforced timeout → `SIGKILL` to the whole **process group** | A policy that just sleeps (which burns no CPU, so `RLIMIT_CPU` never fires) |
| **Memory** | `RLIMIT_AS` | OOMing the host |
| **Fork bombs** | `RLIMIT_NPROC` | Exhausting the process table |
| **Disk** | `RLIMIT_FSIZE`, `RLIMIT_CORE=0` | Filling the disk; core-dumping secrets |
| **File descriptors** | `RLIMIT_NOFILE` | Descriptor exhaustion |
| **Secrets in the environment** | The child gets an **allowlist** (`PATH`, `LANG`, `LC_ALL`, `SSL_CERT_*`), never the evaluator's environment; and with `/proc` outside the Landlock allowlist, the evaluator's environment cannot be recovered from `/proc/<pid>/environ` either | Reading `ASTRO_MINE_BENCH_DB`, `CORE_REPO_TOKEN`, OIDC config out of `os.environ` or another process's `/proc` |
| **GPU** | `CUDA_VISIBLE_DEVICES=""` unless the envelope grants GPUs | Silent GPU use |

**Tier A confines the filesystem with Landlock (bench#36).** A submitted policy is restricted to an
allowlist — the interpreter, its libraries, the submission's import roots, and the run's scratch
directory — so `embargo/*/heldout_seeds.json`, config files, SSH keys, and the rest of the host are
simply not reachable: an `open()` of them returns `EACCES` before a byte is read. The evaluator
hands the held-out set to the worker as *integers on the argv*, never as a file the worker opens, so
nothing legitimate needs the embargo path. This closes the read that earlier revisions of this
document called out as Tier A's gap, and — because `/proc` is off the allowlist — closes the
same-uid `/proc/<pid>/environ` route to the evaluator's secrets along with it.

Two caveats an operator must hold:

- **The confinement is fail-closed but filesystem-dependent.** Landlock needs a kernel that supports
  it (≥ 5.13, present in `CONFIG_LSM`) *and* a filesystem that honours it. Every standard Linux
  filesystem (ext4/xfs/btrfs/overlay/tmpfs) does; some network/9p mounts silently deny even granted
  paths, so a confined worker there cannot start and the submission is **rejected** — never run
  unconfined. On a kernel without Landlock at all, Tier A refuses up front (`SandboxUnavailable`).
- **The metric-float covert channel is narrowed, not a concern for the seeds.** Because the read is
  closed, a submission has no held-out secret to encode in its metrics. The residual channel (§4.3)
  matters only for a secret a *misconfigured* deployment placed inside an import root the allowlist
  grants — a deployment error, not a property of the seeds.

Tier A is therefore sufficient for a public, untrusted leaderboard's **filesystem and network**
boundaries. What it still does not draw — the host's PID/process-table view and the kernel attack
surface — is Tier B's job (namespaces + gVisor), which remains the recommended posture for a
public, internet-facing deployment.

### Tier B — `ContainerSandbox` (required for a public deployment)

Everything in Tier A, **plus** the boundaries a bare subprocess cannot draw — a whole-namespace view
of the host, and a defence against the kernel attack surface itself:

| Control | Mechanism | Denies |
|---|---|---|
| **Filesystem** | `--read-only` rootfs + a `noexec,nosuid` tmpfs + **one** writable bind mount (the result channel) | Reading the held-out seeds, the config, the host at all — a whole-rootfs boundary, where Tier A draws the same line with a Landlock allowlist |
| **Network namespace** | `--network=none` — a namespace with no interface | Egress, at the namespace level rather than syscall-by-syscall |
| **Capabilities** | `--cap-drop=ALL`, `--security-opt no-new-privileges` | Every capability-based escape |
| **Identity** | `--user 65534:65534` (nobody) | Running as root inside the container |
| **cgroups** | `--memory`, `--cpus`, `--pids-limit`, `--ulimit` | Resource exhaustion, enforced by the kernel's cgroup, not just rlimits |
| **Kernel attack surface** | `--runtime=runsc` (**gVisor**) when configured | Container escapes via kernel bugs — gVisor interposes a user-space kernel |

Run it against the **signed, digest-pinned evaluation-runner image**, and configure gVisor:

```python
ContainerSandbox(
    image="ghcr.io/astro-mine/astro-mine-sim@sha256:...",   # signed + digest-pinned
    runtime_flags=("--runtime=runsc",),                     # gVisor
    limits=SandboxLimits(cpu_seconds=120, memory_bytes=4 << 30),
)
```

## 4. What is NOT protected against, in either tier

Stated as plainly as the protections, because these are the residual risks an operator is accepting.

1. **Kernel 0-days.** seccomp and namespaces are kernel features; a bug in the kernel's syscall
   handling can bypass them. gVisor (Tier B) narrows this substantially; it does not eliminate it.
2. **Side channels.** Spectre/Meltdown-class attacks, cache timing, and hyperthread co-residency are
   out of scope. Do not co-locate submission workers with anything holding secrets.
3. **Covert channels through the score itself.** Metric values are attacker-controlled floats that
   the leaderboard publishes. A submission that *can* read a secret can encode it there. Both tiers
   now close the read a public leaderboard cares about — the embargoed held-out seeds (Tier A via
   Landlock, Tier B via the rootfs) — so there is no held-out secret left to encode. The channel
   itself is unbounded and remains open: it would carry any secret a *misconfigured* deployment left
   inside a path the allowlist grants (an import root, the scratch dir). Quantising/rounding
   published metrics would narrow it further but not close it; the durable fix is to keep secrets
   out of the worker's reach, which is what the confinement enforces.
4. **Denial of service by legitimate volume.** Rate limits and per-role quotas (bench#29) bound one
   principal; a botnet of authenticated principals is a capacity problem, not a sandbox one.
5. **A compromised IdP or signing key.** They are trust roots. If the IdP mints a token for an
   attacker, or the pinned cosign key leaks, the sandbox still holds — but the *authorization* and
   *provenance* stories do not. Rotate keys; use short token lifetimes.
6. **Self-reported resource usage.** `WorkerResult.usage` (CPU, max RSS) is reported by the worker
   *about itself* and is **advisory telemetry only**. A hostile submission can lie about it. It
   cannot lie its way past the *limits*, which the kernel enforces and never asks the worker about.
7. **A malicious evaluation-runner image.** The image is a trust root. Verify its signature and pin
   it by digest.
8. **`InProcessScorer`.** It exists for the local tier and the determinism gate — code *you* wrote.
   Wiring it into a hosted `LeaderboardService` re-opens everything above. Don't.

## 5. Fail-closed, everywhere

The sandbox never degrades quietly:

- If the no-egress filter **cannot be installed** on the host (not Linux, unknown architecture), the
  sandbox raises `SandboxUnavailable` and the submission **does not run**. An unenforced boundary
  that looks enforced is worse than no boundary, because the operator believes it is contained.
- If the **Landlock filesystem confinement cannot be enforced** — a kernel without Landlock — the
  default (`FilesystemPolicy.CONFINE`) sandbox likewise raises `SandboxUnavailable` up front. And if
  the kernel supports Landlock but the evaluator's *filesystem* does not honour it (a 9p/drvfs
  mount), the confined worker cannot read its own interpreter and fails to start — the submission is
  rejected, never scored with the host filesystem exposed. Running an untrusted submission with the
  host filesystem visible requires the explicit, auditable `FilesystemPolicy.HOST` opt-in, which is
  for the trusted/local tier only.
- If the container runtime is absent, `ContainerSandbox` refuses rather than falling back.
- A seed that **times out, is killed by a limit, crashes, or reports an error** rejects the *whole
  submission*. It is never scored on the seeds that happened to finish.
- A submission that writes **garbage over the result channel** is reported as a crash, never as a
  score.
- Supply-chain verification (cosign signature + SLSA provenance + SBOM) runs **before execution**,
  and is **not optional**: a deployment can pin *which* signer it trusts, but it cannot switch
  verification off.
- With **no OIDC verifier configured**, every write route returns 503. "No IdP" never means
  "everyone is trusted".

## 6. What stays account-free

The local tier is sacred (CX-LOCAL; conventions.md §7). None of the above touches it:

```bash
astro-mine-bench score          # no account, no token, no network, no database, no container
```

`run(spec, policy)` runs *your own* policy in *your own* process, which is exactly right — it is
your code. Reading the board, a scorecard, a provenance bundle, or a replay from a hosted
deployment also needs no account. Authentication gates the **write** surface only.

## 7. Reporting

Security issues in Astro-Mine go to the process in
[`astro-mine/.github/SECURITY.md`](https://github.com/astro-mine/.github/blob/main/SECURITY.md) —
please do not open a public issue for a sandbox escape.
