"""The Bench command line — clone, run, and score a baseline (RM-P0-BENCH-05; bench.md §12).

The one-command realization of the headline promise (charter §13): from a clean clone,
``astro-mine-bench score`` runs the selected runner's baseline policy on the anchor scenario and
prints its content-addressed scorecard — offline, no account, no cloud. The ``fixture`` runner
scores the reference :class:`~astro_mine.bench.baseline.BaselinePolicy`; a runner that resolves real
content may offer its own through the optional ``DefaultPolicyProvider`` seam, which is how a
capability-aware baseline reaches this command without Bench ever reading a SADF document.

``score --runner fixture|sim`` selects the runner (default ``fixture`` — the dependency-clean
deterministic trace fixture, *not* Sim). Runners are discovered by name through the
``astro_mine.bench.runners`` entry-point group (conventions.md §7); ``sim`` resolves an injected
Sim runner from ``astro-mine-sim[bench]``. The CLI never imports Sim, keeping the base package
dep-clean, and the chosen runner is recorded in the scorecard and folded into its content hash so a
fixture score and a Sim score are distinguishable by provenance, not just by value.

``fetch`` obtains the content a scenario pins (G1.2; bench#56) — it mirrors each pin **by digest**
from the published registry (``ghcr.io/astro-mine``) into a local OCI-layout store, verifies it
fail-closed, and prints the path. It is the only command that touches a network, and it is
idempotent: a second run offline succeeds from what is already local (CX-LOCAL). ``fetch`` then
``score --runner sim --registry <path>`` is the real end-to-end path; both honour
``$ASTRO_MINE_HUB_REGISTRY``, the same convention ``astro-mine-sim run`` uses. Needs the ``[fetch]``
extra for the Hub client, and fails with an install hint — never an ImportError — without it.

``submit`` is the write path to a leaderboard (G2.14). ``--hub-ref`` references an artifact **by
digest** — the contract bench.md §6 fixes, verified fail-closed before it runs — and is the path a
community submission should take; ``--policy-ref`` is the local/dev path, sandboxed like any
submission but not reproducible (nothing pins what the reference resolves to), so it is not
leaderboard-grade. The Hub intake returns a job ticket, which ``--wait`` follows to a verdict. The
bearer token comes from ``$ASTRO_MINE_BENCH_TOKEN`` (or ``--token-file``), never a flag: a token on
argv lands in shell history and in ``ps``. Identity is the token's alone (bench#29). Needs the
``[submit]`` extra for the HTTP client; reads and ``score``/``list`` need neither it nor an account.

The ``eval-worker`` subcommand is the single-seed rollout entry point Cloud fans out for scale-out
evaluation (RM-P1-BENCH-11): ``astro-mine-bench eval-worker --scenario-id … --policy-ref …`` writes
one seed's ``metrics.parquet`` + ``trace.mcap`` (needs the ``[cloud]`` + ``[recording]`` extras). It
is the *same* argv the leaderboard runs inside a sandbox to execute an untrusted submission
(bench#30). It is delegated to :func:`~astro_mine.bench.eval.run_worker`, which owns its own
argument parsing.

``zoo-sync`` and ``zoo-search`` are the hosted-catalog operator commands (bench#33): ``zoo-sync``
is the **migration/seed utility** that populates the Postgres/pgvector catalog from the packaged
zoo's ``scenario.json`` documents, and ``zoo-search`` runs a similarity query against it. Both need
a ``--dsn`` (or ``$ASTRO_MINE_BENCH_CATALOG_DSN``) and the ``[leaderboard]`` extra; neither is
required to score a scenario offline.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from astro_mine.bench.baseline import (
    REFERENCE_EPISODE_RUNNER_ID,
    RunnerNotAvailableError,
    ScoringRefused,
    default_policy_for,
    load_runner_provider,
    run,
)
from astro_mine.bench.content import (
    DEFAULT_CONTENT_SOURCE,
    STORE_ENV,
    FetchError,
    default_store_path,
    fetch_scenario_content,
    resolve_store_path,
)
from astro_mine.bench.leaderboard._jobs import JobRecord
from astro_mine.bench.leaderboard._models import Submission
from astro_mine.bench.metrics import Scorecard
from astro_mine.bench.submit import TOKEN_ENV, SubmitError
from astro_mine.bench.zoo import (
    ANCHOR_SCENARIO_ID,
    CATALOG_DSN_ENV,
    FilesystemCatalog,
    list_scenarios,
    load_scenario,
    open_sql_catalog,
)

__all__ = ["main"]


def _format_scorecard(card: Scorecard) -> str:
    """Render a scorecard as an aligned, human-readable table.

    The runner is named in the header (and folds into ``content_hash``), so a fixture score and a
    Sim score are legible as different provenance rather than only different numbers (G1.1/G1.8).
    """
    lines = [
        f"scenario:  {card.scenario_id}",
        f"runner:    {card.runner}",
        f"scorecard: {card.content_hash}",
        "",
    ]
    width = max((len(m.metric) for m in card.metrics), default=0)
    for m in card.metrics:
        value = "n/a" if m.value is None else f"{m.value:.6g}"
        arrow = "up" if m.direction.value == "higher_better" else "down"
        lines.append(f"  {m.metric:<{width}}  {value:>12} {m.unit:<12} ({arrow}-better, n={m.n})")
    if card.runner == REFERENCE_EPISODE_RUNNER_ID:
        lines.append("")
        lines.append(
            "scored with the reference runner — a deterministic trace fixture, not a physics "
            "engine. Use `--runner sim` for a Sim-backed run."
        )
    return "\n".join(lines)


def _score(args: argparse.Namespace) -> int:
    """Run the baseline on a zoo scenario and print its scorecard.

    ``--runner`` selects the runner by name (default ``fixture``, the always-available
    dependency-clean built-in). Bench discovers the runner, it never imports one — a third party
    like ``sim`` resolves an injected runner from ``astro-mine-sim[bench]`` through the
    ``astro_mine.bench.runners`` entry-point group, and fails with an install hint (not a
    traceback) when it is absent (CX-LOCAL; conventions.md §1.1, §7).
    """
    try:
        spec = load_scenario(args.scenario_id)
    except KeyError as exc:
        print(f"error: {exc}", file=args.stderr)
        return 2
    try:
        provider = load_runner_provider(args.runner)
    except RunnerNotAvailableError as exc:
        print(f"error: {exc}", file=args.stderr)
        return 2
    # Constructing the runner can fail even when the provider is registered — an engine-backed
    # runner may need content or a store it cannot find (e.g. the `sim` runner without
    # $ASTRO_MINE_HUB_REGISTRY). Surface that as a clean error, never a traceback (CX-LOCAL): the
    # provider raises with an actionable message, so print it rather than letting it escape.
    try:
        # `--registry` (else $ASTRO_MINE_HUB_REGISTRY, else the XDG default) is passed as a plain
        # path: the provider Protocol types `store` as `object` precisely so Bench can hand a
        # runner its content store without naming an engine type (conventions.md §1.1). Only the
        # `fixture` built-in ignores it.
        store = _store_argument(args)
        episode_runner = provider.episode_runner(store)
        runner_id = provider.runner_id
        # The policy is the runner's to choose, not this CLI's. A capability-aware baseline has to
        # read each asset's SADF, which Bench never materializes (a ResolvedScenario carries content
        # hashes, not bundles) — so a runner that already resolves the content offers one through
        # the optional DefaultPolicyProvider seam, and Bench asks for it by name. The spec goes
        # with the store: it names *which* pinned assets this run scores, so the policy is built
        # against the same digests the episode runs on. Providers that offer none, the `fixture`
        # built-in included, fall back to BaselinePolicy unchanged (astro-mine-sim#61).
        policy = default_policy_for(provider, spec, store)
    except (RunnerNotAvailableError, RuntimeError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=args.stderr)
        return 2
    seeds = tuple(args.seeds) if args.seeds else None
    # A runner may decline mid-run: `sim` refuses a scenario whose pinned providers did not rebuild,
    # because the scorecard would be a claim about content it never modelled (astro-mine-sim#67).
    # That is a designed outcome with a message a user can act on — it names each unresolved pin and
    # the package that supplies it — so present it as an error, not a 30-line traceback that reads
    # as "this is broken" (#79; CX-LOCAL).
    #
    # Deliberately narrow: only `ScoringRefused`. Catching RuntimeError here would also swallow a
    # genuine engine bug into a clean message and lose the traceback that makes it debuggable.
    try:
        card = run(spec, policy, runner=episode_runner, runner_id=runner_id, seeds=seeds)
    except ScoringRefused as exc:
        print(f"error: {exc}", file=args.stderr)
        return 2
    output = card.model_dump_json(indent=2) if args.json else _format_scorecard(card)
    print(output, file=args.stdout)
    return 0


def _store_argument(args: argparse.Namespace) -> str | None:
    """The content store to hand a runner, or ``None`` to let it resolve its own.

    ``--registry`` wins; then ``$ASTRO_MINE_HUB_REGISTRY``; then the XDG default **if it exists**,
    so ``fetch`` followed by ``score --runner sim`` composes with no environment set up at all.
    Falls back to ``None`` when nothing is configured, which leaves the runner free to raise its own
    actionable "no content store" message rather than have Bench invent a path that is not there.
    """
    explicit = getattr(args, "registry", None)
    if explicit:
        return str(explicit)
    if os.environ.get(STORE_ENV):
        return None  # the runner reads the same variable; let it, unchanged (astro-mine-sim#63)
    default = default_store_path()
    return str(default) if default.exists() else None


def _fetch(args: argparse.Namespace) -> int:
    """Mirror a scenario's pinned content into a local store and print the path.

    Fail-closed throughout (bench.md §5; hub.md §9): a pin that does not resolve, does not
    reproduce its pinned digest, or does not re-verify locally aborts the command. The
    signer-pinning posture is printed rather than left implicit — a reader must not infer a
    guarantee that was not checked (bench#56 D6).
    """
    try:
        spec = load_scenario(args.scenario_id)
    except KeyError as exc:
        print(f"error: {exc}", file=args.stderr)
        return 2

    trusted_key: bytes | None = None
    if args.trusted_key is not None:
        try:
            trusted_key = Path(args.trusted_key).read_bytes()
        except OSError as exc:
            print(f"error: cannot read --trusted-key: {exc}", file=args.stderr)
            return 2

    store_path = resolve_store_path(args.registry)
    print(f"fetching {len(spec.content_refs())} pins into {store_path}", file=args.stdout)
    try:
        pins = fetch_scenario_content(
            spec,
            source=args.source,
            store=args.registry,
            trusted_key_pem=trusted_key,
            on_event=lambda message: print(f"  {message}", file=args.stdout),
        )
    except FetchError as exc:
        print(f"error: {exc}", file=args.stderr)
        return 2

    moved = sum(pin.size_bytes for pin in pins if pin.mirrored)
    fresh = sum(1 for pin in pins if pin.mirrored)
    print("", file=args.stdout)
    print(
        f"{len(pins)} pin(s) verified — {fresh} fetched ({moved / 1e6:.1f} MB), "
        f"{len(pins) - fresh} already present",
        file=args.stdout,
    )
    print(
        "verified by content digest"
        + (
            "; signer pinned to --trusted-key"
            if trusted_key is not None
            else "; signer not pinned (pass --trusted-key to pin)"
        ),
        file=args.stdout,
    )
    print("", file=args.stdout)
    print(f"content store: {store_path}", file=args.stdout)
    print(
        f"score it with:  astro-mine-bench score {args.scenario_id} "
        f"--runner sim --registry {store_path}",
        file=args.stdout,
    )
    return 0


def _list(args: argparse.Namespace) -> int:
    """List the scenario ids registered in the zoo."""
    for scenario_id in list_scenarios():
        print(scenario_id, file=args.stdout)
    return 0


def _dsn(args: argparse.Namespace) -> str | None:
    """The hosted-catalog DSN from ``--dsn`` or the environment; ``None`` if neither is set."""
    return args.dsn or os.environ.get(CATALOG_DSN_ENV)


def _zoo_sync(args: argparse.Namespace) -> int:
    """Seed the Postgres/pgvector catalog from the packaged zoo (bench#33 AC4 — the migration)."""
    dsn = _dsn(args)
    if not dsn:
        print(f"error: pass --dsn or set ${CATALOG_DSN_ENV}", file=args.stderr)
        return 2
    catalog = open_sql_catalog(dsn)
    seeded = catalog.seed_from(FilesystemCatalog())
    for entry in seeded:
        lineage = f" (after {entry.parent_id})" if entry.parent_id else ""
        print(f"indexed {entry.scenario_id} v{entry.version}{lineage}", file=args.stdout)
    backend = "pgvector" if catalog.uses_pgvector else "python cosine (sqlite)"
    print(f"\n{len(seeded)} scenario(s) indexed; similarity search: {backend}", file=args.stdout)
    return 0


def _zoo_search(args: argparse.Namespace) -> int:
    """Similarity-search the hosted catalog (pgvector's ``<=>`` on Postgres) — bench#33 AC2."""
    dsn = _dsn(args)
    if not dsn:
        print(f"error: pass --dsn or set ${CATALOG_DSN_ENV}", file=args.stderr)
        return 2
    hits = open_sql_catalog(dsn).search(" ".join(args.query), limit=args.limit)
    if not hits:
        print("no scenarios indexed; run `astro-mine-bench zoo-sync` first", file=args.stdout)
        return 0
    width = max(len(hit.entry.scenario_id) for hit in hits)
    for hit in hits:
        print(
            f"  {hit.entry.scenario_id:<{width}}  distance={hit.distance:.4f}  {hit.entry.name}",
            file=args.stdout,
        )
    return 0


def _submit(args: argparse.Namespace) -> int:
    """Submit a policy to a leaderboard and, with ``--wait``, follow it to a verdict.

    Two intakes, deliberately not equals. ``--hub-ref`` is the path a community submission should
    take: ``bench.md`` §6 fixes the contract as *"a leaderboard submission references Hub artifacts
    by digest"*, so the artifact is authenticated by content hash and signature and the entry stays
    reproducible. ``--policy-ref`` ships and is the local/dev path, but nothing pins what the
    reference resolves to, so it is not leaderboard-grade.

    Identity is never sent: it comes from the verified OIDC bearer token and nothing else, and the
    token is read from the environment so it never lands in shell history (bench#29).
    """
    from astro_mine.bench.submit import await_job, read_token, submit_hub, submit_policy

    if args.job is None and not args.scenario_id:
        print("error: --scenario-id is required unless --job is given", file=args.stderr)
        return 2

    try:
        token = read_token(args.token_file)
    except SubmitError as exc:
        print(f"error: {exc}", file=args.stderr)
        return 2

    try:
        if args.hub_ref is not None:
            job = submit_hub(
                args.to,
                scenario_id=args.scenario_id,
                hub_ref=args.hub_ref,
                token=token,
                method=args.method,
                author=args.author,
            )
            print(f"submitted: job {job.job_id} ({job.status.value})", file=args.stdout)
            if not args.wait:
                # Handing back a ticket and walking away is what the REST-only path already did.
                print(
                    f"follow it with: astro-mine-bench submit --job {job.job_id} --wait "
                    f"--scenario-id {args.scenario_id} --to {args.to}",
                    file=args.stdout,
                )
                return 0
            job = await_job(args.to, job.job_id)
            return _report_job(args, job)

        if args.job is not None:  # resume: follow a ticket from an earlier submit
            job = await_job(args.to, args.job)
            return _report_job(args, job)

        submission = submit_policy(
            args.to,
            scenario_id=args.scenario_id,
            policy_ref=args.policy_ref,
            token=token,
            method=args.method,
            author=args.author,
        )
    except SubmitError as exc:
        print(f"error: {exc}", file=args.stderr)
        return 1

    _print_submission(args, submission)
    return 0


def _report_job(args: argparse.Namespace, job: JobRecord) -> int:
    """Print a terminal job's outcome — the scored submission, or an honest refusal."""
    from astro_mine.bench.submit import SubmitError, get_submission, is_rejected

    if is_rejected(job):
        print(
            f"error: submission {job.status.value}: {job.detail or 'no detail given'}",
            file=args.stderr,
        )
        return 1
    if job.result_id is None:
        print(f"job finished {job.status.value} but carries no result id", file=args.stderr)
        return 1
    try:
        submission = get_submission(args.to, job.result_id)
    except SubmitError as exc:
        print(f"error: {exc}", file=args.stderr)
        return 1
    _print_submission(args, submission)
    return 0


def _print_submission(args: argparse.Namespace, submission: Submission) -> None:
    """Print the scored submission and, when it is on the board, its rank."""
    from astro_mine.bench.submit import rank_of

    if args.json:
        print(submission.model_dump_json(indent=2), file=args.stdout)
        return
    print(f"submission {submission.submission_id}", file=args.stdout)
    print(f"  scenario:  {submission.scenario_id}", file=args.stdout)
    print(f"  integrity: {submission.integrity}", file=args.stdout)
    for score in submission.scores:
        value = "n/a" if score.value is None else f"{score.value:g} {score.unit}"
        print(f"  {score.metric}: {value}", file=args.stdout)
    entry = rank_of(args.to, submission.scenario_id, submission.submission_id)
    if entry is not None:
        print(f"  rank: #{entry.rank} on {submission.scenario_id}", file=args.stdout)


# --- Per-verb argument sets -------------------------------------------------------------------
#
# Each verb's flags live in one function so they can be attached to *either* parser: this package's
# own `astro-mine-bench <verb>`, and the umbrella's `astro-mine <verb>` (RFC-0011 §3, wired in
# astro_mine.bench.umbrella). Declaring them once is what stops the two surfaces from drifting — a
# flag added here appears in both, and neither can quietly lose one. They are not underscore-
# prefixed because they are exactly that seam; everything else in this module still is.


def add_score_arguments(parser: argparse.ArgumentParser) -> None:
    """`score` — run a policy on a scenario and score it."""
    parser.add_argument(
        "scenario_id",
        nargs="?",
        default=ANCHOR_SCENARIO_ID,
        help=f"zoo scenario id to score (default: {ANCHOR_SCENARIO_ID})",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        metavar="SEED",
        help="seeds to score (default: the public set)",
    )
    parser.add_argument(
        "--runner",
        default="fixture",
        metavar="NAME",
        help=(
            "runner to score with (default: fixture). 'fixture' is a deterministic trace fixture, "
            "not a physics engine; 'sim' runs the real Sim engine and needs astro-mine-sim[bench] "
            "plus fetched content. The runner is recorded in the scorecard and its hash."
        ),
    )
    parser.add_argument(
        "--registry",
        default=None,
        metavar="PATH",
        help=(
            "local content store an engine-backed runner reads (default: "
            f"${STORE_ENV}, else the cache dir `astro-mine-bench fetch` writes to). "
            "Ignored by the fixture runner."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit the scorecard as JSON")


#: Shown under `fetch` on both surfaces — the size and the token are the two things a user needs
#: *before* starting a 461 MB download, not after it fails.
FETCH_DESCRIPTION = (
    "Resolve a scenario's pinned content by digest, mirror it into a local OCI-layout "
    "store, and verify it fail-closed. The anchor pulls ~461 MB (the world bundle is "
    "99.6% of it); a private registry needs $GITHUB_TOKEN with read:packages until the "
    "org flips public. Re-running is idempotent and works offline."
)


def add_fetch_arguments(parser: argparse.ArgumentParser) -> None:
    """`fetch` — download a scenario's pinned content into a local store."""
    parser.add_argument(
        "scenario_id",
        nargs="?",
        default=ANCHOR_SCENARIO_ID,
        help=f"zoo scenario whose pins to fetch (default: {ANCHOR_SCENARIO_ID})",
    )
    parser.add_argument(
        "--registry",
        default=None,
        metavar="PATH",
        help=(f"local store to populate (default: ${STORE_ENV}, else {default_store_path()})"),
    )
    parser.add_argument(
        "--from",
        dest="source",
        default=DEFAULT_CONTENT_SOURCE,
        metavar="REGISTRY",
        help=f"published registry to fetch from (default: {DEFAULT_CONTENT_SOURCE})",
    )
    parser.add_argument(
        "--trusted-key",
        default=None,
        metavar="PATH",
        help=(
            "public key pinning *whose* signature is accepted. Optional: content is verified by "
            "digest regardless, and the pins ship in this package. Without it a signature must "
            "still be present, intact and bound to the artifact — but any key satisfies it."
        ),
    )


#: Shown under `submit` on both surfaces.
SUBMIT_DESCRIPTION = (
    "Submit a policy to a leaderboard and, with --wait, follow it to a verdict. "
    "--hub-ref is the path a community submission should take: the artifact is "
    "referenced by digest and verified fail-closed, so the entry is reproducible. "
    "--policy-ref is the local/dev path — it runs sandboxed like any submission, but "
    "nothing pins what the reference resolves to, so it is not leaderboard-grade. "
    "The bearer token is read from $ASTRO_MINE_BENCH_TOKEN (or --token-file), never "
    "from a flag; identity comes from the token alone."
)


def add_submit_arguments(parser: argparse.ArgumentParser) -> None:
    """`submit` — submit a policy to a leaderboard."""
    intake = parser.add_mutually_exclusive_group(required=True)
    intake.add_argument(
        "--hub-ref",
        default=None,
        metavar="REF",
        help="Hub 'name:version' tag or 'sha256:' digest — the recommended, reproducible path",
    )
    intake.add_argument(
        "--policy-ref",
        default=None,
        metavar="MODULE:ATTR",
        help="importable policy reference (local/dev; not reproducible, so not leaderboard-grade)",
    )
    intake.add_argument(
        "--job",
        default=None,
        metavar="JOB_ID",
        help="follow an existing job from an earlier submit instead of submitting again",
    )
    parser.add_argument(
        "--scenario-id",
        default=None,
        help="the zoo scenario to submit against (required unless --job)",
    )
    parser.add_argument("--to", required=True, metavar="URL", help="leaderboard base URL")
    parser.add_argument("--method", default=None, help="display metadata: the method's name")
    parser.add_argument("--author", default=None, help="display metadata: the author's name")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="poll the job to a terminal status and print the resulting submission + rank",
    )
    parser.add_argument(
        "--token-file",
        default=None,
        metavar="PATH",
        help=f"read the bearer token from this file instead of ${TOKEN_ENV}",
    )
    parser.add_argument("--json", action="store_true", help="emit the submission as JSON")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astro-mine-bench", description=__doc__.splitlines()[0])
    subcommands = parser.add_subparsers(dest="command", required=True)

    score = subcommands.add_parser("score", help="run the baseline on a scenario and score it")
    add_score_arguments(score)
    score.set_defaults(func=_score)

    fetch = subcommands.add_parser(
        "fetch",
        help="download a scenario's pinned content into a local store",
        description=FETCH_DESCRIPTION,
    )
    add_fetch_arguments(fetch)
    fetch.set_defaults(func=_fetch)

    submit = subcommands.add_parser(
        "submit",
        help="submit a policy to a leaderboard",
        description=SUBMIT_DESCRIPTION,
    )
    add_submit_arguments(submit)
    submit.set_defaults(func=_submit)

    listing = subcommands.add_parser("list", help="list the scenarios in the zoo")
    listing.set_defaults(func=_list)

    sync = subcommands.add_parser(
        "zoo-sync", help="seed the Postgres/pgvector catalog from the packaged zoo"
    )
    sync.add_argument("--dsn", default=None, help=f"catalog DSN (default: ${CATALOG_DSN_ENV})")
    sync.set_defaults(func=_zoo_sync)

    search = subcommands.add_parser(
        "zoo-search", help="similarity-search the hosted scenario catalog"
    )
    search.add_argument("query", nargs="+", help="the free-text query to rank scenarios against")
    search.add_argument("--dsn", default=None, help=f"catalog DSN (default: ${CATALOG_DSN_ENV})")
    search.add_argument("--limit", type=int, default=5, help="how many hits to return")
    search.set_defaults(func=_zoo_search)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: object = None, stderr: object = None) -> int:
    """Parse ``argv`` and dispatch a Bench CLI command; returns the process exit code.

    ``stdout``/``stderr`` default to the real streams; tests inject buffers. Kept dependency-clean
    (core + pydantic): the CLI imports no engine and never touches Sim.
    """
    import sys

    args_list = list(sys.argv[1:] if argv is None else argv)
    # `eval-worker` (RM-P1-BENCH-11) owns its own argparse and rides the [cloud]/[recording] extras,
    # so it is delegated verbatim rather than folded into the dependency-clean top-level parser.
    if args_list and args_list[0] == "eval-worker":
        from astro_mine.bench.eval import run_worker

        return run_worker(args_list[1:])

    parser = _build_parser()
    args = parser.parse_args(argv)
    args.stdout = stdout if stdout is not None else sys.stdout
    args.stderr = stderr if stderr is not None else sys.stderr
    result: int = args.func(args)
    return result
