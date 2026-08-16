# `embargo/` — the seeds are not here any more

The **embargoed held-out seed sets** for the Bench scenario zoo — the anti-gaming reserve
(`docs/architecture/bench.md` §9) — used to live in this directory, committed in plaintext.

**They now live in the private repository [`astro-mine/embargo`](https://github.com/astro-mine/embargo).**
This directory holds only this file, so that a reader who lands here from a path, a script or a
stack trace is told where to go rather than finding nothing.

## Why they left

They were committed here deliberately, for CI verifiability, on the standing assumption that
`astro-mine-platform` was private. The public flip retires that assumption — and it retires it for
**every commit in the history**, not just `HEAD`.

That is the part the old version of this file did not say. It instructed a reader to *rotate every
held-out set here* before the flip, which is necessary and not sufficient: rotating and committing
new sets into a soon-to-be-public tree republishes the same problem one commit later. The seeds had
to stop living in a public git tree at all.

The mechanism needed no new code. `astro_mine.bench.leaderboard.resolve_embargo_root` already read
**`$ASTRO_MINE_BENCH_EMBARGO_ROOT`** and fell back to this directory only when unset
(astro-mine-platform#15, wired through the API seeder in astro-mine-api#19). What was missing was the
decision and the move, not the seam.

## Using the store

```bash
git clone https://github.com/astro-mine/embargo.git
export ASTRO_MINE_BENCH_EMBARGO_ROOT="$PWD/embargo"
```

Then `load_heldout_seeds` and `scripts/bench/seal_heldout_seeds.py` both resolve through it. Without
the variable set, both fall back here and fail with a path that leads to this file.

## The old sets are dead

Rotated on 2026-08-16, before the flip. The retired values remain in this repository's git history
and that is accepted — rotating first is precisely what makes publishing the history safe. They were
also *sequential* (`900001`–`900012`, beside a public set of `1001`–`1005`), so anything scored
against them was weaker than it looked even while the repository was private.

## What CI verifies now

`tests/bench/test_zoo_anchor.py` splits the old single assertion in two:

- **absence** — that no `heldout_seeds.json` exists anywhere in the working tree. Unconditional,
  needs no secret, and is a strictly stronger property than the "outside `src/`" check it replaces.
- **the commitment** — that the sealed payload still hashes to the anchor's `heldout_commit`. This
  needs the store, so it runs only where `$ASTRO_MINE_BENCH_EMBARGO_ROOT` resolves to one. A store
  that is present but *stale* fails; an unreachable store skips with a reason that says in as many
  words that nothing was verified.

Give CI a read credential for `astro-mine/embargo` — the same shape as `CORE_REPO_TOKEN`, which the
org already uses for private cross-repo reads — and the commitment check runs there too.

## Still outstanding

The encrypted, eval-time-only disclosure tier (`bench.md` §9) — seeds sealed so that even a reader
of the store cannot see them outside a scoring run. It remains a CX-SEC follow-up. Moving the store
out of a soon-to-be-public tree was the part that could not wait for it.
