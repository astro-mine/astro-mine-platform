# astro-mine-cli

**The command line is not in this repository.** It is a separate distribution and a separate repo:
[`astro-mine/astro-mine-cli`](https://github.com/astro-mine/astro-mine-cli), which depends on this
package and provides the single `astro-mine` executable.

```console
$ pip install astro-mine-cli        # brings astro-mine-platform with it
$ astro-mine <component> <verb>
```

`astro-mine-platform` ships **no console scripts** ([platform#1](https://github.com/astro-mine/astro-mine-platform/issues/1)).
Four `python -m` entry points remain here, each machine-facing plumbing something already depends
on rather than a command a person types — see [Command line](../../../README.md#command-line) in the
root README for the list.

## Why this file is a pointer and not a copy

It used to be the CLI repo's README, copied in by the consolidation. That copy described a design
that no longer exists — verbs discovered through the `astro_mine.cli` entry-point group, a
distribution with "no runtime dependencies, not one", and a separate binary per component — and
every one of those premises was reversed when the CLI was re-architected on the platform and then
moved back out of it.

Keeping a second copy of another repo's README in this one is how that happens: it has no reader
who would notice it going stale, and no test that could. So this file states where the thing lives
and stops. Documentation for the CLI belongs with the CLI.

## Design authority

- [RFC-0011](https://github.com/astro-mine/docs/blob/main/rfc/0011-umbrella-cli.md) — the umbrella
  CLI and the naming rule.
- [`conventions.md` §13](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md) —
  one executable, one grammar. Any `astro-mine-<component>` binary named in an older document is
  historical.
- [`architecture/cli.md`](https://github.com/astro-mine/docs/blob/main/architecture/cli.md) — the
  component design.
