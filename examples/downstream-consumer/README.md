# Downstream consumer example

> **SUPERSEDED — the pattern below is no longer how anything consumes this platform.**
> Consolidation merged all 18 component packages into one distribution and retired the whole
> `[tool.uv.sources]` Git-pin matrix along with the per-component wheels
> (`docs/CONSOLIDATION_PLAN.md` §4). There is no `astro-mine-core` distribution to pin, and no
> sibling repo left copying this block. **Today a consumer writes
> `dependencies = ["astro-mine-platform"]`** — or `astro-mine-cli`, which brings the platform with
> it — and gets every component at once.
>
> It is left here, runnable and unedited, for two reasons. It still resolves: the
> `astro-mine-core` repo and its `v0.1.0` tag both exist, so `uv sync --locked` works and the
> example demonstrates what it always demonstrated. And it is the record of how the private
> incubation actually distributed code before the consolidation — rewriting it would erase that
> rather than correct it, the same reason `CONSOLIDATION_PLAN.md` §4 keeps its own superseded text.
>
> Retargeting it at `astro-mine-platform` would need a tag to pin, and this repo has cut none.
> Pinning a branch is what the example itself argues against.

The pattern, as it was: **consuming `astro-mine-core` during private incubation**
(`docs/VERSIONING.md` §5) via a **tag-pinned `uv` Git source**, resolved reproducibly with
`uv sync --locked`, authenticated in CI with a read-scoped PAT. Every sibling repo
(worlds, prospect, fleet, sim, bench, link, cloud) copied this pattern — bumping the tag at
each integration milestone.

## The mechanism

`pyproject.toml` declares Core as a Git dependency pinned to an **annotated tag** (the
version identity — `VERSIONING.md` §2.1), never `branch = "main"`:

```toml
[tool.uv.sources]
astro-mine-core = { git = "https://github.com/astro-mine/astro-mine-core.git", tag = "v0.1.0" }
```

`uv.lock` records the exact resolved commit, so `uv sync --locked` is byte-for-byte
reproducible and fails if the lock is stale.

## Authentication (`CORE_REPO_TOKEN`)

`astro-mine-core` is **private**, so the Git fetch needs a credential. CI supplies a
read-scoped PAT as the secret **`CORE_REPO_TOKEN`** (an org-level Actions secret visible
to the consuming repos; a fine-grained token with *Contents: Read* on `astro-mine-core`
is sufficient). The job rewrites the HTTPS remote to carry the token:

```bash
git config --global url."https://x-access-token:${CORE_REPO_TOKEN}@github.com/".insteadOf "https://github.com/"
uv sync --locked
```

This only affects `uv`'s fetch inside CI — `origin` remotes stay on plain HTTPS, so
GitHub Desktop is unaffected. The publish side of distribution (the GHCR schema bundle)
uses the workflow's own `GITHUB_TOKEN`, not this PAT.

## Run it

```bash
uv sync --locked          # resolves astro-mine-core from the pinned tag
uv run python check_consumer.py
```

`check_consumer.py` imports Core's typed models and asserts `__version__` matches the
pinned tag — the proof that the wheel (not just the namespace) installed.

## Updating the pin

When Core cuts a new tag, bump `tag = "vX.Y.Z"` in `pyproject.toml`, then
`uv lock` and commit the refreshed `uv.lock`.
