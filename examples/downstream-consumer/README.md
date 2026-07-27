# Downstream consumer example

The canonical pattern for **consuming `astro-mine-core` during private incubation**
(`docs/VERSIONING.md` §5): a **tag-pinned `uv` Git source**, resolved reproducibly with
`uv sync --locked`, authenticated in CI with a read-scoped PAT. Every sibling repo
(worlds, prospect, fleet, sim, bench, link, cloud) copies this pattern — bump the tag at
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
