# Registry inventory — what this tree has published, and where else it exists

`registry-inventory.json` is the committed record of every artifact this repository has published to
a Hub registry: its digest, whether anything in the tree pins it, and whether a second copy exists.
This file says what the fields mean and why the record is kept here rather than read from a registry.

## Why a committed record at all

[`hub.md` §2](https://github.com/astro-mine/docs/blob/main/architecture/hub.md) principle 1 is
unambiguous: *"a given `name:version` resolves to one immutable digest; tags are mutable pointers,
digests are forever."*

Immutability is a property of **the record**, not of the store. A registry enforces it only for
names it currently holds — ask it about a name that was pruned and it answers "nothing here", which
is indistinguishable from "never published". So if the only record of what `name:version` meant lives
inside the store, **pruning the store retroactively undoes immutability**: the slot is free, and the
next publish takes it.

That is not hypothetical. On 2026-08-08 a prune of the workspace store removed
`excavation-gns:0.2.0`–`0.5.0`, tags and blobs. `0.4.0` is the tier the published cost curve in
`bench/zoo/lunar_polar_ice_excavation_fidelity_v1/CROSSOVER.md` was measured against. Its digest
survived only because that document happened to write it down. The other three did not, and are
recorded here as `lost` with no digest — which is the honest state, and the argument for this file.

## Fields

| Field | Meaning |
|---|---|
| `disposition` | `published` · `lost` · `ephemeral` — see below |
| `producer` | The component whose publish path minted it |
| `manifest_digest` | The OCI manifest digest — what `Registry.resolve()` returns and what a scenario `content_hash` pins |
| `bundle_digest` | The signed payload hash from the Core `PluginManifest`'s provenance block. For a surrogate tier this is the hash the signature covers, and it is **not** the manifest digest |
| `mirrored_to` | Keys into the top-level `mirrors` map. **Empty means one copy exists** |
| `pinned_by` | Repository-relative paths that resolve this artifact. Empty for `ephemeral` |
| `migrates_to` | Present when the published name predates `conventions.md` §13 — the conforming name the artifact takes at the flip-time sweep. Ten entries carry one |
| `note` | Why it is what it is |

### Dispositions

- **`published`** — real content. Prune it only if `mirrored_to` is non-empty, and never if
  `pinned_by` is non-empty and it is the last copy.
- **`lost`** — was published, no longer resolvable anywhere. Kept deliberately: the entry is what
  stops the name being silently reused. An entry with a `bundle_digest` can have a rebuild *checked*;
  an entry without one cannot, and republishing it is refused outright.
- **`ephemeral`** — a fixture. Safe to prune; regenerate by re-running whatever made it. Nothing
  resolves it by digest.

## The artifact-name migration worklist

Ten published names predate `conventions.md` §13, and each carries the conforming name it becomes
(`migrates_to`). **This file is the worklist**, because it is the only record of the *published* set;
`tests/hub/test_artifact_names.py` records what the *tree references*, and the two are not the same
list. `shackleton_water_ice_pds_v1` is exactly the difference: it is published, it is
non-conforming, and since prior publishing stopped defaulting to the recipe key
(astro-mine-platform#34) nothing in the tree references it by that name any more. It still has to be
re-published, and this is where that is tracked.

Registry names are immutable, so each is a **re-publish under a new name, not a rename** — new
digests, a re-pinned zoo, and every previously published scorecard still resolving by digest. §13
requires it as one sweep, so the registry never carries a half-migrated set.

## Mirroring, and the one entry still unmirrored

The §13 migration re-published ten artifacts, and for a while the zoo pinned digests that existed
only in the workspace store — the single-point-of-failure this file exists to record, re-created by
fixing a different problem. They are mirrored now (astro-mine-platform#41), and they were re-signed
with the **org** key first: the migration had signed them with the workspace development key, which
`verify` rejects against the published `anchor-signing.pub`, so mirroring them as-is would have
published content that fails closed for every consumer.

`excavation-gns:0.6.0` is the one entry left in `SINGLE_COPY`, and it is the original one. Note that
re-publishing it would not be mirroring it: the tier the tree actually depends on was destroyed by
the 2026-08-08 prune and cannot be rebuilt (astro-mine-platform#42).

## The two entries that need mirroring

`excavation-gns:0.6.0` and `shackleton_water_ice_pds_v1:1.0.0` are `published`, are pinned by
something in the tree, and have `mirrored_to: []`. The workspace store is their only copy, on a
working tree nothing backs up.

The workspace convention says the anchor content set is published to `ghcr.io/astro-mine` and so the
local store is "a convenience, no longer the only source". **That is true of the nine anchor packages
and false of these two** — GHCR holds exactly nine containers, and neither of these is among them.
`astro-mine-platform#41` tracks closing that gap; the exact `oras cp` procedure is in
[`docs/hub/publishing-the-anchor-content-set.md`](docs/hub/publishing-the-anchor-content-set.md)
under "Mirroring the two artifacts that are not part of the nine". Until it runs, a prune of either
is unrecoverable in the way the 2026-08-08 prune already was.

## What consumes this file

`scripts/surrogate/publish_surrogate.py` reads it to decide whether a publish is a rebuild of
something already recorded, and refuses to mint a new artifact under a recorded name unless the
operator states the digest they expect and it matches. See the guard's own docstring for why the
check cannot be founded on the registry.

`tests/hub/test_registry_inventory.py` keeps the record internally consistent and in step with what
the zoo pins. It deliberately does **not** read a registry: the record has to be verifiable on a
clean clone, which is the whole point of committing it.

## Keeping it current

Publishing something the tree will pin means adding its entry in the same change. There is no
generator, and that is deliberate — a file regenerated from a registry would inherit exactly the
weakness it exists to remove.
