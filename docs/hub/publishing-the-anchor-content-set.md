# Publishing the anchor content set

The lunar-polar-ice-prospecting anchor scenario pins **nine content artifacts** by digest — one
world, six fleet assets, one resource prior, one contact plan. This runbook is the **manual,
by-component** procedure for publishing them to the org registry so anyone can fetch them by digest.

It is deliberately **not** a CI workflow. The world bundle is large and is not committed (it is
rebuilt from a multi-GB DEM by a documented local step), and the signing key must never live in a
repo or a CI secret — so publishing is a step a maintainer runs by hand with the org key. See
[hub#30](astro-mine-hub#30) for the decision.

## Prerequisites (two are not code)

1. **The org signing key** — a real ECDSA-P256 key held in the org's secure store (a password
   manager / KMS), **never** committed and **never** a CI secret. Generate it once with the one
   canonical signing-key command, `astro-mine hub keygen --out <secure-dir>` (writes `cosign.key` +
   `cosign.pub`), then move the private key into the secure store. The matching **public** key
   (`cosign.pub`) is published as the pinned trust anchor (below). This replaces the development key
   that seeded the local workspace registry.
2. **Remote-capable producer commands.** Each producer's `publish` must accept a remote registry
   — the `Registry(path)` → `open_registry(location)` change. All four producers are now modules in
   one distribution and the commands are one binary, so this is no longer four cross-repo PRs.
3. **GHCR write access** — a token with `write:packages`, exported as `GITHUB_TOKEN` (the Hub
   client's GHCR auth reads it automatically).
4. **The built artifacts** — the world's prebuilt bundle (`files/data/shackleton-*/bundle`) and the
   inputs the lighter producers rebuild from. The **authoritative per-artifact build recipes** live
   in
   [`src/astro_mine/bench/zoo/lunar_polar_ice_prospecting_v1/PROVENANCE.md`](../../src/astro_mine/bench/zoo/lunar_polar_ice_prospecting_v1/PROVENANCE.md)
   and `pins.json`. This runbook only adds the remote target, the org key, and verification.

## The registry target

```bash
export REGISTRY=ghcr.io/astro-mine          # or a local OCI-layout dir for a dry run, e.g. ./reg
export ORG_KEY=/secure/anchor-signing.key   # the org private key (never committed)
export GITHUB_TOKEN=…                        # a token with write:packages (GHCR auth)
```

`$REGISTRY` is the *only* thing that changes between publishing to GHCR and to a local registry —
the Hub client dispatches path-vs-URL automatically (`open_registry`). Publish to a local dir first
to rehearse the whole procedure offline.

## Publish, by component (signing is mandatory)

Each producer builds its own artifact and signs it with the **supplied** org key. Signing is
required — an unsigned artifact must never reach the trusted registry (hub.md principle 3). The
**world publishes its prebuilt bundle** (no rebuild):

```bash
# world — publish the prebuilt bundle
astro-mine worlds publish files/data/shackleton-0.4.0/bundle --registry "$REGISTRY" --key "$ORG_KEY"

# fleet — the six anchor assets (relay-orbiter, lander, prospecting-rover, excavator, hauler, isru-plant)
astro-mine fleet publish <asset> --oci --registry "$REGISTRY" --sign --key "$ORG_KEY"     # ×6

# prospect — the belief prior (parametric recipe; no rasters)
astro-mine prospect publish shackleton_water_ice_v1 --registry "$REGISTRY" --private-key "$ORG_KEY"

# link — the contact plan (see PROVENANCE.md for the full input set)
python scripts/link/build_anchor_contact_plan.py --registry "$REGISTRY" --key "$ORG_KEY" …
```

Publishing verifies at admission and fails closed: a signed artifact must prove it verifies
(signature + SLSA + SBOM) *before* it is indexed. Re-running against an already-published digest is a
no-op — digests are immutable; a fix is a new version, never an overwrite.

## Verify the nine pins resolve fail-closed

After publishing, confirm every pinned digest resolves and re-verifies from the target registry —
pin-for-pin against the digests the scenario already declares:

```bash
astro-mine hub verify --registry "$REGISTRY" shackleton-de-gerlache-v1:0.4.0 --trusted-key anchor-signing.pub
astro-mine hub verify --registry "$REGISTRY" astro-mine.fleet.relay-orbiter:0.1.0 --trusted-key anchor-signing.pub
# … the remaining seven (see zoo/lunar_polar_ice_prospecting_v1/pins.json for the id → digest list)
```

`--trusted-key anchor-signing.pub` pins verification to the org key (below); without it, `verify`
still checks the signature is intact but not *whose* it is.

`verify` fails closed on any tampered blob, missing attestation, or a signature that does not verify
against the pinned trust anchor. A **downstream** consumer never needs this repo: they fetch with
`astro-mine bench fetch` (bench#56), which pulls each pin by digest and re-verifies on the way in.

## The trust anchor

Consumers verify against a **pinned org public key**, not one carried alongside the artifact. The
org's public key is committed here as [`anchor-signing.pub`](../../anchor-signing.pub) (a public key is
not secret) — the private half stays in the org's secure store and is never committed. Point
verification at it with `astro-mine hub verify … --trusted-key anchor-signing.pub` (or
`HubClient(trusted_public_key_pem=…)` in Python). A registry that served a tampered artifact and a
matching rogue key would still fail, because the trusted key is pinned out of band.

The generic Hub client is *not* hardwired to this key — it takes the trust anchor at run time — so
`anchor-signing.pub` is org/deployment config that lives here (and travels with the anchor scenario's
provenance), not a constant baked into the library.

## Mirroring the two artifacts that are not part of the nine

The nine above are on `ghcr.io/astro-mine`. Two other pieces of real, pinned content are **not**, and
the workspace store is their only copy (`registry-inventory.json`, `astro-mine-platform#41`):

| Reference | Manifest digest | Pinned by |
|---|---|---|
| `excavation-gns:0.6.0` | `sha256:b3c5846b…` | the fidelity-crossover zoo entry |
| `shackleton_water_ice_pds_v1:1.0.0` | `sha256:182595d5…` | `prospect/priors/RECIPE.md` |

**Copy the bytes; do not re-publish from source.** Re-publishing re-stamps the manifest and changes
the digest — which is how the anchor pins drifted on the last re-pin, and is why
`astro_mine.bench.content` mirrors rather than rebuilds. Both artifacts are pinned *by digest*, so a
mirror that changes the digest would not be a mirror. `oras cp` is a byte-exact copy and preserves
it; the OCI manifest does not contain the repository name, so the digest survives even under a
different name.

Both carry three attestations each (signature, SLSA provenance, CycloneDX SBOM) attached through OCI
Referrers. `--recursive` is **not optional** — a mirror without them fails the fail-closed verify at
pull, which is the same as not having mirrored it.

```bash
# oras is not installed by default on this workstation
curl -sLO https://github.com/oras-project/oras/releases/download/v1.2.3/oras_1.2.3_linux_amd64.tar.gz
mkdir -p ~/.local/bin && tar -xzf oras_1.2.3_linux_amd64.tar.gz -C ~/.local/bin oras

export STORE=/mnt/d/MyProjects/AstroMine/files/hub-registry
echo "$GITHUB_TOKEN" | oras login ghcr.io --username "$USER" --password-stdin

# The surrogate tier. `excavation-gns` already conforms to conventions.md §13, so the name carries over.
oras cp --recursive --from-oci-layout "$STORE:excavation-gns:0.6.0" \
        ghcr.io/astro-mine/excavation-gns:0.6.0

# The PDS-conditioned resource field. Published under its **conforming** name (§13: no underscores,
# no version in the name), so the artifact-name sweep does not have to move it a second time.
oras cp --recursive --from-oci-layout "$STORE:shackleton_water_ice_pds_v1:1.0.0" \
        ghcr.io/astro-mine/shackleton-water-ice-pds:1.0.0
```

Then confirm each resolves from the remote and still verifies, and that the digest did not move:

```bash
oras manifest fetch --descriptor ghcr.io/astro-mine/excavation-gns:0.6.0
# expect sha256:b3c5846b5df7d241ea9df1ddf97729c138576c06709577a609aea37cfc923f18
oras manifest fetch --descriptor ghcr.io/astro-mine/shackleton-water-ice-pds:1.0.0
# expect sha256:182595d53cb60a0ff78b79cb0ade5e75e166007f9be24145d223e9adbbdece9b

astro-mine hub verify --registry ghcr.io/astro-mine excavation-gns:0.6.0 \
    --trusted-key anchor-signing.pub
```

Finally, record the mirror: add `"ghcr"` to each entry's `mirrored_to` in `registry-inventory.json`
and remove it from `SINGLE_COPY` in `tests/hub/test_registry_inventory.py`. That test pins the set
exactly, so it fails until both moves are made together — which is what stops the mirror being
claimed before it happened.

## Before the public flip

During private incubation the org's GHCR packages are **private**: a pull still needs a token, so
publishing here makes the content fetchable **for the org**, not yet for outsiders. The set becomes
publicly fetchable when the org flips public (the first-runnable-benchmark milestone) — no republish
needed, the digests are already immutable.
