# Architecture

`astro-mine-seal` is a component of the [Astro-Mine](https://github.com/astro-mine)
platform. Its architecture is specified in the platform **docs** repo, not here — this
file is the conventional pointer back to that source of truth
([`conventions.md` §13](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md)).

- **Design record** —
  [`rfc/0005-seal-supply-chain-companion.md`](https://github.com/astro-mine/docs/blob/main/rfc/0005-seal-supply-chain-companion.md):
  the accepted RFC that creates this package — a thin Core companion owning the whole
  artifact-integrity domain (signing, verification, SLSA provenance, SBOM, verify-twice),
  built on Core's frozen `registry.Signature` / `Verifier` surface, keeping Core
  crypto-free.
- **Component design** —
  [`architecture/seal.md`](https://github.com/astro-mine/docs/blob/main/architecture/seal.md):
  purpose, principles, runtime, data, integration, security, and roadmap alignment for
  this package.
- **Consolidated design** —
  [`architecture/guard.md` §9.5](https://github.com/astro-mine/docs/blob/main/architecture/guard.md):
  "Conventional security & supply chain" — the artifact-integrity design this package
  consolidates (the fail-closed signed-load gate and the Fleet/Hub/Guard signer that
  Seal dedupes).
- **Cross-cutting standards** —
  [`architecture/conventions.md`](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md):
  the platform-wide technology, schema, packaging, security (§9), and versioning
  conventions every component follows.
- **System integration view** —
  [`architecture/system.md`](https://github.com/astro-mine/docs/blob/main/architecture/system.md):
  how the components fit together end to end.

See also the [charter](https://github.com/astro-mine/docs/blob/main/charter/) (vision)
and the [roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/) (phased delivery).

> **Dependencies** — Seal is the platform's single home for the `cryptography` dependency
> (Core ships no crypto, by design). It depends only on tag-pinned `astro-mine-core` and
> `cryptography` — and, deliberately, on **no registry**: the `attest()` / `verify()`
> verify-twice orchestrators (`RM-P1-SEAL-03`) drive the `AttestationStore` port (`str`
> digests, `bytes` payloads), so Hub adapts its OCI registry to Seal rather than Seal
> depending on Hub.
