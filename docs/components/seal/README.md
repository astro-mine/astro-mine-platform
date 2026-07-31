# astro-mine-seal

**Artifact-integrity companion for [Astro-Mine](https://github.com/astro-mine).**
The single home for the platform's **signing, verification, SLSA provenance, and SBOM** —
a thin [Core](https://github.com/astro-mine/astro-mine-core) companion built on Core's
already-frozen `registry.Signature` / `Verifier` surface and the `astro_mine.core.hashing`
content-hash primitive. Every producer (Fleet, Hub, Guard, and the growing publisher
frontier) signs a **seal** on its artifacts, and the intactness of that seal is what
verification tests — hence the name. Core stays crypto-free; **this package is the one
home for the `cryptography` dependency.**

> **Status:** Phase 1 — the signer (`RM-P1-SEAL-02`) and the SLSA / SBOM / verify-twice
> surface (`RM-P1-SEAL-03`) have landed.
> See [RFC-0005](https://github.com/astro-mine/docs/blob/main/rfc/0005-seal-supply-chain-companion.md),
> [`architecture/seal.md`](https://github.com/astro-mine/docs/blob/main/architecture/seal.md),
> and the [Phase-1 roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/phase-1-autonomy-studio.md).

## Layout

```
src/astro_mine/seal/
├── _signing.py        # generate_keypair / sign_digest / verify_signature / make_verifier
├── _attest.py         # build_slsa_provenance / build_cyclonedx_sbom / attest ; AttestationStore
├── _supply_chain.py   # verify-twice: verify(...) ; DEFAULT_REQUIRED = (signature, slsa, sbom)
└── __init__.py        # facade: the public surface (and __version__)
```

The signing/attestation stack consolidates Hub's `supply_chain/` module and dedupes the
byte-compatible signer copies in Fleet and Guard. It is the platform's single home for the
`cryptography` dependency; Core ships no crypto, by design.

## Verify twice, registry-agnostically

`attest()` and `verify()` are the publish- and pull-side orchestrators. They are
**registry-agnostic**: they drive the `AttestationStore` port — `str` digests, `str` media
types, `bytes` payloads, and **no registry type** — so Seal never depends on Hub, and Hub,
Bench, Fleet, or any producer binds whatever content-addressed store it has.

```python
from astro_mine.seal import attest, verify

attest(store, subject, private_key_pem=key, name="rover", version="0.1.0",
       builder_id="astro-mine-hub")          # sign + attach SLSA provenance + SBOM
verify(store, subject, trusted_public_key_pem=pub)   # at admission ... and again at pull
```

`verify()` **fails closed**: it returns `None` only when the artifact's bytes are intact,
*every* attached signature verifies over the subject under the pinned key, SLSA provenance
is present and well-shaped, and an SBOM is present and CycloneDX. A tampered artifact, a
tampered attestation, a missing or bad signature, an untrusted key, a garbage document, an
unknown `require` token, or any error from the store all raise `SupplyChainError` — never a
permissive default.

## Development

Seal is part of the [`astro-mine-platform`](../../../README.md) distribution — one repository, one
environment, one test suite. See [`docs/DEVELOPMENT.md`](../../DEVELOPMENT.md) for setup, then run
this component's suite with its own CI selection:

```bash
python scripts/test.py seal
```

See [CONTRIBUTING.md](https://github.com/astro-mine/.github/blob/main/CONTRIBUTING.md) for the full workflow.

## License

Apache-2.0 — see [LICENSE](../../../LICENSE). Copyright Astro-Mine project contributors.
