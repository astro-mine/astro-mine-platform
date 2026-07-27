# Download gating — license, verification tier, and dual-use capability (RM-P1-HUB-05).
#
# hub.md §3/§9 and §11 specify OPA with "data-driven Rego bundles, so governance/export-control
# rules evolve without code changes". This is that policy: the *logic* lives here, every *rule
# datum* (allowed licenses, verified namespaces, the gated capability tags) lives in `data.json`
# — so widening a license set or gating a new capability is a bundle release, not a code change
# plus a redeploy.
#
# It is the exact rule the pure-Python evaluator in `_policy.py` implements (that one is the
# offline tier-1 fallback, hub.md principle 7 — a laptop must gate correctly with no OPA to run).
# A shared conformance suite feeds identical inputs to both and asserts identical outcomes, so the
# two cannot silently drift.
#
# The gate FAILS CLOSED: `allow` is `false` by default, and every rule below is a reason to deny.
#
#   input := {
#     "reference":         "pol:1.0.0",
#     "license":           "Apache-2.0",      # null when the manifest declares none
#     "namespace":         "open",
#     "capability_tags":   ["operational_targeting"],
#     "grants":            [],                 # capability grants the requester holds
#     "allowed_licenses":  ["Apache-2.0"],     # what the requester will accept (data.json default)
#     "require_verified":  false
#   }
package astro_mine.hub.download

import rego.v1

policy := data.astro_mine.hub.policy

# The bundle revision — recorded alongside every decision for auditability (hub.md §5, §10).
version := policy.version

default allow := false

allow if {
	license_ok
	verified_ok
	count(ungranted) == 0
}

# 1. License must be present and permitted (charter §10.4 — Apache-2.0 is the default).
license_ok if input.license in input.allowed_licenses

# 2. A requester may demand a reviewed artifact; only curated/verified namespaces satisfy it.
verified_ok if not input.require_verified

verified_ok if input.namespace in policy.verified_namespaces

# 3. Every gated (dual-use) capability the artifact declares needs a matching grant — Core's
#    taxonomy (`operational_targeting` et al., conventions.md §12), consumed, never redefined.
ungranted contains tag if {
	some tag in input.capability_tags
	tag in policy.gated_capability_tags
	not tag in input.grants
}

# The first failing rule denies, in the same order the Python evaluator applies them.
code := "license_denied" if not license_ok

code := "verification_required" if {
	license_ok
	not verified_ok
}

code := "capability_gated" if {
	license_ok
	verified_ok
	count(ungranted) > 0
}

code := "allowed" if allow

reason := "allowed" if allow

reason := sprintf("license %v is not permitted", [input.license]) if not license_ok

reason := "a verified/curated artifact is required" if {
	license_ok
	not verified_ok
}

reason := sprintf("gated capability %v requires an access grant", [sort(ungranted)]) if {
	license_ok
	verified_ok
	count(ungranted) > 0
}

# The decision document the Hub policy engine reads back.
decision := {
	"allow": allow,
	"code": code,
	"reason": reason,
	"version": version,
}
