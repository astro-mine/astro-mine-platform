@0xb9f4d2a7c1e80356;
# Message catalog v0.1 — canonical Cap'n Proto schema for the per-tick hot-path
# observation family (RM-P0-CORE-04). Zero-copy decode for per-tick sensor/telemetry
# streams (conventions.md §3; core.md §8). Loaded at runtime by
# astro_mine.core.messages.hotpath; the Pydantic models in
# astro_mine.core.messages.model mirror this schema and a round-trip test guards drift.
#
# Cap'n Proto requires lowerCamelCase field names; the Pydantic snake_case fields map
# 1:1 (battery_soc_j <-> batterySocJ). Optional Float64 scalars carry a companion
# `has*` Bool (Cap'n Proto primitives have no null); optional Text/struct fields use
# pointer presence (`_has`). All quantities SI; spatial values resolve in `frame`.

struct Vec3 {
  x @0 :Float64;
  y @1 :Float64;
  z @2 :Float64;
}

struct Quat {
  x @0 :Float64;
  y @1 :Float64;
  z @2 :Float64;
  w @3 :Float64;
}

struct Transform {
  translationM @0 :Vec3;
  rotationQuatXyzw @1 :Quat;
}

# Typed frame/time primitives (RM-P0-CORE-06; astro_mine.core.units). A `frame` names a
# SPICE frame (name + class [+ optional center]); an `epoch` is SPICE ephemeris time
# (tdbSeconds + scale). Optional Text `center` uses pointer presence.
struct ReferenceFrame {
  name @0 :Text;
  frameClass @1 :Text;
  center @2 :Text;                # optional (pointer presence)
}

struct Epoch {
  tdbSeconds @0 :Float64;
  scale @1 :Text;
}

# Additive hot-path parity for the full units vocabulary (RM-P1-CORE-07; RFC-0007 §1d).
# The control plane (units.proto) carries these; these structs let a future per-tick
# message carry a georeferenced CRS or an epoch window zero-copy without re-deriving the
# shape. Optional Text `projection`/`datum` use pointer presence; no Earth default.
struct PlanetaryCRS {
  body @0 :Text;
  bodyFixedFrame @1 :Text;
  referenceRadiusM @2 :Float64;
  projection @3 :Text;            # optional (pointer presence)
  datum @4 :Text;                 # optional (pointer presence)
}

struct EpochWindow {
  start @0 :Epoch;
  end @1 :Epoch;
}

struct StateSample {
  agentId @0 :Text;
  frame @1 :ReferenceFrame;
  pose @2 :Transform;
  linearVelocityMps @3 :Vec3;     # optional (pointer presence)
  angularVelocityRps @4 :Vec3;    # optional (pointer presence)
  batterySocJ @5 :Float64;
  hasBatterySocJ @6 :Bool;
  temperatureK @7 :Float64;
  hasTemperatureK @8 :Bool;
  mode @9 :Text;                  # optional (pointer presence)
}

struct SensorReading {
  sensor @0 :Text;
  values @1 :List(Float64);
  unit @2 :Text;                  # optional (pointer presence)
  resourceSpecies @3 :Text;       # optional (pointer presence)
  noiseSigma @4 :Float64;
  hasNoiseSigma @5 :Bool;
  valid @6 :Bool;
}

struct PeerLink {
  peer @0 :Text;
  reachable @1 :Bool;
  rateBps @2 :Float64;
  hasRateBps @3 :Bool;
  latencyS @4 :Float64;
  hasLatencyS @5 :Bool;
  marginDb @6 :Float64;
  hasMarginDb @7 :Bool;
}

struct CommsObservationMask {
  agentId @0 :Text;
  links @1 :List(PeerLink);
  earthContact @2 :Bool;
}

struct Observation {
  tick @0 :Int64;
  simTimeS @1 :Float64;
  agentId @2 :Text;
  observable @3 :Bool;
  selfState @4 :StateSample;
  sensors @5 :List(SensorReading);
  comms @6 :CommsObservationMask;            # optional (pointer presence)
  neighbors @7 :List(StateSample);
  epoch @8 :Epoch;                # optional (pointer presence)
}
