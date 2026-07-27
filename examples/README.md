# Examples

Merged from the component repos that shipped an `examples/` tree, preserving
their original repo-root-relative paths (core's schema/format tests glob these
directories directly):

- `assets/` — SADF asset examples (core)
- `objectives/` — ObjectiveSpec examples (core)
- `mission/` — MissionSpec examples (core; RFC-0001)
- `plan/` — Plan/ContingentPlan examples (core; RFC-0006)
- `policy/` — policy/planner examples (core)
- `plugins/` — plugin-registry examples (core)
- `run-provenance/` — run-provenance examples (core)
- `downstream-consumer/` — a standalone consumer package exercising Core as a
  dependency (core)

Every component that defines an authored format also ships its reference
example as **package data** under `src/astro_mine/<comp>/reference/`
(conventions.md §13) — resolve those with `importlib.resources`, never by path
into this tree.

## Guard's examples note (from astro-mine-guard)

The reviewed anchor `SafetySpec` (`anchor.safety.yaml`) used to live here, under
`examples/safety_specs/`. It now ships as **package data** so an installed Guard can resolve it
from a wheel — it moved to `src/astro_mine/guard/reference/safety_specs/anchor.safety.yaml`.

Reach it by name, never by path:

```python
from astro_mine.guard.reference import load_anchor_safety_spec, anchor_safety_spec_text

document = load_anchor_safety_spec()   # parsed + validated SafetyDocument
text = anchor_safety_spec_text()       # the raw YAML
```

Or from the CLI: `astro-mine-guard validate anchor` / `compile anchor` / `sign anchor`.
