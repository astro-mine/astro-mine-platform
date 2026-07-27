# Architecture

`astro-mine-studio` is a component of the [Astro-Mine](https://github.com/astro-mine)
platform. Its architecture is specified in the platform **docs** repo, not here — this
file is the conventional pointer back to that source of truth
([`conventions.md` §13](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md)).

- **Component design** —
  [`architecture/studio.md`](https://github.com/astro-mine/docs/blob/main/architecture/studio.md):
  purpose, principles, runtime, data, integration, security, and roadmap alignment for
  this package (the design front door — intent → trade study → scored Campaign, orchestrating
  the autonomy stack, with optional provider-abstracted LLM-assisted intent capture).
- **Cross-cutting standards** —
  [`architecture/conventions.md`](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md):
  the platform-wide technology, schema, packaging, security, and versioning conventions
  every component follows.
- **System integration view** —
  [`architecture/system.md`](https://github.com/astro-mine/docs/blob/main/architecture/system.md):
  how the components fit together end to end.

> **Web UI.** `ui/` is a React/TypeScript front end on the platform front-end baseline
> ([`conventions.md` §2.1](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md)),
> alongside the FastAPI back end and the plain Python library package (hatchling). It embeds
> `@astro-mine/view` for 3D inspection and renders candidate comparison with uncertainty.
>
> Two known gaps, both Wave 24 rather than defects in this scaffold:
>
> - **It is not yet served.** `api/_app.py` mounts no static assets and the package ships no console
>   script, so running Studio means starting the API and the Vite dev server separately
>   (`astro-mine-studio#33`).
> - **It is not yet a surface.** Under RFC-0010 the UI becomes `@astro-mine/studio-ui`, composed by
>   the console rather than served standalone, and its inline styles retire onto `@astro-mine/ui`
>   (`astro-mine-studio#31`). Its charts move from Plotly to **visx** with that conversion
>   (`conventions.md` §2.1); parallel coordinates is hand-built, since `visx` has no such mark.
>
> **Preserve the uncertainty semantics through that move.** `ui/src/figures.ts` splits bounded and
> unbounded candidates into separate traces on purpose: Plotly coerces a `null` bound to `0` and
> draws a zero-length error bar, which asserts a precision the data does not have. That split is
> correctness, not styling, and a re-theme that merges the traces reintroduces the lie.

See also the [charter](https://github.com/astro-mine/docs/blob/main/charter/) (vision)
and the [roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/) (phased delivery).
