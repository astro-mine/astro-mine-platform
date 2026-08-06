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

> **Web UI — not in this distribution.** Studio's front end is the `/design` pages of the one
> application, in [`astro-mine-ui`](https://github.com/astro-mine/astro-mine-ui). The
> `@astro-mine/studio-ui` console *surface* this section used to describe is retired with the rest
> of the plugin model — see
> [`architecture/ui.md` §11](https://github.com/astro-mine/docs/blob/main/architecture/ui.md).
>
> **One property from it was correctness, not styling.** The old tree split bounded and unbounded
> candidates into separate traces on purpose: Plotly coerces a `null` bound to `0` and draws a
> zero-length error bar, asserting a precision the data does not have. The replacement enforces the
> same thing through `UncertaintyValue` and the design system's chart layer, where a null bound
> renders as an open mark
> ([`architecture/ui.md` §2, §7.1](https://github.com/astro-mine/docs/blob/main/architecture/ui.md)).

See also the [charter](https://github.com/astro-mine/docs/blob/main/charter/) (vision)
and the [roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/) (phased delivery).
