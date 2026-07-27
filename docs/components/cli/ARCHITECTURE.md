# Architecture — astro-mine-cli

The module map and the decisions behind it. The design authority is
**[RFC-0011](https://github.com/astro-mine/docs/blob/main/rfc/0011-umbrella-cli.md)** (accepted);
this document summarizes what it settled and records what implementation decided on top. The
platform-wide view is [`architecture/system.md §4`](https://github.com/astro-mine/docs/blob/main/architecture/system.md),
where `astro-mine-cli` sits in the Backbone row.

## What this package is

A **thin dispatcher**: discovery, routing, and packaging. It owns no schema, no message, no wire
type, and it makes **no change to Core** (`CORE_INTERFACE_VERSIONS` stays `0.1.0`). Its entire job
is to let a user guess an *action* — `astro-mine score` — and reach the component that implements
it.

## The shape, and why it is this one

RFC-0011 §1 chose an **entry-point group plus a static first-party manifest**, over three
alternatives. The two halves solve different problems and neither is redundant:

| Mechanism | Solves |
|---|---|
| **(a) Discovery** — enumerate `astro_mine.cli` via `importlib.metadata` | A component contributes a verb by declaring an entry point, with **no PR to this package**. Enumeration reads distribution metadata and imports nothing. |
| **(b) The first-party manifest** — plain strings, verb → distribution | Pure discovery cannot name a fix for an **uninstalled** component: with no `astro-mine-learn` there is no `train` entry point, and the umbrella could only say "unknown command". The manifest is what turns that into *"`astro-mine train` needs `astro-mine-learn` — `pip install astro-mine-learn`"*. |

The manifest governs **only** that friendly hint. Third-party verbs are discovered dynamically and
need no entry in it, so it does not reintroduce a PR-to-extend chokepoint through the back door.

## The four constraints, and where each is enforced

| Constraint | Enforced by |
|---|---|
| **Zero runtime dependencies** (CX-LOCAL — the deciding constraint) | `pyproject.toml` `dependencies = []`; `tests/test_packaging.py::test_declares_no_runtime_dependencies`; the bare-venv install step in CI; and `test_installed_provider.py`, which asserts a real venv holds nothing else |
| **No PR to extend** | Discovery keys on the group *name*. `tests/fixtures/provider` is a distribution deliberately **not** named `astro-mine-*`, installed for real, whose verbs this repo never mentions — and `astro-mine demo` works |
| **`--help` imports nothing** | The two-phase parse + the static manifest. Asserted as a negative in a clean interpreter (`test_listing_verbs_imports_no_provider`), because an eager `load()` produces correct output and would otherwise be invisible |
| **Degrade honestly** | The manifest turns a missing component into a `pip install` line; a malformed provider and a verb collision are messages, not tracebacks |
| **Additive** | Component CLIs are untouched and keep working when invoked directly |

### Why there is no `CORE_REPO_TOKEN` step in CI

Every sibling repo's CI rewrites GitHub HTTPS fetches with a PAT so `uv sync` can resolve the
private `astro-mine-core` git source. This repo has no such step **because it has nothing private
to resolve** — the dependency list is empty. The absence is deliberate and load-bearing: if that
step ever becomes necessary here, the zero-dependency rule has already been broken upstream of it.
The workflow says so at the point where a copy-paste from a sibling would land.

## Module map

```
src/astro_mine/cli/
  __init__.py     # public surface: __version__, main, build_parser, Subcommand, discover_verbs
  _protocol.py    # the contribution contract + the conformance check (all three groups)
  _manifest.py    # the static first-party tables (verb/kind → distribution, help)
  _discovery.py   # verb enumeration; the only place a provider is imported
  _dispatch.py    # the two-phase parser, built-in verbs, and the dispatch loop
  _validators.py  # the validator contract + discovery (the `validate` federation)
  _validate.py    # the built-in `validate` verb — a router, not a validator
  _scaffolds.py   # the two scaffold groups + discovery (the `new` federation)
  _new.py         # the built-in `new` and `plugin new` verbs — routers, not generators
  _templates.py   # the one scaffold the umbrella owns: `plugin new cli`
  __main__.py     # `python -m astro_mine.cli`
```

## The three built-in verbs: `validate`, `new`, `plugin`

Every other verb belongs to a component. These three belong here because all three *route* rather
than do, and routing is the one job no component can hold without importing its siblings, which the
narrow waist forbids (`conventions.md §1.1`). `validate` works out who owns a document and asks
them; `new` and `plugin new` work out who owns a *kind* and ask them. Nothing else belongs in the
set — a verb that actually does something belongs to the component that does it.

## `validate` — the format's owner owns the checker

It was Core's registered verb at first, and that could never have federated: extending it would
have meant Core consulting Guard and Mind, i.e. Core importing its own consumers. Moving the verb
up here is what lets each owner keep its checker while a user types one command (RFC-0011 §6).

**The umbrella still parses nothing.** It has no YAML parser and gaining one would break the
zero-dependency rule, so routing is inverted: rather than reading a file to decide who owns it, the
umbrella asks each validator `claims(path)` and hands the file to whoever says yes. Every owner
already has the parser and the schema; none of that is duplicated here, and a tenth format needs no
change to this package. A test asserts this file contains no document parsing at all.

The cost, stated plainly: running `validate` imports **every** installed validator, because
ownership cannot be determined without asking. It is paid only by the command the user typed —
never by `--help` — and the alternative, first-claim-wins, would buy fewer imports with a silent
precedence rule.

| Situation | Behaviour |
|---|---|
| Two validators claim one file | Hard error naming both — which checker judged a document is provenance |
| Nobody claims it | Refused, listing who *is* installed. A document is never checked against a guessed schema |
| No validators installed | Names the package that owns the format at hand |
| A distribution advertises a `validate` verb | Hard error: a built-in cannot be shadowed silently |

A document must be **self-describing** to be routed — the umbrella has to identify the owner before
anyone can be told a `--kind`. For a file without that marker, the owning CLI still takes `--kind`
directly.

The console script `astro-mine` resolves to `astro_mine.cli:main`; that name and target are pinned
by a test, because RFC-0011's per-component dispatch (`astro-mine studio serve`) is a thin call into
an already-shipped subcommand and only works if they stay put.

## `new` and `plugin new` — the artifact's owner owns the scaffold

RFC-0011 §7 puts scaffolding in the umbrella because it is a **cross-component authoring concern
with no natural single-component home**: an asset is Fleet's, a stack spec is Mind's, a SafetySpec
is Guard's, a solver plugin is Allocate's. `fleet new` is the exemplar; the umbrella generalizes its
shape and routes each kind to whoever owns it.

**The umbrella generates nothing it does not own.** It has no templating engine and no YAML parser
— the zero-dependency rule forbids both — so the owning component, which already has the schema,
writes the bytes. The one exception proves the rule rather than bending it: `plugin new cli`
scaffolds a package against the `astro_mine.cli` entry-point group, and the umbrella *is* that
group's owner, exactly as it is the `validate` routing problem's owner.

### Two groups, one contract

| Group | What it scaffolds | Entry-point name |
|---|---|---|
| `astro_mine.cli.scaffolds` | an authored **document** (`astro-mine new asset`) | the kind as typed |
| `astro_mine.cli.plugin_scaffolds` | an installable **plugin package** (`astro-mine plugin new solver`) | the kind as typed |

Both resolve to the **same four members `Subcommand` has** — `name`, `help`, `add_arguments`,
`run` — checked by the same function, differing only in the group and noun its error message names.
A second protocol with the same shape under different nouns would have given component authors two
things to learn and this package two checkers to keep in step. So a component contributes a scaffold
by writing the object it already knows how to write, and — as with a verb — nothing imports the
umbrella to do it.

**Why two groups rather than one with a `target` attribute.** Deciding which verb a scaffold belongs
to by reading an attribute would mean *loading every scaffold* to render `astro-mine new --help`,
which is the precise cost the two-phase parse exists to avoid. A group name can be filtered on for
free; an attribute on a loaded object cannot.

**Routing is by name, not by inspection.** Unlike `validate` — which must ask each validator *"is
this file yours?"*, because a path carries no owner — the user types the kind. So discovery stays a
metadata read, `astro-mine new` lists every available kind without importing anything, and adding a
tenth kind needs no change here.

### The two arguments the umbrella owns

Before handing the parser to a scaffold, the umbrella declares `output` (positional) and `--force`.
Every kind therefore has the same skeleton, and a user who has scaffolded one can scaffold the next
without re-reading the help. Everything else is the owner's to declare, and a scaffold must not
re-declare those two.

| Situation | Behaviour |
|---|---|
| Two distributions offer one kind | Hard error naming both — which package generated your starting file is provenance |
| A distribution offers a kind the umbrella owns | Hard error: a built-in cannot be shadowed silently |
| A first-party kind whose component is absent | Names the install, as a missing verb does |
| A first-party kind whose component **is installed but offers no scaffold** | Says exactly that. Telling a user to install what they already have would be the umbrella lying about an environment it can see |
| A kind nobody has ever advertised | Unknown-kind error listing what is available |

**What the scaffold emits must validate.** A freshly scaffolded document passing straight into
`astro-mine validate` with no hand-editing is the only acceptance test that means anything — and it
is a claim each owner keeps, since each owns both the template and the checker. No CI in this repo
can check it: `astro-mine-cli` installs no components by design. What *is* checked here is that a
scaffolded `cli` plugin installs into a bare venv, registers, and runs
(`tests/test_installed_provider.py`), and that a third-party distribution can contribute a kind with
no change to this package (`tests/fixtures/provider`).

### Why parsing happens in two phases

The umbrella cannot build a complete argparse tree up front. Filling in one verb's arguments means
calling that verb's `add_arguments`, which means importing its component — so a single-phase parser
would import **every installed component** just to render `--help`.

So phase one parses only *which verb* (everything after it is `argparse.REMAINDER`), and phase two
loads that one verb and lets it parse its own tail. The user pays for the import of the command they
actually ran, and for nothing else.

The cost is that the top-level listing cannot show a verb's own `help` string — which is the second
job `_manifest.py` does. A verb's *complete* help still comes from the provider, on
`astro-mine <verb> --help`, where paying for one import is exactly what the user asked for.

### Why `build_parser()` is a function, not a module-level constant

The verb set is read from installed metadata **at build time**, so a cached parser would freeze the
environment as it looked at first import. Building per invocation also lets the tests construct a
parser against a fixture environment without reimporting the module.

## The two adapter styles a component can choose

Every Astro-Mine CLI already exposes `_build_parser()` and `main(argv) -> int`, so both styles are
cheap. The umbrella ships neither — the adapter lives in the component, which is the point — but
`tests/fixtures/provider` implements one of each, and those are the copyable references.

| Style | Shape | When |
|---|---|---|
| **Per-verb** | The adapter declares the verb's own arguments and calls its handler | An action worth promoting to the top level: `astro-mine score` |
| **Component passthrough** | `add_arguments` takes `argparse.REMAINDER`; `run` forwards the tail to the component's existing `main(argv)` | The cheap on-ramp, and RFC-0011 §2's `astro-mine <component> <verb>` form: `astro-mine studio serve` |

Passthrough re-declares nothing, so the umbrella's help can never drift from the component's real
flags. Per-verb costs one small adapter and buys top-level discoverability. A component can start at
passthrough and promote individual verbs later, without the umbrella changing.

## Decisions

### Settled here

- **`Development Status :: 3 - Alpha` from the first commit.** The package ships a working console
  script; `1 - Planning` is platform-wide drift
  ([astro-mine/docs#40](https://github.com/astro-mine/docs/issues/40)), not a status to inherit.
- **The empty state exits 0**, and lists the platform's verbs marked *not installed here*. On a bare
  install that turns `astro-mine --help` into a map of the platform rather than an empty shell —
  which is the discovery gap (**UC-A3**) the umbrella exists to close, working before the user has
  installed a single component. An *unrecognized* verb still fails (exit 2).
- **`Subcommand` is a structural `typing.Protocol`, not a base class.** RFC-0011 left the shape to
  implementation. A component forced to write `from astro_mine.cli import Subcommand` would take a
  runtime dependency on the umbrella, inverting the layering this package protects
  (`conventions.md §1.1`). Conformance is therefore checked by shape at dispatch, and a
  non-conforming provider is reported by **name, entry point and missing member** — never as an
  `AttributeError` from inside our own frames.
- **The umbrella exposes only what a component registers**, not a component's full subcommand tree
  (RFC-0011's other open question). The component owns its surface; the umbrella owns routing.
  Re-exposure would make our help text a function of a sibling's internals, and would drift the
  moment that sibling added a flag.
- **A verb claimed by two distributions is a hard error naming both**, never resolved by
  precedence. Which package handled a command is provenance: a silent winner means the same command
  line does different things on two machines and nothing tells the user which they have. (Allocate's
  solver registry takes the same stance for the same reason.)
- **Environment failures are reported; verb failures are not caught.** A collision or a malformed
  provider is somebody else's packaging bug and prints a message with a non-zero exit. An exception
  raised *inside* a verb propagates untouched — swallowing it would make the umbrella the thing you
  have to remove in order to debug your own tool.
- **A verb returning `None` exits 0.** That is what `sys.exit(None)` means everywhere else in
  Python, and a component that completed its work should not be punished with a crash for following
  the convention.
- **A scaffold is a `Subcommand`.** RFC-0011 §7 asked for a scaffold contribution group; it did not
  ask for a second protocol. The four members are the same members, so they are the *same* contract
  checked by the same function, and a component author learns one shape rather than two. Only the
  error message's group and noun vary.
- **`plugin new cli` is a built-in kind**, and the only one. The umbrella hosts the `astro_mine.cli`
  entry-point group, so it owns that group's scaffold — the identical carve-out that makes
  `validate` a built-in verb. Without it this package would ship a dispatcher with nothing to
  dispatch to until eight sibling repos landed, which is the empty-shell problem RFC-0011 §1b
  already rejected once.
- **A component that is installed but offers no scaffold is told so**, rather than told to install
  itself. The probe is an `importlib.metadata` version lookup — free, and no import. The case that
  motivated it was `new world` while Worlds still had no `WorldSpec` scaffold (G2.11, since closed
  by [astro-mine/astro-mine-worlds#57](https://github.com/astro-mine/astro-mine-worlds/issues/57)),
  and the rule outlives it: the verb-level degradation path does not draw this distinction, and the
  kind-level one must, because a kind can be absent from a component that is present.
- **No git tags yet**, so `hatch-vcs` stamps a development version — matching the sibling repos
  during private incubation. The version is *derived*, so it cannot drift.

### Deferred

- **Shell completion** over the discovered verb set (RFC-0011 leaves it to implementation).
- **The scaffolds themselves**, in every component that owns one. This package ships the contract,
  the routing, the degradation and the `cli` plugin kind; `asset`, `stack`, `safety` and the six
  remaining plugin kinds land in their owners' repos, which is the whole point of federating.

### Owned elsewhere

- **The naming rule and the alias/deprecation policy** are normative in
  [`conventions.md §13`](https://github.com/astro-mine/docs/blob/main/architecture/conventions.md),
  not here. The renames themselves (`fleet`/`worlds`/`link`/`prospect`, `astro-mine-train`) land in
  each component's own repo, tracked by
  [astro-mine/docs#57](https://github.com/astro-mine/docs/issues/57).
- **`validate` federation** (RFC §6): the format's owner owns its validator, Core owns `$id`-keyed
  dispatch for Core formats, and the umbrella only routes. No checker is reimplemented here.
- **The umbrella's release cadence** relative to components —
  [`VERSIONING.md`](https://github.com/astro-mine/docs/blob/main/VERSIONING.md).
