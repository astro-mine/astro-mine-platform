# syntax=docker/dockerfile:1
#
# The workload image the live-cluster tests dispatch.
#
# Its entrypoint is the in-pod run harness (astro_mine.cloud.submission.harness), which stages the
# job's content-addressed inputs from the object store, runs `job.command`, captures the declared
# outputs back, and records the RunContext -- through the *same* execute() the local backend calls.
#
# THIS FILE WAS WRITTEN BEFORE CONSOLIDATION AND DID NOT COME ACROSS WITH IT. It installed
# `--extra s3 --extra cluster`, which were `astro-mine-cloud`'s extra names; the platform calls them
# `cloud-s3` and `cloud-cluster`, so `uv sync --locked` re-resolved, refused, and exited non-zero --
# and the `;`-chained RUN below swallowed it, because the exit status of `a; b; c` is `c`'s. The
# image built clean with an empty virtualenv, and every workload pod in the lane died on
# `ModuleNotFoundError: No module named 'astro_mine'`, which is eight of the nine cluster-e2e
# failures. The masking is fixed first (`set -eux`), because a build that cannot install the thing
# it exists to run must not produce an image.
#
# TWO THINGS HERE ARE LOAD-BEARING FOR THE DETERMINISM GATE. Both feed the RunContext content
# address (conventions.md §5), so getting either wrong makes the equivalence test fail for reasons
# that have nothing whatever to do with the run:
#
#   1. code_version. RunContext records the installed distribution's version. The platform pins
#      `version = "0.1.0"` statically in pyproject.toml, so the host and the image agree by
#      construction -- there is nothing to pin at build time any more. (It used to be derived by
#      hatch-vcs from `git describe`, which is why this file carried SETUPTOOLS_SCM_PRETEND_VERSION
#      and installed `git`; maturin took over the build and neither does anything now.) The version
#      is still passed in, and now *asserted* rather than injected: see the smoke check below.
#
#   2. env_lockfile. RunContext records the content address of the active uv.lock, which
#      _active_uv_lock() finds by walking up from the CWD. So uv.lock is COPYed to the WORKDIR --
#      the same bytes the host pins, hence the same address. ASTRO_MINE_ENV_LOCKFILE names it
#      outright, so it holds even if the process runs from somewhere else.

ARG PYTHON_VERSION=3.12

# --- builder: compile the platform wheel ---------------------------------------------------------
#
# The platform is a maturin project that bundles Guard's Rust core as `astro_mine.guard._core`, so
# installing it from source needs a Rust toolchain -- which `python:*-slim` does not have. Building
# it in a discarded stage keeps rustup, cargo and the ~1GB target directory out of the image that
# gets `docker save`d and pushed through crane.
#
# Same base as the runtime stage, deliberately: a wheel with a compiled extension is built against
# the glibc it is built on, and building on the runner instead (which has Rust already) would link
# against ubuntu-latest's glibc and install into a Debian image that may not have it.
FROM python:${PYTHON_VERSION}-slim AS builder

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --no-modify-path --profile minimal --default-toolchain stable

# uv 0.9.9, matching `docker/Dockerfile`. **Not 0.5.29, which this pinned and which cannot read the
# lockfile in this repository**: `uv.lock` is `revision = 3`, 0.5.29 predates that revision, and its
# response to a lockfile it does not understand is to re-resolve -- which `--locked` then refuses.
# Measured rather than guessed: `uv lock --check` fails on 0.5.29 and passes on 0.8.0 and 0.9.9.
#
# The runner installs uv through `setup-uv@v5`, which takes the latest release, so the lockfile's
# revision follows uv forward and a fixed pin here drifts behind it. That is survivable now only
# because the failure is loud: before `set -eux` below, this exact mismatch produced a clean build
# of an empty image.
COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /usr/local/bin/uv

WORKDIR /build
COPY pyproject.toml uv.lock README.md LICENSE /build/
COPY rust /build/rust
COPY src /build/src
# Guard's build script compiles the CompiledSafetyModel wire form from `../schemas/guard/proto`, so
# the crate does not build without it -- `protox failed to compile the proto: … is not in any
# include path`. The protos are the same canonical files the Python side generates from, which is
# what makes the Rust safety core and the Python compiler agree byte-for-byte (`rust/build.rs`), so
# this is a build input rather than package data.
COPY schemas /build/schemas

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/rust/target \
    uv build --wheel --out-dir /wheels

# --- runtime -------------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim

# The version the image must report -- MUST equal the host's (see (1) above). Asserted, not pinned.
ARG ASTRO_MINE_PLATFORM_VERSION=0.1.0

# Ray, for the RayJob half of the dispatch round-trip: KubeRay's head/worker containers run
# `ray start` from *this* image, so `ray` has to be in it. Heavy, and needed by exactly one test
# -- hence a build arg, so a Job-only run can skip it.
ARG INSTALL_RAY=1
# 2.30.0 ships no cp312 wheel (only cp39/310/311) and the image is Python 3.12, so the build died
# on "no wheels with a matching Python ABI tag". 2.56.0 is what the platform's `learn-rllib` extra
# already resolves to -- keeping the workspace on one Ray rather than two.
ARG RAY_VERSION=2.56.0

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/work/.venv \
    PATH=/work/.venv/bin:$PATH

# wget is not decoration: KubeRay injects liveness and readiness probes into the Ray head that shell
# out to it, and python:3.12-slim does not ship it. Without it the probes fail with
# "bash: line 1: wget: command not found", the kubelet kills a perfectly healthy Ray head, it
# crash-loops, and the RayJob sits in Initializing until the dispatch times out -- a fifteen-minute
# failure whose cause is a missing 200KB binary.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates wget \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /usr/local/bin/uv

WORKDIR /work

COPY pyproject.toml uv.lock README.md LICENSE /work/

# The dependency set, from the lockfile and nothing else -- `--locked` fails rather than
# re-resolving, which is the claim `env_lockfile` makes. `--no-install-project` because the project
# itself arrives as the wheel the builder stage compiled; syncing it here would need Rust.
RUN set -eux; \
    uv sync --locked --no-dev --no-install-project --extra cloud-s3 --extra cloud-cluster

# `--no-deps`: everything it needs is already pinned by the sync above, and letting the wheel pull
# its own would silently step outside the lockfile the determinism gate hashes.
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    set -eux; \
    uv pip install --no-deps /wheels/*.whl

RUN set -eux; \
    if [ "${INSTALL_RAY}" = "1" ]; then uv pip install "ray[default]==${RAY_VERSION}"; fi

# The environment pin, by name -- see (2) above.
ENV ASTRO_MINE_ENV_LOCKFILE=/work/uv.lock

# THE SMOKE CHECK THIS FILE DID NOT HAVE. The entrypoint below is a module path, and a module path
# that does not resolve is not a build error -- it is a pod that starts, exits 1, and reports as a
# dispatch failure fifteen minutes later in a test about something else. Importing it here turns
# every way of failing to install the platform into a failed build, at the layer that caused it.
#
# The version equality is the determinism gate's requirement (1) checked rather than assumed: if
# the two ever diverge, `test_a_cluster_run_reproduces_the_laptop_run` fails on a content address
# and says nothing about why.
RUN set -eux; \
    python -c "import astro_mine.cloud.submission.harness"; \
    test "$(python -c 'import astro_mine.cloud as c; print(c.__version__)')" = "${ASTRO_MINE_PLATFORM_VERSION}"

# Non-root, as every Astro-Mine workload image is (cloud.md §9). /work must stay readable: the
# harness runs from it and reads uv.lock out of it.
RUN chown -R 65532:65532 /work
USER 65532:65532

# The harness -- not a workload. The workload's own argv travels in $ASTRO_MINE_JOBSPEC and the
# harness launches it. (The compiled manifests set `command` to the same thing explicitly, since a
# Kubernetes `command` overrides ENTRYPOINT; this keeps `docker run` on the image sane too.)
ENTRYPOINT ["python", "-m", "astro_mine.cloud.submission.harness"]
