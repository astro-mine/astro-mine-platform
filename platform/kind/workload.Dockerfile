# syntax=docker/dockerfile:1
#
# The workload image the live-cluster tests dispatch.
#
# Its entrypoint is the in-pod run harness (astro_mine.cloud.submission.harness), which stages the
# job's content-addressed inputs from the object store, runs `job.command`, captures the declared
# outputs back, and records the RunContext -- through the *same* execute() the local backend calls.
#
# TWO THINGS HERE ARE LOad-BEARING FOR THE DETERMINISM GATE. Both feed the RunContext content
# address (conventions.md §5), so getting either wrong makes the equivalence test fail for reasons
# that have nothing whatever to do with the run:
#
#   1. code_version. RunContext records the installed astro-mine-cloud version, and hatch-vcs
#      derives it from `git describe`. There is no .git in this build context, so without a pin
#      the build would either fail or invent a version different from the host's.
#      SETUPTOOLS_SCM_PRETEND_VERSION (hatch-vcs delegates to setuptools-scm) pins it to the
#      *host's* version, which up.sh reads off the host venv and passes in.
#
#      The fix is emphatically NOT to exclude code_version from content_address(). That field is
#      in the reproducibility minimum on purpose; dropping it to make a test go green would gut
#      conventions.md §5. A test that is sensitive to the code version is the harness working.
#
#   2. env_lockfile. RunContext records the content address of the active uv.lock, which
#      _active_uv_lock() finds by walking up from the CWD. So uv.lock is COPYed to the WORKDIR --
#      the same bytes the host pins, hence the same address. ASTRO_MINE_ENV_LOCKFILE names it
#      outright, so it holds even if the process runs from somewhere else.

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

# The astro-mine-cloud version to build as -- MUST equal the host's (see above).
ARG ASTRO_MINE_CLOUD_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${ASTRO_MINE_CLOUD_VERSION}
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ASTRO_MINE_CLOUD=${ASTRO_MINE_CLOUD_VERSION}

# Ray, for the RayJob half of the dispatch round-trip: KubeRay's head/worker containers run
# `ray start` from *this* image, so `ray` has to be in it. Heavy, and needed by exactly one test
# -- hence a build arg, so a Job-only run can skip it.
ARG INSTALL_RAY=1
# 2.30.0 ships no cp312 wheel (only cp39/310/311) and the image is Python 3.12, so the build died
# on "no wheels with a matching Python ABI tag". 2.56.0 is what astro-mine-learn's lockfile already
# resolves to -- keeping the workspace on one Ray rather than two.
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
 && apt-get install -y --no-install-recommends git ca-certificates wget \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.29 /uv /usr/local/bin/uv

WORKDIR /work

COPY pyproject.toml uv.lock README.md LICENSE /work/
COPY src /work/src

# uv (not pip) resolves the dependencies: astro-mine-core is a *private git source* declared in
# [tool.uv.sources], which pip does not read -- it would go looking for it on PyPI and fail. Using
# `--locked` also guarantees the container's environment is the one uv.lock pins, which is the
# claim `env_lockfile` makes.
#
# The Core repo is private during incubation, so the build needs a token. It arrives as a BuildKit
# *secret*, never a build-arg: an ARG lands in the image history and would leak the credential
# into every layer of an image we then push to a registry.
RUN --mount=type=secret,id=core_token \
    if [ -s /run/secrets/core_token ]; then \
      git config --global \
        "url.https://x-access-token:$(cat /run/secrets/core_token)@github.com/.insteadOf" \
        "https://github.com/"; \
    fi; \
    uv sync --locked --no-dev --extra s3 --extra cluster; \
    rm -f /root/.gitconfig

RUN if [ "${INSTALL_RAY}" = "1" ]; then uv pip install "ray[default]==${RAY_VERSION}"; fi

# The environment pin, by name -- see (2) above.
ENV ASTRO_MINE_ENV_LOCKFILE=/work/uv.lock

# Non-root, as every Astro-Mine workload image is (cloud.md §9). /work must stay readable: the
# harness runs from it and reads uv.lock out of it.
RUN chown -R 65532:65532 /work
USER 65532:65532

# The harness -- not a workload. The workload's own argv travels in $ASTRO_MINE_JOBSPEC and the
# harness launches it. (The compiled manifests set `command` to the same thing explicitly, since a
# Kubernetes `command` overrides ENTRYPOINT; this keeps `docker run` on the image sane too.)
ENTRYPOINT ["python", "-m", "astro_mine.cloud.submission.harness"]
