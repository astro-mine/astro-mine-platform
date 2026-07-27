# Portable, lockfile-pinned execution of the Bench determinism gate (RM-P0-BENCH-04).
#
# The gate itself runs locally with no container (CX-LOCAL); this image is the *portability*
# artifact (conventions.md §7 — pinned base image + uv lockfile). It is not built in CI.
#
# Pin the base image by digest before the repo flips public (CX-REPRO); the tag is pinned here
# during private incubation. astro-mine-core is a private git dependency during incubation, so pass
# a token at build time:
#   docker build --secret id=core_token,env=CORE_REPO_TOKEN -t astro-mine-bench-gate .
#   docker run --rm astro-mine-bench-gate
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# git is needed to resolve the private core git dependency; drop it after the public/PyPI flip.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN --mount=type=secret,id=core_token \
    sh -c 'if [ -f /run/secrets/core_token ]; then \
      git config --global \
        url."https://x-access-token:$(cat /run/secrets/core_token)@github.com/".insteadOf \
        "https://github.com/"; \
    fi' \
    && uv sync --locked --no-dev

# Fail the container on any non-reproducibility.
CMD ["uv", "run", "--no-dev", "python", "scripts/determinism_gate.py"]
