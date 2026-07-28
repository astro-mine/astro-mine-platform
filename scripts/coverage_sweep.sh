#!/usr/bin/env bash
# Full-platform coverage sweep: every component suite, each under its source
# repo's CI selection, accumulated into one .coverage file.
set -u
cd "$(dirname "$0")/.."
rm -f .coverage
declare -A SEL=(
  [sim]="-m|not gpu and not ray and not docker"
  [allocate]="-m|not scale"
  [bench]="-m|not cluster and not gpu"
  [hub]="-m|not integration"
  [mind]="-m|not pddl and not native and not slow and not sim"
  [guard]="-m|not sim and not slow"
  [learn]="-m|not slow and not gpu and not cluster"
)
FAILED=""
for comp in core spice seal worlds prospect link fleet sim bench cloud surrogate mind learn allocate guard hub studio; do
  [ -d "tests/$comp" ] || continue
  args=()
  if [ -n "${SEL[$comp]:-}" ]; then
    IFS='|' read -r flag expr <<< "${SEL[$comp]}"
    args=("$flag" "$expr")
  fi
  echo "=== $comp ${args[*]:-}"
  uv run python -m pytest "tests/$comp" -q "${args[@]}" \
    --cov=astro_mine --cov-append --cov-report= || FAILED="$FAILED $comp"
done
echo "combined report:"
uv run python -m coverage report --format=total
[ -z "$FAILED" ] || { echo "FAILED:$FAILED"; exit 1; }
