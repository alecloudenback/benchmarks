#!/usr/bin/env bash
#
# Build every containerised BasicTerm_ME implementation and run them all against
# the GPU in this machine, reporting the best of N runs for each.
#
#   ./run_all.sh                     # 100M model points, best of 3
#   REPEATS=1 ./run_all.sh           # one run each, for a quick check
#   MULTIPLIER=100 ./run_all.sh      # 1M model points, for a quick check
#   SKIP_BUILD=1 ./run_all.sh        # reuse images already present locally
#
# Requires: docker with the NVIDIA container toolkit (`docker run --gpus all`).

set -euo pipefail
cd "$(dirname "$0")"

MULTIPLIER=${MULTIPLIER:-10000}
REPEATS=${REPEATS:-3}
LOG=${LOG:-run_all.log}
SKIP_BUILD=${SKIP_BUILD:-0}
PYTHON_IMAGE=${PYTHON_IMAGE:-basicterm_me_python}
JULIA_IMAGE=${JULIA_IMAGE:-basicterm_me_julia}

: > "$LOG"
results=()

gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
echo "GPU:         ${gpu:-none detected}"
echo "model points: $((MULTIPLIER * 10000))"
echo "repeats:      $REPEATS"
echo "full output:  $LOG"
echo

if [ "$SKIP_BUILD" != "1" ]; then
    echo "== building images =="
    docker build -q -t "$PYTHON_IMAGE" ./BasicTerm_ME_python
    docker build -q -t "$JULIA_IMAGE" ./BasicTerm_ME_julia
    echo
fi

# run <label> <image> <args...>
run() {
    local label=$1 image=$2
    shift 2
    local best= out= t=
    printf '%-34s' "$label"
    for _ in $(seq 1 "$REPEATS"); do
        if ! out=$(docker run --rm --gpus all "$image" "$@" 2>&1); then
            printf '  FAILED\n'
            { echo "### $label -- FAILED"; echo "$out"; echo; } >> "$LOG"
            results+=("$label"$'\t'"failed")
            return 0
        fi
        { echo "### $label"; echo "$out"; echo; } >> "$LOG"
        t=$(printf '%s\n' "$out" | sed -n 's/^time_in_seconds=//p' | tail -1)
        if [ -n "$t" ] && { [ -z "$best" ] || awk "BEGIN{exit !($t < $best)}"; }; then
            best=$t
        fi
        printf ' .'
    done
    printf '  %ss\n' "${best:-?}"
    results+=("$label"$'\t'"${best:-?}")
}

echo "== running =="
run "python recursive pytorch"   "$PYTHON_IMAGE" --model torch_recursive --multiplier "$MULTIPLIER"
run "python iterative jax"       "$PYTHON_IMAGE" --model jax_iterative   --multiplier "$MULTIPLIER"
run "julia array (rates=inline)" "$JULIA_IMAGE"  --model array  --rates inline --multiplier "$MULTIPLIER"
run "julia array (rates=table)"  "$JULIA_IMAGE"  --model array  --rates table  --multiplier "$MULTIPLIER"
run "julia kernel (rates=table)" "$JULIA_IMAGE"  --model kernel --rates table  --multiplier "$MULTIPLIER"
run "julia kernel (rates=inline)" "$JULIA_IMAGE" --model kernel --rates inline --multiplier "$MULTIPLIER"
run "julia reactant (rates=inline)" "$JULIA_IMAGE" --model reactant --rates inline --multiplier "$MULTIPLIER"
run "julia reactant (rates=table)" "$JULIA_IMAGE"  --model reactant --rates table  --multiplier "$MULTIPLIER"

echo
echo "== best of $REPEATS, $((MULTIPLIER * 10000)) model points, ${gpu:-unknown GPU} =="
printf '| %-32s | %12s |\n' "implementation" "seconds"
printf '|%s|%s|\n' "$(printf -- '-%.0s' {1..34})" "$(printf -- '-%.0s' {1..14})"
for r in "${results[@]}"; do
    printf '| %-32s | %12s |\n' "${r%%$'\t'*}" "${r##*$'\t'}"
done
