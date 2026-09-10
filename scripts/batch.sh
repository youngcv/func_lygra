#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/batch.sh <object_name> [instance_id ...]"
    echo
    echo "Examples:"
    echo "  bash scripts/batch.sh knife"
    echo "  bash scripts/batch.sh knife 000 003 009"
    echo "  bash scripts/batch.sh lighter"
    echo "  VISUALIZE=True bash scripts/batch.sh stapler 000"
    echo "  GPUS=0,1 bash scripts/batch.sh stapler"
    echo "  INPUT_DIR=/data/objects/knife_real OUTPUT_DIR=/data/grasps bash scripts/batch.sh knife"
    echo "  DRY_RUN=True bash scripts/batch.sh knife"
    exit 0
fi

CATEGORY="$1"
shift

SCRIPT="${SCRIPT:-scripts/$CATEGORY.sh}"
if [ ! -f "$SCRIPT" ]; then
    echo "Single-object script not found: $SCRIPT"
    exit 1
fi

INPUT_ROOT="${ROOT_DIR:-assets/object}"
INPUT_PATH="${INPUT_DIR:-$INPUT_ROOT/$CATEGORY}"
OUTPUT_ROOT="${OUTPUT_DIR:-${SAVE_DIR:-cache/grasp}}"
worker_pids=()
status_dir=""

terminate_tree() {
    local pid="$1"
    local sig="${2:-TERM}"
    local child

    for child in $(ps -o pid= --ppid "$pid" 2>/dev/null); do
        terminate_tree "$child" "$sig"
    done

    kill "-$sig" "$pid" 2>/dev/null || true
}

cleanup() {
    local sig="${1:-INT}"
    trap - INT TERM EXIT
    echo
    echo "Stopping batch jobs..."

    for pid in "${worker_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            terminate_tree "$pid" TERM
        fi
    done

    sleep 1

    for pid in "${worker_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            terminate_tree "$pid" KILL
        fi
    done

    if [ -n "$status_dir" ] && [ -d "$status_dir" ]; then
        rm -rf "$status_dir"
    fi

    if [ "$sig" = "TERM" ]; then
        exit 143
    fi
    exit 130
}

trap 'cleanup INT' INT
trap 'cleanup TERM' TERM

if [ ! -d "$INPUT_PATH" ]; then
    echo "Object input directory not found: $INPUT_PATH"
    exit 1
fi

if [ "$#" -gt 0 ]; then
    ids=("$@")
else
    mapfile -t ids < <(find "$INPUT_PATH" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
fi

if [ "${#ids[@]}" -eq 0 ]; then
    echo "No object instances found under $INPUT_PATH"
    exit 1
fi

gpu_spec="${GPUS:-}"
if [ -z "$gpu_spec" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    gpu_spec="$CUDA_VISIBLE_DEVICES"
fi
gpu_spec="${gpu_spec:-0}"
gpu_spec="${gpu_spec//,/ }"
read -r -a gpus <<< "$gpu_spec"

success_count=0
fail_count=0
skip_count=0

run_one() {
    local id="$1"
    local gpu="$2"
    local obj_dir
    obj_dir="$INPUT_PATH/$id"
    if [ ! -f "$obj_dir/labeled.obj" ]; then
        echo "[gpu:$gpu] Skip $id: missing $obj_dir/labeled.obj"
        return 2
    fi

    echo "=== [gpu:$gpu] Processing $CATEGORY/$id from $INPUT_PATH via $SCRIPT ==="
    if [ "${DRY_RUN:-False}" = "True" ] || [ "${DRY_RUN:-False}" = "true" ]; then
        echo "ROBOT=${ROBOT:-sharpa} CUDA_VISIBLE_DEVICES=$gpu INPUT_DIR=$INPUT_PATH OUTPUT_DIR=$OUTPUT_ROOT VISUALIZE=${VISUALIZE:-False} bash $SCRIPT object.asset.instance_id=$id"
        return 0
    fi

    ROBOT="${ROBOT:-sharpa}" CUDA_VISIBLE_DEVICES="$gpu" INPUT_DIR="$INPUT_PATH" OUTPUT_DIR="$OUTPUT_ROOT" VISUALIZE="${VISUALIZE:-False}" bash "$SCRIPT" \
        object.asset.instance_id="$id" &
    local child_pid=$!
    worker_pids+=("$child_pid")
    wait "$child_pid"
}

if [ "${#gpus[@]}" -le 1 ]; then
    for id in "${ids[@]}"; do
        run_one "$id" "${gpus[0]}"
        status=$?
        if [ "$status" -eq 0 ]; then
            success_count=$((success_count + 1))
        elif [ "$status" -eq 2 ]; then
            skip_count=$((skip_count + 1))
        else
            echo "Failed $CATEGORY/$id with exit code $status"
            fail_count=$((fail_count + 1))
        fi
    done

    echo "Batch complete: $success_count succeeded, $fail_count failed, $skip_count skipped."
    exit 0
fi

echo "Using GPUs: ${gpus[*]}"
status_dir="${TMPDIR:-/tmp}/lygra_batch_${CATEGORY}_$$"
mkdir -p "$status_dir"

for gpu_idx in "${!gpus[@]}"; do
    (
        worker_success=0
        worker_fail=0
        worker_skip=0
        gpu="${gpus[$gpu_idx]}"

        for id_idx in "${!ids[@]}"; do
            if [ $((id_idx % ${#gpus[@]})) -ne "$gpu_idx" ]; then
                continue
            fi

            run_one "${ids[$id_idx]}" "$gpu"
            status=$?
            if [ "$status" -eq 0 ]; then
                worker_success=$((worker_success + 1))
            elif [ "$status" -eq 2 ]; then
                worker_skip=$((worker_skip + 1))
            else
                echo "Failed $CATEGORY/${ids[$id_idx]} on gpu:$gpu with exit code $status"
                worker_fail=$((worker_fail + 1))
            fi
        done

        echo "$worker_success $worker_fail $worker_skip" > "$status_dir/gpu_${gpu_idx}.status"
    ) &
    worker_pids+=("$!")
done

for job in "${worker_pids[@]}"; do
    wait "$job"
    status=$?
    if [ "$status" -ne 0 ]; then
        fail_count=$((fail_count + 1))
    fi
done

for status_file in "$status_dir"/*.status; do
    if [ ! -f "$status_file" ]; then
        continue
    fi
    read -r worker_success worker_fail worker_skip < "$status_file"
    success_count=$((success_count + worker_success))
    fail_count=$((fail_count + worker_fail))
    skip_count=$((skip_count + worker_skip))
done

rm -rf "$status_dir"

echo "Batch complete: $success_count succeeded, $fail_count failed, $skip_count skipped."
