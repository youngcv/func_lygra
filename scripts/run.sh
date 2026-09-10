#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/run.sh <object_name> [hydra override ...]"
    exit 1
fi

OBJECT_NAME="$1"
shift

if [ ! -f "configs/object/$OBJECT_NAME.yaml" ]; then
    echo "Unsupported object: $OBJECT_NAME"
    exit 1
fi

INPUT_ROOT="${ROOT_DIR:-assets/object}"
INPUT_PATH="${INPUT_DIR:-$INPUT_ROOT/$OBJECT_NAME}"
OUTPUT_ROOT="${OUTPUT_DIR:-${SAVE_DIR:-cache/grasp}}"

python generate.py \
    robot="${ROBOT:-sharpa}" \
    object="$OBJECT_NAME" \
    visualize="${VISUALIZE:-True}" \
    object.asset.root="$INPUT_ROOT" \
    object.asset.input_dir="$INPUT_PATH" \
    save_dir="$OUTPUT_ROOT" \
    "$@"
