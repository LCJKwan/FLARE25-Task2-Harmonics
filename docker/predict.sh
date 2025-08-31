#!/usr/bin/env bash
set -eu pipefail

# locations are fixed by the challenge harness:
INPUT_DIR=/workspace/inputs
OUTPUT_DIR=/workspace/outputs

# ensure output dir exists
mkdir -p "$OUTPUT_DIR"

# example invocation; adjust flags as needed:
python inference.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR"
