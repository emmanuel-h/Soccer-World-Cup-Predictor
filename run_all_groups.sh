#!/bin/bash

GROUP_DIR="group_stage"
SCRIPT="predictor.py"
RESULTS_DIR="results"

mkdir -p "$RESULTS_DIR"

# Remove stale match CSVs so headers are written fresh
for group_file in "$GROUP_DIR"/group_*.txt; do
    group_letter=$(basename "$group_file" .txt | sed 's/group_//')
    rm -f "$RESULTS_DIR/group_${group_letter}.csv"
done

for group_file in "$GROUP_DIR"/group_*.txt; do
    group_letter=$(basename "$group_file" .txt | sed 's/group_//')
    group_name="Group $group_letter"
    output_file="$RESULTS_DIR/group_${group_letter}.csv"

    mapfile -t teams < "$group_file"

    echo "Running $group_name: ${teams[*]}" >&2

    python3 "$SCRIPT" \
        --matches-csv "$output_file" \
        --group "$group_name" \
        "${teams[@]}"

    echo "  -> $output_file" >&2
done

echo "Done. Results saved to $RESULTS_DIR/" >&2
