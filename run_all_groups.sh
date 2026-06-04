#!/bin/bash

GROUP_DIR="group_stage"
SCRIPT="predictor.py"
RESULTS_DIR="results"

# Output formats: comma-separated list of csv, json, mpp
OUTPUT_FORMAT="${OUTPUT_FORMAT:-csv,json,mpp}"

mkdir -p "$RESULTS_DIR"

# Remove stale output files so headers/arrays are written fresh
# predictor.py lowercases the group slug, so we must match case here
for group_file in "$GROUP_DIR"/group_*.txt; do
    group_letter=$(basename "$group_file" .txt | sed 's/group_//')
    slug=$(echo "group_${group_letter}" | tr '[:upper:]' '[:lower:]')
    rm -f "$RESULTS_DIR/${slug}.csv"
    rm -f "$RESULTS_DIR/${slug}.json"
    rm -f "$RESULTS_DIR/${slug}_mpp.json"
done

for group_file in "$GROUP_DIR"/group_*.txt; do
    group_letter=$(basename "$group_file" .txt | sed 's/group_//')
    group_name="Group $group_letter"

    mapfile -t teams < "$group_file"

    echo "Running $group_name: ${teams[*]}" >&2

    python3 "$SCRIPT" \
        --output-format "$OUTPUT_FORMAT" \
        --output-dir    "$RESULTS_DIR" \
        --group         "$group_name" \
        "${teams[@]}"

    echo "" >&2
done

echo "Done. Results saved to $RESULTS_DIR/" >&2

if [[ "$OUTPUT_FORMAT" == *mpp* ]]; then
    echo "" >&2
    echo "To push all MPP predictions, run:" >&2
    echo "  python3 mpp_push.py results/group_A_mpp.json --championship-id <ID>" >&2
    echo "  (use --list-championships to find your championship ID)" >&2
fi
