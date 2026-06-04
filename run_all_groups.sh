#!/bin/bash

GROUP_DIR="group_stage"
SCRIPT="predictor.py"
RESULTS_DIR="results"

mkdir -p "$RESULTS_DIR"

for group_file in "$GROUP_DIR"/group_*.txt; do
    group_letter=$(basename "$group_file" .txt | sed 's/group_//')
    group_name="Group $group_letter"
    output_file="$RESULTS_DIR/group_${group_letter}.csv"

    mapfile -t teams < "$group_file"

    echo "Running $group_name: ${teams[*]}" >&2

    output=$(python3 "$SCRIPT" "${teams[@]}" 2>&1)

    echo "Group,Rank,Team,MP,W,D,L,GF,GA,GD,Pts,Adv%" > "$output_file"

    echo "$output" | sed 's/✓//' | awk -v group="$group_name" '
        /PREDICTED STANDINGS/ { in_standings=1; next }
        in_standings && /^[[:space:]]*Team/ { past_header=1; rank=0; next }
        in_standings && /^[[:space:]]*[─═]+/ { next }
        in_standings && past_header && /^[[:space:]]*$/ { exit }
        in_standings && past_header && /[0-9]+\.[0-9]+%/ {
            rank++
            line = $0
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)

            n = split(line, f, /[[:space:]]+/)
            adv  = f[n];   gsub(/%/, "", adv)
            pts  = f[n-1]
            gd   = f[n-2]
            ga   = f[n-3]
            gf   = f[n-4]
            l    = f[n-5]
            d    = f[n-6]
            w    = f[n-7]
            mp   = f[n-8]

            team = ""
            for (i = 1; i <= n-9; i++)
                team = (team == "") ? f[i] : team " " f[i]

            printf "%s,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
                group, rank, team, mp, w, d, l, gf, ga, gd, pts, adv
        }
    ' >> "$output_file"

    echo "  -> $output_file" >&2
done

echo "Done. Results saved to $RESULTS_DIR/" >&2
