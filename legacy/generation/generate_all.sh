#!/usr/bin/env bash
set -euo pipefail

OUT="./synthetic_ordos_all"
mkdir -p "$OUT/images"

run() {
  id="$1"; shift
  tmp="$OUT/_run_$id"
  python generate_synthetic_ordonnances.py --out_dir "$tmp" "$@"
  for f in "$tmp"/images/*.png; do
    base="$(basename "$f")"
    mv "$f" "$OUT/images/${id}_${base}"
  done
  for f in "$tmp"/*.fhir.json; do
    base="$(basename "$f")"
    mv "$f" "$OUT/${id}_${base}"
  done
  if [ -f "$tmp/annotations.json" ]; then
    mv "$tmp/annotations.json" "$OUT/annotations_${id}.json"
  fi
  rm -rf "$tmp"
}

run v01 --n 50  --paper A5 --style mixed --blur 0.6 --jpeg 0.5 --skew 2.5
run v02 --n 60  --paper A5 --style typed --blur 0.3 --jpeg 0.3 --skew 1.5 --stains 0.1
run v03 --n 80  --paper A4 --style mixed --blur 0.5 --jpeg 0.6 --skew 2.0 --stains 0.3
run v04 --n 100 --paper A5 --style hand  --blur 0.7 --jpeg 0.5 --skew 3.0 --stains 0.25
run v05 --n 120 --paper A4 --style typed --blur 0.2 --jpeg 0.4 --skew 1.0 --stains 0.2
run v06 --n 150 --paper A5 --style mixed --blur 0.6 --jpeg 0.4 --skew 2.5 --stains 0.35
run v07 --n 90  --paper A4 --style hand  --blur 0.4 --jpeg 0.5 --skew 3.5 --stains 0.3
run v08 --n 200 --paper A5 --style typed --blur 0.5 --jpeg 0.5 --skew 2.2 --stains 0.15
run v09 --n 50  --paper A4 --style mixed --blur 0.8 --jpeg 0.6 --skew 2.8 --stains 0.4
run v10 --n 125 --paper A5 --style hand  --blur 0.3 --jpeg 0.7 --skew 1.8 --stains 0.2
run v11 --n 160 --paper A4 --style typed --blur 0.6 --jpeg 0.2 --skew 2.0 --stains 0.25
run v12 --n 180 --paper A5 --style mixed --blur 0.2 --jpeg 0.6 --skew 1.2 --stains 0.3
run v13 --n 75  --paper A4 --style hand  --blur 0.5 --jpeg 0.3 --skew 2.7 --stains 0.35
run v14 --n 220 --paper A5 --style typed --blur 0.7 --jpeg 0.4 --skew 3.2 --stains 0.4
run v15 --n 140 --paper A4 --style mixed --blur 0.4 --jpeg 0.4 --skew 1.6 --stains 0.2
run v16 --n 260 --paper A5 --style hand  --blur 0.6 --jpeg 0.6 --skew 2.4 --stains 0.45
run v17 --n 300 --paper A4 --style typed --blur 0.3 --jpeg 0.5 --skew 2.9 --stains 0.3
run v18 --n 90  --paper A5 --style mixed --blur 0.5 --jpeg 0.2 --skew 1.4 --stains 0.25
run v19 --n 200 --paper A4 --style hand  --blur 0.8 --jpeg 0.5 --skew 3.0 --stains 0.5
run v20 --n 240 --paper A5 --style typed --blur 0.2 --jpeg 0.3 --skew 1.0 --stains 0.1
