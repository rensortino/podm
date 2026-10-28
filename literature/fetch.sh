#!/usr/bin/env bash
# Download every paper listed in papers.tsv into this folder (skips files already present).
set -euo pipefail
cd "$(dirname "$0")"
grep -v '^#' papers.tsv | while IFS=$'\t' read -r id file _; do
  [ -z "$id" ] && continue
  if [ -f "$file" ]; then echo "have $file"; continue; fi
  echo "fetching $id -> $file"
  curl -sfL -o "$file" "https://arxiv.org/pdf/$id"
done
