#!/usr/bin/env bash
# Build the repo-owned Flash-Next image used by dgxt.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
image="${IMAGE:-dgxt/qwen38-flash-next:latest}"

echo ">> building $image from $project_dir"
docker build --tag "$image" "$project_dir"
echo ">> built $image"
