#!/usr/bin/env bash
# Build the vLLM 0.29 Flash-Next image used by the dgxt v0.29 profile.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
image="${IMAGE:-dgxt/qwen38-flash-next-v029:latest}"

echo ">> building $image from $project_dir"
docker build --file "$project_dir/Dockerfile.v0.29" --tag "$image" "$project_dir"
echo ">> built $image"
