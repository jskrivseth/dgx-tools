# NIM engine module for dgx-tools. Fully implemented.
#
# NVIDIA Inference Microservices — prebuilt, self-tuned containers from
# NVIDIA's NGC registry. Architecturally different from vLLM in ways that
# matter here:
#   - One image PER MODEL (the model choice IS the image tag), not one
#     image + a --model flag. "Model" below means an NGC image tag.
#   - Requires `docker login nvcr.io` with an NGC API key before the image
#     can even be pulled (separate from any HuggingFace auth).
#   - No --max-model-len / --gpu-memory-utilization equivalents — each
#     container auto-tunes itself for the detected GPU. See
#     engine_defines_var()/ENGINE_API_KEY_EXTERNAL usage in dgxt for how
#     these differences are handled generically.
#   - Only a small, NVIDIA-curated catalog of models exist as NIMs — there's
#     no free-form "serve any HF model" option like vLLM. Stick to
#     ENGINE_RECOMMENDED_MODELS (or browse https://build.nvidia.com/spark
#     for more) rather than `dgxt search`/`model-pull`, which are HF-only
#     and don't apply here.
#
# Reference: https://build.nvidia.com/spark/nim-llm/overview

ENGINE_NAME="nim"
ENGINE_STATUS="ready"
ENGINE_CONTAINER_NAME="nim-server"

# Config file env var names this engine reads/writes.
ENGINE_MODEL_VAR="NIM_IMAGE"
ENGINE_API_KEY_VAR="NGC_API_KEY"
ENGINE_PORT_VAR="NIM_PORT"

# NGC_API_KEY is a real credential from ngc.nvidia.com, not something dgxt
# can safely invent — auto-generating a random fallback (like vLLM's
# serving API key) would silently persist a bogus value and fail
# confusingly at `docker login`. See ENGINE_API_KEY_EXTERNAL handling in
# dgxt's cmd_start/cmd_save/cmd_setup.
ENGINE_API_KEY_EXTERNAL="1"
ENGINE_API_KEY_HINT="Get one at: https://ngc.nvidia.com/setup/api-key"

# Curated NIM images verified to exist for DGX Spark-class (GB10 ARM64
# Blackwell) hardware as of this writing. NVIDIA's NIM catalog for ARM64
# is still narrow and evolving — check https://build.nvidia.com/spark for
# newer additions. Format: "image tag|approx size|note"
ENGINE_DEFAULT_MODEL="nvcr.io/nim/meta/llama-3.1-8b-instruct-dgx-spark:latest"
ENGINE_RECOMMENDED_MODELS=(
  "nvcr.io/nim/meta/llama-3.1-8b-instruct-dgx-spark:latest|~16GB|fast, well-tested tool calling (default)"
  "nvcr.io/nim/qwen/qwen3-32b-dgx-spark:latest|~64GB|NVFP4, matches this box's vLLM default model"
  "nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2-dgx-spark:latest|~18GB|hybrid reasoning/non-reasoning, fast"
)

# NIM's own cache dirs — distinct from vLLM/HF's ~/.cache/huggingface/hub
# (see lib/common.sh's ensure_hub_cache). NIM downloads model weights from
# NGC on first run into these, not via `hf`.
NIM_CACHE_DIR="${NIM_CACHE_DIR:-$HOME/.cache/nim}"
NIM_WORKSPACE_DIR="${NIM_WORKSPACE_DIR:-$HOME/.local/share/nim/workspace}"

# Use a dgxt-private Docker config dir for nvcr.io auth instead of the
# user's default ~/.docker/config.json. Some machines (e.g. ones with
# NVIDIA AI Workbench installed) configure a global credHelper for
# nvcr.io that can be broken/misconfigured independent of dgxt entirely —
# this sidesteps that class of problem completely rather than requiring
# users to debug or edit their global Docker credential setup.
NIM_DOCKER_CONFIG_DIR="${NIM_DOCKER_CONFIG_DIR:-$HOME/.config/dgx-tools/docker}"

# Start the NIM container. Called by the generic cmd_start in dgxt after it
# has resolved model/port/api_key — max_len/gpu_mem/tool_call_parser are
# accepted for signature compatibility but unused (NIM has no equivalent
# knobs; see the module comment above).
engine_run_container() {
  local model="$1" _max_len="$2" port="$3" _gpu_mem="$4" ngc_api_key="$5" _tool_call_parser="${6:-}"

  mkdir -p "$NIM_CACHE_DIR" "$NIM_WORKSPACE_DIR" "$NIM_DOCKER_CONFIG_DIR"
  chmod -R a+w "$NIM_CACHE_DIR" "$NIM_WORKSPACE_DIR"

  # docker login is itself idempotent/fast, so just always run it rather
  # than trying (unreliably) to detect prior auth from config.json.
  echo "Logging in to nvcr.io..."
  if ! echo "$ngc_api_key" | docker --config "$NIM_DOCKER_CONFIG_DIR" login nvcr.io --username '$oauthtoken' --password-stdin; then
    echo "" >&2
    echo "ERROR: docker login nvcr.io failed (see docker's error above)." >&2
    echo "  Common causes: the key's scope doesn't include 'NGC Catalog'/container" >&2
    echo "  registry access (check when generating it), or your account hasn't" >&2
    echo "  accepted the EULA for this specific image yet — visit its NGC catalog" >&2
    echo "  page and click 'Get Container' once. $ENGINE_API_KEY_HINT" >&2
    return 1
  fi

  docker --config "$NIM_DOCKER_CONFIG_DIR" run -d \
    --name "$ENGINE_CONTAINER_NAME" \
    --gpus all \
    --shm-size=16GB \
    -e NGC_API_KEY="$ngc_api_key" \
    -v "${NIM_CACHE_DIR}:/opt/nim/.cache" \
    -v "${NIM_WORKSPACE_DIR}:/opt/nim/workspace" \
    -p "${port}:8000" \
    "$model"
}
