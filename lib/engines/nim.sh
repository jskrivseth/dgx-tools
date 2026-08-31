# NIM engine module for dgx-tools. Fully implemented.
#
# NVIDIA Inference Microservices — prebuilt, self-tuned containers from
# NVIDIA's NGC registry. Architecturally different from vLLM in ways that
# matter here:
#   - One image PER MODEL (the model choice IS the image tag), not one
#     image + a --model flag. "Model" below means an NGC image tag.
#   - Requires `docker login nvcr.io` with an NGC API key before the image
#     can even be pulled (separate from any HuggingFace auth).
#   - No --gpu-memory-utilization equivalent — each container auto-tunes
#     GPU memory for the detected hardware. It DOES support a context
#     length override (NIM_MAX_MODEL_LEN), but unlike vLLM there's no
#     reliable way for dgxt to auto-derive "the right" value (see
#     resolve_max_context override below) since the model value here is
#     an NGC image tag, not an HF repo id with a config.json to read.
#     See engine_defines_var()/ENGINE_API_KEY_EXTERNAL in dgxt for how
#     these per-engine differences are handled generically.
#   - Only a small, NVIDIA-curated catalog of models exist as NIMs — there's
#     no free-form "serve any HF model" option like vLLM. `dgxt search`
#     queries NGC's public catalog search API directly (see engine_search
#     below); `dgxt model-pull`/`model-list` pre-pull/list Docker images
#     directly (see engine_model_pull/engine_model_list below) since NIM
#     models aren't HF repos `hf download`/`hf cache ls` could handle.
#
# Reference: https://build.nvidia.com/spark/nim-llm/overview

ENGINE_NAME="nim"
ENGINE_STATUS="ready"
ENGINE_CONTAINER_NAME="nim-server"

# Config file env var names this engine reads/writes.
ENGINE_MODEL_VAR="NIM_IMAGE"
ENGINE_API_KEY_VAR="NGC_API_KEY"
ENGINE_PORT_VAR="NIM_PORT"
ENGINE_MAX_LEN_VAR="NIM_MAX_MODEL_LEN"

# NIM auto-selects a hardware "profile" per container that isn't
# guaranteed to use the model's full native context — e.g. Qwen3-32B
# (native 40960) can auto-select an 8192-token profile. Confirmed NIM's
# vLLM backend enforces the EXACT same native-max validation vLLM itself
# does (same error, same VLLM_ALLOW_LONG_MAX_MODEL_LEN escape hatch) --
# so cmd_start's pre-flight native-max warning is just as valuable here,
# it just needs the real underlying HF repo id, not the NGC image tag.
# Verified via each image's NGC catalog page / actual container logs.
# NOTE: must use `declare -g -A` (global), not plain `declare -A` --
# this file is sourced from inside load_engine(), a bash function, so a
# non-global declare here would scope the array to that function call
# and it would vanish before resolve_max_context() is ever actually
# invoked from cmd_start (a real bug hit and fixed during development:
# an unset array indexed by a non-numeric key like a repo path makes
# bash try to evaluate it as an arithmetic expression instead).
declare -g -A NIM_KNOWN_HF_MODELS=(
  ["nvcr.io/nim/meta/llama-3.1-8b-instruct-dgx-spark:latest"]="meta-llama/Llama-3.1-8B-Instruct"
  ["nvcr.io/nim/qwen/qwen3-32b-dgx-spark:latest"]="Qwen/Qwen3-32B"
  ["nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2-dgx-spark:latest"]="nvidia/NVIDIA-Nemotron-Nano-9B-v2"
)

# Overrides common.sh's generic resolve_max_context (which assumes
# "model" already IS an HF repo id, true for vLLM but not NIM's NGC image
# tags). For a known-mapped image (the table above), delegates to the
# shared HF-based resolver on the real underlying repo id; for anything
# else, returns empty (meaning: don't pass NIM_MAX_MODEL_LEN, let NIM's
# own profile default apply) rather than guessing wrong or silently
# falling back to a meaningless hardcoded 131072.
resolve_max_context() {
  local model="$1"
  local hf_model="${NIM_KNOWN_HF_MODELS[$model]:-}"
  if [[ -n "$hf_model" ]]; then
    resolve_max_context_from_hf_repo "$hf_model"
  else
    echo ""
  fi
}

# NGC_API_KEY is a real credential from ngc.nvidia.com, not something dgxt
# can safely invent — auto-generating a random fallback (like vLLM's
# serving API key) would silently persist a bogus value and fail
# confusingly at `docker login`. See ENGINE_API_KEY_EXTERNAL handling in
# dgxt's cmd_start/cmd_save/cmd_setup.
ENGINE_API_KEY_EXTERNAL="1"
ENGINE_API_KEY_HINT="Generate one at https://org.ngc.nvidia.com/account/api-keys -> 'Generate Personal Key' -> under 'Key Permissions / Services Included', check 'NGC Catalog' (required for docker pull; 'Public API Endpoints' alone is NOT enough)."

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
# has resolved model/max_len/port/api_key — gpu_mem/tool_call_parser are
# accepted for signature compatibility but unused (no equivalent knobs;
# see the module comment above and resolve_max_context override).
engine_run_container() {
  local model="$1" max_len="$2" port="$3" _gpu_mem="$4" ngc_api_key="$5" _tool_call_parser="${6:-}"

  mkdir -p "$NIM_CACHE_DIR" "$NIM_WORKSPACE_DIR" "$NIM_DOCKER_CONFIG_DIR"
  chmod -R a+w "$NIM_CACHE_DIR" "$NIM_WORKSPACE_DIR"

  # docker login is itself idempotent/fast, so just always run it rather
  # than trying (unreliably) to detect prior auth from config.json.
  echo "Logging in to nvcr.io..."
  if ! echo "$ngc_api_key" | docker --config "$NIM_DOCKER_CONFIG_DIR" login nvcr.io --username '$oauthtoken' --password-stdin; then
    echo "" >&2
    echo "ERROR: docker login nvcr.io failed (see docker's error above)." >&2
    echo "  Most likely: the key's Key Permissions don't include 'NGC Catalog'." >&2
    echo "  $ENGINE_API_KEY_HINT" >&2
    echo "  Also possible: your account hasn't accepted this specific image's EULA" >&2
    echo "  yet — open its NGC catalog page and click 'Get Container' once." >&2
    return 1
  fi

  local max_len_args=()
  [[ -n "$max_len" ]] && max_len_args=(-e "NIM_MAX_MODEL_LEN=$max_len")

  docker --config "$NIM_DOCKER_CONFIG_DIR" run -d \
    --name "$ENGINE_CONTAINER_NAME" \
    --gpus all \
    --shm-size=16GB \
    -e NGC_API_KEY="$ngc_api_key" \
    "${max_len_args[@]}" \
    -v "${NIM_CACHE_DIR}:/opt/nim/.cache" \
    -v "${NIM_WORKSPACE_DIR}:/opt/nim/workspace" \
    -p "${port}:8000" \
    "$model"
}

# Search NVIDIA's public NGC catalog for NIM containers. Uses the same
# unauthenticated search API catalog.ngc.nvidia.com's own web UI calls
# (no NGC CLI or API key needed just to browse). Called generically by
# dgxt's cmd_search, which delegates here when this function is defined.
#
# Important caveat: this searches ALL NIM containers, most of which are
# x86_64-only — only results with "-dgx-spark" in the name are confirmed
# ARM64/GB10-native (see the module comment above re: narrow ARM64
# coverage). Flagged in the output rather than filtered out entirely,
# since NVIDIA adds new DGX Spark-native images over time and a name-based
# heuristic could miss ones that don't follow the naming convention yet.
engine_search() {
  local query="$1"
  if ! command -v curl >/dev/null 2>&1 || ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: search requires curl and python3." >&2
    return 1
  fi

  local qparam
  qparam=$(python3 -c "
import urllib.parse, json, sys
print(urllib.parse.quote(json.dumps({'query': sys.argv[1], 'page': 0, 'pageSize': 50})))
" "$query")

  local resp
  resp=$(curl -sf --max-time 10 "https://api.ngc.nvidia.com/v2/search/catalog/resources/CONTAINER?q=${qparam}" 2>/dev/null)
  if [[ -z "$resp" ]]; then
    echo "ERROR: NGC catalog search request failed (network issue?)." >&2
    return 1
  fi

  echo "$resp" | python3 -c "
import json, sys
d = json.load(sys.stdin)
seen = {}
for r in d.get('results', []):
    for item in r.get('resources', []):
        if item.get('orgName') == 'nim':
            seen[item['resourceId']] = item.get('displayName', '')
if not seen:
    print('No NIM containers found for that query.')
else:
    for rid in sorted(seen):
        tag = 'nvcr.io/' + rid + ':latest'
        native = '  [DGX-Spark/ARM64-native]' if 'dgx-spark' in rid else ''
        print(f'  {tag}{native}')
        print(f'      {seen[rid]}')
"
  echo ""
  echo "NOTE: only '[DGX-Spark/ARM64-native]' results are confirmed to run on this"
  echo "hardware. Most NIM images are x86_64-only. Full curated ARM64 list:"
  echo "  https://build.nvidia.com/spark"
}

# Pre-pull the NIM Docker image ahead of `dgxt start`. Note this only
# pre-fetches the image itself, not the model weights inside it — those
# are downloaded by the container from NGC into NIM_CACHE_DIR on first
# run, and NIM exposes no separate "pre-fetch just the weights" step.
# Called generically by dgxt's cmd_model_pull.
engine_model_pull() {
  local model="$1"
  ensure_docker
  local ngc_api_key="${!ENGINE_API_KEY_VAR:-}"
  if [[ -z "$ngc_api_key" ]]; then
    echo "ERROR: $ENGINE_API_KEY_VAR is not set." >&2
    echo "  $ENGINE_API_KEY_HINT" >&2
    return 1
  fi

  mkdir -p "$NIM_DOCKER_CONFIG_DIR"
  echo "Logging in to nvcr.io..."
  if ! echo "$ngc_api_key" | docker --config "$NIM_DOCKER_CONFIG_DIR" login nvcr.io --username '$oauthtoken' --password-stdin; then
    echo "ERROR: docker login nvcr.io failed (see docker's error above)." >&2
    echo "  $ENGINE_API_KEY_HINT" >&2
    return 1
  fi

  echo "Pulling $model..."
  docker --config "$NIM_DOCKER_CONFIG_DIR" pull "$model"
}

# List locally pulled NIM images — docker's own image cache, not a
# separate model-weights cache (those live under NIM_CACHE_DIR, downloaded
# per-container on first run; see engine_model_pull above).
engine_model_list() {
  ensure_docker
  echo "Locally pulled NIM images:"
  docker images --filter "reference=nvcr.io/nim/*" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"
  echo ""
  echo "Downloaded model weights (fetched by the container from NGC on first run):"
  echo "  $NIM_CACHE_DIR"
}
