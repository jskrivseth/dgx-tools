# llama.cpp engine module for dgx-tools.
#
# The official multi-arch CUDA server image avoids a local build and supports
# GGUF models directly from Hugging Face. The IQ4_XS quant, Q8 KV cache, and
# large unified-memory-friendly batches fit Qwen3.8-Flash-Next's full native
# context on one 128 GB DGX Spark without a patched vLLM image.

ENGINE_NAME="llama-cpp"
ENGINE_STATUS="ready"
ENGINE_CONTAINER_NAME="llama-cpp-server"
ENGINE_IMAGE="${LLAMA_CPP_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda}"
ENGINE_HF_APPS_FILTER="llama.cpp"
ENGINE_STARTUP_TIMEOUT="${LLAMA_CPP_STARTUP_TIMEOUT:-3600}"

ENGINE_MODEL_VAR="LLAMA_CPP_MODEL"
ENGINE_MAX_LEN_VAR="LLAMA_CPP_MAX_MODEL_LEN"
ENGINE_API_KEY_VAR="LLAMA_CPP_API_KEY"
ENGINE_PORT_VAR="LLAMA_CPP_PORT"
ENGINE_DEFAULT_PORT="8000"

ENGINE_DEFAULT_MODEL="unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS"
ENGINE_RECOMMENDED_MODELS=(
  "Qwen3.8 Flash-Next GGUF|unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS|~94GB|recommended single-Spark quant; measured full 256K context at ~27 tok/s"
  "Qwen3.8 Flash-Next GGUF|unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q3_K_XL|~90GB|more memory headroom and lower quality"
  "Qwen3.8 Flash-Next GGUF|unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q2_K_XL|~79GB|smaller and faster-loading Flash-Next quant"
)

resolve_default_context_override() {
  case "$1" in
    *Qwen3.8-Flash-Next*) echo "65536" ;;
    *) return 1 ;;
  esac
}

# GGUF repository IDs may include llama.cpp's ":QUANT" selector, which is
# not part of the underlying Hugging Face repository name.
resolve_max_context() {
  case "$1" in
    *Qwen3.8-Flash-Next*) echo "262144" ;;
    *) resolve_max_context_from_hf_repo "${1%%:*}" ;;
  esac
}

# Download only the selected GGUF shard set. A plain `hf download` of this
# repository would fetch every quantization and consume well over a terabyte.
engine_model_pull() {
  local model="$1"
  shift || true
  local repo="${model%%:*}"
  local quant=""
  [[ "$model" == *:* ]] && quant="${model##*:}"

  ensure_hf_cli || return 1
  ensure_hub_cache || return 1
  if [[ -n "$quant" && "$*" != *--include* ]]; then
    hf download "$repo" --include "${quant}/*" "$@"
  else
    hf download "$repo" "$@"
  fi
}

engine_run_container() {
  local model="$1" max_len="$2" port="$3" _gpu_mem="$4" api_key="$5"
  local _tool_call_parser="${6:-}" _reasoning_parser="${7:-}"
  local hf_env=()
  [[ -n "${HF_TOKEN:-}" ]] && hf_env=(-e "HF_TOKEN=$HF_TOKEN")

  local parallel="${LLAMA_CPP_PARALLEL:-1}"
  if [[ ! "$parallel" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: LLAMA_CPP_PARALLEL must be a positive integer." >&2
    return 1
  fi

  docker run -d \
    --name "$ENGINE_CONTAINER_NAME" \
    --gpus all \
    --shm-size=16g \
    "${hf_env[@]}" \
    -v "${HUB_CACHE}:/root/.cache/huggingface/hub" \
    -p "${port}:8080" \
    "$ENGINE_IMAGE" \
    --hf-repo "$model" \
    --port 8080 \
    --api-key "$api_key" \
    --ctx-size "${max_len:-65536}" \
    --parallel "$parallel" \
    --n-gpu-layers all \
    --flash-attn on \
    --fit off \
    --load-mode none \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --batch-size 2048 \
    --ubatch-size 2048 \
    --threads 20 \
    --cache-ram 8192 \
    --no-mmproj \
    --jinja \
    --no-reasoning-preserve \
    --chat-template-kwargs '{"reasoning_effort":"medium"}'
}
