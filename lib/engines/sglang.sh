# SGLang engine module for dgx-tools.
#
# Multi-model SGLang engine for DGX Spark-class (Blackwell sm_121a, ARM64)
# hardware. Each model gets its own dev image (model-specific kernel
# optimizations), speculative-decoding defaults, and architecture flags.
#
# Supported models:
#   - Qwen3.8-27B NVFP4 (dense hybrid-attention VLM; DFlash2/DSpark/MTP)
#   - Nemotron 3.5 Lightning 30B-A3B NVFP4 (hybrid Mamba-Transformer MoE;
#     DSpark/MTP/DFlash; 1M context; ~5,400 tok/s prefill)
#   - Qwen3-Coder-Next NVFP4 GB10 (80B/3B hybrid GDN coder; DFlash;
#     150 tok/s short code; requires SGLang patches)
#   - Qwen3.6 35B A3B NVFP4 (MoE 3B active; MTP speculation; agent-ready)
#
# All models use FlashInfer attention and FP8 KV cache.

ENGINE_NAME="sglang"
ENGINE_STATUS="ready"
ENGINE_CONTAINER_NAME="sglang-server"
# Default image (used for display and as fallback). The actual image is
# resolved per-model in resolve_engine_image().
ENGINE_IMAGE="lmsysorg/sglang:dev-qwen38-27b-dflash2"
ENGINE_HF_APPS_FILTER="sglang"

# SGLang is intentionally a sibling service: it uses a different port and
# starting it must not offer to stop an already-running vLLM service.
ENGINE_ALLOW_OTHER_ENGINES_RUNNING="1"

ENGINE_MODEL_VAR="SGLANG_MODEL"
ENGINE_API_KEY_VAR="SGLANG_API_KEY"
ENGINE_PORT_VAR="SGLANG_PORT"
ENGINE_GPU_MEM_VAR="SGLANG_GPU_MEM"
ENGINE_TOOL_CALL_PARSER_VAR="SGLANG_TOOL_CALL_PARSER"
ENGINE_REASONING_PARSER_VAR="SGLANG_REASONING_PARSER"
ENGINE_SPECULATIVE_MODE_VAR="SGLANG_SPECULATIVE_MODE"
ENGINE_SPECULATIVE_MODEL_VAR="SGLANG_SPECULATIVE_MODEL"
ENGINE_SPECULATIVE_TOKENS_VAR="SGLANG_SPECULATIVE_TOKENS"
ENGINE_DEFAULT_MODEL="RadixArk/Qwen3.8-27B-NVFP4"
ENGINE_DEFAULT_TOOL_CALL_PARSER="qwen3_coder"
ENGINE_DEFAULT_REASONING_PARSER="qwen3"
ENGINE_DEFAULT_SPECULATIVE_MODE="dflash2"
ENGINE_DEFAULT_SPECULATIVE_TOKENS="8"
ENGINE_DEFAULT_MAX_RUNNING_REQUESTS="8"

# SGLang's model card and cookbook use 30000. 8001 keeps the sibling usable
# while the existing vLLM server continues to own port 8000.
ENGINE_DEFAULT_PORT="8001"

ENGINE_RECOMMENDED_MODELS=(
  "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4|~22GB|hybrid Mamba-MoE, 3B active, 1M context, DSpark -- fastest prefill on Spark"
  "saricles/Qwen3-Coder-Next-NVFP4-GB10|~43GB|80B/3B hybrid GDN coder, DFlash -- 150 tok/s short code; requires SGLang patches (see README)"
  "nvidia/Qwen3.6-35B-A3B-NVFP4|~18GB|MoE 3B active, MTP speculation, agent-ready tool calling"
  "RadixArk/Qwen3.8-27B-NVFP4|~20GB|dense hybrid-attention VLM, DFlash2/DSpark/MTP speculation"
)

# ---- Model-specific resolvers -------------------------------------------

# SGLang ships model-specific dev images with optimized kernels. Using the
# wrong image may lack the architecture support or speculative-decoding
# integration for that model.
resolve_engine_image() {
  local model="$1"
  case "$model" in
    *Nemotron-3.5-Lightning*|*nemotron-3.5-lightning*)
      echo "lmsysorg/sglang:dev-nemotron3-5-lightning" ;;
    *Qwen3-Coder-Next*|*qwen3-coder-next*)
      echo "lmsysorg/sglang:nightly-dev-cu13-20260415-2c9e76d3" ;;
    *Qwen3.6*|*qwen3.6*)
      echo "lmsysorg/sglang:nightly-dev-cu13-20260415-2c9e76d3" ;;
    *Qwen3.8*|*qwen3.8*)
      echo "lmsysorg/sglang:dev-qwen38-27b-dflash2" ;;
    *)
      echo "lmsysorg/sglang:dev-qwen38-27b-dflash2" ;;
  esac
}

# Model-specific GPU memory fraction. Nemotron's official Spark recipe uses
# 0.85 (validated by NVIDIA); Qwen3.8 uses 0.80.
resolve_gpu_memory() {
  local model="$1" _max_len="${2:-}"
  if [[ -n "${SGLANG_GPU_MEM:-}" ]]; then
    echo "$SGLANG_GPU_MEM"
    return
  fi
  case "$model" in
    *Nemotron-3.5-Lightning*|*nemotron-3.5-lightning*)
      echo "0.85" ;;
    *Qwen3-Coder-Next*|*qwen3-coder-next*)
      # 0.55 is conservative for the ~43GB NVFP4 checkpoint; 0.60 is safe
      # once stable. Higher risks unified-memory OOM requiring a power cycle.
      echo "0.55" ;;
    *Qwen3.6*|*qwen3.6*)
      # 0.60 matches the vLLM Spark recipe; the ~18GB NVFP4 checkpoint
      # leaves ample unified memory for KV cache at native 256K context.
      echo "0.60" ;;
    *)
      echo "0.80" ;;
  esac
}

# Model-specific default speculative mode. DSpark is the best speculator
# for Nemotron on DGX Spark (per SGLang's day-0 announcement); DFlash2 is
# the best for Qwen3.8-27B.
resolve_default_speculative_mode() {
  local model="$1"
  case "$model" in
    *Nemotron-3.5-Lightning*|*nemotron-3.5-lightning*)
      echo "dspark" ;;
    *Qwen3-Coder-Next*|*qwen3-coder-next*)
      echo "dflash" ;;
    *Qwen3.6*|*qwen3.6*)
      echo "mtp" ;;
    *Qwen3.8*|*qwen3.8*)
      echo "dflash2" ;;
    *)
      echo "none" ;;
  esac
}

# Model-specific default draft model for speculative decoding.
resolve_default_speculative_model() {
  local model="$1"
  case "$model" in
    *Nemotron-3.5-Lightning*|*nemotron-3.5-lightning*)
      echo "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark" ;;
    *Qwen3-Coder-Next*|*qwen3-coder-next*)
      echo "z-lab/Qwen3-Coder-Next-DFlash" ;;
    *Qwen3.6*|*qwen3.6*)
      # MTP head is baked into the checkpoint; no separate draft model.
      echo "" ;;
    *Qwen3.8*|*qwen3.8*)
      echo "incoai/Qwen3.8-27B-DFlash2" ;;
    *)
      echo "" ;;
  esac
}

# Model-specific default max running requests. Qwen3-Coder-Next's ~43GB
# checkpoint leaves less unified memory for KV cache, so 4 concurrent
# requests is the validated safe default.
resolve_max_running_requests() {
  local model="$1"
  case "$model" in
    *Qwen3-Coder-Next*|*qwen3-coder-next*)
      echo "4" ;;
    *)
      echo "$ENGINE_DEFAULT_MAX_RUNNING_REQUESTS" ;;
  esac
}

# Model-specific reasoning parser (called by resolve_reasoning_parser_for_engine
# in common.sh when the engine defines this function).
resolve_reasoning_parser() {
  local model="$1"
  case "$model" in
    *Nemotron-3.5-Lightning*|*nemotron-3.5-lightning*)
      echo "nemotron_3" ;;
    *Qwen3-Coder-Next*|*qwen3-coder-next*|*Qwen3.6*|*qwen3.6*|*Qwen3.8*|*qwen3.8*)
      echo "qwen3" ;;
    *)
      echo "" ;;
  esac
}

# ---- Container launch ---------------------------------------------------

engine_run_container() {
  local model="$1" _max_len="$2" port="$3" gpu_mem="$4" api_key="$5"
  local tool_call_parser="${6:-}" reasoning_parser="${7:-}"
  local parser_args=()
  local arch_args=()
  local image
  image=$(resolve_engine_image "$model")

  # Resolve model-specific defaults for speculative decoding.
  local default_spec_mode
  default_spec_mode=$(resolve_default_speculative_mode "$model")
  local speculative_mode="${SGLANG_SPECULATIVE_MODE:-$default_spec_mode}"
  local speculative_tokens="${SGLANG_SPECULATIVE_TOKENS:-$ENGINE_DEFAULT_SPECULATIVE_TOKENS}"
  local default_spec_model
  default_spec_model=$(resolve_default_speculative_model "$model")
  local default_max_running
  default_max_running=$(resolve_max_running_requests "$model")
  local max_running_requests="${SGLANG_MAX_RUNNING_REQUESTS:-$default_max_running}"
  local speculative_args=()

  if [[ ! "$max_running_requests" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: SGLANG_MAX_RUNNING_REQUESTS must be a positive integer." >&2
    return 1
  fi

  [[ -n "$tool_call_parser" ]] && parser_args+=(--tool-call-parser "$tool_call_parser")
  [[ -n "$reasoning_parser" ]] && parser_args+=(--reasoning-parser "$reasoning_parser")

  # Architecture-specific flags (e.g. Mamba backend for hybrid models).
  case "$model" in
    *Nemotron-3.5-Lightning*|*nemotron-3.5-lightning*)
      arch_args=(
        --mamba-backend flashinfer
        --mamba-radix-cache-strategy extra_buffer
      )
      ;;
    *Qwen3-Coder-Next*|*qwen3-coder-next*)
      arch_args=(
        --mamba-scheduler-strategy extra_buffer
        --disable-cuda-graph
      )
      ;;
  esac

  case "$speculative_mode" in
    dflash|dflash2)
      speculative_args=(
        --speculative-algorithm DFLASH
        --speculative-draft-model-path "${SGLANG_SPECULATIVE_MODEL:-$default_spec_model}"
        --speculative-num-draft-tokens "$speculative_tokens"
      )
      ;;
    dspark)
      speculative_args=(
        --speculative-algorithm DSPARK
        --speculative-draft-model-path "${SGLANG_SPECULATIVE_MODEL:-$default_spec_model}"
        --speculative-draft-attention-backend flashinfer
      )
      ;;
    eagle|mtp)
      speculative_args=(
        --speculative-algorithm EAGLE
        --speculative-num-steps 3
        --speculative-eagle-topk 1
        --speculative-num-draft-tokens 4
        --enable-linear-replayssm-spec
      )
      ;;
    none)
      ;;
    *)
      echo "ERROR: SGLANG_SPECULATIVE_MODE must be dflash2, dspark, eagle, or none." >&2
      return 1
      ;;
  esac

  # Model-specific environment variables.
  local env_args=()
  case "$model" in
    *Qwen3-Coder-Next*|*qwen3-coder-next*)
      # DeepGEMM disabled: the scale format of this checkpoint doesn't match
      # what DeepGEMM expects on Blackwell, causing accuracy degradation.
      env_args=(
        -e SGLANG_ENABLE_JIT_DEEPGEMM=0
        -e SGLANG_ENABLE_DEEP_GEMM=0
      )
      ;;
  esac

  docker run -d \
    --name "$ENGINE_CONTAINER_NAME" \
    --gpus all \
    --ipc host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --entrypoint "" \
    -p "${port}:30000" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v "${HUB_CACHE}:/root/.cache/huggingface/hub" \
    "${env_args[@]}" \
    "$image" \
    python3 -m sglang.launch_server \
      --trust-remote-code \
      --model-path "$model" \
      --host 0.0.0.0 \
      --port 30000 \
      --kv-cache-dtype fp8_e4m3 \
      --mem-fraction-static "$gpu_mem" \
      --max-running-requests "$max_running_requests" \
      --attention-backend flashinfer \
      --chunked-prefill-size 2048 \
      "${arch_args[@]}" \
      "${speculative_args[@]}" \
      --api-key "$api_key" \
      "${parser_args[@]}"
}
