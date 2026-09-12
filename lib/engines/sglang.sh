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
ENGINE_MAX_LEN_VAR="SGLANG_MAX_MODEL_LEN"
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
  "Nemotron 3.5 Lightning|nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4|~22GB|hybrid Mamba-MoE, 3B active, 1M context, DSpark"
  "Qwen3.6-35B-A3B|nvidia/Qwen3.6-35B-A3B-NVFP4|~18GB|MoE 3B active, MTP speculation, agent-ready tool calling"
  "Qwen3.8-27B|RadixArk/Qwen3.8-27B-NVFP4|~20GB|dense hybrid-attention VLM, DFlash2/DSpark/MTP speculation"
)

# ---- Model-specific resolvers -------------------------------------------

# Curated model IDs map to stable internal profiles. Runtime overrides below
# dispatch on this exact key; an unregistered/manual model gets "generic"
# instead of accidentally inheriting a sibling model's settings.
model_profile() {
  case "$1" in
    nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) echo "nemotron35" ;;
    nvidia/Qwen3.6-35B-A3B-NVFP4) echo "qwen36" ;;
    RadixArk/Qwen3.8-27B-NVFP4) echo "qwen38" ;;
    *) echo "generic" ;;
  esac
}

# SGLang ships model-specific dev images with optimized kernels. Using the
# wrong image may lack the architecture support or speculative-decoding
# integration for that model.
resolve_engine_image() {
  local model="$1"
  case "$(model_profile "$model")" in
    nemotron35)
      echo "lmsysorg/sglang:dev-nemotron3-5-lightning" ;;
    qwen36)
      echo "lmsysorg/sglang:nightly-dev-cu13-20260415-2c9e76d3" ;;
    qwen38)
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
  case "$(model_profile "$model")" in
    nemotron35)
      echo "0.85" ;;
    qwen36)
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
  case "$(model_profile "$model")" in
    nemotron35)
      echo "dspark" ;;
    qwen36)
      echo "mtp" ;;
    qwen38)
      echo "dflash2" ;;
    *)
      echo "none" ;;
  esac
}

# Model-specific default draft model for speculative decoding.
resolve_default_speculative_model() {
  local model="$1"
  case "$(model_profile "$model")" in
    nemotron35)
      echo "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark" ;;
    qwen36)
      # MTP head is baked into the checkpoint; no separate draft model.
      echo "" ;;
    qwen38)
      echo "incoai/Qwen3.8-27B-DFlash2" ;;
    *)
      echo "" ;;
  esac
}

# Model-specific default speculative token count. MTP heads predict a small
# number of tokens (Qwen3.6's vLLM recipe uses 3); DFlash block drafting
# uses 8.
resolve_default_speculative_tokens() {
  local model="$1"
  case "$(model_profile "$model")" in
    qwen36)
      echo "3" ;;
    *)
      echo "$ENGINE_DEFAULT_SPECULATIVE_TOKENS" ;;
  esac
}

# Model-specific default max running requests.
resolve_max_running_requests() {
  local model="$1"
  case "$(model_profile "$model")" in
    *)
      echo "$ENGINE_DEFAULT_MAX_RUNNING_REQUESTS" ;;
  esac
}

# Check whether an explicitly-set speculative draft model is compatible with
# the target model. Returns 0 (compatible) if the draft model name contains
# the target model's family name, or if the draft is empty (MTP has no
# separate draft). Returns 1 (incompatible) otherwise. This catches the
# common mistake of switching models without clearing SGLANG_SPECULATIVE_MODEL,
# which would otherwise load a draft trained for a different architecture.
spec_model_compatible() {
  local target="$1" draft="$2"
  [[ -z "$draft" ]] && return 0
  case "$(model_profile "$target")" in
    nemotron35)
      [[ "$draft" == *Nemotron* || "$draft" == *nemotron* ]] && return 0 ;;
    qwen38)
      [[ "$draft" == *Qwen3.8* || "$draft" == *qwen3.8* ]] && return 0 ;;
    qwen36)
      [[ "$draft" == *Qwen3.6* || "$draft" == *qwen3.6* ]] && return 0 ;;
  esac
  return 1
}

# Model-specific reasoning parser (called by resolve_reasoning_parser_for_engine
# in common.sh when the engine defines this function).
resolve_reasoning_parser() {
  local model="$1"
  case "$(model_profile "$model")" in
    nemotron35)
      echo "nemotron_3" ;;
    qwen36|qwen38)
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
  local default_spec_tokens
  default_spec_tokens=$(resolve_default_speculative_tokens "$model")
  local speculative_tokens="${SGLANG_SPECULATIVE_TOKENS:-$default_spec_tokens}"
  local default_spec_model
  default_spec_model=$(resolve_default_speculative_model "$model")

  # Guard: if the user explicitly set SGLANG_SPECULATIVE_MODEL but it doesn't
  # match the target model's family, warn and fall back to the model-specific
  # defaults (both mode and draft model). This catches the common mistake of
  # switching models without clearing the previous model's speculative settings.
  if [[ -n "${SGLANG_SPECULATIVE_MODEL:-}" ]]; then
    if ! spec_model_compatible "$model" "$SGLANG_SPECULATIVE_MODEL"; then
      echo "WARNING: SGLANG_SPECULATIVE_MODEL ($SGLANG_SPECULATIVE_MODEL) doesn't match $model."
      echo "  Falling back to model defaults: mode=$default_spec_mode, draft=${default_spec_model:-<none>}"
      echo "  (clear SGLANG_SPECULATIVE_MODEL and SGLANG_SPECULATIVE_MODE from ~/.dgxtrc to fix permanently)"
      SGLANG_SPECULATIVE_MODEL=""
      speculative_mode="$default_spec_mode"
    fi
  fi
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
  case "$(model_profile "$model")" in
    nemotron35)
      arch_args=(
        --mamba-backend flashinfer
        --mamba-radix-cache-strategy extra_buffer
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
        --speculative-num-draft-tokens "$speculative_tokens"
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
  # No curated profile currently needs additional environment overrides.

  # Context length: only pass when explicitly set (SGLang auto-detects the
  # model's native max otherwise).
  local context_args=()
  if [[ -n "$_max_len" && "$_max_len" != "auto" ]]; then
    context_args=(--context-length "$_max_len")
  fi

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
      "${context_args[@]}" \
      "${arch_args[@]}" \
      "${speculative_args[@]}" \
      --api-key "$api_key" \
      "${parser_args[@]}"
}
