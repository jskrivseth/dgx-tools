# SGLang engine module for dgx-tools.
#
# This profile follows the SGLang Qwen3.8-27B cookbook's DGX Spark recipe:
# FlashInfer attention, FP8 KV cache, 0.80 unified-memory fraction, a 2048
# token prefill chunk, and configurable speculative decoding. DFlash2 is the
# default; DSpark remains available as a fallback.

ENGINE_NAME="sglang"
ENGINE_STATUS="ready"
ENGINE_CONTAINER_NAME="sglang-server"
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
  "RadixArk/Qwen3.8-27B-NVFP4|~20GB|DGX Spark NVFP4 W4A4 recipe with in-checkpoint MTP"
)

resolve_gpu_memory() {
  echo "${SGLANG_GPU_MEM:-0.80}"
}

engine_run_container() {
  local model="$1" _max_len="$2" port="$3" gpu_mem="$4" api_key="$5"
  local tool_call_parser="${6:-}" reasoning_parser="${7:-}"
  local parser_args=()
  local speculative_mode="${SGLANG_SPECULATIVE_MODE:-$ENGINE_DEFAULT_SPECULATIVE_MODE}"
  local speculative_tokens="${SGLANG_SPECULATIVE_TOKENS:-$ENGINE_DEFAULT_SPECULATIVE_TOKENS}"
  local max_running_requests="${SGLANG_MAX_RUNNING_REQUESTS:-$ENGINE_DEFAULT_MAX_RUNNING_REQUESTS}"
  local speculative_args=()

  if [[ ! "$max_running_requests" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: SGLANG_MAX_RUNNING_REQUESTS must be a positive integer." >&2
    return 1
  fi

  [[ -n "$tool_call_parser" ]] && parser_args+=(--tool-call-parser "$tool_call_parser")
  [[ -n "$reasoning_parser" ]] && parser_args+=(--reasoning-parser "$reasoning_parser")

  case "$speculative_mode" in
    dflash2)
      speculative_args=(
        --speculative-algorithm DFLASH
        --speculative-draft-model-path "${SGLANG_SPECULATIVE_MODEL:-incoai/Qwen3.8-27B-DFlash2}"
        --speculative-num-draft-tokens "$speculative_tokens"
      )
      ;;
    dspark)
      speculative_args=(
        --speculative-algorithm DSPARK
        --speculative-draft-model-path "${SGLANG_SPECULATIVE_MODEL:-RadixArk/Qwen3.8-27B-DSpark}"
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
    "$ENGINE_IMAGE" \
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
      "${speculative_args[@]}" \
      --api-key "$api_key" \
      "${parser_args[@]}"
}
