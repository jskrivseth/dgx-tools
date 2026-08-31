# vLLM engine module for dgx-tools. Fully implemented — the reference
# pattern for adding other engines (see lib/engines/*.sh placeholders).
#
# NVIDIA's own vLLM deployment guidance for DGX Spark-class (Blackwell
# sm_121 + ARM64) hardware is Docker-only: upstream vLLM's native build
# targets don't yet fully cover this combo, so a maintained container image
# is the practical path rather than a native/pip install.

ENGINE_NAME="vllm"
ENGINE_STATUS="ready"
ENGINE_CONTAINER_NAME="vllm-server"
ENGINE_IMAGE="vllm/vllm-openai:latest"
ENGINE_HF_APPS_FILTER="vllm"

# Config file env var names this engine reads/writes (kept as the original
# VLLM_* names for continuity with existing ~/.vllmrc-based configs).
ENGINE_MODEL_VAR="VLLM_MODEL"
ENGINE_MAX_LEN_VAR="VLLM_MAX_MODEL_LEN"
ENGINE_API_KEY_VAR="VLLM_API_KEY"
ENGINE_PORT_VAR="VLLM_PORT"
ENGINE_GPU_MEM_VAR="VLLM_GPU_MEM"
ENGINE_TOOL_CALL_PARSER_VAR="VLLM_TOOL_CALL_PARSER"

# Recommended models for a DGX Spark-class box (128GB unified memory).
# Edit this list for your own hardware/preferences — nothing else depends
# on these specific values. Format: "id|approx size|note"
ENGINE_DEFAULT_MODEL="Qwen/Qwen3-32B"
ENGINE_RECOMMENDED_MODELS=(
  "nvidia/Qwen3.6-35B-A3B-NVFP4|~18GB|best perf, DGX-optimized quantization"
  "Qwen/Qwen3.6-35B-A3B|~70GB|full precision"
  "Qwen/Qwen3-32B|~64GB|full precision (default)"
  "Qwen/Qwen3-8B|~16GB|fast, smaller"
)

# vLLM's OpenAI-compatible server rejects any request with tool/function
# definitions (as every agentic coding CLI sends) unless tool calling is
# explicitly enabled with a parser matched to the model's tool-call output
# format. All models recommended above are Qwen3-family, which emit
# <tool_call>...</tool_call> XML — hence "qwen3_xml". This is only used as
# a last-resort fallback (see resolve_tool_call_parser below); override
# VLLM_TOOL_CALL_PARSER (or set it to empty) if serving a different model
# family; see: https://docs.vllm.ai/en/latest/features/tool_calling.html
ENGINE_DEFAULT_TOOL_CALL_PARSER="qwen3_xml"

# Best-effort auto-detection of the right --tool-call-parser for a given
# model. There's no official/automatic way to do this: HuggingFace has no
# standard "tool call format" field, and vLLM itself has no auto-detect
# mode — it requires an explicit --tool-call-parser matched to a
# hand-maintained model-family table (see the docs link above). This
# mirrors that table, keyed off the model's own config.json
# "architectures" field (the same file the engine itself reads), so
# plugging in a non-default model still gets a sensible parser without
# manual lookup. Falls back to ENGINE_DEFAULT_TOOL_CALL_PARSER only when
# detection can't run at all (e.g. no network); returns empty (tool
# calling disabled) for a recognized-but-unmapped architecture, since
# guessing wrong silently produces malformed tool calls at runtime rather
# than a clean, obvious failure.
resolve_tool_call_parser() {
  local model="$1"
  if ! command -v curl >/dev/null 2>&1 || ! command -v python3 >/dev/null 2>&1; then
    echo "$ENGINE_DEFAULT_TOOL_CALL_PARSER"
    return
  fi

  local config_json
  config_json=$(curl -sfL "https://huggingface.co/${model}/raw/main/config.json" 2>/dev/null)
  if [[ -z "$config_json" ]]; then
    echo "$ENGINE_DEFAULT_TOOL_CALL_PARSER"
    return
  fi

  local arch
  arch=$(echo "$config_json" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    archs = d.get('architectures') or []
    print(archs[0] if archs else '')
except Exception:
    pass
" 2>/dev/null)

  case "$arch" in
    *Qwen3Coder*|*Qwen3_5Moe*|*Qwen3Moe*|*Qwen3*) echo "qwen3_xml" ;;
    *Qwen2*) echo "hermes" ;;
    *Llama4*) echo "llama4_pythonic" ;;
    *Llama*) echo "llama3_json" ;;
    *Mistral*|*Mixtral*) echo "mistral" ;;
    *Granite*) echo "granite" ;;
    *InternLM*) echo "internlm" ;;
    *Jamba*) echo "jamba" ;;
    *GptOss*|*GPTOss*) echo "openai" ;;
    *Glm4*) echo "glm45" ;;
    *) echo "" ;;
  esac
}

# Start the vLLM container. Called by the generic cmd_start in dgxt
# after it has resolved model/max_len/port/gpu_mem/api_key/tool_call_parser.
engine_run_container() {
  local model="$1" max_len="$2" port="$3" gpu_mem="$4" api_key="$5" tool_call_parser="${6:-}"
  local tool_args=()
  if [[ -n "$tool_call_parser" ]]; then
    tool_args=(--enable-auto-tool-choice --tool-call-parser "$tool_call_parser")
  fi
  docker run -d \
    --name "$ENGINE_CONTAINER_NAME" \
    --gpus all \
    --ipc host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --entrypoint "" \
    -p "${port}:8000" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e VLLM_API_KEY="$api_key" \
    -v "${HUB_CACHE}:/root/.cache/huggingface/hub" \
    "$ENGINE_IMAGE" \
    vllm serve "$model" \
    --max-model-len "$max_len" \
    --gpu-memory-utilization "$gpu_mem" \
    --api-key "$api_key" \
    "${tool_args[@]}"
}
