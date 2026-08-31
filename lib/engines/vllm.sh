# vLLM engine module for dgx-tools. Fully implemented — the reference
# pattern for a "one image + --model flag, HF-hosted" engine (see
# lib/engines/*.sh placeholders). For the opposite shape — one image per
# model, external registry auth — see lib/engines/nim.sh instead.
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
# vLLM can genuinely extend a model past its native context via YaRN RoPE
# scaling (unlike NIM's precompiled engines, which have no such knob) --
# see engine_run_container's rope_args and cmd_start's native-max-context
# check in dgxt for how this gets triggered.
ENGINE_SUPPORTS_ROPE_SCALING="1"

# Config file env var names this engine reads/writes (kept as the original
# VLLM_* names for continuity with existing ~/.vllmrc-based configs).
ENGINE_MODEL_VAR="VLLM_MODEL"
ENGINE_MAX_LEN_VAR="VLLM_MAX_MODEL_LEN"
ENGINE_API_KEY_VAR="VLLM_API_KEY"
ENGINE_PORT_VAR="VLLM_PORT"
ENGINE_GPU_MEM_VAR="VLLM_GPU_MEM"
ENGINE_TOOL_CALL_PARSER_VAR="VLLM_TOOL_CALL_PARSER"
ENGINE_REASONING_PARSER_VAR="VLLM_REASONING_PARSER"

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

# Same rationale as above but for --reasoning-parser: all models
# recommended above are Qwen3-family, which use <think></think> reasoning
# delimiters — hence "qwen3" as the last-resort fallback (see
# resolve_reasoning_parser below). Override VLLM_REASONING_PARSER (or set
# it to empty to disable reasoning output) if serving a different model
# family; see: https://docs.vllm.ai/en/latest/features/reasoning_outputs.html
ENGINE_DEFAULT_REASONING_PARSER="qwen3"

# Shared helper: fetches a model's config.json from HuggingFace and prints
# its first "architectures" entry (empty on any failure — no network tools,
# fetch failure, gated/private repo, or unparseable JSON). Used by both
# resolve_tool_call_parser and resolve_reasoning_parser below so a
# `dgxt start` only pays for one HF lookup instead of two.
resolve_model_architecture() {
  local model="$1"
  if ! command -v curl >/dev/null 2>&1 || ! command -v python3 >/dev/null 2>&1; then
    return
  fi

  local config_json
  config_json=$(curl -sfL "https://huggingface.co/${model}/raw/main/config.json" 2>/dev/null)
  [[ -z "$config_json" ]] && return

  echo "$config_json" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    archs = d.get('architectures') or []
    print(archs[0] if archs else '')
except Exception:
    pass
" 2>/dev/null
}

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
  local arch
  arch=$(resolve_model_architecture "$model")
  if [[ -z "$arch" ]]; then
    echo "$ENGINE_DEFAULT_TOOL_CALL_PARSER"
    return
  fi

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

# Same idea as resolve_tool_call_parser, but for --reasoning-parser: vLLM's
# reasoning/chain-of-thought extraction is likewise model-family-specific
# (Qwen3 uses <think>...</think>, DeepSeek-R1 uses its own delimiters,
# Granite/Mistral/GLM4 etc. each have their own parser), so hardcoding
# "qwen3" would silently break (or vllm serve would flat-out reject it)
# for any non-Qwen3 model in ENGINE_RECOMMENDED_MODELS or a manually
# configured VLLM_MODEL. Falls back to ENGINE_DEFAULT_REASONING_PARSER
# only when detection can't run at all; returns empty (reasoning output
# disabled, --enable-reasoning omitted) for a recognized-but-unmapped
# architecture, same fail-clean rationale as the tool-call parser above.
# See: https://docs.vllm.ai/en/latest/features/reasoning_outputs.html
resolve_reasoning_parser() {
  local model="$1"
  local arch
  arch=$(resolve_model_architecture "$model")
  if [[ -z "$arch" ]]; then
    echo "$ENGINE_DEFAULT_REASONING_PARSER"
    return
  fi

  case "$arch" in
    *Qwen3*) echo "qwen3" ;;
    *DeepseekV3*|*DeepSeekV3*|*DeepseekV2*) echo "deepseek_v3" ;;
    *Deepseek*|*DeepSeek*) echo "deepseek_r1" ;;
    *Granite*) echo "granite" ;;
    *Glm4*) echo "glm45" ;;
    *Mistral*|*Mixtral*) echo "mistral" ;;
    *GptOss*|*GPTOss*) echo "openai_gptoss" ;;
    *) echo "" ;;
  esac
}

# Start the vLLM container. Called by the generic cmd_start in dgxt
# after it has resolved model/max_len/port/gpu_mem/api_key/tool_call_parser/
# reasoning_parser.
engine_run_container() {
  local model="$1" max_len="$2" port="$3" gpu_mem="$4" api_key="$5" tool_call_parser="${6:-}" reasoning_parser="${7:-}"
  local tool_args=()
  if [[ -n "$tool_call_parser" ]]; then
    tool_args=(--enable-auto-tool-choice --tool-call-parser "$tool_call_parser")
  fi
  local reasoning_args=()
  if [[ -n "$reasoning_parser" ]]; then
    # --enable-reasoning was deprecated in vLLM v0.9.0 and removed in
    # v0.10.0+ (which `vllm/vllm-openai:latest` now tracks) -- passing it
    # makes `vllm serve` reject the whole command with "unrecognized
    # arguments". Just --reasoning-parser now implicitly enables
    # reasoning-content extraction.
    reasoning_args=(--reasoning-parser "$reasoning_parser")
  fi
  # Set (as a bare global, not passed positionally) by cmd_start's
  # native-max-context check when the user explicitly confirmed they
  # want to exceed the model's derived native context length.
  local allow_long_env=()
  [[ "${ALLOW_LONG_MAX_MODEL_LEN:-0}" == "1" ]] && allow_long_env=(-e "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1")

  # Real YaRN RoPE-scaling context extension (not just the validator
  # bypass above). Without this, tokens past the model's actual trained
  # position range are simply out-of-distribution -- YaRN interpolates
  # position embeddings so the model can meaningfully attend beyond its
  # native range. Set as bare globals by cmd_start's native-max-context
  # check (auto-computed from the requested/native ratio), or settable
  # directly for manual control. See:
  # https://qwen.readthedocs.io/en/latest/deployment/vllm.html#context-length
  local rope_args=()
  local rope_factor="${VLLM_ROPE_SCALING_FACTOR:-${ROPE_SCALING_FACTOR:-}}"
  if [[ -n "$rope_factor" ]]; then
    local rope_original="${VLLM_ROPE_SCALING_ORIGINAL_MAX:-${ROPE_SCALING_ORIGINAL_MAX:-}}"
    rope_args=(--rope-scaling "{\"rope_type\":\"yarn\",\"factor\":${rope_factor},\"original_max_position_embeddings\":${rope_original}}")
  fi

  # Only pass --max-model-len if dgxt actually resolved a value. If
  # resolution failed for any reason (HF lookup unreachable/timed out,
  # etc.), omit the flag entirely rather than passing an empty string --
  # vllm serve crashes outright on `--max-model-len ''`. Omitting it
  # lets vLLM derive its own default from the model's config.json itself
  # (fetched from inside the container, which has its own working
  # network path independent of whatever failed on the host side).
  local max_len_args=()
  [[ -n "$max_len" ]] && max_len_args=(--max-model-len "$max_len")

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
    "${allow_long_env[@]}" \
    -v "${HUB_CACHE}:/root/.cache/huggingface/hub" \
    "$ENGINE_IMAGE" \
    vllm serve "$model" \
    "${max_len_args[@]}" \
    --gpu-memory-utilization "$gpu_mem" \
    --api-key "$api_key" \
    "${tool_args[@]}" \
    "${rope_args[@]}" \
    "${reasoning_args[@]}"
}
