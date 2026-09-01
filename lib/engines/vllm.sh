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
# MoE backend for NVFP4 models on Blackwell (SM120). vLLM's own "auto"
# oracle (vllm.config.kernel.KernelConfig.moe_backend, default) already
# picks the best backend per MoE layer -- including falling back to
# Marlin for weight-only (W4A16) NVFP4 layers that CUTLASS/CuteDSL can't
# handle, which some "mixed" NVFP4 checkpoints (nvidia/modelopt_mixed
# quant_algo, e.g. combining NVFP4/W4A16_NVFP4/MXFP8 across layers) do
# contain. Forcing a single backend for the whole model skips that
# per-layer fallback and can crash on load ("does not support the
# deployment configuration") for such checkpoints, so we leave this
# unset (auto) by default. Only set VLLM_MOE_BACKEND if you've verified
# your specific checkpoint is pure NVFP4 (W4A4) end to end, in which case
# "cutlass" forces the native FP4 tensor cores instead of auto's pick.
# NOTE: there is no vLLM env var of this name -- engine_run_container
# translates it into the --moe-backend CLI flag (see there for why).
ENGINE_MOE_BACKEND_VAR="VLLM_MOE_BACKEND"

# Recommended models for a DGX Spark-class box (128GB unified memory).
# Edit this list for your own hardware/preferences — nothing else depends
# on these specific values. Format: "id|approx size|note"
ENGINE_DEFAULT_MODEL="nvidia/Qwen3.6-35B-A3B-NVFP4"
ENGINE_RECOMMENDED_MODELS=(
  "nvidia/Qwen3.6-35B-A3B-NVFP4|~18GB|best perf, DGX-optimized quantization (default)"
  "openai/gpt-oss-120b|~65GB|stronger quality, native MXFP4 MoE, still fast"
  "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4|~40GB|larger MoE (80B/3B active); benchmarks below default on GPQA/agentic tasks despite the size -- try before trusting the param count"
  "unsloth/Qwen3.8-27B-NVFP4|~16GB|dense hybrid-attention VLM, needs FlashInfer + atomic-add workaround (handled automatically)"
  "Qwen/Qwen3.6-35B-A3B|~70GB|full precision"
  "Qwen/Qwen3-32B|~64GB|full precision, dense"
  "Qwen/Qwen3-8B|~16GB|fast, smaller"
)

# vLLM's OpenAI-compatible server rejects any request with tool/function
# definitions (as every agentic coding CLI sends) unless tool calling is
# explicitly enabled with a parser matched to the model's tool-call output
# format. Most models recommended above are Qwen3-family, which emit
# <tool_call>...</tool_call> XML — hence "qwen3_xml" as the last-resort
# fallback below. gpt-oss-120b is the one exception (matched by its own
# GptOss case in resolve_tool_call_parser/resolve_reasoning_parser, so it
# never hits this fallback). Override VLLM_TOOL_CALL_PARSER (or set it to
# empty) if serving some other model family that also falls through; see:
# https://docs.vllm.ai/en/latest/features/tool_calling.html
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
    # Dense qwen3_5 (Qwen3_5ForConditionalGeneration, e.g. Qwen3.8-27B) is
    # matched before the generic *Qwen3* catch-all below: it emits
    # OpenAI-style tool calls, not the <tool_call> XML the MoE Qwen3
    # family (Qwen3_5MoeForConditionalGeneration etc.) uses -- confirmed
    # against the model's own published vLLM recipe. Don't broaden this
    # pattern to also match the Moe variant; that path is qwen3_xml and
    # already verified working.
    *Qwen3_5ForConditionalGeneration*) echo "qwen3_coder" ;;
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
  # directly for manual control.
  #
  # NOTE: the standalone `--rope-scaling` CLI flag was removed in vLLM
  # v0.11.1+ (the version `vllm/vllm-openai:latest` now tracks) -- passing
  # it makes `vllm serve` reject the whole command with "unrecognized
  # arguments", the same failure mode as the old --enable-reasoning flag.
  # RoPE/YaRN scaling must now be injected via --hf-overrides, and current
  # transformers configs key this as "rope_parameters" (older configs used
  # "rope_scaling" -- vLLM's HF-overrides merge accepts either name, but
  # "rope_parameters" is what newer model configs actually read). See:
  # https://docs.vllm.ai/en/latest/features/context_extension/
  local rope_args=()
  local rope_factor="${VLLM_ROPE_SCALING_FACTOR:-${ROPE_SCALING_FACTOR:-}}"
  if [[ -n "$rope_factor" ]]; then
    local rope_original="${VLLM_ROPE_SCALING_ORIGINAL_MAX:-${ROPE_SCALING_ORIGINAL_MAX:-}}"
    rope_args=(--hf-overrides "{\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":${rope_factor},\"original_max_position_embeddings\":${rope_original}}}")
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

  # NVFP4 MoE backend override (see ENGINE_MOE_BACKEND_VAR above for why
  # this defaults to unset/auto rather than forcing "cutlass"). There is
  # no VLLM_MOE_BACKEND *env var* -- vLLM's kernel selection
  # (vllm.config.kernel.KernelConfig.moe_backend) is CLI-only, set via
  # --moe-backend; an env var of the same name is silently ignored (and
  # logged as "Unknown vLLM environment variable"). We keep
  # VLLM_MOE_BACKEND as dgxt's own config var name for continuity, but
  # translate it into --moe-backend here, only when the user has set it.
  local moe_backend="${VLLM_MOE_BACKEND:-}"
  local moe_backend_args=()
  [[ -n "$moe_backend" ]] && moe_backend_args=(--moe-backend "$moe_backend")

  # gpt-oss (native MXFP4, attention sinks) needs two Blackwell
  # (SM120/SM121) correctness workarounds that no other recommended model
  # does. Auto-selected backends are actively broken here, not just slow:
  #   - CUTLASS/FlashInfer MXFP4 kernels corrupt output on this hardware
  #     (garbled tokens / null content on the first response). Marlin is
  #     the only MXFP4 backend confirmed correct here. This IS a real env
  #     var (unlike VLLM_MOE_BACKEND above) -- vLLM reads it directly.
  #   - FlashInfer attention doesn't implement gpt-oss's attention-sinks
  #     mechanism at all: hard startup crash ("attention sinks not
  #     supported"), not a silent fallback. TRITON_ATTN is the only
  #     working attention backend for this model on this hardware.
  # See: https://github.com/vllm-project/vllm/issues/37030 and
  # https://conselara.dev/notes/gpt-oss-120b-single-dgx-spark/
  local gptoss_env=() gptoss_args=() gptoss_vol=()
  case "$model" in
    *gpt-oss*|*GptOss*|*GPTOss*)
      gptoss_env=(-e "VLLM_MXFP4_BACKEND=marlin")
      gptoss_args=(--attention-backend TRITON_ATTN)
      # Harmony tokenizer vocab fetch (see ensure_tiktoken_cache in
      # lib/common.sh for the full story) -- pre-fetch it to a persistent
      # host dir and point the container at it, so this never depends on
      # in-container network at request time. Best-effort: if priming
      # fails (offline right now, no curl, etc.) we still set the env var
      # and mount the (possibly empty) dir -- vLLM just falls back to its
      # own live fetch into that same mounted dir instead of /tmp, which
      # is at least no worse than today, and self-heals on next start.
      ensure_tiktoken_cache || true
      gptoss_env+=(-e "TIKTOKEN_RS_CACHE_DIR=/root/tiktoken_cache")
      gptoss_vol=(-v "${TIKTOKEN_CACHE_DIR}:/root/tiktoken_cache")
      ;;
  esac

  # Qwen3.8-27B (dense, hybrid Gated DeltaNet + Gated Attention, qwen3_5
  # arch) needs two workarounds on this hardware that no other
  # recommended model does, per the day-zero DGX Spark recipe this was
  # verified against:
  #   - FlashInfer is the attention backend that actually gets picked
  #     (sm121 xqa decode kernel) and supports the FP8 KV cache this
  #     model's recipe uses; other recommended models don't need it
  #     forced since their defaults already land elsewhere.
  #   - VLLM_MARLIN_USE_ATOMIC_ADD=1 is a hardware-specific Marlin kernel
  #     workaround copied from the working recipe rather than derived --
  #     without it this model's quantized layers can hit incorrect
  #     accumulation on this GPU.
  # See: https://blog.kubesimplify.com/qwen3-8-27b-on-dgx-spark
  local qwen38_env=() qwen38_args=()
  case "$model" in
    *Qwen3.8*|*qwen3.8*)
      qwen38_env=(-e "VLLM_MARLIN_USE_ATOMIC_ADD=1")
      qwen38_args=(--attention-backend FLASHINFER)
      ;;
  esac

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
    "${gptoss_env[@]}" \
    "${qwen38_env[@]}" \
    -v "${HUB_CACHE}:/root/.cache/huggingface/hub" \
    "${gptoss_vol[@]}" \
    "$ENGINE_IMAGE" \
    vllm serve "$model" \
    "${max_len_args[@]}" \
    "${moe_backend_args[@]}" \
    "${gptoss_args[@]}" \
    "${qwen38_args[@]}" \
    --gpu-memory-utilization "$gpu_mem" \
    --api-key "$api_key" \
    "${tool_args[@]}" \
    "${rope_args[@]}" \
    "${reasoning_args[@]}"
}
