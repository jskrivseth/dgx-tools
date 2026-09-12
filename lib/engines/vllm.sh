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
# The stable v0.28.0 image still misroutes RadixArk's Qwen3 DSpark draft.
# Nightly contains the upstream architecture normalization and publishes an
# ARM64 variant for DGX Spark.
ENGINE_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:nightly}"
ENGINE_HF_APPS_FILTER="vllm"
ENGINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

normalize_model_alias() {
  case "$1" in
    qwen38-flash-next|qwen3.8-flash-next|Qwen3.8-Flash-Next|RadixArk/Qwen3.8-Flash-Next-NVFP4|qwen38-flash-next-v029)
      echo "qwen38-flash-next-v029" ;;
    *)
      echo "$1" ;;
  esac
}

is_flash_next_model() {
  case "$1" in
    qwen38-flash-next-v029) return 0 ;;
    *) return 1 ;;
  esac
}

flash_next_checkpoint_model() {
  echo "RadixArk/Qwen3.8-Flash-Next-NVFP4"
}

# Per-model image resolution. Most models use the default vLLM nightly image,
# but Flash-Next requires the repo-owned image with PLE mmap offload patches
# (the 51 GB table must be mmapped from NVMe to fit in 128 GB). Override with
# VLLM_FLASH_NEXT_V029_IMAGE if you built or tagged a different image.
resolve_engine_image() {
  local model="$1"
  case "$model" in
    qwen38-flash-next-v029)
      echo "${VLLM_FLASH_NEXT_V029_IMAGE:-dgxt/qwen38-flash-next-v029:latest}" ;;
    *)
      echo "$ENGINE_IMAGE" ;;
  esac
}

ensure_flash_next_v029_image() {
  local image="$1"
  if docker image inspect "$image" >/dev/null 2>&1; then
    return 0
  fi
  if [[ "${VLLM_FLASH_NEXT_AUTO_BUILD:-1}" == "0" ]]; then
    echo "ERROR: Flash-Next v0.29 image '$image' is not available locally." >&2
    echo "  Build it with: $ENGINE_ROOT/containers/qwen3.8-flash-next/scripts/build-v029-image.sh" >&2
    return 1
  fi
  IMAGE="$image" \
    "$ENGINE_ROOT/containers/qwen3.8-flash-next/scripts/build-v029-image.sh"
}

# vLLM can genuinely extend a model past its native context via YaRN RoPE
# scaling (unlike NIM's precompiled engines, which have no such knob) --
# see engine_run_container's rope_args and cmd_start's native-max-context
# check in dgxt for how this gets triggered.
ENGINE_SUPPORTS_ROPE_SCALING="1"

# Qwen3.6-Fast's last verified DGX Spark profile is native 256K. The
# current nightly can reach an illegal GDN memory access during warmup when
# this checkpoint is launched at 512K with the generic YaRN extension.
resolve_default_context_override() {
  case "$1" in
    unsloth/Qwen3.6-35B-A3B-NVFP4-Fast) echo "262144" ;;
    *) return 1 ;;
  esac
}

# Prevent a stale VLLM_MAX_MODEL_LEN from re-enabling the known-bad extended
# context for this exact checkpoint. Other models retain their configured
# context and RoPE behavior.
normalize_context_for_engine() {
  local model="$1"
  local max_len="$2"
  if [[ "$model" == "unsloth/Qwen3.6-35B-A3B-NVFP4-Fast" &&
        "$max_len" =~ ^[0-9]+$ ]] && (( max_len > 262144 )); then
    echo "WARNING: capping Qwen3.6-Fast at its verified DGX Spark context of 262144 tokens." >&2
    echo "  The current vLLM nightly can fail during GDN warmup above native context." >&2
    echo "262144"
  elif is_flash_next_model "$model" &&
       [[ "$max_len" =~ ^[0-9]+$ ]] && (( max_len > 262144 )); then
    # Qwen's published Flash-Next extension uses fixed 4x YaRN from the
    # native 262,144-token window, including the 500K prefix-cache profile.
    ALLOW_LONG_MAX_MODEL_LEN=1
    ROPE_SCALING_FACTOR=4
    ROPE_SCALING_ORIGINAL_MAX=262144
    echo "$max_len"
  else
    echo "$max_len"
  fi
}

# NVIDIA's Spark recipe starts the multimodal Omni checkpoint at 131072
# tokens even though its published native ceiling is 256K. The lower
# allocation leaves unified memory for media processing and KV cache.
resolve_default_context_override() {
  case "$1" in
    *Nemotron-3-Nano-Omni*|*nemotron-3-nano-omni*) echo "131072" ;;
    qwen38-flash-next-v029) echo "500000" ;;
    *) return 1 ;;
  esac
}

# Config file env var names this engine reads/writes (kept as the original
# VLLM_* names for continuity with existing ~/.vllmrc-based configs).
ENGINE_MODEL_VAR="VLLM_MODEL"
ENGINE_MAX_LEN_VAR="VLLM_MAX_MODEL_LEN"
ENGINE_API_KEY_VAR="VLLM_API_KEY"
ENGINE_PORT_VAR="VLLM_PORT"
ENGINE_GPU_MEM_VAR="VLLM_GPU_MEM"
ENGINE_SERVED_MODEL_NAME_VAR="VLLM_SERVED_MODEL_NAME"
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

# Repetition-penalty default for the OpenAI-compatible server. This exists
# because of a real, observed failure mode: Qwen3-family models (all
# recommended models above except gpt-oss-120b) can fall into repetition
# loops -- repeating the same sentence, or the same no-op tool call, over
# and over -- especially in long agentic/tool-use sessions. Qwen's own
# model card recommends presence_penalty=1.5 to fix this, but
# presence_penalty can ONLY be set per-request by the calling client
# (vLLM has no server-side default/fallback for it -- see
# ModelConfig.get_diff_sampling_param, which only recognizes
# repetition_penalty/temperature/top_k/top_p/min_p/max_new_tokens as
# overridable server-side defaults). Most agentic CLI/IDE harnesses don't
# send presence_penalty at all, so the model never gets the correction
# it needs regardless of what the server operator wants. repetition_penalty
# IS one of those server-side-overridable params, so we default it to a
# conservative 1.1 here (within Qwen's own suggested 1.05-1.15 troubleshooting
# range) applied transparently to every request via --override-generation-config,
# regardless of whether the client sets anything. Set to empty to disable
# (e.g. if you find it makes output blander for your workload) or override
# to tune the strength.
ENGINE_REPETITION_PENALTY_VAR="VLLM_REPETITION_PENALTY"

# Nemotron's official DGX Spark recipe uses 0.85. NVIDIA's Qwen3.6 Spark
# recipe's 0.4 setting is too low for the current nightly's CUDA-graph
# footprint on DGX Spark: it cannot reserve enough KV cache for one 256K
# request. Use the validated 0.6 Spark setting at native context and
# increase that automatically for longer contexts. Keep an explicit
# VLLM_GPU_MEM override authoritative.
resolve_gpu_memory() {
  local model="$1"
  local max_len="${2:-}"
  if [[ -n "${VLLM_GPU_MEM:-}" ]]; then
    echo "$VLLM_GPU_MEM"
    return
  fi
  case "$model" in
    *Nemotron-3.5-Lightning*|*nemotron-3.5-lightning*) echo "0.85" ;;
    nvidia/Qwen3.6-35B-A3B-NVFP4)
      if [[ "$max_len" =~ ^[0-9]+$ ]] && (( max_len > 524288 )); then
        echo "0.7"
      elif [[ "$max_len" =~ ^[0-9]+$ ]] && (( max_len > 262144 )); then
        echo "0.6"
      else
        # The current nightly needs 0.6 to leave enough KV cache after
        # CUDA-graph capture; 0.4 fails before serving a 256K request.
        echo "0.6"
      fi
      ;;
    unsloth/Qwen3.6-35B-A3B-NVFP4-Fast)
      if [[ "$max_len" =~ ^[0-9]+$ ]] && (( max_len > 262144 )); then
        # No Fast-specific Spark fraction is published; reserve more
        # unified memory for the KV cache when leaving the native context.
        echo "0.7"
      else
        echo "0.8"
      fi
      ;;
    ornith-ai/Ornith-1.5-35B-A3B-NVFP4)
      if [[ "$max_len" =~ ^[0-9]+$ ]] && (( max_len > 262144 )); then
        echo "0.7"
      else
        # Community GB10 measurements report stable 256K serving around
        # 0.85 for the official mixed ModelOpt checkpoint. Leave headroom
        # for the optional MTP predictor when extending beyond native
        # context.
        echo "0.85"
      fi
      ;;
    *Qwen3-Coder-Next*|*qwen3-coder-next*|*Qwen3-Next*|*qwen3-next*)
      # The NVFP4-GB10 quant is ~46GB; 0.85 leaves generous KV
      # headroom at native 256K (cheap 4-KV-head cache). Drop the fraction
      # when stretching past native context so the larger KV pool still
      # reserves.
      if [[ "$max_len" =~ ^[0-9]+$ ]] && (( max_len > 262144 )); then
        echo "0.7"
      else
        echo "0.85"
      fi
      ;;
    qwen38-flash-next-v029)
      # The released v0.29 recipe reserves a conservative 20% of the
      # unified pool for the host, PLE page faults, and long-context growth.
      echo "0.80"
      ;;
    *) echo "0.8" ;;
  esac
}

# Recommended models for a DGX Spark-class box (128GB unified memory).
# Edit this list for your own hardware/preferences — nothing else depends
# on these specific values. Format: "id|approx size|note"
ENGINE_DEFAULT_MODEL="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
ENGINE_RECOMMENDED_MODELS=(
  "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4|~22GB|recommended DGX Spark default -- 1M context and DSpark speculative decoding"
  "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4|~21GB|multimodal reasoning, tool use, and long-context chat; DGX Spark NVFP4 recipe"
  "nvidia/Qwen3.6-35B-A3B-NVFP4|~18GB|NVIDIA's recommended agent-ready model for tool calling and reasoning"
  "nvidia/Llama-3.3-70B-Instruct-FP4|~50GB|general-purpose instruction following and coding; official optimized path is TensorRT-LLM"
  "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16|~62GB|Nemotron 3 Omni compatibility fallback; use NVFP4 on DGX Spark"
  "unsloth/Qwen3.6-35B-A3B-NVFP4-Fast|~22GB|best perf on this hardware -- verified ~71.5 tok/s single-stream via FlashInfer B12X MoE backend, same accuracy as the plain NVFP4 checkpoint"
  "ornith-ai/Ornith-1.5-35B-A3B-NVFP4|~22GB|coding and agentic reasoning specialist; DGX Spark NVFP4 profile with native 256K context"
  "openai/gpt-oss-120b|~65GB|stronger quality, native MXFP4 MoE, still fast"
  "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4|~40GB|larger MoE (80B/3B active); benchmarks below default on GPQA/agentic tasks despite the size -- try before trusting the param count"
  "ucbye/Qwen3-Coder-Next-NVFP4-GB10|~46GB|pinned, ungated 80B/3B hybrid Gated-DeltaNet coder; FlashInfer+Marlin NVFP4 recipe, native 256K context"
  "RadixArk/Qwen3.8-27B-NVFP4|~16GB|dense hybrid-attention VLM, native MTP or matching DSpark draft; GB10 workarounds handled automatically"
  "unsloth/Qwen3.8-27B-NVFP4|~16GB|same dense VLM but Unsloth Dynamic V3.0 NVFP4 (compressed-tensors, auto-detect) -- MTP speculation, no DSpark draft; measured ~20 tok/s single-stream with MTP"
  "qwen38-flash-next-v029|~135GB|RECOMMENDED: vLLM 0.29, fixed prefix cache, deterministic GB10 top-k, 500K YaRN"
  "Qwen/Qwen3.6-35B-A3B|~70GB|full precision"
  "Qwen/Qwen3-32B|~64GB|full precision, dense"
  "Qwen/Qwen3-8B|~16GB|fast, smaller"
)

# vLLM's OpenAI-compatible server rejects any request with tool/function
# definitions (as every agentic coding CLI sends) unless tool calling is
# explicitly enabled with a parser matched to the model's tool-call output
# format. Most models recommended above are Qwen3-family, which emit
# <tool_call>...</tool_call> XML — hence "qwen3_xml" as the last-resort
# fallback below. Nemotron 3.5 Lightning uses the Qwen-compatible
# qwen3_coder parser, while gpt-oss-120b is matched by its own GptOss case
# in resolve_tool_call_parser/resolve_reasoning_parser. Override
# VLLM_TOOL_CALL_PARSER (or set it to empty) if serving some other model
# family that also falls through; see:
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

  # NVIDIA's official Nemotron 3.5 Lightning vLLM recipe uses the
  # Qwen-compatible parser. Match the model ID before the HF lookup so a
  # temporary metadata/network failure cannot silently disable or misparse
  # tool calls for this known model.
  case "$model" in
    *Nemotron-3.5-Lightning*|*nemotron-3.5-lightning*|*Nemotron-3-Nano-Omni*|*nemotron-3-nano-omni*)
      echo "qwen3_coder"
      return
      ;;
    qwen38-flash-next-v029)
      echo "qwen3_coder"
      return
      ;;
    *Llama-3.3-70B*|*llama-3.3-70b*) echo "llama3_json"; return ;;
  esac

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
    *Qwen3Next*|*Qwen3-Next*) echo "qwen3_coder" ;;
    *Qwen3Coder*|*Qwen3_5Moe*|*Qwen3Moe*|*Qwen3*) echo "qwen3_xml" ;;
    *Qwen2*) echo "hermes" ;;
    # Qwen3-Coder-Next / Qwen3-Next (Qwen3NextForCausalLM) emit the
    # Qwen3-coder tool-call format; match before the fallthrough so tool
    # calling stays enabled for these. Their architecture hits none of the
    # cases below, so without this the Qwen3-Coder-Next checkpoint and
    # nvidia's official Qwen3-Next checkpoints would silently ship with
    # tool calling disabled.
    *DeepseekV4*|*DeepSeekV4*) echo "qwen3_coder" ;;
    *Llama4*) echo "llama4_pythonic" ;;
    *Llama*) echo "llama3_json" ;;
    *Mistral*|*Mixtral*) echo "mistral" ;;
    *Granite*) echo "granite" ;;
    *InternLM*) echo "internlm" ;;
    *Jamba*) echo "jamba" ;;
    *GptOss*|*GPTOss*) echo "openai" ;;
    *NemotronH*) echo "qwen3_coder" ;;
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

  # Nemotron 3.5 Lightning emits reasoning in the Nemotron v3 format.
  # Match its model ID before the HF lookup for the same offline-safe
  # behavior as resolve_tool_call_parser().
  case "$model" in
    *Nemotron-3.5-Lightning*|*nemotron-3.5-lightning*|*Nemotron-3-Nano-Omni*|*nemotron-3-nano-omni*)
      echo "nemotron_v3"
      return
      ;;
  esac

  # Qwen3-Next's published chat template has no <think> delimiters and this
  # checkpoint emits ordinary answers directly. Applying the generic qwen3
  # parser would therefore classify the whole answer as reasoning content.
  # Qwen3.8-Flash-Next uses the standard qwen3 reasoning parser. Must match
  # before the *Qwen3*Next* catch-all below, which would otherwise return
  # empty and disable reasoning output for Flash-Next.
  case "$model" in
    qwen38-flash-next-v029) echo "qwen3"; return ;;
  esac

  case "$model" in
    *Qwen3*Next*|*qwen3*next*) echo ""; return ;;
  esac

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
    *NemotronH*) echo "nemotron_v3" ;;
    *) echo "" ;;
  esac
}

is_qwen38_model() {
  case "$1" in
    *Qwen3.8*|*qwen3.8*|qwen38-flash-next-v029) return 0 ;;
    *) return 1 ;;
  esac
}

configure_qwen38_profile() {
  local model="$1"
  local max_len="${2:-}"
  local default_speculative_mode="mtp"
  local default_speculative_tokens=5
  local max_num_seqs=4
  local batch_tokens=8192
  local speculative_mode
  local speculative_tokens="${VLLM_SPECULATIVE_TOKENS:-${VLLM_MTP_TOKENS:-}}"

  QWEN38_ENV_ARGS=(-e "VLLM_MARLIN_USE_ATOMIC_ADD=1")
  # Unsloth's checkpoint is compressed-tensors/NVFP4 (Dynamic V3.0) and
  # auto-detects its quantization, so we intentionally omit --quantization
  # here (unlike the NVIDIA modelopt path). The FlashInfer + bounded-batching
  # profile below matches the same qwen3_5 dense hybrid-attention architecture
  # shared with RadixArk's checkpoint.
  QWEN38_ARGS=(
    --attention-backend FLASHINFER
    --enable-chunked-prefill
    --distributed-executor-backend mp
  )
  QWEN38_SPECULATIVE_ARGS=()
  QWEN38_SERVE_COMMAND=(vllm serve "$model")

  case "$model" in
    RadixArk/Qwen3.8-27B-NVFP4|radixark/Qwen3.8-27B-NVFP4)
      # RadixArk ships a matching DSpark sparse draft, so speculative
      # decoding defaults to the dspark method with that draft model.
      default_speculative_mode="dspark"
      ;;
    unsloth/Qwen3.8-27B-NVFP4|unsloth/qwen3.8-27b-nvfp4)
      # Unsloth's NVFP4 has no third-party DSpark draft; its trained
      # predictor is MTP, so speculation stays on the model's own MTP head.
      default_speculative_mode="mtp"
      ;;
    qwen38-flash-next-v029)
      # The PLE table (~48 GB) is mmapped from NVMe; v0.29 adds the
      # prefix-cache, GB10 kernel, and deterministic top-k fixes.
      default_speculative_mode="mtp"
      default_speculative_tokens=2
      max_num_seqs=8
      batch_tokens=8192
      QWEN38_ENV_ARGS+=(
        -e "VLLM_PLE_MMAP=1"
        -e "VLLM_PLE_MMAP_WORKERS=32"
        -e "VLLM_PLE_MMAP_TRIM_AVAILABLE_MIB=${VLLM_FLASH_NEXT_PLE_TRIM_MIB:-8192}"
        -e "VLLM_PLE_MMAP_TRIM_MIN_ROWS=${VLLM_FLASH_NEXT_PLE_TRIM_MIN_ROWS:-1024}"
      )
      QWEN38_ARGS+=(--load-format safetensors --no-enable-flashinfer-autotune)
      QWEN38_ARGS+=(
        -cc.cudagraph_mode=PIECEWISE
        '-cc.splitting_ops=["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen4_exp_compute_ple_ngram_ids","vllm::qwen4_exp_ple_short_conv","vllm::qwen4_exp_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup_ids"]'
      )
      QWEN38_ENV_ARGS+=(
        -e "VLLM_QSA_DET_TOPK=1"
        -e "VLLM_QSA_DET_LIB=/opt/llm/kernel-det/_C_det.so"
        -e "VLLM_MTP_DRAFT_VOCAB=/opt/llm/draft_vocab_65536.npy"
        -e "VLLM_PLE_MMAP_MADVISE=random"
      )
      ;;
  esac

  QWEN38_ARGS+=(--max-num-batched-tokens "$batch_tokens")
  if is_flash_next_model "$model"; then
    if [[ "${VLLM_FLASH_NEXT_PREFIX_CACHING:-1}" != "0" ]]; then
      QWEN38_ARGS+=(--enable-prefix-caching)
    else
      QWEN38_ARGS+=(--no-enable-prefix-caching)
    fi
  else
    QWEN38_ARGS+=(--enable-prefix-caching)
  fi
  QWEN38_ARGS+=(--max-num-seqs "$max_num_seqs")
  speculative_mode="${VLLM_SPECULATIVE_MODE:-$default_speculative_mode}"

  case "${speculative_mode,,}" in
    none)
      ;;
    mtp)
      speculative_tokens="${speculative_tokens:-$default_speculative_tokens}"
      if [[ "$speculative_tokens" == "0" ]]; then
        :
      elif [[ "$speculative_tokens" =~ ^[1-9][0-9]*$ ]]; then
        local speculative_config
        speculative_config="{\"method\":\"mtp\",\"num_speculative_tokens\":${speculative_tokens}}"
        if is_flash_next_model "$model" &&
           [[ "$max_len" =~ ^[0-9]+$ ]]; then
          # The YaRN override is not propagated to the MTP draft model.
          speculative_config="{\"method\":\"mtp\",\"num_speculative_tokens\":${speculative_tokens},\"max_model_len\":${max_len}}"
        fi
        QWEN38_SPECULATIVE_ARGS=(
          --speculative-config
          "$speculative_config"
        )
      else
        echo "ERROR: VLLM_SPECULATIVE_TOKENS must be 0 or a positive integer." >&2
        return 1
      fi
      ;;
    dspark)
      speculative_tokens="${speculative_tokens:-7}"
      if [[ ! "$speculative_tokens" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: VLLM_SPECULATIVE_TOKENS must be a positive integer." >&2
        return 1
      fi
      local dspark_model="${VLLM_SPECULATIVE_MODEL:-RadixArk/Qwen3.8-27B-DSpark}"
      QWEN38_SPECULATIVE_ARGS=(
        --speculative-config
        "{\"method\":\"dspark\",\"model\":\"${dspark_model}\",\"num_speculative_tokens\":${speculative_tokens}}"
      )
      ;;
    dflash)
      # z-lab/incoai's DFlash2 block-diffusion draft, trained against
      # Qwen/Qwen3.8-27B (quant-agnostic). Nightly vLLM implements the
      # "dflash" method; z-lab's H200 evals put it ahead of both the
      # built-in MTP and RadixArk's DSpark on acceptance length.
      speculative_tokens="${speculative_tokens:-7}"
      if [[ ! "$speculative_tokens" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: VLLM_SPECULATIVE_TOKENS must be a positive integer." >&2
        return 1
      fi
      local dflash_model="${VLLM_SPECULATIVE_MODEL:-incoai/Qwen3.8-27B-DFlash2}"
      QWEN38_SPECULATIVE_ARGS=(
        --speculative-config
        "{\"method\":\"dflash\",\"model\":\"${dflash_model}\",\"num_speculative_tokens\":${speculative_tokens}}"
      )
      ;;
    *)
      echo "ERROR: VLLM_SPECULATIVE_MODE must be mtp, dspark, dflash, or none for Qwen3.8." >&2
      return 1
      ;;
  esac
}

# Start the vLLM container. Called by the generic cmd_start in dgxt
# after it has resolved model/max_len/port/gpu_mem/api_key/tool_call_parser/
# reasoning_parser.
engine_run_container() {
  local model="$1" max_len="$2" port="$3" gpu_mem="$4" api_key="$5" tool_call_parser="${6:-}" reasoning_parser="${7:-}"
  local served_model_name_args=()
  if is_flash_next_model "$model"; then
    local flash_aliases="${VLLM_FLASH_NEXT_MODEL_ALIASES:-qwen3.8-flash-next,gpt-5.4-nano}"
    local -a flash_alias_args=()
    local alias
    IFS=',' read -r -a flash_alias_args <<< "$flash_aliases"
    served_model_name_args=(--served-model-name)
    for alias in "${flash_alias_args[@]}"; do
      [[ -n "$alias" ]] && served_model_name_args+=("$alias")
    done
    served_model_name_args+=(qwen3.8-flash-next-v029)
  elif [[ -n "${VLLM_SERVED_MODEL_NAME:-}" ]]; then
    if [[ "${VLLM_SERVED_MODEL_NAME}" =~ [[:space:]] ]]; then
      echo "ERROR: VLLM_SERVED_MODEL_NAME must be a single model alias without whitespace." >&2
      return 1
    fi
    served_model_name_args=(--served-model-name "$VLLM_SERVED_MODEL_NAME" "$model")
  fi
  local tool_args=()
  if [[ -n "$tool_call_parser" ]]; then
    tool_args=(--enable-auto-tool-choice --tool-call-parser "$tool_call_parser")
  fi
  local reasoning_args=()
  if [[ -n "$reasoning_parser" ]]; then
    # --enable-reasoning was deprecated in vLLM v0.9.0 and removed in
    # v0.10.0+ (which `vllm/vllm-openai:nightly` now tracks) -- passing it
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
  # v0.11.1+ (the version `vllm/vllm-openai:nightly` now tracks) -- passing
  # it makes `vllm serve` reject the whole command with "unrecognized
  # arguments", the same failure mode as the old --enable-reasoning flag.
  # RoPE/YaRN scaling must now be injected via --hf-overrides. Most models
  # read rope_parameters at the top level, but Qwen3.6 and Qwen3.8 store it
  # under text_config and require the complete multimodal RoPE object because
  # hf-overrides replaces nested dictionaries rather than merging them. See:
  # https://docs.vllm.ai/en/latest/features/context_extension/
  local rope_args=()
  local rope_factor="${VLLM_ROPE_SCALING_FACTOR:-${ROPE_SCALING_FACTOR:-}}"
  if [[ -n "$rope_factor" ]]; then
    local rope_original="${VLLM_ROPE_SCALING_ORIGINAL_MAX:-${ROPE_SCALING_ORIGINAL_MAX:-}}"
    case "$model" in
      *Qwen3.6*|*qwen3.6*|*Qwen3.8*|*qwen3.8*)
        rope_args=(--hf-overrides "{\"text_config\":{\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":${rope_factor},\"original_max_position_embeddings\":${rope_original},\"mrope_interleaved\":true,\"mrope_section\":[11,11,10],\"partial_rotary_factor\":0.25,\"rope_theta\":10000000}}}")
        ;;
      *)
        rope_args=(--hf-overrides "{\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":${rope_factor},\"original_max_position_embeddings\":${rope_original}}}")
        ;;
    esac
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

  # Repetition-penalty default (see ENGINE_REPETITION_PENALTY_VAR above
  # for the full story on why this exists and why repetition_penalty,
  # not presence_penalty, is the right lever here). Applied via
  # --override-generation-config so it's a transparent server-side
  # default -- still overridden per-request by any client that sends its
  # own repetition_penalty, and skipped entirely if set to empty.
  local repetition_penalty="${VLLM_REPETITION_PENALTY-1.1}"
  local repetition_penalty_args=()
  if [[ -n "$repetition_penalty" ]]; then
    repetition_penalty_args=(--override-generation-config "{\"repetition_penalty\":${repetition_penalty}}")
  fi

  # NVFP4 MoE backend override (see ENGINE_MOE_BACKEND_VAR above for why
  # this defaults to unset/auto rather than forcing "cutlass"). There is
  # no VLLM_MOE_BACKEND *env var* -- vLLM's kernel selection
  # (vllm.config.kernel.KernelConfig.moe_backend) is CLI-only, set via
  # --moe-backend; an env var of the same name is silently ignored (and
  # logged as "Unknown vLLM environment variable"). We keep
  # VLLM_MOE_BACKEND as dgxt's own config var name for continuity, but
  # translate it into --moe-backend here, only when the user has set it.
  # Unsloth's Qwen3.6 "-Fast" NVFP4 checkpoint is a different, purely-NVFP4
  # calibration from NVIDIA's mixed-quant checkpoint. Its recipe calls for
  # flashinfer_b12x (verified live on this box at ~71.5 tok/s single-stream),
  # but do not broaden this to every future "*NVFP4-Fast*" model: the
  # selected backend also affects speculative draft-model initialization.
  local moe_backend="${VLLM_MOE_BACKEND:-}"
  case "$model" in
    unsloth/Qwen3.6-35B-A3B-NVFP4-Fast)
      moe_backend="${VLLM_MOE_BACKEND:-flashinfer_b12x}"
      ;;
  esac
  local moe_backend_args=()
  [[ -n "$moe_backend" ]] && moe_backend_args=(--moe-backend "$moe_backend")

  # NVIDIA's Nemotron 3.5 Lightning GB10 recipe uses FlashInfer for the
  # Mamba path, aligned Mamba caches, FP8 KV cache, prefix caching, and
  # Marlin for MoE. Long-prefill scheduling is bounded to 2048 tokens so up
  # to 16 low-concurrency requests can interleave instead of one running
  # prefill consuming the entire scheduler budget. Keep these settings
  # model-specific: other checkpoints have different backend requirements.
  # An explicit VLLM_MOE_BACKEND still overrides the model-specific Marlin
  # default above.
  local nemotron_args=()
  case "$model" in
    *Nemotron-3.5-Lightning*|*nemotron-3.5-lightning*)
      [[ -n "$moe_backend" ]] || moe_backend="marlin"
      moe_backend_args=(--moe-backend "$moe_backend")
      nemotron_args=(
        --trust-remote-code
        --mamba-backend flashinfer
        --mamba-cache-mode align
        --mamba-ssm-cache-dtype float16
        --enable-mamba-cache-stochastic-rounding
        --mamba-cache-philox-rounds 5
        --kv-cache-dtype fp8
        --enable-prefix-caching
        --max-num-batched-tokens 32768
        --long-prefill-token-threshold 2048
        --max-num-seqs 16
      )
      ;;
  esac

  # Same "-Fast" checkpoint family also needs CUTE_DSL_ARCH set for its
  # CuteDSL-based kernels to target this GPU's actual compute capability
  # -- the checkpoint's own DGX Spark docs call out "you will get 2x
  # slower inference" without it. See:
  # https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4-Fast
  local fast_env=()
  case "$model" in
    unsloth/Qwen3.6-35B-A3B-NVFP4-Fast)
      fast_env=(-e "CUTE_DSL_ARCH=sm_121a")
      ;;
  esac

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

  local qwen38_env=() qwen38_args=() speculative_args=()
  local qwen38_serve_command=(vllm serve "$model")
  if is_qwen38_model "$model"; then
    configure_qwen38_profile "$model" "$max_len" || return 1
    qwen38_env=("${QWEN38_ENV_ARGS[@]}")
    qwen38_args=("${QWEN38_ARGS[@]}")
    speculative_args=("${QWEN38_SPECULATIVE_ARGS[@]}")
    qwen38_serve_command=("${QWEN38_SERVE_COMMAND[@]}")
  fi
  if is_flash_next_model "$model"; then
    local flash_checkpoint_model
    flash_checkpoint_model=$(flash_next_checkpoint_model)
    local flash_repo_dir="$HUB_CACHE/models--${flash_checkpoint_model//\//--}"
    local flash_snapshot_host
    flash_snapshot_host=$(find "$flash_repo_dir/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null || true)
    if [[ -z "$flash_snapshot_host" ]]; then
      echo "ERROR: Flash-Next checkpoint snapshot was not found under $flash_repo_dir." >&2
      echo "  Pull it first with: dgxt model-pull $flash_checkpoint_model" >&2
      return 1
    fi
    local flash_snapshot_container="/root/.cache/huggingface/hub/${flash_repo_dir#"$HUB_CACHE"/}/snapshots/$(basename "$flash_snapshot_host")"
    qwen38_serve_command=(vllm serve "$flash_snapshot_container")
  fi

  # NVIDIA's Qwen3.6 DGX Spark recipe (vLLM >= 0.28) uses the NVFP4
  # checkpoint with FP8 KV cache, Marlin MoE kernels, bounded batching,
  # async scheduling, prefix caching, and a three-token MTP draft. Keep
  # these defaults specific to NVIDIA's checkpoint: the community "-Fast"
  # checkpoint above has a different, verified FlashInfer B12X profile.
  local qwen36_env=() qwen36_args=() qwen36_speculative_args=() qwen36_fast_speculative_args=()
  local ornith_args=() ornith_speculative_args=()
  local codernext_env=() codernext_args=()
  case "$model" in
    nvidia/Qwen3.6-35B-A3B-NVFP4)
      # vLLM recommends this experimental Marlin path for the model's small
      # expert shapes on DGX Spark. Keep it scoped to the NVIDIA checkpoint;
      # VLLM_MARLIN_USE_ATOMIC_ADD=0 remains an explicit opt-out.
      qwen36_env=(-e "VLLM_MARLIN_USE_ATOMIC_ADD=${VLLM_MARLIN_USE_ATOMIC_ADD:-1}")
      qwen36_args=(
        --host 0.0.0.0
        --tensor-parallel-size 1
        --trust-remote-code
        --quantization modelopt
        --kv-cache-dtype fp8
        --attention-backend flashinfer
        --max-num-seqs 4
        --max-num-batched-tokens 8192
        --enable-chunked-prefill
        --async-scheduling
        --enable-prefix-caching
        --load-format fastsafetensors
      )
      # Do not force Marlin here. The modelopt quantization config selects
      # Marlin for the quantized target automatically, while an explicit
      # global backend is also inherited by the unquantized MTP predictor in
      # some vLLM builds. Leaving the target on auto lets each model choose a
      # compatible backend; VLLM_MOE_BACKEND remains an explicit override.
      [[ -n "$moe_backend" ]] && qwen36_args+=(--moe-backend "$moe_backend")

      local qwen36_speculative_mode="${VLLM_SPECULATIVE_MODE:-mtp}"
      local qwen36_speculative_tokens="${VLLM_SPECULATIVE_TOKENS:-3}"
      if [[ "${qwen36_speculative_mode,,}" == "dspark" || "${qwen36_speculative_mode,,}" == "dflash" ]]; then
        echo "WARNING: VLLM_SPECULATIVE_MODE=$qwen36_speculative_mode is not supported for Qwen3.6; using mtp." >&2
        echo "  Set VLLM_SPECULATIVE_MODE=none to disable speculative decoding." >&2
        qwen36_speculative_mode="mtp"
      fi
      case "${qwen36_speculative_mode,,}" in
        none)
          ;;
        mtp)
          if [[ ! "$qwen36_speculative_tokens" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: VLLM_SPECULATIVE_TOKENS must be a positive integer." >&2
            return 1
          fi
          qwen36_speculative_args=(
            --speculative-config
            "{\"method\":\"mtp\",\"num_speculative_tokens\":${qwen36_speculative_tokens},\"moe_backend\":\"triton\"}"
          )
          ;;
        *)
          echo "ERROR: VLLM_SPECULATIVE_MODE must be mtp or none for Qwen3.6." >&2
          return 1
          ;;
      esac
      ;;
    unsloth/Qwen3.6-35B-A3B-NVFP4-Fast)
      # The verified Fast run used the generic serving path: no speculative
      # draft, KV-cache override, or extra scheduler flags. The current
      # nightly's TorchInductor/Triton autotuning path can hit an illegal
      # CUDA address during GDN warmup, so use eager execution for this
      # exact compatibility profile.
      qwen36_args=(--enforce-eager)
      ;;
    ornith-ai/Ornith-1.5-35B-A3B-NVFP4)
      # Ornith is a Qwen3.5 MoE-compatible ModelOpt NVFP4 checkpoint.
      # Keep the target MoE backend on auto so mixed-precision layers and
      # the unquantized MTP predictor can choose compatible kernels.
      ornith_args=(
        --host 0.0.0.0
        --tensor-parallel-size 1
        --trust-remote-code
        --quantization modelopt
        --kv-cache-dtype fp8
        --max-num-seqs 16
        --max-num-batched-tokens 4096
        --enable-chunked-prefill
        --enable-prefix-caching
      )

      # The official card does not enable speculative decoding. Start with
      # that stable baseline; the checkpoint contains one MTP layer and
      # supports opt-in one/two-token experiments on compatible vLLM builds.
      local ornith_speculative_mode="${VLLM_SPECULATIVE_MODE:-none}"
      local ornith_speculative_tokens="${VLLM_SPECULATIVE_TOKENS:-1}"
      case "${ornith_speculative_mode,,}" in
        none)
          ;;
        mtp)
          if [[ ! "$ornith_speculative_tokens" =~ ^[1-2]$ ]]; then
            echo "ERROR: VLLM_SPECULATIVE_TOKENS must be 1 or 2 for Ornith MTP." >&2
            return 1
          fi
          ornith_speculative_args=(
            --speculative-config
            "{\"method\":\"mtp\",\"num_speculative_tokens\":${ornith_speculative_tokens},\"moe_backend\":\"triton\"}"
          )
          ;;
        *)
          echo "ERROR: VLLM_SPECULATIVE_MODE must be mtp or none for Ornith." >&2
          return 1
          ;;
      esac
      ;;
    *Qwen3-Coder-Next*|*qwen3-coder-next*|*Qwen3-Next*|*qwen3-next*)
      # The NVFP4-GB10 quant and its ucbye mirror use the same
      # 80B/3B hybrid Gated-DeltaNet
      # coder (Qwen3NextForCausalLM). FlashInfer handles attention; the
      # FP4 linear and MoE GEMMs are explicitly routed to Marlin. The
      # current vLLM nightly uses CLI backend flags; the older
      # VLLM_NVFP4_GEMM_BACKEND environment variable is deprecated.
      # No MTP/DSpark draft ships with this quant, so speculative decoding
      # defaults to none -- opt in with the separate -DSpark checkpoint.
      codernext_env=(
        -e "VLLM_MARLIN_USE_ATOMIC_ADD=1"
      )
      codernext_args=(
        --host 0.0.0.0
        --tensor-parallel-size 1
        --trust-remote-code
        --kv-cache-dtype fp8
        --attention-backend flashinfer
        --linear-backend marlin
        --moe-backend marlin
        --enable-prefix-caching
        --enable-chunked-prefill
        --max-num-batched-tokens 8192
        --max-num-seqs 8
      )
      ;;
  esac

  # DSpark is NVIDIA's recommended speculative decoder for Nemotron on
  # DGX Spark and low-concurrency interactive workloads. Keep alternatives
  # opt-in because their draft checkpoints and compatibility vary by vLLM
  # release. Set VLLM_SPECULATIVE_MODE=none to restore a non-speculative
  # baseline, or select mtp/dflash for explicit experiments.
  case "$model" in
    *Nemotron-3.5-Lightning*|*nemotron-3.5-lightning*)
      local speculative_mode="${VLLM_SPECULATIVE_MODE:-dspark}"
      local speculative_tokens="${VLLM_SPECULATIVE_TOKENS:-3}"
      case "${speculative_mode,,}" in
        none)
          ;;
        dspark)
          if [[ ! "$speculative_tokens" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: VLLM_SPECULATIVE_TOKENS must be a positive integer." >&2
            return 1
          fi
          local dspark_model="${VLLM_SPECULATIVE_MODEL:-${model}-DSpark}"
          speculative_args=(
            --speculative_config.method dspark
            --speculative_config.num_speculative_tokens "$speculative_tokens"
            --speculative_config.model "$dspark_model"
          )
          ;;
        mtp)
          if [[ ! "$speculative_tokens" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: VLLM_SPECULATIVE_TOKENS must be a positive integer." >&2
            return 1
          fi
          speculative_args=(
            --speculative_config.method mtp
            --speculative_config.num_speculative_tokens "$speculative_tokens"
          )
          ;;
        dflash)
          if [[ ! "$speculative_tokens" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: VLLM_SPECULATIVE_TOKENS must be a positive integer." >&2
            return 1
          fi
          local dflash_model="${VLLM_SPECULATIVE_MODEL:-${model}-DFlash}"
          speculative_args=(
            --speculative_config.method dflash
            --speculative_config.num_speculative_tokens "$speculative_tokens"
            --speculative_config.model "$dflash_model"
          )
          ;;
        *)
          echo "ERROR: VLLM_SPECULATIVE_MODE must be dspark, none, mtp, or dflash." >&2
          return 1
          ;;
      esac
      ;;
  esac

  # Model profiles contribute only their matching environment, flags, and
  # mounts; the container invocation below remains engine-generic.
  local model_env=("${gptoss_env[@]}" "${qwen38_env[@]}" "${qwen36_env[@]}" "${fast_env[@]}" "${codernext_env[@]}")
  local omni_args=()
  case "$model" in
    *Nemotron-3-Nano-Omni*|*nemotron-3-nano-omni*)
      # NVIDIA's vLLM Spark recipe for the Omni checkpoint. The multimodal
      # limits prevent unbounded media batching; video pruning and explicit
      # frame sampling reduce prefill cost without changing text behavior.
      omni_args=(
        --host 0.0.0.0
        --tensor-parallel-size 1
        --trust-remote-code
        --video-pruning-rate 0.5
        --max-num-seqs 8
        --allowed-local-media-path /
        --media-io-kwargs '{"video":{"fps":2,"num_frames":256}}'
        --limit-mm-per-prompt '{"video":1,"image":1,"audio":1}'
        --enable-prefix-caching
        --max-num-batched-tokens 32768
        --kv-cache-dtype fp8
      )
      ;;
  esac
  local model_args=("${nemotron_args[@]}" "${gptoss_args[@]}" "${qwen38_args[@]}" "${qwen36_args[@]}" "${ornith_args[@]}" "${omni_args[@]}" "${codernext_args[@]}")
  local model_volumes=("${gptoss_vol[@]}")

  local image
  image=$(resolve_engine_image "$model")
  case "$model" in
    qwen38-flash-next-v029)
      ensure_flash_next_v029_image "$image" || return 1
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
    "${model_env[@]}" \
    -v "${HUB_CACHE}:/root/.cache/huggingface/hub" \
    "${model_volumes[@]}" \
    "$image" \
    "${qwen38_serve_command[@]}" \
    "${served_model_name_args[@]}" \
    "${max_len_args[@]}" \
    "${repetition_penalty_args[@]}" \
    "${moe_backend_args[@]}" \
    "${model_args[@]}" \
    "${speculative_args[@]}" \
    "${qwen36_speculative_args[@]}" \
    "${qwen36_fast_speculative_args[@]}" \
    "${ornith_speculative_args[@]}" \
    --gpu-memory-utilization "$gpu_mem" \
    --api-key "$api_key" \
    "${tool_args[@]}" \
    "${rope_args[@]}" \
    "${reasoning_args[@]}"
}
