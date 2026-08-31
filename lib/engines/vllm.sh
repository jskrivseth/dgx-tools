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

# Start the vLLM container. Called by the generic cmd_start in dgxt
# after it has resolved model/max_len/port/gpu_mem/api_key.
engine_run_container() {
  local model="$1" max_len="$2" port="$3" gpu_mem="$4" api_key="$5"
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
    --api-key "$api_key"
}
