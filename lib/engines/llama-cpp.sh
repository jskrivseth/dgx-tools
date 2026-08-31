# llama.cpp engine module for dgx-tools — PLACEHOLDER, not yet implemented.
# Lightweight GGUF-based inference, NVIDIA-backed.
#
# To implement: copy lib/engines/vllm.sh as a template and fill in
# ENGINE_CONTAINER_NAME/ENGINE_IMAGE/ENGINE_*_VAR/ENGINE_RECOMMENDED_MODELS,
# then replace engine_run_container() below with a real `docker run`.

ENGINE_NAME="llama-cpp"
ENGINE_STATUS="planned"
ENGINE_CONTAINER_NAME="llama-cpp-server"

engine_run_container() {
  echo "ERROR: llama.cpp support isn't implemented in dgx-tools yet." >&2
  echo "Contributions welcome — see lib/engines/vllm.sh for the pattern to follow." >&2
  return 1
}
