# dgx-tools shared library — engine-agnostic helpers shared by every engine
# module (Docker lifecycle, HF cache/CLI passthroughs, config file, API
# key/context resolution). Sourced by dgxt; not meant to run directly.

# ---- Config -------------------------------------------------------------
DEFAULT_CONFIG_FILE="$HOME/.dgxtrc"

# Match huggingface_hub's own cache resolution so dgxt and the `hf` CLI
# always agree on where models live — shared across every engine.
HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"

# Persistent host cache dir for the tiktoken-rs vocab file gpt-oss's
# Harmony tokenizer needs (see ensure_tiktoken_cache below).
TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$HOME/.cache/dgxt/tiktoken}"

# gpt-oss's Harmony tokenizer (openai_harmony, a Rust/tiktoken-rs
# extension) lazily fetches its o200k_base vocab file from a hardcoded
# Microsoft blob-storage CDN URL on the FIRST real chat request, not at
# model load -- so `dgxt start`'s healthcheck can pass clean and your
# first actual request still 500s with "failed to download or load vocab
# file" (openai_harmony.HarmonyError). We've observed this fail
# intermittently even when the container's network path to that exact
# URL works fine (plain curl succeeds) -- tiktoken-rs's own tiny fetch
# client apparently isn't as reliable. Avoid depending on in-container
# network for this at all: pre-fetch the vocab file here, into a
# persistent host dir, using the *exact* cache-key filename tiktoken-rs
# expects (sha1 hex of the vocab URL) -- its own cache-hit logic then
# finds it pre-populated with a matching hash and skips the network
# fetch entirely. See lib/engines/vllm.sh's gpt-oss case in
# engine_run_container for where this cache dir gets mounted in.
# Best-effort: any failure here (no curl, no network right now, etc.)
# just leaves the cache unprimed, and vLLM falls back to its own
# (unreliable) live fetch instead -- not a hard error.
ensure_tiktoken_cache() {
  mkdir -p "$TIKTOKEN_CACHE_DIR" 2>/dev/null || return 1
  local url="https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
  local expected_sha256="446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
  command -v python3 >/dev/null 2>&1 || return 1
  local cache_key
  cache_key=$(python3 -c "import hashlib,sys; print(hashlib.sha1(sys.argv[1].encode()).hexdigest())" "$url" 2>/dev/null)
  [[ -z "$cache_key" ]] && return 1
  local dest="$TIKTOKEN_CACHE_DIR/$cache_key"

  actual_sha256() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }

  if [[ -f "$dest" ]] && [[ "$(actual_sha256 "$dest")" == "$expected_sha256" ]]; then
    return 0
  fi
  command -v curl >/dev/null 2>&1 || return 1
  curl -fsSL -o "$dest.tmp" "$url" 2>/dev/null || { rm -f "$dest.tmp"; return 1; }
  if [[ "$(actual_sha256 "$dest.tmp")" != "$expected_sha256" ]]; then
    rm -f "$dest.tmp"
    return 1
  fi
  mv "$dest.tmp" "$dest"
}

# Load config file (key=value, one per line). Env vars already set in the
# environment take priority and are left untouched.
load_config() {
  local config_file="${1:-$DEFAULT_CONFIG_FILE}"
  if [[ -f "$config_file" ]]; then
    while IFS='=' read -r key value; do
      [[ "$key" =~ ^[[:space:]]*# ]] && continue
      [[ -z "$key" ]] && continue
      key=$(echo "$key" | xargs)
      value=$(echo "$value" | xargs)
      # Only set if not already present in the environment, so real env vars win.
      if [[ -z "${!key+x}" ]]; then
        export "$key"="$value"
      fi
    done < "$config_file"
  fi
}

# Set KEY=VALUE in $DEFAULT_CONFIG_FILE, creating the file and replacing any
# existing line for KEY.
save_config_value() {
  local key="$1" value="$2"
  touch "$DEFAULT_CONFIG_FILE"
  if grep -q "^${key}=" "$DEFAULT_CONFIG_FILE" 2>/dev/null; then
    sed -i "s|^${key}=.*|${key}=${value}|" "$DEFAULT_CONFIG_FILE"
  else
    echo "${key}=${value}" >> "$DEFAULT_CONFIG_FILE"
  fi
}

# ---- Prerequisites --------------------------------------------------------

ensure_docker() {
  if ! docker ps >/dev/null 2>&1; then
    echo "Docker not accessible. Try: newgrp docker  (or log out/in), then re-run."
    exit 1
  fi
}

# Check that the `hf` CLI (huggingface_hub) is available. Everything
# HuggingFace-related beyond this (search, download, cache management,
# auth) is delegated straight to `hf` — this is the one place dgxt touches
# HF tooling directly, and only to bootstrap it onto a fresh box.
# If missing and running interactively, offers to install it via pip
# (never installs silently/non-interactively — always asks first).
ensure_hf_cli() {
  if command -v hf >/dev/null 2>&1; then
    return 0
  fi

  echo "HuggingFace CLI ('hf') not found on PATH." >&2

  if [[ ! -t 0 ]]; then
    echo "Install it with:" >&2
    echo "  pip3 install huggingface_hub" >&2
    echo "  (add --break-system-packages if pip refuses due to PEP 668)" >&2
    return 1
  fi

  local reply
  read -rp "Install it now with pip3? [Y/n] " reply
  if [[ "$reply" =~ ^[Nn] ]]; then
    echo "Skipping. Install manually with: pip3 install huggingface_hub" >&2
    return 1
  fi

  echo "Installing huggingface_hub..."
  # Debian/Ubuntu's PEP 668 "externally-managed-environment" protection can
  # make a plain `pip install` fail; retry with the override before giving
  # up. Guarded by `if` so a failure here can't kill the script under `set -e`.
  if ! pip3 install -q huggingface_hub 2>/dev/null; then
    if ! pip3 install -q --break-system-packages huggingface_hub; then
      echo "ERROR: pip install failed. Try manually:" >&2
      echo "  pip3 install --break-system-packages huggingface_hub" >&2
      return 1
    fi
  fi

  if ! command -v hf >/dev/null 2>&1; then
    echo "hf still not found on PATH after install." >&2
    echo "It likely installed to ~/.local/bin — open a new shell and retry, or:" >&2
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
    return 1
  fi

  echo "hf installed: $(command -v hf)"
  return 0
}

# Make sure the HF cache dir exists and is writable, creating it if needed.
# Must run BEFORE any `docker run` bind-mount of this path (see engine
# modules), otherwise Docker (running the container as root) will
# auto-create it as root on a fresh box and block every future non-container
# write (hf download, model-pull, etc.) — the one recurring local gotcha.
ensure_hub_cache() {
  mkdir -p "$HUB_CACHE"
  if [[ ! -w "$HUB_CACHE" ]]; then
    echo "ERROR: $HUB_CACHE is not writable by $(whoami)."
    echo "This usually happens when a container (running as root) wrote to"
    echo "this bind-mounted directory first. Fix ownership with:"
    echo "  sudo chown -R \"$(whoami)\":\"$(whoami)\" \"$HUB_CACHE\""
    return 1
  fi
}

# ---- Container lifecycle (generic — engines just supply a container name) -

is_running() {
  local container_name="$1"
  docker ps --filter "name=^${container_name}$" --format '{{.Names}}' 2>/dev/null | grep -q "^${container_name}$"
}

cmd_stop() {
  local container_name="$1"
  ensure_docker
  if ! is_running "$container_name"; then
    echo "Container '$container_name' is not running."
    return 0
  fi
  echo "Stopping container..."
  docker stop "$container_name"
  docker rm "$container_name"
  echo "Stopped."
}

cmd_logs() {
  local container_name="$1"
  ensure_docker
  if ! is_running "$container_name"; then
    echo "Container '$container_name' is not running."
    return 1
  fi
  docker logs -f "$container_name"
}

cmd_status() {
  local container_name="$1"
  ensure_docker
  docker ps -a --filter "name=^${container_name}$" --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
}

# Stream a container's logs in the background, poll a health endpoint until
# ready (or timeout), then clean up the log stream either way.
wait_for_ready() {
  local container_name="$1" port="$2"

  docker logs -f "$container_name" 2>&1 &
  local log_pid=$!
  # Self-clearing: a RETURN trap isn't scoped to this function alone — left
  # armed, it fires again on the caller's next return too, where $log_pid
  # is out of scope (unbound under `set -u`). Clear it as its own first act.
  trap 'kill "$log_pid" 2>/dev/null || true; trap - RETURN' RETURN

  local elapsed=0
  local timeout=900
  while [[ $elapsed -lt $timeout ]]; do
    if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
      kill "$log_pid" 2>/dev/null || true
      echo ""
      echo "Server is ready! (${elapsed}s) — listening on port ${port}."
      echo "See the README for a curl example, or: dgxt logs"
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done

  kill "$log_pid" 2>/dev/null || true
  echo "Timeout waiting for server. Check logs: dgxt logs"
  return 1
}

# ---- Shared value resolution ---------------------------------------------

# Get the value of $var_name, generating a random one if unset. Avoids ever
# serving on the LAN with a fixed, widely-known default key.
resolve_api_key() {
  local var_name="$1"
  local current="${!var_name:-}"
  if [[ -n "$current" ]]; then
    printf '%s' "$current"
  else
    openssl rand -hex 24 2>/dev/null || head -c 48 /dev/urandom | xxd -p | head -c 48
  fi
}

# Parse a human-friendly context-length string like "64k", "256k", "1m"
# (binary K/M, matching the existing 131072-for-"128k" convention already
# used by this codebase's defaults) into a raw token-count integer. Plain
# integers and "auto"/empty pass through unchanged. Used wherever a
# max-context value is read from a CLI flag or config/env var, so users
# can write --max-context 256k or VLLM_MAX_MODEL_LEN=1m instead of typing
# out 262144/1048576.
parse_context_size() {
  local input="$1"
  case "$input" in
    auto|"") echo "$input" ;;
    *[Kk]) echo $(( ${input%[Kk]} * 1024 )) ;;
    *[Mm]) echo $(( ${input%[Mm]} * 1024 * 1024 )) ;;
    *) echo "$input" ;;
  esac
}

# Resolve an HF repo's max context length from its actual config.json (the
# same file the serving engine itself reads); falls back to 131072 if it
# can't be found. Reads the repo file directly (not via `hf`) since it
# needs to work before a model is downloaded and without requiring `hf`.
# Named distinctly from resolve_max_context() (a thin wrapper around this
# by default) so engines whose "model" isn't an HF repo id (e.g. nim.sh,
# where it's an NGC image tag) can override the wrapper while still
# reusing this for any known-mapped HF repo id underneath.
resolve_max_context_from_hf_repo() {
  local model="$1"
  local max_ctx=""
  if command -v curl >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
    local config_json
    config_json=$(curl -sfL --max-time 10 "https://huggingface.co/${model}/raw/main/config.json" 2>/dev/null)
    if [[ -n "$config_json" ]]; then
      max_ctx=$(echo "$config_json" | python3 -c "
import sys, json

def find_key(obj, keys):
    if isinstance(obj, dict):
        for k in keys:
            if isinstance(obj.get(k), int):
                return obj[k]
        for v in obj.values():
            found = find_key(v, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_key(v, keys)
            if found is not None:
                return found
    return None

try:
    d = json.load(sys.stdin)
    val = find_key(d, ['max_position_embeddings', 'seq_length', 'n_positions', 'max_sequence_length'])
    if val is not None:
        print(val)
except Exception:
    pass
" 2>/dev/null)
    fi
  fi
  echo "${max_ctx:-131072}"
}

# Default resolve_max_context: for engines whose "model" value already IS
# an HF repo id (vLLM), this is a direct passthrough. Engines where it
# isn't (e.g. nim.sh) override this function entirely.
resolve_max_context() {
  resolve_max_context_from_hf_repo "$1"
}

# Preferred default context length when nothing else specifies one (no
# --max-context flag, no persisted VLLM_MAX_MODEL_LEN/NIM_MAX_MODEL_LEN).
# Aim for the 1M window supported by the default Nemotron deployment, while
# still capping other models to whatever they actually support.
DEFAULT_PREFERRED_CONTEXT=1048576

# The actual "auto" resolution used by cmd_start/cmd_save/cmd_config when
# no explicit context was requested: aim for DEFAULT_PREFERRED_CONTEXT,
# but never silently exceed what the model actually supports (its real
# derived native max from resolve_max_context) — so a long-context model
# (e.g. Nemotron 3.5 Lightning, native 1048576) gets the full 1M
# automatically, while a shorter-context model (e.g. Qwen3-32B, native
# 40960) safely falls back to its own real ceiling instead of requesting
# more than it can serve. If the native max is unknown (resolve_max_context
# returns empty — e.g. an unmapped NIM image), this also returns empty
# rather than guessing, matching resolve_max_context's own "don't guess"
# convention.
resolve_default_context() {
  local model="$1"
  local engine_default
  if declare -f resolve_default_context_override >/dev/null 2>&1; then
    if engine_default=$(resolve_default_context_override "$model") && [[ -n "$engine_default" ]]; then
      echo "$engine_default"
      return
    fi
  fi

  local native_max
  native_max=$(resolve_max_context "$model")
  if [[ -z "$native_max" ]]; then
    echo ""
  elif [[ "$native_max" =~ ^[0-9]+$ ]] && (( native_max < DEFAULT_PREFERRED_CONTEXT )); then
    echo "$native_max"
  else
    echo "$DEFAULT_PREFERRED_CONTEXT"
  fi
}

# True if the current engine actually maps this setting to a real config
# var name (vs. load_engine()'s "_DGXT_UNSET" placeholder sentinel). Some
# settings (context length, GPU memory fraction, tool-call parser) don't
# apply to every engine — e.g. NIM's containers are fully self-tuned per
# GPU profile and have no equivalent knobs. Used to skip printing/saving
# a setting (and, critically, to avoid writing a line literally named
# "_DGXT_UNSET=..." to ~/.dgxtrc) for engines that don't define it.
engine_defines_var() {
  [[ "$1" != "_DGXT_UNSET" ]]
}

# Delegates to the current engine module's own resolve_tool_call_parser()
# if it defines one (only vllm.sh does today — parser names/detection are
# engine-specific). Falls back to ENGINE_DEFAULT_TOOL_CALL_PARSER (or
# disabled) for engines that don't implement this.
resolve_tool_call_parser_for_engine() {
  local model="$1"
  if declare -f resolve_tool_call_parser >/dev/null 2>&1; then
    resolve_tool_call_parser "$model"
  else
    echo "${ENGINE_DEFAULT_TOOL_CALL_PARSER:-}"
  fi
}

# Same idea, for --reasoning-parser (only vllm.sh implements
# resolve_reasoning_parser today). Falls back to
# ENGINE_DEFAULT_REASONING_PARSER (or disabled) for engines that don't.
resolve_reasoning_parser_for_engine() {
  local model="$1"
  if declare -f resolve_reasoning_parser >/dev/null 2>&1; then
    resolve_reasoning_parser "$model"
  else
    echo "${ENGINE_DEFAULT_REASONING_PARSER:-}"
  fi
}

# Same idea, for the GPU memory fraction. Engines may choose a
# model-specific default while still honoring their configured value.
resolve_gpu_memory_for_engine() {
  local model="$1"
  local max_len="${2:-}"
  if declare -f resolve_gpu_memory >/dev/null 2>&1; then
    resolve_gpu_memory "$model" "$max_len"
  else
    echo "${!ENGINE_GPU_MEM_VAR:-0.8}"
  fi
}

# ---- Model search/passthroughs (engine-agnostic) --------------------------

# Search models. If the current engine defines engine_search() (only
# nim.sh does today — model catalogs vary wildly by engine), delegates to
# that; otherwise falls back to the generic `hf models ls` passthrough
# (optionally filtered to ENGINE_HF_APPS_FILTER-compatible models).
cmd_search() {
  local query="${1:-}"
  shift || true
  if [[ -z "$query" ]]; then
    read -rp "Search query: " query
  fi
  if declare -f engine_search >/dev/null 2>&1; then
    engine_search "$query"
    return
  fi
  ensure_hf_cli || return 1
  if [[ -n "${ENGINE_HF_APPS_FILTER:-}" ]]; then
    hf models ls --search "$query" --apps "$ENGINE_HF_APPS_FILTER" --sort downloads "$@"
  else
    hf models ls --search "$query" --sort downloads "$@"
  fi
}

# Download a model. If the current engine defines engine_model_pull()
# (only nim.sh does today — "pulling a model" means something different
# per engine), delegates to that; otherwise falls back to `hf download`
# (uses the dgxt cache dir). `hf` handles resuming/validating partial
# downloads itself; we only guarantee the cache dir is writable first.
cmd_model_pull() {
  local model="${1:-${!ENGINE_MODEL_VAR:-$ENGINE_DEFAULT_MODEL}}"
  [[ $# -gt 0 ]] && shift
  if declare -f engine_model_pull >/dev/null 2>&1; then
    engine_model_pull "$model" "$@"
    return
  fi
  ensure_hf_cli || return 1
  ensure_hub_cache || return 1
  # vLLM only ever loads the top-level safetensors + config/tokenizer
  # files. Some repos also ship an original/ (reference bf16/fp32
  # checkpoint, meant for other runtimes) or metal/ (Apple MLX) folder
  # alongside the vLLM-ready weights -- openai/gpt-oss-120b is a prime
  # example: ~63GB of MXFP4 safetensors plus 100GB+ of original/ that
  # vLLM never reads, more than doubling the download for nothing.
  # Exclude both by default; skip this if the caller already passed
  # their own --include/--exclude (trust their judgement over ours).
  local exclude_args=()
  if [[ "$*" != *--include* && "$*" != *--exclude* ]]; then
    exclude_args=(--exclude "original/*" --exclude "metal/*")
  fi
  hf download "$model" "${exclude_args[@]}" "$@"
}

# List downloaded models. Delegates to engine_model_list() if the current
# engine defines one (only nim.sh today); otherwise `hf cache ls`.
cmd_model_list() {
  if declare -f engine_model_list >/dev/null 2>&1; then
    engine_model_list
    return
  fi
  ensure_hf_cli || return 1
  hf cache ls
}

# Print "repo_id|size" for models already fully cached locally, one per
# line, largest-complete-download-looking first. Used by cmd_setup's
# model picker so users can select something they've already pulled
# instead of retyping a HF id from memory. Best-effort: silently prints
# nothing if `hf` isn't installed or cache is empty/unparseable — this is
# a convenience list, not a hard requirement.
#
# Filters out repos under MIN_CACHED_MODEL_BYTES: `hf download` leaves a
# tiny metadata-only cache entry (just config.json etc, often <1MB) the
# moment a download starts, even if it's incomplete or was only used to
# probe --arch. Those aren't actually servable, so they'd be misleading
# in a "ready to serve now" list.
MIN_CACHED_MODEL_BYTES=$((1024 * 1024 * 1024)) # 1 GiB
list_cached_models() {
  command -v hf >/dev/null 2>&1 || return 0
  command -v python3 >/dev/null 2>&1 || return 0
  hf cache ls --format json 2>/dev/null | python3 -c '
import json, sys

def to_bytes(size_str):
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    size_str = size_str.strip()
    if size_str and size_str[-1] in units:
        try:
            return float(size_str[:-1]) * units[size_str[-1]]
        except ValueError:
            return 0
    try:
        return float(size_str)
    except ValueError:
        return 0

try:
    entries = json.load(sys.stdin)
except Exception:
    sys.exit(0)

rows = []
for e in entries:
    if e.get("repo_type") != "model":
        continue
    size_bytes = to_bytes(e.get("size", "0"))
    if size_bytes < '"$MIN_CACHED_MODEL_BYTES"':
        continue
    rows.append((size_bytes, e.get("repo_id", ""), e.get("size", "")))

for size_bytes, repo_id, size in sorted(rows, reverse=True):
    print(f"{repo_id}|{size}")
' 2>/dev/null
}
