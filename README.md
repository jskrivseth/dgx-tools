# dgx-tools

Get an LLM inference engine running on a fresh DGX Spark-class box (or
similar single-GPU ARM64 NVIDIA workstation) as fast as possible — no
native builds, no dependency wrangling beyond Docker and the official
HuggingFace CLI.

Currently implemented: **vLLM** and **NIM**. Placeholders exist for
TensorRT-LLM, SGLang, llama.cpp, and Ollama (see [Engines](#engines)).

dgxt does **not** reimplement anything the `hf` CLI already does well —
searching, downloading, cache management, and auth are all thin
passthroughs to `hf`. dgxt's job is only the Docker lifecycle (start,
stop, config) around whichever engine you pick.

## Getting dgxt

```bash
git clone https://github.com/jskrivseth/dgx-tools.git ~/dgx-tools
cd ~/dgx-tools
./dgxt help
```

No build step, no install step — it's a single bash script (`dgxt`) plus
a small `lib/` of sourced helpers. Optionally put it on your `PATH`:

```bash
ln -s ~/dgx-tools/dgxt ~/bin/dgxt   # if ~/bin is already on your PATH
```

Needs Docker (with the NVIDIA Container Toolkit) and the `hf` CLI. Both
are checked at runtime — if `hf` is missing, dgxt offers to install it
for you via pip right then. See [Prerequisites](#prerequisites) in the
appendix if either check fails.

## Quick start

```bash
./dgxt setup
```

Walks through, in order: engine selection (default `vllm`; `nim` for
NVIDIA's curated containers), a Docker check, HuggingFace auth, an API
key (generated for you, or pasted for NIM), and a model pick from a
short recommended list (or type your own model ID) — then starts the
container and streams logs until the server is healthy.

## Everyday commands

```bash
./dgxt start [model] [--max-context N]   # start serving (default model/context if omitted)
./dgxt stop                              # stop and remove the container
./dgxt restart [model]                   # stop then start
./dgxt logs                              # tail container logs
./dgxt status                            # is it running?
./dgxt config                            # show effective settings (file + env, merged)
./dgxt save                              # persist current settings to ~/.dgxtrc
```

`--max-context` accepts plain token counts or K/M shorthand (`64k` ==
`65536`, `256k` == `262144`, `1m` == `1048576`). Left unset, dgxt targets
1M tokens capped to whatever the model actually natively supports. With the
default Nemotron model, DSpark speculative decoding is enabled automatically.
Asking for more than a model's native max works too — see
[Context extension](#context-extension-past-a-models-native-max) in the
appendix for how (and where it can't).

For `RadixArk/Qwen3.8-27B-NVFP4`, dgxt automatically selects the matching
`RadixArk/Qwen3.8-27B-DSpark` draft checkpoint with its recommended seven-token
draft block, in addition to the FlashInfer/atomic-add workarounds and bounded
batching settings from the DGX Spark reference recipe. Selecting that model is
therefore sufficient; no speculative-decoding flags are required. Set
`VLLM_SPECULATIVE_MODE=none` for a baseline, or use `mtp` to select the target
checkpoint's native MTP head. `VLLM_SPECULATIVE_MODEL` and
`VLLM_SPECULATIVE_TOKENS` override the DSpark checkpoint and depth.
`VLLM_MTP_TOKENS` remains accepted as a backwards-compatible alias for the MTP
token count.

For `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, dgxt automatically
applies NVIDIA's GB10 backend recipe: FlashInfer for Mamba, aligned Mamba
caches, FP8 KV cache, prefix caching, Marlin for MoE, and DSpark speculative
decoding with three tokens. It uses 2048-token prefill chunks inside a
32768-token batch budget so up to 16 low-concurrency requests can interleave
instead of queueing behind one long prefill. It also uses the recipe's `0.85`
GPU-memory fraction unless `VLLM_GPU_MEM` is explicitly set. Set
`VLLM_SPECULATIVE_MODE=none` for a non-speculative baseline, or select `mtp`
or `dflash` for explicit experiments. `VLLM_SPECULATIVE_MODEL` overrides the
derived DSpark/DFlash draft model ID, and `VLLM_SPECULATIVE_TOKENS` controls
the draft length.

The setup model picker also includes these NVIDIA DGX Spark model recipes:

| Model | Hugging Face ID | Best for |
|---|---|---|
| Nemotron 3 Nano Omni 30B A3B Reasoning | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | Multimodal reasoning, tool use, and long-context chat |
| Qwen3.6 35B A3B NVFP4 | `nvidia/Qwen3.6-35B-A3B-NVFP4` | Agentic workloads, tool calling, and reasoning |
| Llama 3.3 70B Instruct FP4 | `nvidia/Llama-3.3-70B-Instruct-FP4` | General-purpose chat, reasoning, and code generation (TensorRT-LLM optimized) |

On DGX Spark, prefer the NVFP4 checkpoints. BF16 is retained as a
Nemotron Omni compatibility fallback, but it uses roughly 62 GB versus
roughly 21 GB for the official NVFP4 checkpoint and leaves substantially
less unified memory available for KV cache and the operating system. The
Qwen3.6 NVFP4 entry uses NVIDIA's Spark recipe automatically: FP8 KV cache,
FlashInfer attention, modelopt's Marlin-compatible quantized MoE path,
bounded chunked prefill, async scheduling, prefix caching, fast safetensors
loading, and three-token MTP speculative decoding. dgxt also enables vLLM's
experimental Marlin atomic-add path for the small expert shapes; set
`VLLM_MARLIN_USE_ATOMIC_ADD=0` to opt out. Set
`VLLM_SPECULATIVE_MODE=none` for a baseline. The target backend is left on
auto so the unquantized MTP predictor can select a compatible backend.

The `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` variant uses its DGX Spark-specific
`CUTE_DSL_ARCH=sm_121a` and `flashinfer_b12x` MoE path, plus FP8 KV cache,
bounded batching, chunked prefill, async scheduling, prefix caching, and its
built-in two-token MTP draft. It does not receive `--quantization modelopt`;
Unsloth's compressed-tensors checkpoint is auto-detected. Set
`VLLM_SPECULATIVE_MODE=none` when comparing against a non-speculative
baseline. Its automatic GPU-memory setting is `0.8` through native 256K
context and `0.7` for longer YaRN contexts; explicit `VLLM_GPU_MEM` still
overrides this heuristic.

The NVIDIA Qwen3.6 recipe's original `0.4` GPU-memory fraction is too low
for the current vLLM nightly's CUDA-graph footprint on Spark: it can fail
to reserve enough KV cache for one 256K request. dgxt therefore uses `0.6`
through 512K and `0.7` above 512K. It is a KV-cache/concurrency budget, not
a direct single-request speed setting. An explicit `VLLM_GPU_MEM`
environment or config value always overrides these tiers; monitor unified
memory pressure before trying `0.8` or higher.

The Llama 3.3 FP4 checkpoint is included for model selection, but NVIDIA's
model card documents TensorRT-LLM as its optimized inference engine rather
than a DGX Spark vLLM recipe. Treat it as an experimental vLLM choice until
the TensorRT-LLM engine is enabled in dgxt; use TensorRT-LLM directly for
the NVIDIA-validated Llama deployment path.

The Omni recipe defaults to a 131072-token context on Spark even though the
checkpoint supports up to 256K; pass `--max-context 256k` only when the
additional unified-memory allocation is appropriate for your workload.

They can also be selected directly without running setup:

```bash
dgxt start nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4
dgxt start nvidia/Qwen3.6-35B-A3B-NVFP4
dgxt start nvidia/Llama-3.3-70B-Instruct-FP4
```

Once running, the server speaks the OpenAI-compatible API:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: ****** VLLM_API_KEY>" \
  -d '{"model":"<model>","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## Not reimplemented — use `hf` directly

dgxt only wraps three things as thin conveniences; everything else is a
direct passthrough to `hf` (or NGC, for NIM — see the
[appendix](#hf--ngc-passthrough-details) for how that differs):

```bash
./dgxt search <query>       # -> hf models ls --search <query> --sort downloads
./dgxt model-pull <model>   # -> hf download <model>  (pre-fetch before starting)
./dgxt model-list           # -> hf cache ls
```

For anything beyond that — browsing by author/params, verifying cache
integrity, pruning incomplete downloads, switching HF accounts — just use
`hf` directly:

```bash
hf models ls --search llama --num-parameters min:6B,max:32B --sort downloads
hf cache ls
hf cache prune          # clean up incomplete/orphaned downloads
hf auth login / hf auth whoami
```

## Configuration

Settings live in `~/.dgxtrc` (created by `dgxt save` or `dgxt setup`), as
plain `KEY=value` lines. **Environment variables always take priority
over the file** — export something and it wins, no editing required:

```bash
# ~/.dgxtrc
DGXT_ENGINE=vllm
VLLM_MODEL=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
VLLM_MAX_MODEL_LEN=1m
VLLM_API_KEY=<random, generated for you>
VLLM_PORT=8000
VLLM_GPU_MEM=0.85                       # omit for model/context auto-tuning
# Model-specific: dspark for Lightning, mtp for Qwen3.6
VLLM_SPECULATIVE_MODE=dspark
VLLM_SPECULATIVE_TOKENS=3
HF_TOKEN=hf_...
```

`VLLM_SPECULATIVE_MODE` is interpreted for the selected model. If a
configuration carried over from Lightning sets `dspark` or `dflash` while
Qwen3.6 is selected, dgxt warns and falls back to Qwen3.6's supported MTP
mode. Use `VLLM_SPECULATIVE_MODE=none` to disable speculative decoding.

NIM uses a different set of keys — see the
[appendix](#full-config-file-examples) for its example and the
differences (no GPU-memory knob, an external API key that can't be
auto-generated).

## Engines

```bash
./dgxt engine            # show current engine + list all, with status
./dgxt engine <name>      # switch (persists to ~/.dgxtrc)
```

| Engine | Status |
|---|---|
| `vllm` | ready |
| `nim` | ready |
| `tensorrt-llm` | planned |
| `sglang` | planned |
| `llama-cpp` | planned |
| `ollama` | planned |

Adding a new engine means writing one small file in `lib/engines/` — a
handful of variables (container name, config-file variable names,
recommended models) plus one function that runs `docker run`. Everything
else — Docker lifecycle, health-check waiting, log streaming, HF
passthroughs, config save/load, random API key generation — is shared in
`lib/common.sh` and just works once the engine module is filled in.
`lib/engines/vllm.sh` is the reference for a "one image + `--model` flag,
HF-hosted, auto-generated API key" engine; `lib/engines/nim.sh` is the
reference for the opposite shape — one image per model, an external
registry API key that must never be auto-generated, and no
context-length/GPU-memory knobs at all.

## Troubleshooting / Appendix

### Prerequisites

This assumes a genuinely clean box — nothing pre-set-up, nothing assumed
except what typically ships on a DGX Spark / DGX OS image:

| Tool | Usually already present on DGX OS? | If missing |
|---|---|---|
| `docker` + NVIDIA Container Toolkit | Yes | See [NVIDIA Container Toolkit install docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| `python3` / `pip3` | Yes | `sudo apt install python3 python3-pip` |
| `git` | Yes | `sudo apt install git` |
| `hf` (HuggingFace CLI) | **No — dgxt offers to install it for you** | `pip3 install huggingface_hub`, or just say yes when prompted (see below) |

dgxt checks for Docker and `hf` at runtime and tells you exactly what's
missing. For `hf`, it goes one step further: if it's missing and you're
running interactively, dgxt asks whether to install it via pip right then
(it never installs anything silently/non-interactively).

### Installing the `hf` CLI

Not included with the OS. Any command that needs it (`setup`, `search`,
`model-pull`, `model-list`, or `start` before a model is cached) will
detect it's missing and prompt:

```
HuggingFace CLI ('hf') not found on PATH.
Install it now with pip3? [Y/n]
```

Answer yes and dgxt installs it for you (retrying with
`--break-system-packages` automatically if Debian/Ubuntu's PEP 668
protection blocks the plain install). To install it yourself instead:

```bash
pip3 install huggingface_hub
# or, if pip refuses with an externally-managed-environment error:
pip3 install --break-system-packages huggingface_hub
```

Then confirm it's on your `PATH`:

```bash
hf --help
```

(If `hf` isn't found afterward, open a new shell — pip installs to
`~/.local/bin`, which needs a fresh `PATH` to pick up.)

### Docker group membership

If `docker ps` gives a permission error, you're not (yet) in the `docker`
group:

```bash
sudo usermod -aG docker "$USER"
newgrp docker      # or just log out and back in
```

### Context extension past a model's native max

**Default context (no flag/config value set)**: dgxt aims for 1M tokens,
capped to whatever the model actually natively supports (checked against
the model's own `config.json`) — Nemotron gets its full 1M window
automatically, while a shorter-context model safely falls back to its own
real ceiling instead of silently over-requesting.

For Qwen3.8, use the model selection directly; the RadixArk checkpoint
automatically enables the separately trained DSpark drafter:

```bash
dgxt restart RadixArk/Qwen3.8-27B-NVFP4
```

The default matching draft model is
`RadixArk/Qwen3.8-27B-DSpark`, with its recommended seven-token draft block.
To explicitly override that default, set
`VLLM_SPECULATIVE_MODE=dspark VLLM_SPECULATIVE_TOKENS=7`. This is a
single-GPU, low-concurrency configuration; keep `VLLM_GPU_MEM` at or below
`0.8` initially on GB10 because the CPU and GPU share the same 128 GB
unified-memory pool. Use `VLLM_SPECULATIVE_MODE=mtp` to select the target
checkpoint's native MTP head, or `none` for a baseline.

To opt into the model's documented 1M YaRN extension, add
`--max-context 1m` to the command above. dgxt preserves the model's
multimodal RoPE fields and nests the override under `text_config`; using a
partial top-level override will not correctly extend this checkpoint. dgxt
does not silently apply YaRN to models whose native context is shorter than
1M, because static YaRN can reduce short-context quality. The forum
measurements used about `0.60` GPU memory utilization for 1M; the default
`0.8` leaves more KV capacity but less unified-memory headroom, so lower
`VLLM_GPU_MEM` if other workloads share the machine.

The vLLM engine uses the ARM64 `vllm/vllm-openai:nightly` image because the
stable v0.28.0 image misroutes RadixArk's `DSparkDraftModel` architecture.
The nightly image includes the upstream native routing fix.

**Explicitly requesting more than a model's native max**: dgxt warns and
asks before starting, since exceeding it risks incorrect output or CUDA
errors. On confirmation:
- **vLLM**: genuinely extends context via YaRN RoPE scaling
  (`--hf-overrides`), not just a bypass flag — the factor/original-max are
  auto-computed from the model's native max, and `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`
  is also set. Override with `VLLM_ROPE_SCALING_FACTOR` /
  `VLLM_ROPE_SCALING_ORIGINAL_MAX` if you know a better-validated value for
  a specific model (e.g. Qwen's own docs recommend `32768` as the YaRN base
  for Qwen3 dense models, not their extended `config.json` default).
- **NIM**: no equivalent exists. Many DGX-Spark NIM containers ship a
  precompiled, hardware-optimized engine with context baked in at
  NVIDIA's build time — `NIM_MAX_MODEL_LEN` only has any effect at all on
  locally-*buildable* TensorRT-LLM profiles (per NVIDIA's own docs), and
  even then there's no RoPE-scaling override available. dgxt tells you
  this plainly rather than pretending the bypass fixed anything; switch to
  vLLM (`dgxt engine vllm`) if you need genuine context extension past a
  model's native max.

### hf / NGC passthrough details

All three convenience commands (`search`, `model-pull`, `model-list`)
work for NIM too, just against NGC instead of HF: `search` queries NGC's
public catalog search API directly (no NGC CLI/API key needed just to
browse) and flags which results are confirmed ARM64/DGX-Spark-native vs.
likely x86_64-only; `model-pull` pre-pulls the Docker image (not the
model weights inside it — those download from NGC into `~/.cache/nim` on
first container start, NIM exposes no separate step for that);
`model-list` shows locally pulled NIM images instead of the HF cache.

```bash
./dgxt search qwen   # (with DGXT_ENGINE=nim)
#   nvcr.io/nim/qwen/qwen3-32b-dgx-spark:latest  [DGX-Spark/ARM64-native]
#       Qwen3-32B NIM for DGX Spark
#   nvcr.io/nim/qwen/qwen-2.5-72b-instruct:latest
#       qwen-2.5-72b-instruct                     <- likely x86_64-only
```

`model-pull` also excludes each repo's `original/` and `metal/` folders
by default (unless you pass your own `--include`/`--exclude`) — vLLM
never reads these, but `hf download` grabs them anyway if unfiltered.
`openai/gpt-oss-120b` is the case that matters most: ~63GB of MXFP4
safetensors vLLM actually loads vs. 100GB+ of `original/` bf16 reference
weights it doesn't.

### Full config file examples

```bash
# ~/.dgxtrc with DGXT_ENGINE=vllm
DGXT_ENGINE=vllm
VLLM_MODEL=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
VLLM_MAX_MODEL_LEN=1m
VLLM_API_KEY=<random, generated for you>
VLLM_PORT=8000
VLLM_GPU_MEM=0.85
VLLM_SPECULATIVE_MODE=dspark
VLLM_SPECULATIVE_TOKENS=3
HF_TOKEN=hf_...
```

NIM uses a different set of keys (no GPU-memory knob — each container
auto-tunes GPU memory itself; it does support a context-length override,
since NIM's auto-selected profile isn't always the model's full native
context):

```bash
# ~/.dgxtrc with DGXT_ENGINE=nim
DGXT_ENGINE=nim
NIM_IMAGE=nvcr.io/nim/meta/llama-3.1-8b-instruct-dgx-spark:latest
NGC_API_KEY=<real key from org.ngc.nvidia.com/account/api-keys, with 'NGC Catalog' permission checked>
NIM_PORT=8000
# NIM_MAX_MODEL_LEN=64k   # optional -- unset lets NIM's own profile decide
```

### Common errors

**"Docker not accessible"** — you're not in the `docker` group yet, or
just joined it and haven't started a new shell. See
[Docker group membership](#docker-group-membership) above.

**Cache directory not writable** — usually means a container (running as
root inside Docker) wrote to your HuggingFace cache directory before you
did, leaving it root-owned. Fix:

```bash
sudo chown -R "$(whoami)":"$(whoami)" ~/.cache/huggingface
```

dgxt creates this directory itself (as your user) before every `docker
run`, specifically to prevent this from happening on a fresh box — but it
can still occur if you ran the container manually outside of dgxt first.

**Gated model downloads failing** — you need to be logged in and have
accepted the model's license on huggingface.co:

```bash
hf auth login
```
