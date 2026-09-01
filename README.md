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
256K tokens capped to whatever the model actually natively supports.
Asking for more than a model's native max works too — see
[Context extension](#context-extension-past-a-models-native-max) in the
appendix for how (and where it can't).

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
VLLM_MODEL=nvidia/Qwen3.6-35B-A3B-NVFP4
VLLM_MAX_MODEL_LEN=131072
VLLM_API_KEY=<random, generated for you>
VLLM_PORT=8000
VLLM_GPU_MEM=0.8
HF_TOKEN=hf_...
```

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

**Default context (no flag/config value set)**: dgxt aims for 256K tokens,
capped to whatever the model actually natively supports (checked against
the model's own `config.json`) — a long-context model gets the full 256K
automatically, while a shorter-context model safely falls back to its own
real ceiling instead of silently over-requesting.

**Explicitly requesting more than a model's native max**: dgxt warns and
asks before starting, since exceeding it risks incorrect output or CUDA
errors. On confirmation:
- **vLLM**: genuinely extends context via YaRN RoPE scaling
  (`--rope-scaling`), not just a bypass flag — the factor/original-max are
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
VLLM_MODEL=nvidia/Qwen3.6-35B-A3B-NVFP4
VLLM_MAX_MODEL_LEN=131072
VLLM_API_KEY=<random, generated for you>
VLLM_PORT=8000
VLLM_GPU_MEM=0.8
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
