---
license: apache-2.0
base_model:
- RadixArk/Qwen3.8-Flash-Next-NVFP4
library_name: vllm
pipeline_tag: text-generation
tags:
- qwen3.8
- dgx-spark
- gb10
- long-context
- 1m-context
- nvfp4
- vllm
- speculative-decoding
---

# One million tokens on one 128 GB DGX Spark

On an ASUS GX10, this Qwen3.8-Flash-Next setup handled a 989,734-token prompt,
returned all five buried values, and generated 67 more tokens. Short prompts
decoded at a 26.7 tok/s median, and the same profile scored 92.7% on HumanEval+
Mini.

The upstream
[Qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)
project got this model to 500K on the same class of machine. Its original memory
policy ran out of room before 1M. I kept the model and its mmap-based PLE
offload, then changed how those mapped pages are handled under memory pressure.

The PLE lookup table stays memory-mapped on NVMe. Linux is told to expect random
access, and clean PLE pages are released before they squeeze the growing KV
cache. That was enough to serve a near-limit request while keeping native MTP
speculative decoding enabled.

> This directory contains serving code, **not model
> weights**. The code is Apache-2.0. The Qwen/RadixArk checkpoint has separate
> terms; review the source model card before use.

## Measured on the GX10

| Test | Result |
|---|---:|
| Advertised context | 1,000,000 tokens |
| Allocated KV capacity | 1,095,163 tokens |
| Validated request | 989,734 prompt + 67 output = **989,801 tokens** |
| Distributed needle retrieval | **5/5** at 5%, 25%, 50%, 75%, 95% |
| Median decode, five coding languages | **26.712 tok/s** |
| Minimum decode | **26.630 tok/s** |
| Median TTFT | **0.257 s** |
| HumanEval | **156/164 (95.1%)** |
| HumanEval+ Mini | **152/164 (92.7%)** |
| Minimum available memory during 989K prefill | **3.512 GiB** |
| Swap-free change during 989K prefill | **-0.010 GiB** |

Hardware: NVIDIA GX10 / GB10 with 121.63 GiB usable unified memory. The model
service was the only large workload on the box.

## Quickstart

Requirements:

- DGX Spark, ASUS GX10, or compatible GB10 system with 128 GB unified memory
- NVIDIA container runtime and Docker
- about 130 GB of fast local storage; NVMe is strongly recommended
- roughly 10 minutes for a cold model load on the tested machine

Within `dgxt`, `qwen3.8-flash-next`, `qwen38-flash-next-v029`, and the full
checkpoint ID all select the released vLLM 0.29 profile. The API advertises
the short model name instead of the full Hugging Face repository ID.

The v0.29 profile builds
`vllm/vllm-openai:v0.29.0` with the re-targeted PLE mmap hook, prefix-cache
block-size fix, GB10 FLA/top-k fixes, and reduced MTP draft vocabulary:

```bash
dgxt setup
# select: qwen38-flash-next-v029
dgxt start qwen38-flash-next-v029 --max-context 512k
```

The v0.29 profile defaults to the published NVFP4 checkpoint layout rather
than the optional hybrid checkpoint, keeps KV cache dtype on `auto`, and uses
PIECEWISE CUDA graphs. The 500K profile is the recommended starting point;
768K and 1M remain experimental on a single 128 GB GB10/GX10.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-flash-next",
    "messages": [{"role":"user","content":"Write a lock-free ring buffer in Rust."}],
    "temperature": 0,
    "max_tokens": 1024,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

## Where the extra memory came from

Flash-Next has an enormous n-gram/PLE lookup table. The upstream recipe already
memory-maps that table instead of permanently loading it beside the GPU
weights.

At ultra-long context, Linux's page cache becomes the next bottleneck. PLE rows
are hash-selected, so normal sequential readahead is mostly waste. Meanwhile,
every cached PLE page competes with a KV cache that is growing toward 30 GiB.

The runtime integration adds three controls:

1. `MADV_RANDOM` disables inappropriate sequential readahead.
2. Below a memory watermark, `MADV_DONTNEED` and `POSIX_FADV_DONTNEED` release
   clean PLE pages that can always be read again from NVMe.
3. Trimming only runs after large gathers, not during token-by-token decode.

## Operational limits

- **Prompt plus output must remain below one million.** The validated 989,801-
  token request leaves 10,199 tokens of headroom.
- **This is a single-sequence profile.** It is optimized for one huge context,
  not aggregate multi-user throughput.
- **Do not co-locate another large model.** CPU, page cache, GPU allocations, and
  KV all share the same 128 GB pool.
- **The PLE table stays on fast local storage.** Slow network storage will hurt
  page-fault latency.
- **The checkpoint is public but tagged as a candidate export.** Pin the tested
  revision instead of silently following future repository updates.
- Multimodal inputs were not part of this validation.

## Credits

- Qwen team / Alibaba for Qwen3.8-Flash-Next.
- [RadixArk](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) for the
  NVFP4 checkpoint.
- [blazux](https://github.com/blazux/qwen3.8-Flash-DGX) for discovering and
  implementing the mmap-PLE approach that made Flash-Next practical on GB10.
- [jschmied](https://github.com/jschmied/qwen38-flash-next-gb10) for independent
  upstream reproduction and concurrency/offload investigation.
- vLLM and NVIDIA Model Optimizer for the serving and quantization stack.

If you try this on another Spark, please open an issue with the KV capacity,
minimum `MemAvailable`, storage model, and result JSON. I would especially like
to see results from other NVMe drives and OEM GB10 systems.
