"""Quick vLLM API benchmark — tokens/sec for prefill (decode) and latency."""
import time
import requests

BASE = "http://localhost:8000/v1/chat/completions"
KEY = "e3362985877cfe981bc5406b5059d3b791287117e14dc862"
HEADERS = {
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json",
}

def make_prompt_tokens(num_tokens: int) -> list[int]:
    return list(range(1, num_tokens + 1))  # dummy token IDs

results = []

# ---- Latency test (short prompt, measure per-token decode) ----
print("=" * 70)
print("LATENCY TEST: short prompt → 256 output tokens")
print("=" * 70)

prompt_short = make_prompt_tokens(64)
resp_short = requests.post(
    BASE, json={
        "model": "Qwen/Qwen3.6-35B-A3B",
        "messages": [{"role": "user", "content": "test"}],
        "prompt_token_ids": prompt_short,
        "max_tokens": 256,
        "temperature": 0,
        "stream": False,
    },
    headers=HEADERS, timeout=120,
).json()

latency_total = resp_short.get("usage", {}).get("total_time_ms", resp_short.get("usage", {}).get("completion_time_ms", 0))
prompt_tokens = resp_short.get("usage", {}).get("prompt_tokens", 64)
completion_tokens = resp_short.get("usage", {}).get("completion_tokens", 256)
latency_decode_ms = resp_short.get("usage", {}).get("completion_time_ms", 0)

print(f"  Total latency:        {latency_total/1000:.2f}s")
print(f"  Prefill latency:      {latency_total/1000 - latency_decode_ms/1000:.2f}s")
print(f"  Decode latency:       {latency_decode_ms/1000:.2f}s")
print(f"  Prefill tokens/sec:   {prompt_tokens / (latency_total/1000 - latency_decode_ms/1000):.1f}" if latency_total != latency_decode_ms else "  Prefill tokens/sec:   N/A")
print(f"  Decode tokens/sec:    {completion_tokens / (latency_decode_ms/1000):.1f}")
print(f"  Output tokens:        {completion_tokens}")

# ---- Throughput tests at different context lengths ----
print()
print("=" * 70)
print("THROUGHPUT TESTS: varying context lengths → 512 output tokens")
print("=" * 70)

for ctx_tokens in [1_000, 4_000, 16_000, 64_000]:
    prompt_toks = make_prompt_tokens(ctx_tokens)
    t0 = time.monotonic()
    r = requests.post(
        BASE, json={
            "model": "Qwen/Qwen3.6-35B-A3B",
            "messages": [{"role": "user", "content": "test"}],
            "prompt_token_ids": prompt_toks,
            "max_tokens": 512,
            "temperature": 0,
            "stream": False,
        },
        headers=HEADERS, timeout=300,
    )
    elapsed = time.monotonic() - t0
    resp = r.json()
    usage = resp.get("usage", {})
    pt = usage.get("prompt_tokens", ctx_tokens)
    ct = usage.get("completion_tokens", 512)
    prefill_ms = usage.get("prefill_time_ms", 0)
    decode_ms = usage.get("completion_time_ms", 0)

    prefill_tok_s = pt / (prefill_ms / 1000) if prefill_ms > 0 else 0
    decode_tok_s = ct / (decode_ms / 1000) if decode_ms > 0 else 0

    print(f"  Context {ctx_tokens:>7,}: prefill={prefill_ms/1000:.2f}s ({prefill_tok_s:>7.0f} tok/s)  decode={decode_ms/1000:.2f}s ({decode_tok_s:>6.0f} tok/s)  total={elapsed:.1f}s")
    results.append({"ctx": ctx_tokens, "pt": pt, "ct": ct, "prefill_ms": prefill_ms, "decode_ms": decode_ms, "total_s": round(elapsed, 2)})

# ---- Summary table ----
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  {'Context':>10}  {'Prefill':>8}  {'Prefill/s':>10}  {'Decode':>8}  {'Decode/s':>10}  {'Total':>6}")
print(f"  {'tokens':>10}  {'(ms)':>8}  {'(tokens)':>10}  {'(ms)':>8}  {'(tokens)':>10}  {'(s)':>6}")
for r in results:
    print(f"  {r['ctx']:>10,}  {r['prefill_ms']:>8.0f}  {r['pt']/(r['prefill_ms']/1000) if r['prefill_ms']>0 else 0:>10.0f}  {r['decode_ms']:>8.0f}  {r['ct']/(r['decode_ms']/1000) if r['decode_ms']>0 else 0:>10.0f}  {r['total_s']:>6.1f}")

print()
print(f"GPU: GB10 (Blackwell)")
print(f"Model: Qwen/Qwen3.6-35B-A3B (MoE, ~3B activated)")
print(f"max_model_len: 262,144")
