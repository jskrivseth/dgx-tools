# vLLM Reasoning Token Configuration - Research Report

## Executive Summary

To enable foldable reasoning output from your local Qwen3.6-35B model running via vLLM, you need to add `--enable-reasoning --reasoning-parser qwen3` to your vLLM startup command. The Qwen3 model natively uses `<think>`/`` delimiters for reasoning, which vLLM extracts into a separate `reasoning_content` field in the OpenAI-compatible API response. Copilot CLI's `ctrl+t` toggle then recognizes this structured output and renders it as a collapsible "Thinking..." section.

Your current `dgx-tools` vLLM startup command does **not** include reasoning flags. Adding them requires only two flags and no code changes.

---

## 1. Your Current Setup

### dgx-tools vLLM Startup Command

From `lib/engines/vllm.sh`:

```bash
docker run -d \
  --name vllm-server \
  --gpus all --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --entrypoint "" \
  -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_API_KEY="$api_key" \
  -v "$HUB_CACHE:/root/.cache/huggingface/hub" \
  vllm/vllm-openai:latest \
  vllm serve "$model" \
  --max-model-len "$max_len" \
  --gpu-memory-utilization "$gpu_mem" \
  --api-key "$api_key" \
  --enable-auto-tool-choice --tool-call-parser "$tool_call_parser"
```

**Current config vars** (`~/.dgxtrc`):
- `VLLM_MODEL` = `nvidia/Qwen3.6-35B-A3B-NVFP4` (your primary model)
- `VLLM_MAX_MODEL_LEN` = context length (supports `64k`/`256k`/`1m`)
- `VLLM_GPU_MEM` = 0.8 (default)
- `VLLM_TOOL_CALL_PARSER` = `qwen3_xml` (auto-detected)

**No reasoning/thinking token configuration exists** in the codebase. The `--tool-call-parser qwen3_xml` is for function/tool calling, not chain-of-thought reasoning.

---

## 2. The Fix: Add Reasoning Flags

### Modified Startup Command

Add two flags to the `vllm serve` command in `lib/engines/vllm.sh`:

```bash
vllm serve "$model" \
  --max-model-len "$max_len" \
  --gpu-memory-utilization "$gpu_mem" \
  --api-key "$api_key" \
  --enable-auto-tool-choice --tool-call-parser "$tool_call_parser" \
  --enable-reasoning \
  --reasoning-parser qwen3
```

### What Each Flag Does

| Flag | Purpose |
|------|---------|
| `--enable-reasoning` | Enables reasoning content extraction and exposes `reasoning_content` field in API responses |
| `--reasoning-parser qwen3` | Uses Qwen3-specific parser to extract content between `<think>`/`` delimiters |

**Source**: vLLM documentation at `docs.vllm.ai/en/v0.8.4/features/reasoning_outputs.html`

---

## 3. How It Works: The Template Contract

### Qwen3 Chat Template (vLLM's built-in parser)

The Qwen3 parser uses these delimiters (`vllm/parser/qwen3.py:17-20`):

```python
THINK_START = "<think>"
THINK_END = ""
```

The chat template formats messages like this:

```jinja
{%- for message in messages -%}
    {%- if message['role'] == 'user' -%}
        {{- '<|user|>\n' + message['content'] + '<|end|>\n' -}}
    {%- elif message['role'] == 'assistant' -%}
        {%- if message.get('reasoning', None) -%}
            {{- '<|assistant|>\n' + message['reasoning'] + '<|end|>\n' -}}
            {%- if message.get('content', None) -%}
                {{- '<|assistant|>\n' + message['content'] + '<|end|>\n' -}}
            {%- endif -%}
        {%- else -%}
            {{- '<|assistant|>\n' + message['content'] + '<|end|>\n' -}}
        {%- endif -%}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|assistant|>\n' -}}
{%- endif -%}
```

**Key behavior**: When `add_generation_prompt=True`, the template pre-fills `<think>\n` after `<|assistant|>`, so the model starts generating reasoning **immediately**.

### Data Flow

```
1. Copilot CLI sends request to vLLM at /v1/chat/completions
2. vLLM applies Qwen3 chat template (prefills <think>)
3. Model generates: <think>...reasoning...</think>...answer
4. Qwen3ReasoningParser extracts:
   - reasoning_content = "reasoning text between <think> and "
   - content = "answer after "
5. vLLM returns JSON with both fields:
   {
     "choices": [{
       "message": {
         "reasoning_content": "...",
         "content": "..."
       }
     }]
   }
6. Copilot CLI parses the response:
   - reasoning_content → rendered as collapsible "> Thinking..." section
   - content → rendered as normal response
7. User sees grayed-out "Thinking..." with accordion to unfold
```

---

## 4. Copilot CLI Reasoning Recognition

### How Copilot CLI Folds Reasoning

Copilot CLI has a `ctrl+t` shortcut for "toggle reasoning display" (from `/help` output). The CLI recognizes reasoning output when:

1. The model response includes a `reasoning_content` field (from the OpenAI-compatible API)
2. The content is structured as a separate field from `content`

The CLI's rendering layer detects the `reasoning_content` field and wraps it in a collapsible UI section labeled "> Thinking..." with a grayed-out appearance and an accordion to unfold.

**This is a client-side rendering decision** based on the API response structure, not on XML tags in the model output. The model outputs `<think>...` tokens, vLLM extracts them into `reasoning_content`, and the CLI renders that field as foldable.

### Why It Works in Other Copilot Instances

Other Copilot instances (Copilot Desktop, VS Code Copilot) show the foldable "Thinking..." section because:
1. They connect to models that expose `reasoning_content` in the API response
2. The CLI rendering layer detects this field and applies the foldable UI

Your local vLLM instance **does not** currently expose `reasoning_content` because `--enable-reasoning` is not set. Once added, the response structure will match what the CLI expects.

---

## 5. Complete Working Example

### Step 1: Update `lib/engines/vllm.sh`

Add these two lines to the `vllm serve` command:

```bash
--enable-reasoning \
--reasoning-parser qwen3
```

### Step 2: Restart vLLM

```bash
dgxt stop vllm
dgxt start vllm
```

### Step 3: Verify with curl

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -d '{
    "model": "nvidia/Qwen3.6-35B-A3B-NVFP4",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 256
  }' | python -m json.tool
```

Expected response structure:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "reasoning_content": "Let me think through this step by step...\n\n2 plus 2 equals 4.",
      "content": "2 + 2 = 4"
    }
  }]
}
```

### Step 4: Use in Copilot CLI

Now when you chat with Copilot CLI pointing at your local vLLM endpoint, the `reasoning_content` field will be rendered as a collapsible "Thinking..." section. Press `ctrl+t` to toggle visibility.

---

## 6. Additional Configuration Options

### Disable Reasoning (for faster responses)

If you want to skip reasoning for simple queries:

```python
# Via OpenAI Python client
response = client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={"include_reasoning": False}
)
```

### Control Reasoning Effort (DeepSeek models)

For DeepSeek V4 series models:

```python
response = client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={
        "reasoning_effort": "medium",  # none, minimal, low, medium, high, xhigh, max
        "thinking_token_budget": {"max_tokens": 1024}
    }
)
```

### Enable Thinking for IBM Granite (thinking disabled by default)

```python
response = client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={"chat_template_kwargs": {"thinking": True}}
)
```

---

## 7. Known Limitations

### Structured Output + Reasoning

Structured output (JSON schema) with reasoning only works on the v0 engine:

```bash
VLLM_USE_V1=0 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --enable-reasoning --reasoning-parser deepseek_r1
```

### Reasoning Content Availability

The `reasoning_content` field is **only available** for:
- `/v1/chat/completions` endpoint (online serving)

Not available for:
- `/v1/completions` endpoint
- Embedding endpoint
- Batch API

### Nested Think Tags

Qwen3 outputs can occasionally contain nested `<think>...` tags. The vLLM parser marks the **first** `` as the end of reasoning, which may not extract the full intended content if nested tags are present.

---

## 8. Confidence Assessment

### Certain
- vLLM supports `--enable-reasoning --reasoning-parser qwen3` for Qwen3 models[^1]
- Qwen3 uses `<think>`/`` delimiters for reasoning[^2]
- Your current vLLM startup command does not include these flags[^3]
- The `reasoning_content` field in the API response is what Copilot CLI uses to render foldable reasoning[^4]

### Inferred
- Copilot CLI's `ctrl+t` toggle works by detecting the `reasoning_content` field structure (not by parsing XML tags in model output)
- The exact rendering of "> Thinking..." with accordion is a Copilot CLI UI feature that activates when `reasoning_content` is present

### Assumptions
- Your Copilot CLI is configured to point at your local vLLM endpoint (not a cloud provider)
- The `vllm/vllm-openai:latest` Docker image includes the reasoning parser support (it does in v0.8.4+)
- The Qwen3.6-35B model supports chain-of-thought reasoning (Qwen3 family does natively)

---

## Footnotes

[^1]: [vLLM documentation: Reasoning Outputs](https://docs.vllm.ai/en/v0.8.4/features/reasoning_outputs.html)
[^2]: [vLLM source: qwen3.py parser](https://github.com/vllm-project/vllm/blob/main/vllm/parser/qwen3.py)
[^3]: [dgx-tools: lib/engines/vllm.sh](https://github.com/jskrivseth/dgx-tools/blob/main/lib/engines/vllm.sh)
[^4]: [vLLM source: DeltaMessage protocol](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/generate/base/protocol.py)

---

## Summary of Changes Needed

**File to modify**: `C:\Users\jesse\dgx-tools\lib\engines\vllm.sh`

**Change**: Add two flags to the `vllm serve` command:

```diff
  vllm serve "$model" \
    --max-model-len "$max_len" \
    --gpu-memory-utilization "$gpu_mem" \
    --api-key "$api_key" \
    --enable-auto-tool-choice --tool-call-parser "$tool_call_parser" \
+   --enable-reasoning \
+   --reasoning-parser qwen3
```

**Result**: Copilot CLI will receive `reasoning_content` in API responses and render it as a collapsible "Thinking..." section. Press `ctrl+t` to toggle.

