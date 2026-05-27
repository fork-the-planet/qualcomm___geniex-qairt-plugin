# auto_llm

Demonstrates loading a basic LLM (anything representable by `LLMModel`)
entirely at runtime, with no model name or model spec hardcoded into the
caller's code. Everything the runtime needs comes from files in the bundle
directory.

## What this example shows

A "basic LLM" today still requires the caller to write a per-family file
(`models/qwen3/qwen3.h`, `models/llama3/llama3.h`, …) that picks the right
chat template formatter and the right input providers. This example shows
that's no longer necessary for any LLM that:

- Can be represented by `LLMModel` (no custom decode loop, prefill flow,
  or KV management).
- Uses the same two CPU-side input providers — embedding + RoPE — that
  every existing LLM family already uses.

The two blockers historically preventing a generic loader were:

1. **Chat template** had to be a `ChatTemplateFunc` bound at pipeline
   creation. The `Pipeline` class in [`auto_llm.h`](./auto_llm.h) lifts
   that constraint by reading the Jinja chat template from the bundle's
   `tokenizer_config.json` at runtime, via
   `geniex-proc::Tokenizer::apply_chat_template`.
2. **Input providers** had to be added by family-specific code. This
   example assumes the embedding + RoPE pair, sourced from the bundle's
   metadata via `geniex::makeEmbeddingProvider` /
   `geniex::makeRoPEProvider`. No other LLM provider exists in the
   codebase today, so the assumption holds for every LLM family.

The result is one entry point that runs any text-only QAIRT LLM bundle:

```cpp
auto pipe = geniex::auto_llm::makePipeline(runtime_cfg, model_cfg);
auto out  = pipe->generateChat(messages, gen_cfg);
```

No model name in the caller's code, no model spec, no template binding.

## Bundle layout

`--model-dir` must point at a QAIRT export bundle containing:

| File                             | Required | Purpose                                  |
|----------------------------------|----------|------------------------------------------|
| `metadata.json`                  | yes      | Tensor shapes, shard wiring, model_id    |
| `genie_config.json`              | yes      | Dialog type, BOS/EOS, RoPE config        |
| `tokenizer.json`                 | yes      | HuggingFace fast tokenizer               |
| `tokenizer_config.json`          | yes      | Chat template (Jinja) + bos/eos strings  |
| `*.bin`                          | yes      | Compiled context-binary shards           |
| `htp_backend_ext_config.json`    | optional | HTP backend extensions                   |
| `embedding_weights.raw` / `embed_tokens.npy` | optional | CPU-side embedding LUT (for bundles that move embedding off-graph) |

This is the standard layout produced by AI Hub's QAIRT export today.

## Build

From the `geniex-qairt` repo root:

```pwsh
cmake -S . -B build -DGENIEX_BUILD_EXAMPLES=ON
cmake --build build --target auto_llm --config Release
```

The binary lands at `build\bin\Release\auto_llm.exe`.

## Run

```pwsh
.\build\bin\Release\auto_llm.exe `
    --model-dir <path>\<to>\<model_dir> `
    --max-tokens 256 `
    --verbose
```

Multi-turn REPL: type a prompt, the model streams the reply, repeat. Type
`exit` / `quit` or send EOF to leave.

### Flags

| Flag | Description |
|------|-------------|
| `--model-dir <path>`        | **Required.** Bundle directory. |
| `--tokenizer-config <path>` | Override path to `tokenizer_config.json` (default: `<model-dir>/tokenizer_config.json`). |
| `--system <text>`           | System prompt, applied once at startup. |
| `--max-tokens <n>`          | Max tokens generated per turn (default 512). |
| `--enable-thinking`         | Plumbs `{"enable_thinking":true}` into the Jinja context for reasoning models that read the field (Qwen3). No-op on templates that don't. |
| `--verbose`                 | Print TTFT / TPS metrics each turn. |
