# auto_llm

Family-free LLM example. Runs any text-only QAIRT bundle without a per-model
header — every numeric hyperparameter comes from `metadata.json`, every
runtime knob from `genie_config.json`, and the chat template from
`tokenizer_config.json` via `geniex-proc::Tokenizer::apply_chat_template`.

The encapsulation in `auto_llm.h` (`geniex::auto_llm::makeModel`,
`geniex::auto_llm::makePipeline`, `geniex::auto_llm::Pipeline`) mirrors the
shape of the per-family files (`models/qwen3/qwen3.h` etc.) but with no
hardcoded `LLMSpec`, no `ChatTemplateFunc` binding, and full message-history
chat-template rendering. It's a sketch of where those per-family files can
converge once the runtime stops needing model-specific specs.

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

## Usage

```pwsh
.\build\bin\Release\auto_llm.exe `
    --model-dir <path>\modelfiles\qwen3_4b `
    --max-tokens 256 `
    --verbose
```

Flags:

| Flag | Description |
|------|-------------|
| `--model-dir <path>`        | **Required.** Bundle directory. |
| `--tokenizer-config <path>` | Override path to `tokenizer_config.json` (default: `<model-dir>/tokenizer_config.json`). |
| `--system <text>`           | System prompt, applied once at startup. |
| `--max-tokens <n>`          | Max tokens generated per turn (default 512). |
| `--enable-thinking`         | Plumbs `{"enable_thinking":true}` into the Jinja context for reasoning models that read the field (Qwen3). No-op on templates that don't. |
| `--verbose`                 | Print TTFT / TPS metrics each turn. |

## Multi-turn

The pipeline keeps the model's KV cache aligned with the conversation
history; the example never calls `pipe.reset()` between successful turns.
On a generation failure, the user turn that triggered it is dropped and the
KV cache is reset so the next turn starts clean.

## Composition compared to per-family files

```cpp
// Per-family (e.g. models/qwen3/qwen3.h):
auto pipe = qwen3::makePipeline(runtime_cfg, model_cfg);
pipe->setSystemPrompt("You are helpful.");          // first turn only
auto prompt = pipe->applyChatTemplate(user_text);   // last user msg only
auto result = pipe->generate(prompt, gen_cfg);

// Generic (auto_llm.h):
auto pipe = geniex::auto_llm::makePipeline(runtime_cfg, model_cfg);
auto result = pipe->generateChat(messages, gen_cfg, opts);
//                                ^^^^^^^^
//                                Full history, every turn.
```

The per-family path passes only the latest user message into a stateful
`ChatTemplateFunc`. The generic path passes the full message vector into a
Jinja template so multi-turn details (Qwen3 thinking-tag stripping, tool
call / tool response interleaving, reasoning content carryover) work as the
upstream model card specifies.
