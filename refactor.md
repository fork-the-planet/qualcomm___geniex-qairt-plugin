# Pipeline Interface Changes — Migration Guide

This document describes breaking interface changes to `LLMPipeline` and the chat template system. Downstream consumers must update their calling patterns accordingly.

---

## 1. `LLMPipeline::create()` — new signature

**Before:**
```cpp
bool create(const std::string& model_name,
            LLMModel model,
            const QnnRuntimeConfig& runtime_cfg,
            const ModelConfig& model_cfg);
```

**After:**
```cpp
bool create(ChatTemplateFunc chat_template,
            LLMModel model,
            const QnnRuntimeConfig& runtime_cfg,
            const ModelConfig& model_cfg);
```

The first parameter changed from a model name string (e.g. `"qwen3"`) to a `ChatTemplateFunc` function pointer. The internal `findChatTemplate()` lookup table has been removed.

**Migration:** Replace the model name string with the appropriate chat template function:

| Old `model_name` | New `ChatTemplateFunc` | Defined in |
|---|---|---|
| `"qwen3"`, `"qwen2.5"` | `chatMLTemplate` | `pipeline/chat_template.h` (core) |
| `"phi4"`, `"phi3.5"` | `phiChatTemplate` | `pipeline/chat_template.h` (core) |
| `"granite4"` | `granite4ChatTemplate` | `models/granite4/granite4.h` |
| `"llama3"`, `"llama3.1"`, `"llama3.2"` | `llama3ChatTemplate` | `models/llama3/llama3.h` |
| *(Falcon3)* | `falcon3ChatTemplate` | `models/falcon3/falcon3.h` |

Example:
```cpp
// Before:
pipe.create("qwen3", qwen3_4b::makeModel(), runtime_cfg, model_cfg);

// After:
pipe.create(chatMLTemplate, qwen3_4b::makeModel(), runtime_cfg, model_cfg);
// — or use the model namespace's chatTemplate constant:
pipe.create(qwen3_4b::chatTemplate, qwen3_4b::makeModel(), runtime_cfg, model_cfg);
```

---

## 2. `isReady()` — new method

```cpp
bool isReady() const;
```

Returns `true` after a successful `create()`, `false` otherwise. Use this for diagnostics or when constructing a pipeline via the manual `create()` path.

---

## 3. `makePipeline()` — now returns `std::optional<LLMPipeline>`

**Before:**
```cpp
inline LLMPipeline makePipeline(const QnnRuntimeConfig& runtime_cfg,
                                const ModelConfig& model_cfg);
```

**After:**
```cpp
inline std::optional<LLMPipeline> makePipeline(const QnnRuntimeConfig& runtime_cfg,
                                               const ModelConfig& model_cfg);
```

Returns `std::nullopt` if pipeline creation fails (QNN init failure, missing tokenizer, etc.). This makes failure unignorable at the type level.

**Migration:**
```cpp
// Before:
auto pipe = qwen3_4b::makePipeline(runtime_cfg, model_cfg);
pipe.setSystemPrompt("You are a helpful assistant.");
pipe.generate(...);

// After:
auto pipe = qwen3_4b::makePipeline(runtime_cfg, model_cfg);
if (!pipe) {
    std::cerr << "Failed to create pipeline.\n";
    return 1;
}
pipe->setSystemPrompt("You are a helpful assistant.");
pipe->generate(...);
```

Note: all member access changes from `pipe.` to `pipe->` when using the factory.

---

## 4. Chat template reorganization

### Common templates (in `core/pipeline/chat_template.h`)

These are shared across multiple model families and remain in core:

- `chatMLTemplate` — ChatML format (`<|im_start|>...<|im_end|>`)
- `phiChatTemplate` — Phi format (`<|system|>...<|end|>`) *(renamed from `phiTemplate`)*

### Model-specific templates (moved to model headers)

These are now `inline` functions in their respective model headers:

- `granite4ChatTemplate` — in `models/granite4/granite4.h`
- `llama3ChatTemplate` — in `models/llama3/llama3.h`
- `falcon3ChatTemplate` — in `models/falcon3/falcon3.h`

### Removed

- `findChatTemplate(const std::string& model_name)` — deleted entirely. Use direct function pointers instead.
- `granite4Template` — renamed to `granite4ChatTemplate` and moved to model header.
- `llama3Template` — renamed to `llama3ChatTemplate` and moved to model header.
- `phiTemplate` — renamed to `phiChatTemplate`, stays in core.

---

## 5. Per-model `chatTemplate` constant

Each model namespace now exports a `ChatTemplateFunc` constant that selects the correct template:

```cpp
namespace geniex::qwen3_4b {
    inline ChatTemplateFunc chatTemplate = chatMLTemplate;
}
namespace geniex::granite4_micro {
    inline ChatTemplateFunc chatTemplate = granite4ChatTemplate;
}
// etc.
```

Use `<model_namespace>::chatTemplate` when calling `create()` manually — it always resolves to the correct template for that model.

---

## Quick reference — two calling patterns

### Pattern A: Per-model factory (recommended)

```cpp
#include "qwen3.h"  // includes pipeline headers transitively

auto pipe = geniex::qwen3_4b::makePipeline(runtime_cfg, model_cfg);
if (!pipe) return 1;

pipe->setSystemPrompt("You are a helpful assistant.");
std::string prompt = pipe->applyChatTemplate("Hello!");
auto result = pipe->generate(prompt, gen_cfg, [](const char* t) {
    std::cout << t << std::flush;
    return true;
});
```

### Pattern B: Manual create (when you need more control)

```cpp
#include "pipeline/llm_pipeline.h"
#include "qwen3.h"

geniex::LLMPipeline pipe;
if (!pipe.create(geniex::qwen3_4b::chatTemplate,
                 geniex::qwen3_4b::makeModel(),
                 runtime_cfg, model_cfg)) {
    return 1;
}

pipe.setSystemPrompt("You are a helpful assistant.");
// ... same as before, using pipe. (not pipe->)
```
