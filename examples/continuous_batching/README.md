# Continuous Batching Examples

This folder contains demos of **continuous batching (CB)** on top of `geniex_core`.
CB lets a single `batch_size=1` QNN model binary serve multiple concurrent
sessions by **concatenating inputs along the sequence dimension**, so 128-token
decode windows are filled with real tokens from several sessions instead of
padding from one.

> `examples/` is intentionally kept separate from `core/`. The shared CB
> library under `common/cb/` is **header-only** and not part of `geniex_core`
> — CB is a demo of an advanced feature, not a first-class runtime API.

## Why Concatenate Instead of Stack?

On GPUs, batching stacks inputs along a batch dimension because the hardware
is compute-bound and exploits batch parallelism. QNN inference on Snapdragon
NPUs is **memory-bandwidth-bound**, so there is no performance difference
between stacking and concatenating — and concatenation has one major
advantage: **it works with existing `batch_size=1` model exports, no
re-compilation**.

A standard single-session decode step wastes 127 of 128 input slots on
padding. Continuous batching reclaims those slots for other sessions.

---

## Directory Layout

```
examples/continuous_batching/
├── CMakeLists.txt          # adds each model demo as a subdirectory
├── README.md               # (this file)
├── common/
│   └── cb/                 # header-only, model-agnostic CB library
│       ├── cb.h            # umbrella include
│       ├── session.h       # Session + SessionStatus
│       ├── scheduler.h     # Scheduler — who runs in this step?
│       ├── kv_cache_manager.h  # per-session KV segments, attn-mask, pos-ids
│       ├── input_provider.h    # CBStepContext + CBInputProvider interface
│       ├── cb_llm_model.h      # CBLLMModel — subclass of LLMModel
│       └── token_sampler.h     # optional greedy argmax extractor
└── qwen3/                  # concrete demo for Qwen3-4B Instruct 2507
    ├── CMakeLists.txt
    ├── qwen3_cb.h          # CBInputProvider impls + makeModel()
    └── qwen3_cb_example.cpp
```

## Architecture

```
             ┌──────────────┐
 user prompt │  Scheduler   │  addSession / getNextBatch / updateSession
             └──────┬───────┘
                    │ selects active sessions
                    ▼
             ┌──────────────┐
             │KVCacheManager│  allocate / extend / compact / shiftForGrowth
             └──────┬───────┘  getPositionIds / getAttentionMask
                    │ builds per-session offsets + block-diagonal mask
                    ▼
             ┌──────────────┐        ┌──────────────────────────┐
             │ CBStepContext│───────▶│ CBInputProvider (model)  │
             └──────┬───────┘        │  - write input_ids /     │
                    │                │    input_embeds          │
                    │                │  - write RoPE cos/sin    │
                    ▼                └──────────────────────────┘
             ┌──────────────┐
             │  CBLLMModel  │  runs every shard on concatenated input,
             │              │  copies KV per session, handles CL promotion,
             │              │  samples next token per session
             └──────────────┘
```

### 1. `Scheduler` (`common/cb/scheduler.h`)

Tracks every session's state (`WAITING → RUNNING → COMPLETED`) and packs
active sessions into the next forward pass up to the `seq_len` budget.

### 2. `KVCacheManager` (`common/cb/kv_cache_manager.h`)

Each session owns a contiguous slice of the shared KV buffer. The manager
keeps bookkeeping only (start_pos, length) — the caller applies the
resulting `MoveOp`s to the real KV tensors.

```
|<-- Session A (len=20) -->|<-- Session B (len=8) -->|   free   |
```

Provides two static helpers used by both CBLLMModel and model-specific
providers:

- **`getPositionIds`** — each session's positions run `[kv_length_i,
  kv_length_i + in_len)`, never a single global range.
- **`getAttentionMask`** — block-diagonal causal mask. Without it session A
  would attend to session B's KV cache and produce garbage.

### 3. `CBStepContext` + `CBInputProvider` (`common/cb/input_provider.h`)

A per-step context bundle that carries **everything a model needs to build
per-session inputs**:

| field             | what it is                                               |
|-------------------|----------------------------------------------------------|
| `sessions`        | active sessions for this step                            |
| `in_segs[i]`      | `{start_offset, length}` of session i in the concat input |
| `kv_segs[i]`      | KV snapshot *before* this step's growth                  |
| `concat_tokens`   | token IDs, zero-padded to `seq_len`                      |
| `seq_len` `kv_len`| graph dimensions for the active CL                       |

### 4. `CBLLMModel` (`common/cb/cb_llm_model.h`)

Subclass of `LLMModel`. Owns the model-agnostic pieces of the CB loop:

- Session scheduling
- Concatenated input buffer construction
- Attention mask build + write
- Shard execution + inter-shard hidden-state transfer
- Per-session KV copy after each shard
- CL promotion + in-place KV reshape
- Next-token sampling via `LLMModel::sampleNextToken`
- KV defragmentation after a session completes

Everything else — **what** to write into `input_ids` / `input_embeds` /
`position_ids_cos` / `position_ids_sin` — is delegated to
`CBInputProvider`s the model registers via `addCBProvider`.

---

## Adding CB Support for a New Model

The existing model under `models/<name>/` already has an `LLMSpec` and a set
of single-session `InputProvider`s. To make it CB-capable, you do **not**
modify `models/<name>/`. Instead, add a new subfolder under
`examples/continuous_batching/<name>/` that supplies CB versions of the
providers.

### Step 1 — Create the folder

```
examples/continuous_batching/<name>/
├── CMakeLists.txt
├── <name>_cb.h
└── <name>_cb_example.cpp
```

### Step 2 — Implement CBInputProviders

For every single-session `InputProvider` the model registers in
`models/<name>/<name>.h`, write a CB counterpart that handles concatenated
sessions. In practice each LLM needs two:

#### (a) Token-id or embedding writer

Writes the concatenated token buffer (or the corresponding embedding rows)
into shard 0's input tensor.

```cpp
class MyCBTokenIdProvider : public cb::CBInputProvider {
public:
    explicit MyCBTokenIdProvider(std::string tensor_name = "input_ids",
                                 int32_t pad = 0)
        : tensor_name_(std::move(tensor_name)), pad_(pad) {}

    void write(Graph& g, const cb::CBStepContext& ctx) override {
        if (!g.hasInput(tensor_name_)) return;
        const auto& spec = g.inputSpec(tensor_name_);
        size_t capacity = 1;
        for (auto d : spec.shape) capacity *= d;

        std::vector<int32_t> buf(capacity, pad_);
        const size_t n = std::min(ctx.concat_tokens.size(), capacity);
        std::copy_n(ctx.concat_tokens.begin(), n, buf.begin());
        g.write(tensor_name_, buf.data(), buf.size());
    }
private:
    std::string tensor_name_;
    int32_t     pad_;
};
```

For models with a **CPU-side embedding table** (`EmbeddingInputProvider`
pattern), load the table in `onInitialized` from `model_cfg.embedding_path`
and emit `ctx.concat_tokens.size() * hidden_size` floats in `write`.

#### (b) Position encoder (RoPE / LongRoPE / PartialRoPE / Llama3 RoPE)

Build per-session position IDs from the step context, then feed them to the
same `RotaryEmbedding` / `LongRoPEEmbedding` / `PartialRoPEEmbedding` class
the single-session provider already uses.

```cpp
class MyCBRoPEProvider : public cb::CBInputProvider {
public:
    MyCBRoPEProvider(size_t head_dim, float theta)
        : rope_(head_dim, theta) {}

    void write(Graph& g, const cb::CBStepContext& ctx) override {
        if (!g.hasInput("position_ids_cos") && !g.hasInput("position_ids_sin"))
            return;

        std::vector<int32_t> pos_ids;
        cb::KVCacheManager::getPositionIds(
            ctx.kv_segs, ctx.in_segs, ctx.seq_len, pos_ids);

        auto [cos, sin] = rope_.forward(pos_ids);
        if (g.hasInput("position_ids_cos"))
            g.write("position_ids_cos", cos.data(), cos.size());
        if (g.hasInput("position_ids_sin"))
            g.write("position_ids_sin", sin.data(), sin.size());
    }
private:
    RotaryEmbedding rope_;
};
```

### Step 3 — Write `makeModel()` factories (one per size)

Mirror the layout of `models/<name>/<name>.h`: it has one inner namespace per
size/variant (e.g. `llama3_2_1b`, `llama3_2_3b`), each with its own
`makeSpec()`. Do the same here — one CB inner namespace per size, each with
its own `makeModel()` that reuses the matching spec. Shared providers and
the chat template live in the outer `<name>_cb` namespace.

```cpp
#include "cb/cb.h"
#include "<name>.h"  // pulls <name>_<size>::makeSpec() from models/<name>/

namespace geniex { namespace <name>_cb {

// Shared providers (token-id writer, RoPE writer, …) and chat template here.
// ...

namespace <name>_<size_a> {
inline cb::CBLLMModel makeModel() {
    cb::CBLLMModel m(geniex::<name>_<size_a>::makeSpec());
    m.addCBProvider(std::make_unique<MyCBTokenIdProvider>(/* ... */));
    m.addCBProvider(std::make_unique<MyCBRoPEProvider>(
        geniex::<name>_<size_a>::kHeadDim,
        geniex::<name>_<size_a>::kRopeTheta));
    return m;
}
}  // namespace <name>_<size_a>

// Add more size namespaces (<name>_<size_b>, …) the same way.

}}  // namespace
```

Callers invoke `geniex::<name>_cb::<name>_<size>::makeModel()` — the call
pattern mirrors `geniex::<name>_<size>::makeSpec()` exactly.

### Step 4 — Write the example executable

Pattern-match `qwen3/qwen3_cb_example.cpp` or `llama3_2/llama3_2_cb_example.cpp`:
parse args, build `QnnRuntimeConfig` + `ModelConfig`, call
`geniex::<name>_cb::<name>_<size>::makeModel().initialize(...)`, drive a
`cb::Scheduler` + `cb::KVCacheManager` through `generateBatch`.

### Step 5 — Add `CMakeLists.txt`

```cmake
add_executable(<name>_cb <name>_cb_example.cpp)

target_include_directories(<name>_cb PRIVATE
    ${CMAKE_SOURCE_DIR}/examples/continuous_batching/common
    ${CMAKE_SOURCE_DIR}/examples/continuous_batching/<name>
)

target_link_libraries(<name>_cb PRIVATE geniex_core geniex-proc)

set_target_properties(<name>_cb PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
```

Then add the subdirectory in `examples/continuous_batching/CMakeLists.txt`:

```cmake
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/<name>)
```

### Checklist — Which provider type matches which model?

| Single-session provider       | CB counterpart                             |
|-------------------------------|--------------------------------------------|
| `TokenIdInputProvider`        | concat-tokens writer (see §a above)        |
| `EmbeddingInputProvider`      | concat-tokens → CPU embedding-lookup writer |
| `RoPEInputProvider`           | per-session positions → `RotaryEmbedding`  |
| `LongRoPEInputProvider`       | per-session positions → `LongRoPEEmbedding`† |
| `PartialRoPEInputProvider`    | per-session positions → `PartialRoPEEmbedding` |
| `Llama3RoPEInputProvider`     | per-session positions → Llama3 RoPE table  |

† **LongRoPE caveat**: the short/long factor switch depends on sequence
length. Under CB each session has its own effective length, so decide the
scaling **per session**, not once per batch.

---

## Building

```shell
cmake -B build -A ARM64
cmake --build build --config Release --target qwen3_cb -j32
```

## Running (qwen3 demo)

```shell
./build/bin/Release/qwen3_cb.exe --max-tokens 50 --verbose
```

```
Prompt 1 (or 'go'/'exit'): Hello
Prompt 2 (or 'go'/'exit'): What is 2+2?
Prompt 3 (or 'go'/'exit'): go
```

## Known Limitations / Future Work

- **VLM models** (`VLMModel`, `PrecomputedEmbeddingProvider`, `MRoPEInputProvider`)
  are not supported — precomputed embeddings are per-request buffers and
  M-RoPE uses 3-D positions. That's a second phase.
- **KV compaction cost** is O(`kv_len × num_layers × bytes_per_token`)
  memmoves on the CPU-visible mirror. At large CLs (4096 × 36 layers ×
  Qwen3-8B) this becomes throughput-dominant — measure before scaling up.
- **Sampling** is greedy argmax only; temperature / top-p sampling and
  per-session sampling params are not wired through.
