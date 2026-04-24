#pragma once

// Continuous-batching adapter for the Llama-3.2 family (1B / 3B / …).
//
// This file reuses the existing single-session specs from
// `models/llama3_2/llama3_2.h` (namespaces `llama3_2_1b`, `llama3_2_3b`, …) —
// no architecture duplication. The only Llama-3.2-specific code here is:
//   1. Llama32CBTokenIdProvider — writes concatenated `input_ids` to shard 0.
//   2. Llama32CBRoPEProvider    — builds per-session positions and writes cos/sin.
//
// Llama 3.2 uses standard RoPE (no frequency scaling), so the providers are
// identical in structure to the Qwen3 CB providers — only the head_dim and
// theta constants differ per size.

#include "cb/cb.h"
#include "llm/llm_utils.h"
#include "pipeline/chat_template.h"
#include "llama3_2.h"  // models/llama3_2/llama3_2.h — reused specs

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace geniex {
namespace llama3_2_cb {

// Default pad token for Llama-3 (EOS "end_of_text" 128001).
static constexpr int32_t kPadTokenId = 128001;

// ────────────────────────────────────────────────────────────────────────────
// Token-id writer: writes the concatenated, zero-padded input_ids buffer
// verbatim into shard 0's `input_ids` tensor. Silently no-ops on other shards.
// ────────────────────────────────────────────────────────────────────────────
class Llama32CBTokenIdProvider : public cb::CBInputProvider {
public:
    explicit Llama32CBTokenIdProvider(std::string tensor_name = "input_ids",
                                      int32_t pad_token_id = kPadTokenId)
        : tensor_name_(std::move(tensor_name)), pad_token_id_(pad_token_id) {}

    void write(Graph& g, const cb::CBStepContext& ctx) override {
        if (!g.hasInput(tensor_name_)) return;

        const auto& spec = g.inputSpec(tensor_name_);
        size_t capacity = 1;
        for (auto d : spec.shape) capacity *= d;

        std::vector<int32_t> buf(capacity, pad_token_id_);
        const size_t n = std::min(ctx.concat_tokens.size(), capacity);
        std::copy_n(ctx.concat_tokens.begin(), n, buf.begin());

        g.write(tensor_name_, buf.data(), buf.size());
    }

private:
    std::string tensor_name_;
    int32_t     pad_token_id_;
};

// ────────────────────────────────────────────────────────────────────────────
// RoPE writer: builds per-session position IDs from the CB step context and
// writes the resulting cos/sin tables into the RoPE tensors.
// ────────────────────────────────────────────────────────────────────────────
class Llama32CBRoPEProvider : public cb::CBInputProvider {
public:
    Llama32CBRoPEProvider(size_t head_dim, float theta,
                          std::string cos_name = "position_ids_cos",
                          std::string sin_name = "position_ids_sin")
        : rope_(head_dim, theta),
          cos_name_(std::move(cos_name)),
          sin_name_(std::move(sin_name)) {}

    void write(Graph& g, const cb::CBStepContext& ctx) override {
        const bool has_cos = g.hasInput(cos_name_);
        const bool has_sin = g.hasInput(sin_name_);
        if (!has_cos && !has_sin) return;

        std::vector<int32_t> pos_ids;
        cb::KVCacheManager::getPositionIds(ctx.kv_segs, ctx.in_segs, ctx.seq_len, pos_ids);

        auto [cos_vec, sin_vec] = rope_.forward(pos_ids);
        if (has_cos) g.write(cos_name_, cos_vec.data(), cos_vec.size());
        if (has_sin) g.write(sin_name_, sin_vec.data(), sin_vec.size());
    }

private:
    RotaryEmbedding rope_;
    std::string     cos_name_;
    std::string     sin_name_;
};

inline ChatTemplateFunc chatTemplate = llama3ChatTemplate;

// ────────────────────────────────────────────────────────────────────────────
// Per-size CB namespaces. Each mirrors the matching single-session namespace
// in models/llama3_2/llama3_2.h (llama3_2_1b, llama3_2_3b, …) and exposes a
// `makeModel()` factory. To add a new size, add a new inner namespace with
// its own `makeModel()` that reuses the corresponding spec.
// ────────────────────────────────────────────────────────────────────────────

namespace llama3_2_3b {
inline cb::CBLLMModel makeModel() {
    cb::CBLLMModel m(geniex::llama3_2_3b::makeSpec());
    m.addCBProvider(std::make_unique<Llama32CBTokenIdProvider>("input_ids", kPadTokenId));
    m.addCBProvider(std::make_unique<Llama32CBRoPEProvider>(
        geniex::llama3_2_3b::kHeadDim, geniex::llama3_2_3b::kRopeTheta));
    return m;
}
}  // namespace llama3_2_3b

namespace llama3_2_1b {
inline cb::CBLLMModel makeModel() {
    cb::CBLLMModel m(geniex::llama3_2_1b::makeSpec());
    m.addCBProvider(std::make_unique<Llama32CBTokenIdProvider>("input_ids", kPadTokenId));
    m.addCBProvider(std::make_unique<Llama32CBRoPEProvider>(
        geniex::llama3_2_1b::kHeadDim, geniex::llama3_2_1b::kRopeTheta));
    return m;
}
}  // namespace llama3_2_1b

}  // namespace llama3_2_cb
}  // namespace geniex
