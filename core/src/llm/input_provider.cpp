#include "llm/input_provider.h"
#include "types.h"

#include "xtensor/io/xnpy.hpp"

#include <algorithm>
#include <stdexcept>

namespace geniex {

// ── EmbeddingInputProvider ────────────────────────────────────────────────────

EmbeddingInputProvider::EmbeddingInputProvider(std::string tensor_name)
    : tensor_name_(std::move(tensor_name)) {}

bool EmbeddingInputProvider::loadTable(const std::string& path,
                                       size_t             vocab_size,
                                       size_t             hidden_size) {
    auto arr = xt::load_npy<float>(path);
    if (arr.size() != vocab_size * hidden_size) {
        throw std::runtime_error(
            "EmbeddingInputProvider: table size mismatch in " + path);
    }
    table_.assign(arr.begin(), arr.end());
    hidden_size_ = hidden_size;
    return true;
}

void EmbeddingInputProvider::onInitialized(const ModelConfig& model_cfg) {
    if (!table_.empty()) return;  // already loaded via loadTable()
    if (model_cfg.embedding_path.empty()) return;

    auto arr = xt::load_npy<float>(model_cfg.embedding_path);
    // npy shape: [vocab_size, hidden_size]
    if (arr.dimension() != 2) {
        throw std::runtime_error(
            "EmbeddingInputProvider: expected 2-D npy, got " +
            std::to_string(arr.dimension()) + "-D");
    }
    hidden_size_ = arr.shape(1);
    table_.assign(arr.begin(), arr.end());
}

void EmbeddingInputProvider::write(Graph& g, const LLMRunContext& ctx) {
    if (!g.hasInput(tensor_name_)) return;
    if (table_.empty()) return;
    auto embeds = tokensToEmbedding(ctx.token_ids, table_.data(), hidden_size_);
    g.write(tensor_name_, embeds.data(), embeds.size());
}

// ── TokenIdInputProvider ──────────────────────────────────────────────────────

TokenIdInputProvider::TokenIdInputProvider(std::string tensor_name,
                                           int32_t     pad_token_id)
    : tensor_name_(std::move(tensor_name))
    , pad_token_id_(pad_token_id) {}

void TokenIdInputProvider::write(Graph& g, const LLMRunContext& ctx) {
    if (!g.hasInput(tensor_name_)) return;

    const auto& spec = g.inputSpec(tensor_name_);
    size_t capacity = 1;
    for (auto d : spec.shape) capacity *= d;

    std::vector<int32_t> buf(capacity, pad_token_id_);
    const size_t n = std::min(ctx.token_ids.size(), capacity);
    std::copy_n(ctx.token_ids.begin(), n, buf.begin());

    g.write(tensor_name_, buf.data(), buf.size());
}

// ── RoPEInputProvider ─────────────────────────────────────────────────────────

RoPEInputProvider::RoPEInputProvider(size_t head_dim, float theta,
                                     std::string cos_name,
                                     std::string sin_name)
    : rope_(head_dim, theta)
    , cos_name_(std::move(cos_name))
    , sin_name_(std::move(sin_name)) {}

void RoPEInputProvider::write(Graph& g, const LLMRunContext& ctx) {
    const bool has_cos = g.hasInput(cos_name_);
    const bool has_sin = g.hasInput(sin_name_);
    if (!has_cos && !has_sin) return;

    auto [cos_vec, sin_vec] = rope_.forward(get_position_ids(ctx.n_past, ctx.curr_len));
    if (has_cos) g.write(cos_name_, cos_vec.data(), cos_vec.size());
    if (has_sin) g.write(sin_name_, sin_vec.data(), sin_vec.size());
}

// ── LongRoPEInputProvider ─────────────────────────────────────────────────────

LongRoPEInputProvider::LongRoPEInputProvider(size_t head_dim, float theta,
                                             std::vector<float> ext_factors,
                                             int max_position_embeddings,
                                             int original_max_position_embeddings,
                                             std::string cos_name,
                                             std::string sin_name)
    : rope_(head_dim, theta, std::move(ext_factors), max_position_embeddings, original_max_position_embeddings)
    , cos_name_(std::move(cos_name))
    , sin_name_(std::move(sin_name)) {}

void LongRoPEInputProvider::write(Graph& g, const LLMRunContext& ctx) {
    const bool has_cos = g.hasInput(cos_name_);
    const bool has_sin = g.hasInput(sin_name_);
    if (!has_cos && !has_sin) return;

    auto [cos_vec, sin_vec] = rope_.forward(get_position_ids(ctx.n_past, ctx.curr_len));
    if (has_cos) g.write(cos_name_, cos_vec.data(), cos_vec.size());
    if (has_sin) g.write(sin_name_, sin_vec.data(), sin_vec.size());
}

// ── PartialRoPEInputProvider ─────────────────────────────────────────────────────

PartialRoPEInputProvider::PartialRoPEInputProvider(size_t head_dim, float theta,
                                                   float rope_fraction, float scale,
                                                   std::string cos_name,
                                                   std::string sin_name)
    : rope_(head_dim, theta, rope_fraction, scale)
    , cos_name_(std::move(cos_name))
    , sin_name_(std::move(sin_name)) {}

void PartialRoPEInputProvider::write(Graph& g, const LLMRunContext& ctx) {
    const bool has_cos = g.hasInput(cos_name_);
    const bool has_sin = g.hasInput(sin_name_);
    if (!has_cos && !has_sin) return;

    auto [cos_vec, sin_vec] = rope_.forward(get_position_ids(ctx.n_past, ctx.curr_len));
    if (has_cos) g.write(cos_name_, cos_vec.data(), cos_vec.size());
    if (has_sin) g.write(sin_name_, sin_vec.data(), sin_vec.size());
}

} // namespace geniex
