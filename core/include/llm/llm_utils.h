#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "geniex_export.h"

namespace geniex {

// Computes rotary positional embedding frequencies for a given head dimension
// and base frequency theta. Constructed once and reused across forward calls.
class GENIEX_API RotaryEmbedding {
public:
    RotaryEmbedding() = default;
    RotaryEmbedding(size_t head_dim, float theta = 10000.f);

    // Returns {cos, sin}, each flat [n * half_dim] where n = position_ids.size()
    // and half_dim = head_dim / 2.
    std::pair<std::vector<float>, std::vector<float>>
    forward(const std::vector<int32_t>& position_ids) const;

    size_t halfDim() const;

private:
    std::vector<float> inv_freq_;  // [half_dim]
    size_t half_dim_ = 0;
};

// LongRoPE with dynamic scaling and per-dimension extension factors.
// ext_factors must have at least half_dim elements; shorter vectors are padded with 1.0.
// Produces cos/sin of size [n * half_dim] like standard RoPE.
class GENIEX_API LongRoPEEmbedding {
public:
    LongRoPEEmbedding() = default;
    LongRoPEEmbedding(size_t head_dim, float theta,
                      std::vector<float> ext_factors,
                      int max_position_embeddings = 131072,
                      int original_max_position_embeddings = 4096);

    std::pair<std::vector<float>, std::vector<float>>
    forward(const std::vector<int32_t>& position_ids) const;

    size_t halfDim() const;

private:
    std::vector<float> ext_factors_;  // [half_dim], per-dimension extension factors
    float base_ = 10000.f;
    int dim_ = 0;  // full head_dim (half_dim_ * 2)
    int max_position_embeddings_ = 131072;
    int original_max_position_embeddings_ = 4096;
    size_t half_dim_ = 0;
};

// Partial RoPE: rotates only (rope_fraction * head_dim) dimensions, with a post-scale factor.
// Produces cos/sin of size [n * rope_half_dim] where rope_half_dim = int(head_dim * rope_fraction) / 2.
class GENIEX_API PartialRoPEEmbedding {
public:
    PartialRoPEEmbedding() = default;
    PartialRoPEEmbedding(size_t head_dim, float theta = 10000.f,
                         float rope_fraction = 1.0f, float scale = 1.0f);

    std::pair<std::vector<float>, std::vector<float>>
    forward(const std::vector<int32_t>& position_ids) const;

    size_t halfDim() const;

private:
    std::vector<float> inv_freq_;  // [rope_half_dim]
    float scale_ = 1.f;
    size_t rope_half_dim_ = 0;
};

// Returns position IDs [n_past, n_past + count) as a flat int32 vector.
GENIEX_API std::vector<int32_t> get_position_ids(size_t n_past, size_t count);

// Returns {cos, sin}, each flat [n * half_dim] for the given position IDs.
// half_dim = head_dim / 2.
GENIEX_API std::pair<std::vector<float>, std::vector<float>>
get_cos_sin(const std::vector<int32_t>& position_ids,
            size_t                      head_dim,
            float                       rope_theta = 10000.f);

// Returns a causal attention mask, flat [seq_len * (kv_len + seq_len)].
// Logically shaped [1, 1, seq_len, kv_len + seq_len]:
//   - Columns [0, n_past)          in rows [0, curr_len): 0.0  (visible past)
//   - Causal columns in current chunk: 0.0; everything else: -1e9
GENIEX_API std::vector<float> get_attention_mask(size_t n_past,
                                      size_t curr_len,
                                      size_t seq_len,
                                      size_t kv_len);

// Looks up and concatenates embeddings for each token ID.
// embedded_tokens: flat row-major [vocab_size * hidden_size] float32 table.
// Returns flat [token_ids.size() * hidden_size].
GENIEX_API std::vector<float> tokensToEmbedding(const std::vector<int32_t>& token_ids,
                                     const float*                 embedded_tokens,
                                     size_t                       hidden_size);

// Returns a zero-initialised buffer for one KV tensor (key or value) with
// size = num_kv_heads * head_dim * kv_len elements.
std::vector<float> get_kv_cache(size_t num_kv_heads,
                                size_t head_dim,
                                size_t kv_len);

} // namespace geniex
