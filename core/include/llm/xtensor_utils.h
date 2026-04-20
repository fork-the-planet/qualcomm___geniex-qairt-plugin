#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "xtensor-all.hpp"
#include "graph.h"
#include "types.h"

namespace geniex {
namespace xt_utils {

// Rotary positional embedding using xtensor.
//   dim   : head dimension actually rotated (e.g. 128)
//   theta : RoPE base (e.g. 1000000)
//   scale : post-processing multiplier on cos/sin (usually 1.0)
class RotaryEmbedding {
public:
    xt::xarray<float> inv_freq;
    float scale = 1.f;

    RotaryEmbedding() = default;
    RotaryEmbedding(int dim, float theta = 10000.f, float scale = 1.f);

    // position_ids: shape (B, S). Returns {cos, sin} each (B, S, D) where D = dim/2.
    std::pair<xt::xarray<float>, xt::xarray<float>>
    forward(const xt::xarray<int32_t>& position_ids) const;
};

// Xtensor-based LLM utility class for running multi-shard inference.
//
// This class manages CPU-side tensor data as std::vector<std::vector<float>>
// (one inner vector per tensor, ordered by graph tensor index).
// It provides embedding lookup, RoPE, attention mask, KV cache operations,
// and Graph I/O helpers (writeInputs / readOutputs).
class XTensorLLMUtils {
public:
    XTensorLLMUtils(size_t seq_len, size_t kv_len, size_t hidden_size,
                    size_t num_heads, size_t num_kv_heads, size_t head_dim,
                    size_t start_layer_idx, size_t end_layer_idx, size_t vocab_size,
                    const std::string& in_states_name = "",
                    const std::string& out_states_name = "",
                    float rope_theta = 10000.f,
                    int32_t eos_token_id = 151645);

    // Discover tensor names and ordering from a Graph's specs.
    void initializeTensorMappings(const Graph& graph);

    // ── Embedding ────────────────────────────────────────────────────────────
    // Batch: looks up token_ids in embedding_table, pads to seq_len_.
    // Returns shape [1, seq_len_, hidden_size_].
    xt::xarray<float> tokensToEmbedding(const std::vector<int32_t>& token_ids,
                                         const xt::xarray<float>& embedding_table) const;

    // Single token. Returns shape [1, 1, hidden_size_].
    xt::xarray<float> tokenToEmbedding(int32_t token_id,
                                        const xt::xarray<float>& embedding_table) const;

    // ── Position encoding ────────────────────────────────────────────────────
    xt::xarray<int32_t> get_position_ids(size_t n_past, size_t curr_len) const;

    // Returns {cos, sin} each shaped [1, 1, curr_len, head_dim/2].
    std::pair<xt::xarray<float>, xt::xarray<float>>
    get_cos_sin(const xt::xarray<int32_t>& position_ids) const;

    // ── Attention mask ───────────────────────────────────────────────────────
    // Returns causal mask shaped [1, 1, seq_len_, total_len_].
    xt::xarray<float> get_attention_mask(size_t n_past, size_t curr_len) const;

    // ── KV cache ─────────────────────────────────────────────────────────────
    // Returns zero-initialized KV cache tensors (2 per layer: key + value).
    std::vector<xt::xarray<float>> get_kv_cache() const;

    // ── Input preparation ────────────────────────────────────────────────────
    // Builds a map of tensor_name -> flat float vector for all inputs
    // (embeds, attention mask, cos, sin, KV cache).
    std::map<std::string, std::vector<float>>
    prepare_inputs(const xt::xarray<float>& input_embeds,
                   size_t n_past, size_t curr_len) const;

    // ── KV cache operations ──────────────────────────────────────────────────
    // Copies newly computed KV from output_data into input_data at position n_past.
    void update_kv_cache(std::vector<std::vector<float>>& input_data,
                         std::vector<std::vector<float>>& output_data,
                         int n_past, int curr_len, size_t out_seq_len);

    // Transfers [0, n_past) of KV cache from src to dst (e.g. ARN -> AR1).
    void transfer_kv_cache(std::vector<std::vector<float>>& dst,
                           std::vector<std::vector<float>>& src,
                           int n_past);

    // ── Graph I/O helpers ────────────────────────────────────────────────────
    // Writes all tensors in input_data to the Graph (float -> native dtype).
    void writeInputs(Graph& graph,
                     const std::vector<std::vector<float>>& input_data) const;

    // Reads all output tensors from the Graph into output_data (native dtype -> float).
    void readOutputs(const Graph& graph,
                     std::vector<std::vector<float>>& output_data) const;

    // ── Public tensor mappings ───────────────────────────────────────────────
    std::vector<std::string> input_tensor_names;
    std::vector<std::string> output_tensor_names;
    std::map<std::string, int> input_tensor_order;
    std::map<std::string, int> output_tensor_order;

    // ── Public architecture parameters ───────────────────────────────────────
    size_t seq_len_, kv_len_, total_len_, hidden_size_;
    size_t head_dim_, num_kv_heads_, vocab_size_;
    size_t start_layer_idx_, end_layer_idx_, num_layers_;
    std::string in_states_name_, out_states_name_;
    int32_t eos_token_id_;
    RotaryEmbedding rope_;
};

} // namespace xt_utils
} // namespace geniex
