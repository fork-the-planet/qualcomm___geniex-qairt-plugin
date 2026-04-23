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

// CPU-side tensor utilities for xtensor-based LLM prototyping.
class XTensorLLMUtils {
public:
    XTensorLLMUtils(size_t seq_len, size_t kv_len, size_t hidden_size,
                    size_t num_heads, size_t num_kv_heads, size_t head_dim,
                    size_t start_layer_idx, size_t end_layer_idx, size_t vocab_size,
                    const std::string& in_states_name = "",
                    const std::string& out_states_name = "",
                    float rope_theta = 10000.f,
                    int32_t eos_token_id = 151645);

    void initializeTensorMappings(const Graph& graph);

    // Returns shape [1, seq_len_, hidden_size_].
    xt::xarray<float> tokensToEmbedding(const std::vector<int32_t>& token_ids,
                                         const xt::xarray<float>& embedding_table) const;

    // Returns shape [1, 1, hidden_size_].
    xt::xarray<float> tokenToEmbedding(int32_t token_id,
                                        const xt::xarray<float>& embedding_table) const;

    xt::xarray<int32_t> get_position_ids(size_t n_past, size_t curr_len) const;

    // Returns {cos, sin} each shaped [1, 1, curr_len, head_dim/2].
    std::pair<xt::xarray<float>, xt::xarray<float>>
    get_cos_sin(const xt::xarray<int32_t>& position_ids) const;

    // Returns causal mask shaped [1, 1, seq_len_, total_len_].
    xt::xarray<float> get_attention_mask(size_t n_past, size_t curr_len) const;

    std::vector<xt::xarray<float>> get_kv_cache() const;

    std::map<std::string, std::vector<float>>
    prepare_inputs(const xt::xarray<float>& input_embeds,
                   size_t n_past, size_t curr_len) const;

    void update_kv_cache(std::vector<std::vector<float>>& input_data,
                         std::vector<std::vector<float>>& output_data,
                         int n_past, int curr_len, size_t out_seq_len);

    // Propagates [0, n_past) KV positions from src to dst (e.g. prefill -> decode graph).
    void transfer_kv_cache(std::vector<std::vector<float>>& dst,
                           std::vector<std::vector<float>>& src,
                           int n_past);

    void writeInputs(Graph& graph,
                     const std::vector<std::vector<float>>& input_data) const;

    void readOutputs(const Graph& graph,
                     std::vector<std::vector<float>>& output_data) const;

    std::vector<std::string> input_tensor_names;
    std::vector<std::string> output_tensor_names;
    std::map<std::string, int> input_tensor_order;
    std::map<std::string, int> output_tensor_order;

    size_t seq_len_, kv_len_, total_len_, hidden_size_;
    size_t head_dim_, num_kv_heads_, vocab_size_;
    size_t start_layer_idx_, end_layer_idx_, num_layers_;
    std::string in_states_name_, out_states_name_;
    int32_t eos_token_id_;
    RotaryEmbedding rope_;
};

} // namespace xt_utils
} // namespace geniex
