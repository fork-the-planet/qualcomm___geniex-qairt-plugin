#include "llm/xtensor_utils.h"

#include <algorithm>
#include <cmath>

namespace geniex {
namespace xt_utils {

// ── RotaryEmbedding ──────────────────────────────────────────────────────────

RotaryEmbedding::RotaryEmbedding(int dim, float theta, float scale) {
    auto idx = xt::arange<float>(0, dim, 2);
    this->inv_freq = 1.f / xt::pow(theta, idx / static_cast<float>(dim));
    this->scale = scale;
}

std::pair<xt::xarray<float>, xt::xarray<float>>
RotaryEmbedding::forward(const xt::xarray<int32_t>& position_ids) const {
    std::size_t B = position_ids.shape()[0];
    std::size_t S = position_ids.shape()[1];
    std::size_t D = inv_freq.shape()[0];

    xt::xarray<float> pos_f = xt::cast<float>(position_ids);
    auto pos3d = xt::reshape_view(pos_f, std::vector<std::size_t>{B, S, 1});
    auto inv3d = xt::broadcast(inv_freq, std::vector<std::size_t>{B, S, D});
    xt::xarray<float> freqs = pos3d * inv3d;
    xt::xarray<float> cos_out = xt::cos(freqs) * scale;
    xt::xarray<float> sin_out = xt::sin(freqs) * scale;
    return {cos_out, sin_out};
}

// ── XTensorLLMUtils ──────────────────────────────────────────────────────────

XTensorLLMUtils::XTensorLLMUtils(
    size_t seq_len, size_t kv_len, size_t hidden_size,
    size_t /*num_heads*/, size_t num_kv_heads, size_t head_dim,
    size_t start_layer_idx, size_t end_layer_idx, size_t vocab_size,
    const std::string& in_states_name,
    const std::string& out_states_name,
    float rope_theta, int32_t eos_token_id)
{
    seq_len_         = seq_len;
    kv_len_          = kv_len;
    hidden_size_     = hidden_size;
    total_len_       = seq_len + kv_len;
    head_dim_        = head_dim;
    num_kv_heads_    = num_kv_heads;
    start_layer_idx_ = start_layer_idx;
    end_layer_idx_   = end_layer_idx;
    num_layers_      = end_layer_idx - start_layer_idx + 1;
    vocab_size_      = vocab_size;
    in_states_name_  = in_states_name;
    out_states_name_ = out_states_name;
    eos_token_id_    = eos_token_id;
    rope_            = RotaryEmbedding(static_cast<int>(head_dim), rope_theta, 1.f);
}

void XTensorLLMUtils::initializeTensorMappings(const Graph& graph) {
    input_tensor_names.clear();
    output_tensor_names.clear();
    input_tensor_order.clear();
    output_tensor_order.clear();

    const auto& in_specs = graph.inputSpecs();
    for (size_t i = 0; i < in_specs.size(); ++i) {
        input_tensor_names.push_back(in_specs[i].name);
        input_tensor_order[in_specs[i].name] = static_cast<int>(i);
    }

    const auto& out_specs = graph.outputSpecs();
    for (size_t i = 0; i < out_specs.size(); ++i) {
        output_tensor_names.push_back(out_specs[i].name);
        output_tensor_order[out_specs[i].name] = static_cast<int>(i);
    }
}

// ── Embedding ────────────────────────────────────────────────────────────────

xt::xarray<float> XTensorLLMUtils::tokensToEmbedding(
    const std::vector<int32_t>& token_ids,
    const xt::xarray<float>& embedding_table) const
{
    xt::xarray<float> result = xt::zeros<float>(
        std::vector<std::size_t>{1, seq_len_, hidden_size_});

    std::size_t n = std::min(token_ids.size(), seq_len_);
    for (std::size_t i = 0; i < n; ++i) {
        xt::view(result, 0, i, xt::all()) =
            xt::view(embedding_table, token_ids[i], xt::all());
    }
    return result;
}

xt::xarray<float> XTensorLLMUtils::tokenToEmbedding(
    int32_t token_id,
    const xt::xarray<float>& embedding_table) const
{
    xt::xarray<float> result = xt::zeros<float>(
        std::vector<std::size_t>{1, static_cast<std::size_t>(1), hidden_size_});

    xt::view(result, 0, 0, xt::all()) =
        xt::view(embedding_table, token_id, xt::all());
    return result;
}

// ── Position encoding ────────────────────────────────────────────────────────

xt::xarray<int32_t> XTensorLLMUtils::get_position_ids(
    std::size_t n_past, std::size_t curr_len) const
{
    return xt::arange<int32_t>(
        static_cast<int32_t>(n_past),
        static_cast<int32_t>(n_past + curr_len));
}

std::pair<xt::xarray<float>, xt::xarray<float>>
XTensorLLMUtils::get_cos_sin(const xt::xarray<int32_t>& position_ids) const {
    auto pos2d = xt::reshape_view(position_ids,
        std::array<std::size_t, 2>{1, position_ids.shape()[0]});
    auto [c, s] = rope_.forward(pos2d);
    return {xt::expand_dims(c, 0), xt::expand_dims(s, 0)};
}

// ── Attention mask ───────────────────────────────────────────────────────────

xt::xarray<float> XTensorLLMUtils::get_attention_mask(
    std::size_t n_past, std::size_t curr_len) const
{
    xt::xarray<float> mask = xt::ones<float>(
        std::array<std::size_t, 2>{seq_len_, total_len_}) * (-1000000000.f);

    if (curr_len == 0)
        return xt::expand_dims(xt::expand_dims(mask, 0), 0);

    // Visible past tokens
    if (n_past > 0) {
        xt::view(mask,
                 xt::range<std::size_t>(0, curr_len),
                 xt::range<std::size_t>(0, n_past)) = 0.f;
    }

    // Causal mask inside current chunk
    xt::xarray<int> rows = xt::reshape_view(
        xt::arange<int>(0, static_cast<int>(curr_len)),
        std::array<std::size_t, 2>{curr_len, 1});
    xt::xarray<int> cols = xt::arange<int>(0, static_cast<int>(curr_len));
    auto cond = xt::greater_equal(rows, cols);
    xt::xarray<float> causal = xt::where(cond, 0.f, -1000000000.f);
    xt::view(mask,
             xt::range<std::size_t>(0, curr_len),
             xt::range<std::size_t>(kv_len_, kv_len_ + curr_len)) = causal;

    return xt::expand_dims(xt::expand_dims(mask, 0), 0);
}

// ── KV cache ─────────────────────────────────────────────────────────────────

std::vector<xt::xarray<float>> XTensorLLMUtils::get_kv_cache() const {
    std::vector<xt::xarray<float>> out;
    out.reserve(num_layers_ * 2);
    std::array<std::size_t, 4> kshape{num_kv_heads_, 1, head_dim_, kv_len_};
    std::array<std::size_t, 4> vshape{num_kv_heads_, 1, kv_len_, head_dim_};
    for (std::size_t i = 0; i < num_layers_; ++i) {
        out.emplace_back(xt::zeros<float>(kshape));
        out.emplace_back(xt::zeros<float>(vshape));
    }
    return out;
}

// ── Input preparation ────────────────────────────────────────────────────────

std::map<std::string, std::vector<float>>
XTensorLLMUtils::prepare_inputs(
    const xt::xarray<float>& input_embeds,
    std::size_t n_past, std::size_t curr_len) const
{
    auto attn = get_attention_mask(n_past, curr_len);
    auto pos  = get_position_ids(n_past, seq_len_);
    auto [cos, sin] = get_cos_sin(pos);
    auto kv_cache = get_kv_cache();

    std::map<std::string, std::vector<float>> d;
    d[in_states_name_]    = std::vector<float>(input_embeds.begin(), input_embeds.end());
    d["attention_mask"]   = std::vector<float>(attn.begin(), attn.end());
    d["position_ids_cos"] = std::vector<float>(cos.begin(), cos.end());
    d["position_ids_sin"] = std::vector<float>(sin.begin(), sin.end());

    for (std::size_t i = start_layer_idx_; i <= end_layer_idx_; ++i) {
        std::size_t ci = i - start_layer_idx_;
        d["past_key_"   + std::to_string(i) + "_in"] =
            std::vector<float>(kv_cache[2 * ci].begin(), kv_cache[2 * ci].end());
        d["past_value_" + std::to_string(i) + "_in"] =
            std::vector<float>(kv_cache[2 * ci + 1].begin(), kv_cache[2 * ci + 1].end());
    }
    return d;
}

// ── KV cache update ──────────────────────────────────────────────────────────

void XTensorLLMUtils::update_kv_cache(
    std::vector<std::vector<float>>& input_data,
    std::vector<std::vector<float>>& output_data,
    int n_past, int curr_len, size_t out_seq_len)
{
    for (std::size_t i = start_layer_idx_; i <= end_layer_idx_; ++i) {
        std::string key_in  = "past_key_"   + std::to_string(i) + "_in";
        std::string key_out = "past_key_"   + std::to_string(i) + "_out";
        std::string val_in  = "past_value_" + std::to_string(i) + "_in";
        std::string val_out = "past_value_" + std::to_string(i) + "_out";

        // Key: [num_kv_heads, 1, head_dim, kv_len]
        auto ki = xt::adapt(input_data[input_tensor_order[key_in]],
                            std::vector<std::size_t>{num_kv_heads_, 1, head_dim_, kv_len_});
        auto ko = xt::adapt(output_data[output_tensor_order[key_out]],
                            std::vector<std::size_t>{num_kv_heads_, 1, head_dim_, out_seq_len});
        xt::view(ki, xt::all(), xt::all(), xt::all(),
                 xt::range(n_past, n_past + curr_len)) =
            xt::view(ko, xt::all(), xt::all(), xt::all(),
                     xt::range(0, curr_len));

        // Value: [num_kv_heads, 1, kv_len, head_dim]
        auto vi = xt::adapt(input_data[input_tensor_order[val_in]],
                            std::vector<std::size_t>{num_kv_heads_, 1, kv_len_, head_dim_});
        auto vo = xt::adapt(output_data[output_tensor_order[val_out]],
                            std::vector<std::size_t>{num_kv_heads_, 1, out_seq_len, head_dim_});
        xt::view(vi, xt::all(), xt::all(),
                 xt::range(n_past, n_past + curr_len), xt::all()) =
            xt::view(vo, xt::all(), xt::all(),
                     xt::range(0, curr_len), xt::all());

        // Write back
        input_data[input_tensor_order[key_in]] =
            std::vector<float>(ki.begin(), ki.end());
        input_data[input_tensor_order[val_in]] =
            std::vector<float>(vi.begin(), vi.end());
    }
}

// ── KV cache transfer (ARN -> AR1) ──────────────────────────────────────────

void XTensorLLMUtils::transfer_kv_cache(
    std::vector<std::vector<float>>& dst,
    std::vector<std::vector<float>>& src,
    int n_past)
{
    for (std::size_t i = start_layer_idx_; i <= end_layer_idx_; ++i) {
        std::string key_name = "past_key_"   + std::to_string(i) + "_in";
        std::string val_name = "past_value_" + std::to_string(i) + "_in";

        // Transfer key cache
        if (input_tensor_order.count(key_name)) {
            int idx = input_tensor_order.at(key_name);
            std::size_t src_kv = src[idx].size() / (num_kv_heads_ * head_dim_);
            std::size_t dst_kv = dst[idx].size() / (num_kv_heads_ * head_dim_);

            auto sk = xt::adapt(src[idx],
                std::vector<std::size_t>{num_kv_heads_, 1, head_dim_, src_kv});
            auto dk = xt::adapt(dst[idx],
                std::vector<std::size_t>{num_kv_heads_, 1, head_dim_, dst_kv});
            xt::view(dk, xt::all(), xt::all(), xt::all(),
                     xt::range(0, n_past)) =
                xt::view(sk, xt::all(), xt::all(), xt::all(),
                         xt::range(0, n_past));
            dst[idx] = std::vector<float>(dk.begin(), dk.end());
        }

        // Transfer value cache
        if (input_tensor_order.count(val_name)) {
            int idx = input_tensor_order.at(val_name);
            std::size_t src_kv = src[idx].size() / (num_kv_heads_ * head_dim_);
            std::size_t dst_kv = dst[idx].size() / (num_kv_heads_ * head_dim_);

            auto sv = xt::adapt(src[idx],
                std::vector<std::size_t>{num_kv_heads_, 1, src_kv, head_dim_});
            auto dv = xt::adapt(dst[idx],
                std::vector<std::size_t>{num_kv_heads_, 1, dst_kv, head_dim_});
            xt::view(dv, xt::all(), xt::all(),
                     xt::range(0, n_past), xt::all()) =
                xt::view(sv, xt::all(), xt::all(),
                         xt::range(0, n_past), xt::all());
            dst[idx] = std::vector<float>(dv.begin(), dv.end());
        }
    }
}

// ── Graph I/O helpers ────────────────────────────────────────────────────────

void XTensorLLMUtils::writeInputs(
    Graph& graph,
    const std::vector<std::vector<float>>& input_data) const
{
    for (std::size_t i = 0; i < input_tensor_names.size(); ++i) {
        const auto& name = input_tensor_names[i];
        if (i < input_data.size() && graph.hasInput(name)) {
            graph.write(name, input_data[i].data(), input_data[i].size());
        }
    }
}

void XTensorLLMUtils::readOutputs(
    const Graph& graph,
    std::vector<std::vector<float>>& output_data) const
{
    output_data.resize(output_tensor_names.size());
    for (std::size_t i = 0; i < output_tensor_names.size(); ++i) {
        const auto& name = output_tensor_names[i];
        if (graph.hasOutput(name)) {
            const auto& spec = graph.outputSpec(name);
            output_data[i].resize(spec.elementCount());
            graph.read(name, output_data[i].data(), output_data[i].size());
        }
    }
}

} // namespace xt_utils
} // namespace geniex
