#include "ssd_model.h"
#include "llm/llm_utils.h"
#include "utils.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace geniex {

// ── Constructor ───────────────────────────────────────────────────────────────

SSDModel::SSDModel(LLMSpec spec, SSDConfig ssd_cfg)
    : LLMModel(std::move(spec)), ssd_cfg_(std::move(ssd_cfg)),
      rope_(spec_.head_dim, ssd_cfg_.rope_theta) {
    draft_levels_ = ssd_cfg_.branches.size();
    attention_map_ = genAttentionMap();
}

// ── onInitialized ─────────────────────────────────────────────────────────────

bool SSDModel::onInitialized() {
    if (!LLMModel::onInitialized()) return false;

    // ── Pre-cache KV tensor pointers and specs ───────────────────────────────
    {
        const auto& kv_block = spec_.state_blocks[kv_state_block_idx_];
        kv_tensor_cache_.clear();

        for (size_t s = 0; s < spec_.shards.size(); ++s) {
            const auto& layer_ranges = kv_block.shard_layer_ranges[s];
            if (layer_ranges.empty()) continue;

            const size_t gi = graphIndex(1, s, active_cl_idx_);
            Graph& g = graph(gi);

            for (const auto& [begin, end] : layer_ranges) {
                for (size_t l = begin; l <= end; ++l) {
                    KVTensorInfo info;
                    info.shard = s;
                    info.layer = l;

                    const std::string key_in  = fmtPattern(kv_block.key_in_pattern, l);
                    const std::string key_out = fmtPattern(kv_block.key_out_pattern, l);
                    const std::string val_in  = fmtPattern(kv_block.value_in_pattern, l);
                    const std::string val_out = fmtPattern(kv_block.value_out_pattern, l);

                    const auto& ki_spec = g.inputSpec(key_in);
                    const auto& ko_spec = g.outputSpec(key_out);
                    info.key_in_ptr      = g.inputPtr(key_in);
                    info.key_out_ptr     = g.outputPtr(key_out);
                    info.key_in_kv_len   = ki_spec.shape[3];
                    info.key_out_seq_len = ko_spec.shape[3];
                    info.key_elem_size   = ki_spec.elementSize();
                    info.key_n_rows      = spec_.num_kv_heads * spec_.head_dim;

                    const auto& vi_spec = g.inputSpec(val_in);
                    const auto& vo_spec = g.outputSpec(val_out);
                    info.val_in_ptr      = g.inputPtr(val_in);
                    info.val_out_ptr     = g.outputPtr(val_out);
                    info.val_in_kv_len   = vi_spec.shape[2];
                    info.val_out_seq_len = vo_spec.shape[2];
                    info.val_token_size  = spec_.head_dim * vi_spec.elementSize();
                    info.val_n_heads     = spec_.num_kv_heads;

                    kv_tensor_cache_.push_back(info);
                }
            }
        }
    }

    // Load the forecast prefix KV cache from disk.
    // FIX 1: Load into the PREFILL graph (phase=0) which has the smaller kv_len
    // stride. The data will be shared with the decode graph via QNN's shared
    // buffer mechanism. Loading into decode graph (phase=1) uses a larger stride
    // that gets corrupted when reshapeKV is called during prefill→decode transition.
    if (!ssd_cfg_.forecast_prefix_path.empty()) {
        if (!loadForecastPrefix()) {
            fprintf(stderr, "Warning: Failed to load forecast prefix KV cache.\n");
        } else {
            n_past_ = ssd_cfg_.forecast_prefix;
        }
    }

    // FIX 2: Set n_past_ to forecast_prefix after initialization so that
    // prefill starts writing KV after the pre-loaded forecast prefix entries.
    n_past_ = ssd_cfg_.forecast_prefix;

    return true;
}

// ── Attention map generation ──────────────────────────────────────────────────

std::vector<int32_t> SSDModel::genAttentionMap() {
    std::vector<int32_t> tree = {-1};

    samples_per_draft_level_.clear();
    nodes_per_draft_level_.clear();

    size_t start_idx = 0;
    for (size_t d = 0; d < draft_levels_; ++d) {
        const size_t end_idx = tree.size();
        const size_t branch_count = ssd_cfg_.branches[d];

        samples_per_draft_level_.push_back(branch_count + 1);

        for (size_t node_idx = start_idx; node_idx < end_idx; ++node_idx) {
            for (size_t c = 0; c < branch_count; ++c) {
                tree.push_back(static_cast<int32_t>(node_idx));
            }
        }

        nodes_per_draft_level_.push_back(tree.size() - end_idx);
        start_idx = end_idx;
    }

    num_draft_nodes_ = tree.size();

    const size_t end_idx = tree.size();
    for (size_t node_idx = 0; node_idx < end_idx; ++node_idx) {
        tree.push_back(static_cast<int32_t>(node_idx));
        for (size_t d = 1; d < draft_levels_; ++d) {
            tree.push_back(static_cast<int32_t>(tree.size() - 1));
        }
    }

    return tree;
}

std::vector<int32_t> SSDModel::genForecastTokens(size_t repeat) const {
    std::vector<int32_t> forecast(draft_levels_);
    std::iota(forecast.begin(), forecast.end(),
              static_cast<int32_t>(spec_.vocab_size));

    std::vector<int32_t> result;
    result.reserve(repeat * draft_levels_);
    for (size_t i = 0; i < repeat; ++i)
        result.insert(result.end(), forecast.begin(), forecast.end());
    return result;
}

// ── Draft tree building ───────────────────────────────────────────────────────

std::vector<int32_t> SSDModel::topKLogits(const float* logits_row, size_t k) const {
    std::vector<int32_t> indices(spec_.vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    if (k > spec_.vocab_size) k = spec_.vocab_size;
    std::partial_sort(indices.begin(), indices.begin() + static_cast<ptrdiff_t>(k),
                      indices.end(),
                      [logits_row](int32_t a, int32_t b) {
                          return logits_row[a] > logits_row[b];
                      });
    indices.resize(k);
    return indices;
}

int32_t SSDModel::argmaxLogits(const float* logits_row) const {
    return static_cast<int32_t>(
        std::max_element(logits_row, logits_row + spec_.vocab_size) - logits_row);
}

std::vector<int32_t> SSDModel::buildSampleTree(int32_t last_token,
                                                size_t phase,
                                                size_t start_offset) const {
    const size_t V = spec_.vocab_size;
    std::vector<int32_t> tree = {last_token};

    size_t draft_level = 0;
    size_t draft_node_idx = 0;
    std::vector<int32_t> samples;
    size_t sample_idx = 0;

    // Reusable buffer for one logit row.
    std::vector<float> row_buf(V);

    for (size_t cur_idx = 1; draft_level < draft_levels_; ++cur_idx) {
        const int32_t parent_idx = attention_map_[cur_idx];

        if (parent_idx != attention_map_[cur_idx - 1]) {
            sample_idx = 0;
        }

        if (draft_node_idx == 0) {
            readLogitsAt(phase, start_offset + draft_level, row_buf.data());
            samples = topKLogits(row_buf.data(), samples_per_draft_level_[draft_level]);
        }

        if (samples[sample_idx] == tree[static_cast<size_t>(parent_idx)]) {
            sample_idx++;
        }
        tree.push_back(samples[sample_idx++]);

        if (++draft_node_idx >= nodes_per_draft_level_[draft_level]) {
            draft_level++;
            draft_node_idx = 0;
        }
    }

    return tree;
}

// ── Draft tree verification ───────────────────────────────────────────────────

std::pair<std::vector<int32_t>, std::vector<int32_t>>
SSDModel::verifyDraftTree(const std::vector<int32_t>& draft_tree,
                          size_t phase) const {
    const size_t V = spec_.vocab_size;

    // Reusable buffer for one logit row.
    std::vector<float> row_buf(V);

    // Sample root node.
    readLogitsAt(phase, 0, row_buf.data());
    std::vector<int32_t> accepted_ids = {0};
    std::vector<int32_t> accepted_tokens = {argmaxLogits(row_buf.data())};

    for (int32_t eos_id : spec_.eos_token_ids) {
        if (accepted_tokens.back() == eos_id)
            return {accepted_tokens, accepted_ids};
    }

    for (size_t cur_idx = 1; cur_idx < num_draft_nodes_; ++cur_idx) {
        const int32_t parent_idx = attention_map_[cur_idx];
        if (parent_idx == accepted_ids.back() &&
            draft_tree[cur_idx] == accepted_tokens.back()) {
            readLogitsAt(phase, cur_idx, row_buf.data());
            int32_t verified = argmaxLogits(row_buf.data());
            accepted_tokens.push_back(verified);
            accepted_ids.push_back(static_cast<int32_t>(cur_idx));

            for (int32_t eos_id : spec_.eos_token_ids) {
                if (verified == eos_id)
                    return {accepted_tokens, accepted_ids};
            }
        }
    }

    return {accepted_tokens, accepted_ids};
}

// ── Tree attention mask (with forecast-prefix support) ────────────────────────

std::vector<float> SSDModel::buildTreeAttentionMask(size_t n_past, size_t num_tokens,
                                                     size_t seq_len, size_t kv_len,
                                                     size_t kv_prefix_offset) const {
    // Mask shape: [seq_len, kv_len + seq_len]
    // But we only fill [num_tokens, kv_len + num_tokens] within that.
    const size_t row_len = kv_len + seq_len;
    const size_t fp = ssd_cfg_.forecast_prefix;
    std::vector<float> mask(seq_len * row_len, -1e9f);

    // kv-prefix-skip / kv-prefix-offset semantics (matching Genie):
    //   All positions default to skipping the prefix [0, fp).
    //   Positions >= kv_prefix_offset have the skip UNDONE → see [0, n_past).
    //   Positions <  kv_prefix_offset keep the skip       → see [fp, n_past) only.

    for (size_t i = 0; i < num_tokens; ++i) {
        float* row = mask.data() + i * row_len;

        if (i < kv_prefix_offset) {
            // Draft tree token: skip prefix, attend to [fp, n_past).
            for (size_t j = fp; j < n_past; ++j) {
                row[j] = 0.0f;
            }
        } else {
            // Forecast token: skip undone, attend to ALL past [0, n_past).
            for (size_t j = 0; j < n_past; ++j) {
                row[j] = 0.0f;
            }
        }

        // Attend to self.
        row[kv_len + i] = 0.0f;

        // Attend to ancestors in the tree.
        int32_t ancestor = attention_map_[i];
        while (ancestor >= 0) {
            row[kv_len + static_cast<size_t>(ancestor)] = 0.0f;
            ancestor = attention_map_[static_cast<size_t>(ancestor)];
        }
    }

    return mask;
}

// ── Read logits at a single position (on-demand dequantization) ───────────────

void SSDModel::readLogitsAt(size_t phase, size_t position, float* dst) const {
    const size_t V = spec_.vocab_size;
    const size_t last_shard = spec_.shards.size() - 1;
    const size_t g_idx = graphIndex(phase, last_shard, active_cl_idx_);
    const Graph& g = graph(g_idx);

    g.read(spec_.shards.back().out_state_name, dst, V, position * V);
}

// ── Optimized selective KV update ─────────────────────────────────────────────
// Uses pre-cached tensor pointers and batches consecutive accepted positions
// into single memcpy calls instead of per-element copies.

void SSDModel::selectiveKVUpdate(const std::vector<bool>& selected,
                                  size_t n_accepted) {
    // Pre-compute copy runs: consecutive accepted positions → single memcpy.
    struct CopyRun { size_t src_start; size_t count; };
    std::vector<CopyRun> runs;
    runs.reserve(n_accepted);

    for (size_t pos = 0; pos < selected.size(); ++pos) {
        if (!selected[pos]) continue;
        if (!runs.empty() && runs.back().src_start + runs.back().count == pos) {
            runs.back().count++;
        } else {
            runs.push_back({pos, 1});
        }
    }

    // Apply copy runs to all cached KV tensors (all shards, all layers).
    for (const auto& info : kv_tensor_cache_) {
        // ── Key: [H, 1, hd, kv_len] input ← [H, 1, hd, seq_len] output ─────
        {
            auto* dst = static_cast<uint8_t*>(info.key_in_ptr);
            const auto* src = static_cast<const uint8_t*>(info.key_out_ptr);
            const size_t es = info.key_elem_size;
            const size_t in_stride  = info.key_in_kv_len;
            const size_t out_stride = info.key_out_seq_len;

            size_t dst_col = n_past_;
            for (const auto& run : runs) {
                const size_t copy_bytes = run.count * es;
                for (size_t row = 0; row < info.key_n_rows; ++row) {
                    std::memcpy(dst + (row * in_stride + dst_col) * es,
                                src + (row * out_stride + run.src_start) * es,
                                copy_bytes);
                }
                dst_col += run.count;
            }
        }

        // ── Value: [H, 1, kv_len, hd] input ← [H, 1, seq_len, hd] output ───
        {
            auto* dst = static_cast<uint8_t*>(info.val_in_ptr);
            const auto* src = static_cast<const uint8_t*>(info.val_out_ptr);
            const size_t ts = info.val_token_size;
            const size_t in_stride  = info.val_in_kv_len;
            const size_t out_stride = info.val_out_seq_len;

            size_t dst_row = n_past_;
            for (const auto& run : runs) {
                const size_t copy_bytes = run.count * ts;
                for (size_t h = 0; h < info.val_n_heads; ++h) {
                    std::memcpy(dst + (h * in_stride + dst_row) * ts,
                                src + (h * out_stride + run.src_start) * ts,
                                copy_bytes);
                }
                dst_row += run.count;
            }
        }
    }
}

// ── Tree-depth-based position IDs ─────────────────────────────────────────────
// Mirrors Genie's AttentionMask::m_cached_attention_counts logic.
// Each node's position = parent's position + 1.
// Root position = n_past - forecast_prefix (subtracting the pre-loaded prefix
// entries that occupy KV slots but are not "real" past tokens for position
// numbering purposes).

std::vector<int32_t> SSDModel::computeTreePositionIds(size_t n_past,
                                                       size_t num_tokens) const {
    const size_t fp = ssd_cfg_.forecast_prefix;
    std::vector<int32_t> pos_ids(num_tokens, 0);

    for (size_t i = 0; i < num_tokens; ++i) {
        if (attention_map_[i] < 0) {
            // Root node: position = n_past - forecast_prefix
            // (The -1 in attention_map encodes root; Genie uses
            //  n_past - |parent_value| - skip + 1, where |parent_value|=1, skip=fp)
            pos_ids[i] = static_cast<int32_t>(n_past - fp);
        } else {
            // Child node: parent's position + 1
            const size_t parent_idx = static_cast<size_t>(attention_map_[i]);
            pos_ids[i] = pos_ids[parent_idx] + 1;
        }
    }

    return pos_ids;
}

// ── Run shards with tree mask ─────────────────────────────────────────────────

void SSDModel::runShardsWithTreeMask(const std::vector<int32_t>& tokens,
                                      size_t phase, size_t n_past,
                                      size_t kv_prefix_offset) {
    const size_t num_tokens = tokens.size();
    const size_t kv_len = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_decode;
    const size_t seq_len = spec_.seq_len_decode;

    // Build tree attention mask with integrated prefix-skip.
    auto mask = buildTreeAttentionMask(n_past, num_tokens, seq_len, kv_len, kv_prefix_offset);

    // Compute tree-based position IDs and RoPE cos/sin.
    auto tree_pos = computeTreePositionIds(n_past, num_tokens);
    auto [cos_vec, sin_vec] = rope_.forward(tree_pos);

    const LLMRunContext ctx{tokens, n_past, num_tokens, phase};

    for (size_t s = 0; s < spec_.shards.size(); ++s) {
        const size_t gi = graphIndex(phase, s, active_cl_idx_);
        Graph& g = graph(gi);

        if (g.hasInput(spec_.attention_mask_name)) {
            g.write(spec_.attention_mask_name, mask.data(), mask.size());
        }

        // Write inputs via providers (token IDs, etc.).
        for (auto& provider : input_providers_) {
            provider->write(g, ctx);
        }

        // Override RoPE with tree-based position IDs (after providers write sequential ones).
        if (g.hasInput("position_ids_cos"))
            g.write("position_ids_cos", cos_vec.data(), cos_vec.size());
        if (g.hasInput("position_ids_sin"))
            g.write("position_ids_sin", sin_vec.data(), sin_vec.size());

        // Execute the graph.
        TimeLog tl;
        if (!g.execute(tl)) {
            throw std::runtime_error(
                "SSD graph execute failed: phase=" + std::to_string(phase) +
                " shard=" + std::to_string(s));
        }

        if (s + 1 < spec_.shards.size()) {
            if (phase == 0) {
                applyConnections({shard_hidden_state_[active_cl_idx_][s]});
            } else {
                applyConnections({decode_shard_hidden_state_[active_cl_idx_][s]});
            }
        }
    }
}

// ── Forecast prefix loading ───────────────────────────────────────────────────

bool SSDModel::loadForecastPrefix() {
    const std::string& path = ssd_cfg_.forecast_prefix_path;
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Cannot open forecast prefix file: %s\n", path.c_str());
        return false;
    }

    KVCacheFileHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file || header.magic != 0xC0DE) {
        fprintf(stderr, "Invalid forecast prefix file (bad magic): %s\n", path.c_str());
        return false;
    }

    const size_t n_valid = header.update_size;
    if (n_valid != ssd_cfg_.forecast_prefix) {
        fprintf(stderr, "Forecast prefix size mismatch: file has %zu, config expects %zu\n",
                n_valid, ssd_cfg_.forecast_prefix);
        return false;
    }

    const size_t n_heads_file = header.n_heads;
    const size_t embed_dim = header.embed_dim;
    const size_t H = spec_.num_kv_heads;
    const size_t hd = spec_.head_dim;

    size_t bytes_per_elem = 1;
    if (header.dtype == 1) bytes_per_elem = 2;
    else if (header.dtype == 2) bytes_per_elem = 4;

    const auto& kv_block = spec_.state_blocks[kv_state_block_idx_];

    auto fmtPat = [](const std::string& pattern, size_t idx) -> std::string {
        std::string result = pattern;
        auto pos = result.find("{}");
        if (pos != std::string::npos)
            result.replace(pos, 2, std::to_string(idx));
        return result;
    };

    struct ShardLayer { size_t shard; size_t layer; };
    std::vector<ShardLayer> all_layers;
    for (size_t s = 0; s < spec_.shards.size(); ++s) {
        for (const auto& [begin, end] : kv_block.shard_layer_ranges[s]) {
            for (size_t l = begin; l <= end; ++l) {
                all_layers.push_back({s, l});
            }
        }
    }

    // Load key data: for each layer, read H*hd rows of n_valid elements.
    // Use the prefill graph (phase=0) as handle — all phase/CL variants share
    // the same physical buffer.  Data must be loaded at prefill stride so that
    // the subsequent reshapeKV(prefill→decode) converts it together with prompt KV.
    for (const auto& [shard, layer] : all_layers) {
        const std::string key_in_name = fmtPat(kv_block.key_in_pattern, layer);

        const size_t gi = graphIndex(0, shard, active_cl_idx_);
        Graph& g = graph(gi);

        if (!g.hasInput(key_in_name)) {
            file.seekg(static_cast<std::streamoff>(n_heads_file * embed_dim * n_valid * bytes_per_elem),
                       std::ios::cur);
            continue;
        }

        const TensorSpec& in_spec = g.inputSpec(key_in_name);
        const size_t in_kv_len = in_spec.shape[3];
        const size_t elem_size = in_spec.elementSize();
        auto* dst = static_cast<uint8_t*>(g.inputPtr(key_in_name));

        const size_t n_rows = H * hd;
        for (size_t row = 0; row < n_rows; ++row) {
            file.read(reinterpret_cast<char*>(dst + row * in_kv_len * elem_size),
                      static_cast<std::streamsize>(n_valid * elem_size));
        }
        if (n_heads_file > H) {
            file.seekg(static_cast<std::streamoff>((n_heads_file - H) * embed_dim * n_valid * bytes_per_elem),
                       std::ios::cur);
        }
    }

    for (const auto& [shard, layer] : all_layers) {
        const std::string val_in_name = fmtPat(kv_block.value_in_pattern, layer);

        const size_t gi = graphIndex(0, shard, active_cl_idx_);
        Graph& g = graph(gi);

        if (!g.hasInput(val_in_name)) {
            file.seekg(static_cast<std::streamoff>(n_heads_file * n_valid * embed_dim * bytes_per_elem),
                       std::ios::cur);
            continue;
        }

        const TensorSpec& in_spec = g.inputSpec(val_in_name);
        const size_t in_kv_len = in_spec.shape[2];
        const size_t elem_size = in_spec.elementSize();
        const size_t token_size = hd * elem_size;
        auto* dst = static_cast<uint8_t*>(g.inputPtr(val_in_name));

        for (size_t h = 0; h < H; ++h) {
            file.read(reinterpret_cast<char*>(dst + h * in_kv_len * token_size),
                      static_cast<std::streamsize>(n_valid * token_size));
        }
        if (n_heads_file > H) {
            file.seekg(static_cast<std::streamoff>((n_heads_file - H) * n_valid * embed_dim * bytes_per_elem),
                       std::ios::cur);
        }
    }

    fprintf(stderr, "Loaded forecast prefix: %zu KV entries from %s\n", n_valid, path.c_str());
    return true;
}

// ── resetKVCache ──────────────────────────────────────────────────────────────

void SSDModel::resetKVCache() {
    LLMModel::resetKVCache();

    if (!ssd_cfg_.forecast_prefix_path.empty()) {
        loadForecastPrefix();
    }
    n_past_ = ssd_cfg_.forecast_prefix;
}

// ── generate (SSD override) ───────────────────────────────────────────────────

std::vector<int32_t> SSDModel::generate(const std::vector<int32_t>& prompt_tokens,
                                         const GenerationConfig& gen_cfg,
                                         std::function<void(int32_t)> token_callback) {
    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 1: Prefill (uses AR-128 graphs, with kv-prefix-skip)
    // ═══════════════════════════════════════════════════════════════════════════
    // Prompt tokens must NOT attend to the forecast prefix KV [0, fp).
    // We write a custom mask that skips positions [0, forecast_prefix).
    const size_t fp = ssd_cfg_.forecast_prefix;
    size_t tokens_processed = 0;
    const size_t total_tokens = prompt_tokens.size();
    size_t last_chunk_size = 0;

    while (tokens_processed < total_tokens) {
        const size_t remaining = total_tokens - tokens_processed;
        const size_t chunk_size = std::min(remaining, spec_.seq_len_prefill);
        last_chunk_size = chunk_size;

        const std::vector<int32_t> chunk(
            prompt_tokens.begin() + static_cast<ptrdiff_t>(tokens_processed),
            prompt_tokens.begin() + static_cast<ptrdiff_t>(tokens_processed + chunk_size));

        const size_t prefill_kv_len = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_prefill;
        const size_t prefill_seq_len = spec_.seq_len_prefill;
        const size_t row_len = prefill_kv_len + prefill_seq_len;

        // Build causal mask that skips forecast prefix [0, fp).
        std::vector<float> mask(prefill_seq_len * row_len, -1e9f);
        for (size_t row = 0; row < chunk_size; ++row) {
            float* row_ptr = mask.data() + row * row_len;
            // Attend to past KV positions [fp, n_past_) — skip prefix.
            for (size_t col = fp; col < n_past_; ++col)
                row_ptr[col] = 0.0f;
            // Causal positions in the current chunk.
            for (size_t col = 0; col <= row; ++col)
                row_ptr[prefill_kv_len + col] = 0.0f;
        }

        // Position IDs: offset by -fp so prompt starts at position 0.
        const size_t rope_n_past = n_past_ - fp;
        const LLMRunContext ctx{chunk, n_past_, chunk_size, /*phase=*/0};

        // Compute RoPE with prefix-adjusted positions.
        std::vector<int32_t> prefill_pos(chunk_size);
        for (size_t i = 0; i < chunk_size; ++i)
            prefill_pos[i] = static_cast<int32_t>(rope_n_past + i);
        auto [pf_cos, pf_sin] = rope_.forward(prefill_pos);

        for (size_t s = 0; s < spec_.shards.size(); ++s) {
            const size_t gi = graphIndex(0, s, active_cl_idx_);
            Graph& g = graph(gi);

            // Write our custom mask (skipping prefix).
            if (g.hasInput(spec_.attention_mask_name)) {
                g.write(spec_.attention_mask_name, mask.data(), mask.size());
            }

            // Write inputs via providers.
            for (auto& provider : input_providers_) {
                provider->write(g, ctx);
            }

            // Override RoPE with prefix-adjusted positions.
            if (g.hasInput("position_ids_cos"))
                g.write("position_ids_cos", pf_cos.data(), pf_cos.size());
            if (g.hasInput("position_ids_sin"))
                g.write("position_ids_sin", pf_sin.data(), pf_sin.size());

            // Execute.
            TimeLog tl;
            if (!g.execute(tl)) {
                throw std::runtime_error(
                    "SSD prefill execute failed: shard=" + std::to_string(s));
            }

            updateKV(s, /*phase=*/0, n_past_, chunk_size);
            if (s + 1 < spec_.shards.size()) {
                applyConnections({shard_hidden_state_[active_cl_idx_][s]});
            }
        }

        n_past_ += chunk_size;
        tokens_processed += chunk_size;
    }

    // Reshape KV: prefill kv_len → decode kv_len.
    {
        const size_t prefill_kv = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_prefill;
        const size_t decode_kv  = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_decode;
        for (size_t s = 0; s < spec_.shards.size(); ++s)
            reshapeKV(s, prefill_kv, decode_kv, n_past_);
    }

    // Sample first token from prefill logits.
    const size_t last_chunk_offset = last_chunk_size - 1;
    int32_t first_token = sampleNextToken(/*phase=*/0, last_chunk_offset);

    for (int32_t eos_id : spec_.eos_token_ids) {
        if (first_token == eos_id) {
            const size_t decode_kv  = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_decode;
            const size_t prefill_kv = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_prefill;
            for (size_t s = 0; s < spec_.shards.size(); ++s)
                reshapeKV(s, decode_kv, prefill_kv, n_past_);
            return {};
        }
    }

    std::vector<int32_t> output_tokens;
    output_tokens.push_back(first_token);
    if (token_callback) token_callback(first_token);

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 2: Initial SSD inference
    // ═══════════════════════════════════════════════════════════════════════════
    // Run [first_token, forecast_0, forecast_1] through AR-32.
    // kv_prefix_offset = 1: the real token (pos 0) skips the forecast prefix,
    // forecast tokens (pos >= 1) attend to the prefix. Matches Genie's behavior.
    {
        std::vector<int32_t> init_tokens = {first_token};
        for (size_t i = 0; i < draft_levels_; ++i) {
            init_tokens.push_back(static_cast<int32_t>(spec_.vocab_size + i));
        }

        // kv_prefix_offset=1: position 0 (real token) skips prefix,
        // forecast tokens (positions >= 1) attend to prefix.
        runShardsWithTreeMask(init_tokens, /*phase=*/1, n_past_,
                              /*kv_prefix_offset=*/1);

        // Commit only the real token (position 0).
        for (size_t s = 0; s < spec_.shards.size(); ++s) {
            updateKV(s, /*phase=*/1, n_past_, /*n_tok=*/1);
        }
        n_past_ += 1;
    }

    // Read logits from the initial inference and build first draft tree.
    // Only need to read forecast logit positions (1..draft_levels_), not all.
    int32_t last_accepted_token = first_token;
    auto draft_tree = buildSampleTree(last_accepted_token, /*phase=*/1, /*start_offset=*/1);

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 3: SSD Generation Loop
    // ═══════════════════════════════════════════════════════════════════════════
    const auto forecast_tokens = genForecastTokens(num_draft_nodes_);
    const size_t total_ssd_tokens = num_draft_nodes_ + forecast_tokens.size();

    // Pre-allocate reusable buffers.
    std::vector<int32_t> tokens;
    tokens.reserve(total_ssd_tokens);
    std::vector<bool> selected(total_ssd_tokens, false);

    for (int step = 0; step < gen_cfg.max_tokens; ++step) {
        if (n_past_ + total_ssd_tokens > spec_.context_lengths[active_cl_idx_]) {
            fprintf(stderr, "SSD: Context limit reached (%zu + %zu > %zu)\n",
                    n_past_, total_ssd_tokens, spec_.context_lengths[active_cl_idx_]);
            break;
        }

        // Concatenate draft tree + forecast tokens.
        tokens.clear();
        tokens.insert(tokens.end(), draft_tree.begin(), draft_tree.end());
        tokens.insert(tokens.end(), forecast_tokens.begin(), forecast_tokens.end());

        // Run all shards with tree attention mask (AR-32 variant).
        // Draft tree positions [0, num_draft_nodes_) skip prefix;
        // forecast positions [num_draft_nodes_, ...) attend to prefix.
        runShardsWithTreeMask(tokens, /*phase=*/1, n_past_,
                              /*kv_prefix_offset=*/num_draft_nodes_);

        // Verify the draft tree (reads logits on-demand, only for accepted path).
        auto [accepted_tokens, accepted_ids] = verifyDraftTree(draft_tree, /*phase=*/1);

        // Selective KV update with pre-cached pointers and batched copies.
        std::fill(selected.begin(), selected.end(), false);
        for (int32_t id : accepted_ids) {
            selected[static_cast<size_t>(id)] = true;
        }

        selectiveKVUpdate(selected, accepted_tokens.size());
        n_past_ += accepted_tokens.size();

        // Emit accepted tokens.
        bool hit_eos = false;
        for (const int32_t tok : accepted_tokens) {
            for (int32_t eos_id : spec_.eos_token_ids) {
                if (tok == eos_id) { hit_eos = true; break; }
            }
            if (hit_eos) break;
            output_tokens.push_back(tok);
            if (token_callback) token_callback(tok);
        }

        if (hit_eos) break;
        if (static_cast<int>(output_tokens.size()) >= gen_cfg.max_tokens) break;

        // Build next draft tree from the forecast logits of the last accepted node.
        const size_t next_draft_offset =
            num_draft_nodes_ + static_cast<size_t>(accepted_ids.back()) * draft_levels_;
        last_accepted_token = accepted_tokens.back();
        draft_tree = buildSampleTree(last_accepted_token, /*phase=*/1, next_draft_offset);
    }

    // Reshape KV back: decode kv_len → prefill kv_len.
    {
        const size_t decode_kv  = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_decode;
        const size_t prefill_kv = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_prefill;
        for (size_t s = 0; s < spec_.shards.size(); ++s)
            reshapeKV(s, decode_kv, prefill_kv, n_past_);
    }

    return output_tokens;
}

} // namespace geniex
