// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace geniex {

// Context describing a single forward-pass step in an LLM inference loop.
struct LLMRunContext {
    const std::vector<int32_t>& token_ids;  // token IDs for the current chunk/step
    size_t                      n_past;     // KV positions already filled
    size_t                      curr_len;   // number of tokens in this chunk/step
    size_t                      phase;      // 0 = prefill, 1 = decode
};

// Inclusive KV layer range [begin, end] owned by a single shard.
struct LayerRange {
    size_t begin;
    size_t end;  // inclusive
};

// Per-shard descriptor for hidden-state wiring between adjacent shards.
struct ShardSpec {
    std::string in_state_name;
    std::string out_state_name;
    bool        lm_head_only = false;
};

enum class StateBlockKind {
    KV,
};

// State/cache declaration for a shard-partitioned decoder runtime.
// For KV blocks, layer ranges are defined per shard and patterns map to
// key/value in/out tensors with {} replaced by layer index.
struct StateBlockSpec {
    std::string    name = "kv_default";
    StateBlockKind kind = StateBlockKind::KV;

    // Per-shard ownership of this state block.
    // Each shard may own zero, one, or many disjoint layer ranges.
    std::vector<std::vector<LayerRange>> shard_layer_ranges;

    std::string key_in_pattern    = "past_key_{}_in";
    std::string value_in_pattern  = "past_value_{}_in";
    std::string key_out_pattern   = "past_key_{}_out";
    std::string value_out_pattern = "past_value_{}_out";
};

inline std::vector<std::vector<LayerRange>> makeShardLayerRanges(
    std::vector<std::optional<LayerRange>> shard_layer_ranges) {
    std::vector<std::vector<LayerRange>> ranges;
    ranges.reserve(shard_layer_ranges.size());
    for (auto&& range : shard_layer_ranges) {
        if (range) {
            ranges.push_back({*range});
        } else {
            ranges.push_back({});
        }
    }
    return ranges;
}

inline StateBlockSpec makeKVOnlyStateBlock(
    std::vector<std::optional<LayerRange>> shard_layer_ranges, std::string name = "kv_default") {
    StateBlockSpec block;
    block.name               = std::move(name);
    block.kind               = StateBlockKind::KV;
    block.shard_layer_ranges = makeShardLayerRanges(std::move(shard_layer_ranges));
    return block;
}

inline StateBlockSpec makeKVOnlyStateBlock(
    std::vector<std::vector<LayerRange>> shard_layer_ranges, std::string name = "kv_default") {
    StateBlockSpec block;
    block.name               = std::move(name);
    block.kind               = StateBlockKind::KV;
    block.shard_layer_ranges = std::move(shard_layer_ranges);
    return block;
}

// Architecture and tensor naming parameters for a split-decoder LLM.
struct LLMSpec {
    std::vector<ShardSpec>      shards;
    std::vector<StateBlockSpec> state_blocks;

    // Tensor-shape-derived hyperparameters; populated by the loader.
    size_t hidden_size  = 0;
    size_t num_kv_heads = 0;
    size_t head_dim     = 0;
    size_t vocab_size   = 0;

    // Populated by LLMModel::onInitialized from the loaded QNN graph names.
    // Empty until the model has initialized.
    size_t              seq_len_prefill = 0;
    size_t              seq_len_decode  = 0;
    std::vector<size_t> context_lengths;

    std::string attention_mask_name = "attention_mask";

    std::vector<int32_t> eos_token_ids;

    // BOS token id from genie_config.json's `dialog.context.bos-token`.
    // -1 = not configured / no BOS for this model.
    int32_t bos_token_id = -1;
};

}  // namespace geniex
