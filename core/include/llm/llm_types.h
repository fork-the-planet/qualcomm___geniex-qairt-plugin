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

// Per-shard descriptor for hidden-state wiring between adjacent shards.
struct ShardSpec {
    std::string in_state_name;
    std::string out_state_name;
    bool        lm_head_only = false;
};

enum class StateBlockKind {
    KV,
};

// The four graph tensor names that carry one key/value cache entry.
struct KVTensorPair {
    std::string key_in;
    std::string key_out;
    std::string value_in;
    std::string value_out;
};

// Declares one key/value cache owned by a shard-partitioned decoder.
// The patterns name its tensors ({} = layer index); shard_pairs holds the
// tensors each shard owns, resolved during initialization.
struct StateBlockSpec {
    std::string    name = "kv_default";
    StateBlockKind kind = StateBlockKind::KV;

    std::string key_in_pattern    = "past_key_{}_in";
    std::string value_in_pattern  = "past_value_{}_in";
    std::string key_out_pattern   = "past_key_{}_out";
    std::string value_out_pattern = "past_value_{}_out";

    std::vector<std::vector<KVTensorPair>> shard_pairs;
};

inline StateBlockSpec makeKVStateBlock(std::string name = "kv_default") {
    StateBlockSpec block;
    block.name = std::move(name);
    block.kind = StateBlockKind::KV;
    return block;
}

// Architecture and tensor naming parameters for a split-decoder LLM.
struct LLMSpec {
    std::vector<ShardSpec>      shards;
    std::vector<StateBlockSpec> state_blocks;

    // Inferred from the loaded graph tensors.
    size_t hidden_size  = 0;
    size_t num_kv_heads = 0;
    size_t head_dim     = 0;
    size_t vocab_size   = 0;

    // Inferred from the loaded graph names.
    size_t              seq_len_prefill = 0;
    size_t              seq_len_decode  = 0;
    std::vector<size_t> context_lengths;

    std::string attention_mask_name = "attention_mask";

    std::vector<int32_t> eos_token_ids;

    // -1 = no BOS configured.
    int32_t bos_token_id = -1;
};

}  // namespace geniex
