// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "cb/kv_cache_manager.h"
#include "cb/session.h"
#include "graph.h"

namespace geniex {
namespace cb {

// Per-step input bundle for a CB forward pass. The CB counterpart of
// single-session LLMRunContext — carries per-session offsets so providers
// can build position-dependent inputs (position IDs, RoPE cos/sin, embeddings).
//
// Invariants:
//   - concat_tokens.size() == seq_len (padding is zero).
//   - in_segs[i] = {start, len} of session i inside concat_tokens.
//   - kv_segs[i].length = past tokens session i already has in KV.
//   - sum of in_segs[i].len <= seq_len.
struct CBStepContext {
    // Same ordering across sessions / in_segs / kv_segs.
    std::vector<Session*>            sessions;
    std::vector<std::pair<int, int>> in_segs;

    // Snapshot taken *before* this step's growth — position IDs start at kv_segs[i].length.
    std::vector<KVCacheSegment> kv_segs;

    std::vector<int32_t> concat_tokens;

    int seq_len = 0;  // graph AR length (e.g. 128)
    int kv_len  = 0;  // per-graph KV capacity for the active CL variant
};

// Writes model-specific per-step inputs into a graph — the CB counterpart
// of single-session `InputProvider`. Must no-op silently when its target
// tensor isn't present on the current shard (check Graph::hasInput()).
//
// A model typically registers a token-id / embedding writer and a RoPE
// cos/sin writer. The attention mask is block-diagonal causal and is
// written by CBLLMModel itself via KVCacheManager::getAttentionMask.
class CBInputProvider {
   public:
    virtual ~CBInputProvider() = default;

    // Hook for one-time setup after Model init (e.g. loading a CPU-side
    // embedding table from model_cfg.embedding_path).
    virtual void onInitialized(const ModelConfig& /*model_cfg*/) {}

    virtual void write(Graph& g, const CBStepContext& ctx) = 0;
};

}  // namespace cb
}  // namespace geniex
