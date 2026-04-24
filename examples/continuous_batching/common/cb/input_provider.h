#pragma once

#include "cb/kv_cache_manager.h"
#include "cb/session.h"
#include "graph.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace geniex {
namespace cb {

// ────────────────────────────────────────────────────────────────────────────
// CBStepContext
//
// Everything a CBInputProvider needs for a single forward pass. Unlike the
// single-session LLMRunContext (which carries one `n_past` and `curr_len`),
// a CB step runs multiple sessions concatenated along the sequence axis, so
// per-session offsets are needed to build position-dependent inputs
// (position IDs, RoPE cos/sin, embeddings, etc.).
//
// Shapes & invariants:
//   - concat_tokens.size() == seq_len (graph AR length); padding is zero.
//   - in_segs[i] = {start, len} of session i inside concat_tokens.
//   - kv_segs[i].length = how many past tokens session i already has in KV.
//   - sum of in_segs[i].len <= seq_len.
// ────────────────────────────────────────────────────────────────────────────
struct CBStepContext {
    // Active sessions for this step, same ordering as in_segs / kv_segs.
    std::vector<Session*> sessions;

    // Input-side layout: {start_offset, length} in the concatenated buffer.
    std::vector<std::pair<int, int>> in_segs;

    // KV-side snapshot taken before this step's growth — so `kv_segs[i].length`
    // is the number of past tokens session i has, and position IDs start there.
    std::vector<KVCacheSegment> kv_segs;

    // Concatenated input token IDs, zero-padded to seq_len.
    std::vector<int32_t> concat_tokens;

    int seq_len = 0;  // graph AR length (e.g. 128)
    int kv_len  = 0;  // per-graph KV capacity for the active CL variant
};

// ────────────────────────────────────────────────────────────────────────────
// CBInputProvider
//
// Interface a model plugs in to write model-specific per-step inputs into a
// graph. Exactly analogous to the single-session `InputProvider`, but takes a
// `CBStepContext` so implementations can build per-session outputs.
//
// Each implementation must silently no-op if its target tensor isn't present
// on the current shard (use Graph::hasInput()).
//
// Canonical implementations a model typically needs:
//   1. Token-id / embedding writer  — writes `input_ids` or `input_embeds`.
//   2. RoPE cos/sin writer          — writes position_ids_cos / position_ids_sin.
//   3. Attention mask writer        — writes the attention mask.
//
// The attention mask is a standard block-diagonal causal mask
// (KVCacheManager::getAttentionMask) and is written by CBLLMModel itself,
// so models typically only need to implement (1) and (2).
// ────────────────────────────────────────────────────────────────────────────
class CBInputProvider {
public:
    virtual ~CBInputProvider() = default;

    // Called once after the Model is fully initialised, e.g. to load an
    // embedding table from model_cfg.embedding_path.
    virtual void onInitialized(const ModelConfig& /*model_cfg*/) {}

    // Write tensor(s) into g for this CB step.
    virtual void write(Graph& g, const CBStepContext& ctx) = 0;
};

}  // namespace cb
}  // namespace geniex
