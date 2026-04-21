#pragma once

#include "llm/llm_model.h"
#include "llm/llm_utils.h"
#include "ssd_types.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace geniex {

// Self-Speculative Decoding model.
//
// Subclasses LLMModel to override the decode loop with tree-based speculative
// generation. Prefill is inherited unchanged from LLMModel.
//
// The SSD model uses two AR variants per shard:
//   - AR-128 (seq_len_prefill): standard prompt prefill
//   - AR-32  (seq_len_decode):  SSD tree verification (draft tree + forecast tokens)
//
// The decode loop builds a draft tree from forecast logits, runs all candidates
// through the AR-32 variant in one forward pass, verifies greedily, and commits
// only accepted KV positions — typically producing 2-3 tokens per iteration.
class SSDModel : public LLMModel {
public:
    SSDModel(LLMSpec spec, SSDConfig ssd_cfg);

    // Overrides the standard generate() to use SSD decode loop.
    // Return false from the callback to stop generation early (user stop).
    std::vector<int32_t> generate(const std::vector<int32_t>& prompt_tokens,
                                  const GenerationConfig& gen_cfg = {},
                                  std::function<bool(int32_t)> token_callback = nullptr) override;

    void resetKVCache() override;

protected:
    bool onInitialized() override;

private:
    // ── SSD decode loop ──────────────────────────────────────────────────────

    // Builds the static attention map from the branch configuration.
    // Returns a flat array of parent indices encoding the draft tree structure
    // and forecast token links. Also populates num_draft_nodes_,
    // samples_per_draft_level_, and nodes_per_draft_level_.
    // Called once from the constructor.
    std::vector<int32_t> genAttentionMap();

    // Generates forecast token IDs repeated `repeat` times.
    // Forecast tokens are [vocab_size, vocab_size + draft_levels - 1].
    std::vector<int32_t> genForecastTokens(size_t repeat) const;

    // Builds a draft tree from the forecast logits of the previous iteration.
    // Reads logits on-demand from the graph output at start_offset.
    std::vector<int32_t> buildSampleTree(int32_t last_token,
                                         size_t phase,
                                         size_t start_offset) const;

    // Verifies the draft tree against verified logits. Reads logits on-demand.
    // Returns:
    //   first:  accepted token IDs (in tree-walk order)
    //   second: accepted node indices within the draft tree
    std::pair<std::vector<int32_t>, std::vector<int32_t>>
    verifyDraftTree(const std::vector<int32_t>& draft_tree,
                    size_t phase) const;

     // Reads logits for a single position directly from the quantized output buffer.
    // Dequantizes only the requested row (vocab_size elements).
    void readLogitsAt(size_t phase, size_t position, float* dst) const;

    // Argmax over a single logit row.
    int32_t argmaxLogits(const float* logits_row) const;

    // Top-k sampling from a single logit row. Returns k token IDs sorted by
    // descending logit value.
    std::vector<int32_t> topKLogits(const float* logits_row, size_t k) const;

    // Builds a tree-structured attention mask for SSD decode.
    // The mask is shaped [num_tokens, kv_len + num_tokens] where num_tokens
    // is the total draft tree + forecast tokens.
    // kv_prefix_offset: token positions < offset skip the forecast prefix KV
    // entries [0, forecast_prefix); positions >= offset attend to the prefix.
    std::vector<float> buildTreeAttentionMask(size_t n_past, size_t num_tokens,
                                              size_t seq_len, size_t kv_len,
                                              size_t kv_prefix_offset) const;

    // Selectively commits KV cache entries for accepted positions only.
    // selected[i] = true means position i should be kept.
    void selectiveKVUpdate(const std::vector<bool>& selected, size_t n_accepted);

    // Loads the forecast prefix KV cache from disk into the graph input buffers.
    bool loadForecastPrefix();

    // Runs all shards for a given phase with tree attention mask override.
    // kv_prefix_offset: positions < offset skip forecast prefix in attention;
    //                   positions >= offset attend to the full KV including prefix.
    void runShardsWithTreeMask(const std::vector<int32_t>& tokens,
                               size_t phase, size_t n_past,
                               size_t kv_prefix_offset);

    // Computes tree-depth-based position IDs for SSD tokens.
    // Each node's position = parent's position + 1.
    // Root position = n_past - forecast_prefix (matching Genie's logic).
    std::vector<int32_t> computeTreePositionIds(size_t n_past,
                                                 size_t num_tokens) const;

    // ── SSD state ────────────────────────────────────────────────────────────
    SSDConfig ssd_cfg_;
    RotaryEmbedding rope_;  // for tree-based RoPE override during SSD decode

    size_t draft_levels_ = 0;           // = branches.size()
    size_t num_draft_nodes_ = 0;        // total nodes in draft tree (including root)
    std::vector<int32_t> attention_map_; // static tree structure
    std::vector<size_t> samples_per_draft_level_; // top-k count per level
    std::vector<size_t> nodes_per_draft_level_;   // node count per level

    // ── Pre-cached KV tensor info (populated at onInitialized) ───────────────
    // Avoids re-parsing tensor names and looking up specs on every SSD iteration.
    struct KVTensorInfo {
        size_t shard;
        size_t layer;
        // Key: [H, 1, hd, kv_len] input / [H, 1, hd, seq_len] output
        void*       key_in_ptr  = nullptr;
        const void* key_out_ptr = nullptr;
        size_t key_in_kv_len  = 0;  // stride in token dimension
        size_t key_out_seq_len = 0;
        size_t key_elem_size  = 0;
        size_t key_n_rows     = 0;  // H * hd
        // Value: [H, 1, kv_len, hd] input / [H, 1, seq_len, hd] output
        void*       val_in_ptr  = nullptr;
        const void* val_out_ptr = nullptr;
        size_t val_in_kv_len  = 0;
        size_t val_out_seq_len = 0;
        size_t val_token_size = 0;  // hd * elem_size
        size_t val_n_heads    = 0;  // H
    };
    std::vector<KVTensorInfo> kv_tensor_cache_;
};

} // namespace geniex
