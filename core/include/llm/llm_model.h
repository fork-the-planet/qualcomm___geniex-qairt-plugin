#pragma once

#include "model.h"
#include "types.h"
#include "llm/llm_types.h"
#include "llm/input_provider.h"
#include "geniex_export.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace geniex {

class GENIEX_API LLMModel : public Model {
public:
    explicit LLMModel(LLMSpec spec);

    // Returns generated token IDs (excluding the prompt). Runs prefill then decode.
    // If token_callback is provided, it is called with each token immediately as it is
    // sampled (before the next decode step), enabling streaming output.
    // Return false from the callback to stop generation early (user stop).
    virtual std::vector<int32_t> generate(const std::vector<int32_t>& prompt_tokens,
                                  const GenerationConfig& gen_cfg = {},
                                  std::function<bool(int32_t)> token_callback = nullptr);

    virtual void resetKVCache();  // zeroes KV buffers and resets n_past_ / active_cl_idx_
    void saveKVCacheToFile(const std::string& path) const;
    void loadKVCacheFromFile(const std::string& path);

    size_t nPast() const;

    // Registers a CPU-side input provider.
    // Must be called before initialize().
    void addInputProvider(std::unique_ptr<InputProvider> provider);

protected:
    bool onInitialized() override;
    int32_t sampleNextToken(size_t phase, size_t token_offset = 0) const;  // argmax over logits of last shard in given phase

    static std::string fmtPattern(const std::string& pattern, size_t layer_idx);
    const StateBlockSpec& requireKVStateBlock() const;

    // phase * (shard_count_ * num_cl_) + shard * num_cl_ + cl_idx
    // phase: 0 = prefill, 1 = decode
    size_t graphIndex(size_t phase, size_t shard, size_t cl_idx) const;

    void runShard(size_t shard, size_t phase, size_t cl_idx, const LLMRunContext& ctx);

    // KV helpers (see llm_model.cpp for layout docs).
    //
    // copyKV: strided copy between two distinct buffers (output→input after execution).
    void copyKV(Graph& src_g, const std::string& src_name, bool src_is_output,
                Graph& dst_g, const std::string& dst_name,
                size_t src_off, size_t dst_off, size_t n_tok, bool is_key);
    void updateKV(size_t s, size_t phase, size_t dst_off, size_t n_tok);

    // reshapeKV: in-place KV rearrangement for kv_len stride change within the shared buffer.
    // Moves the first n_valid token positions from the old layout (stride=old_kv_len) to
    // the new layout (stride=new_kv_len).  Expanding iterates backward; contracting forward.
    // Uses memmove to handle overlapping regions.
    void reshapeKV(size_t shard, size_t old_kv_len, size_t new_kv_len, size_t n_valid);

    LLMSpec spec_;
    // CPU-side input providers (embeddings, positional encoding, etc.).
    std::vector<std::unique_ptr<InputProvider>> input_providers_;

    // Total graphs = 2 × shard_count_ × num_cl_
    size_t shard_count_ = 0;
    size_t num_cl_      = 0;
    size_t active_cl_idx_ = 0;  // index into spec_.context_lengths; advances during prefill

    // Outer index = CL variant.
    std::vector<std::vector<Connection>> shard_hidden_state_;       // hidden state across adjacent prefill shards
    std::vector<std::vector<Connection>> decode_shard_hidden_state_; // hidden state across adjacent decode shards

    size_t kv_state_block_idx_ = std::numeric_limits<size_t>::max();
    size_t n_past_ = 0;

private:
    void buildConnections();  // populates the four connection tables

    // Returns the set of all KV input tensor names across all shards and layers,
    // derived from the configured KV state block patterns. Used by
    // resetKVCache / saveKVCacheToFile / loadKVCacheFromFile.
    std::unordered_set<std::string> buildKVInputNameSet() const;
};

} // namespace geniex
