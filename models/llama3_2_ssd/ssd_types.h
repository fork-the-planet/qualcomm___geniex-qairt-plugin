#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace geniex {

// Configuration for Self-Speculative Decoding (SSD).
//
// SSD accelerates autoregressive generation by speculatively drafting multiple
// candidate tokens in a tree structure, verifying them in a single forward pass
// using a larger AR variant (e.g. AR-32), and accepting as many as match.
//
// The same model is used for both drafting and verification — no separate draft
// model is required. Forecast tokens (special token IDs beyond the vocabulary)
// allow the model to produce logits for future speculation positions.
struct SSDConfig {
    // Tree branching factor per draft level. Array length = number of draft
    // levels. Each element = number of child branches per parent node at that
    // level.
    //
    // Example: branches = {3, 2} → 2-level draft tree:
    //   root → 3 candidates → 2 children each = 10 nodes total.
    //   Each node gets `draft_levels` forecast positions → 20 forecast tokens.
    //   Total tokens per SSD iteration = 10 + 20 = 30.
    std::vector<size_t> branches = {3, 2};

    // Number of pre-computed KV entries for forecast tokens, loaded from disk
    // at initialization. The KV cache starts at this position.
    size_t forecast_prefix = 16;

    // Full path to the forecast prefix KV cache file.
    std::string forecast_prefix_path;

    // RoPE base frequency for tree-based position encoding during SSD decode.
    // head_dim is taken from LLMSpec.
    float rope_theta = 500000.0f;
};

// Binary file header for the Genie KV cache format.
// Used to load forecast prefix KV entries from disk.
#pragma pack(push, 1)
struct KVCacheFileHeader {
    uint32_t num_tensors;   // Number of K+V tensors (2× per layer)
    uint32_t magic;         // Magic number: 0xC0DE
    uint8_t  dtype;         // Data type (0=uint8, 1=float16, 2=float32)
    uint8_t  pad;
    uint16_t n_heads;       // Number of KV heads
    uint16_t embed_dim;     // Embedding dimension per head (head_dim)
    uint16_t update_size;   // Number of KV entries stored
};
#pragma pack(pop)
static_assert(sizeof(KVCacheFileHeader) == 16, "KVCacheFileHeader must be 16 bytes");

} // namespace geniex
