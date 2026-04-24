#pragma once

#include "cb/session.h"

#include <cstdint>
#include <string>
#include <vector>

namespace geniex {
namespace cb {

// Result of sampling one step for a single session.
struct TokenResult {
    std::string session_id;
    int32_t     next_token_id;
};

// Greedy argmax on float32 logits laid out as [total_seq_len, vocab_size].
//
// For each session, reads the logit row at the last position of that
// session's concatenated input segment. Useful if you already have the
// dequantised logits buffer in hand; CBLLMModel otherwise uses
// sampleNextToken() which handles the graph's native output dtype.
inline std::vector<TokenResult> extractNextTokens(
    const float* logits,
    const std::vector<Session*>& sessions,
    const std::vector<int>& seg_lengths,
    int vocab_size) {
    std::vector<TokenResult> results;
    results.reserve(sessions.size());
    int offset = 0;
    for (size_t i = 0; i < sessions.size(); ++i) {
        const float* last_logit = logits + static_cast<size_t>(offset + seg_lengths[i] - 1) * vocab_size;

        int32_t best     = 0;
        float   best_val = last_logit[0];
        for (int v = 1; v < vocab_size; ++v) {
            if (last_logit[v] > best_val) {
                best_val = last_logit[v];
                best     = v;
            }
        }

        results.push_back({sessions[i]->id, best});
        offset += seg_lengths[i];
    }
    return results;
}

}  // namespace cb
}  // namespace geniex
