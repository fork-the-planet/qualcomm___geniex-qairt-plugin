// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "cb/session.h"

namespace geniex {
namespace cb {

struct TokenResult {
    std::string session_id;
    int32_t     next_token_id;
};

// Greedy argmax over an already-dequantised float32 logits buffer
// [total_seq_len, vocab_size], picking each session's last-position row.
//
// Not used by CBLLMModel — it samples through sampleNextToken() which
// handles the graph's native output dtype. Kept for callers that already
// have a float logits buffer in hand.
inline std::vector<TokenResult> extractNextTokens(
    const float* logits, const std::vector<Session*>& sessions, const std::vector<int>& seg_lengths, int vocab_size) {
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
