// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace geniex {
namespace cb {

// One session's slice inside the concatenated KV cache buffer.
//
//   |<-- Session A (len=20) -->|<-- Session B (len=8) -->|   free   |
struct KVCacheSegment {
    std::string session_id;
    int         start_pos = 0;  // token offset of this session's cache
    int         length    = 0;  // valid tokens currently held
};

// Bookkeeping for per-session segments inside the graph's single contiguous
// KV buffer. Speaks only in integer offsets: callers apply the returned
// `MoveOp` lists to the physical tensors themselves.
class KVCacheManager {
   public:
    struct MoveOp {
        int src;
        int dst;
        int len;
    };

    // Append a new segment at the tail. Returns the assigned start position.
    int allocate(const std::string& session_id, int length) {
        int start = 0;
        if (!segments_.empty()) {
            const auto& last = segments_.back();
            start            = last.start_pos + last.length;
        }
        segments_.push_back({session_id, start, length});
        return start;
    }

    // Grow a session's segment after its forward pass produced new tokens.
    void extend(const std::string& session_id, int additional_length) {
        if (auto* seg = getSegment(session_id)) seg->length += additional_length;
    }

    // Drop a session's segment. Leaves a hole; call compact() to reclaim it.
    void release(const std::string& session_id) {
        for (auto it = segments_.begin(); it != segments_.end(); ++it) {
            if (it->session_id == session_id) {
                segments_.erase(it);
                return;
            }
        }
    }

    // Shift every segment left to eliminate gaps. The returned moves must
    // be applied to the physical KV tensor by the caller.
    std::vector<MoveOp> compact() {
        std::vector<MoveOp> moves;
        int                 next_pos = 0;
        for (auto& seg : segments_) {
            if (seg.start_pos != next_pos && seg.length > 0) {
                moves.push_back({seg.start_pos, next_pos, seg.length});
                seg.start_pos = next_pos;
            }
            next_pos = seg.start_pos + seg.length;
        }
        return moves;
    }

    // Shift segments right to make room for each session's growth.
    //
    // For a session growing by G tokens, every segment after it shifts right
    // by G. MoveOps are emitted right-to-left so callers can memmove them in
    // order without overwriting live data. Zero-length segments emit no
    // MoveOp — only their `start_pos` is updated so the upcoming KV write
    // lands in the right place.
    std::vector<MoveOp> shiftForGrowth(const std::vector<std::pair<std::string, int>>& growth_list) {
        std::unordered_map<std::string, int> growth;
        for (const auto& [id, g] : growth_list) growth[id] = g;

        std::vector<int> shift(segments_.size(), 0);
        int              cumul = 0;
        for (size_t i = 0; i < segments_.size(); ++i) {
            shift[i] = cumul;
            auto it  = growth.find(segments_[i].session_id);
            if (it != growth.end()) cumul += it->second;
        }

        std::vector<MoveOp> moves;
        for (int i = static_cast<int>(segments_.size()) - 1; i >= 0; --i) {
            auto& seg = segments_[i];
            if (shift[i] > 0) {
                if (seg.length > 0) moves.push_back({seg.start_pos, seg.start_pos + shift[i], seg.length});
                seg.start_pos += shift[i];
            }
        }

        // Zero-length segments have no data to move — seat them right after
        // the predecessor's post-growth end so the next KV write lands there.
        for (size_t i = 0; i < segments_.size(); ++i) {
            if (segments_[i].length == 0) {
                int pos = 0;
                if (i > 0) {
                    auto it        = growth.find(segments_[i - 1].session_id);
                    int  prev_grow = (it != growth.end()) ? it->second : 0;
                    pos            = segments_[i - 1].start_pos + segments_[i - 1].length + prev_grow;
                }
                segments_[i].start_pos = pos;
            }
        }

        return moves;
    }

    // Build position IDs for the concatenated input.
    //
    // Session i's input tokens get positions [kv_length_i, kv_length_i + in_len).
    // Padding slots stay 0 and must be ignored via the attention mask.
    //
    // Resizes `out_pos_ids` to `padded_len`; returns the total valid token count.
    static int getPositionIds(const std::vector<KVCacheSegment>& segs, const std::vector<std::pair<int, int>>& in_segs,
        int padded_len, std::vector<int32_t>& out_pos_ids) {
        out_pos_ids.assign(padded_len, 0);
        int total = 0;
        for (size_t i = 0; i < segs.size(); ++i) {
            const auto [in_start, in_len] = in_segs[i];
            for (int j = 0; j < in_len; ++j) out_pos_ids[in_start + j] = segs[i].length + j;
            total += in_len;
        }
        return total;
    }

    // Block-diagonal causal attention mask for concatenated sessions.
    // Required for correctness: without it, session A would attend to B's
    // KV cache.
    //
    // Layout (flat [seq_len * (kv_len + seq_len)]):
    //          KV cache columns     Current input columns
    //          A-seg    B-seg       A-input    B-input
    //  A row: [ allow    block       causal     block   ]
    //  B row: [ block    allow       block      causal  ]
    static void getAttentionMask(const std::vector<KVCacheSegment>& kv_segs,
        const std::vector<std::pair<int, int>>& in_segs, int seq_len, int kv_len, std::vector<float>& out_mask) {
        const int W = kv_len + seq_len;
        out_mask.assign(static_cast<size_t>(seq_len) * W, -1e9f);

        for (size_t si = 0; si < kv_segs.size(); ++si) {
            const auto& kv          = kv_segs[si];
            const auto [in_s, in_l] = in_segs[si];

            for (int r = 0; r < in_l; ++r) {
                float* row = out_mask.data() + (in_s + r) * W;
                for (int c = kv.start_pos; c < kv.start_pos + kv.length; ++c) row[c] = 0.f;
                for (int c = 0; c <= r; ++c) row[kv_len + in_s + c] = 0.f;
            }
        }
    }

    KVCacheSegment* getSegment(const std::string& session_id) {
        for (auto& seg : segments_)
            if (seg.session_id == session_id) return &seg;
        return nullptr;
    }

    const std::vector<KVCacheSegment>& segments() const { return segments_; }

    // End of the rightmost segment, i.e. the first free offset.
    int totalUsed() const {
        int end = 0;
        for (const auto& seg : segments_) end = std::max(end, seg.start_pos + seg.length);
        return end;
    }

   private:
    std::vector<KVCacheSegment> segments_;
};

}  // namespace cb
}  // namespace geniex
