#pragma once

#include "llm/llm_model.h"
#include "llm/llm_types.h"
#include "llm/input_provider.h"
#include "llm/llm_utils.h"
#include "pipeline/chat_template.h"
#include "utils.h"
#include "logging.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace geniex {
namespace qwen3_cb {

static constexpr size_t kHeadDim   = 128;
static constexpr float  kRopeTheta = 1000000.0f;

// ════════════════════════════════════════════════════════════════════════════
// 1. Scheduler
//
// Manages the lifecycle of concurrent sessions. Tracks each session's pending
// query, how many tokens have been processed, and current status.
// ════════════════════════════════════════════════════════════════════════════

enum class SessionStatus { WAITING, RUNNING, COMPLETED };

struct Session {
    std::string          id;
    std::vector<int32_t> query_tokens;
    int                  query_len         = 0;
    int                  processed_length  = 0;
    SessionStatus        status            = SessionStatus::WAITING;
    std::vector<int32_t> generated_tokens;
    int                  generated_len     = 0;

    // Per-session generation config.
    int                  max_tokens        = 512;
    int32_t              pending_token     = 0;   // last sampled token (fed back in decode)
};

class Scheduler {
public:
    // Register a new session with its query.
    void addSession(const std::string& session_id,
                    const std::vector<int32_t>& query_tokens,
                    int max_tokens = 512) {
        Session s;
        s.id           = session_id;
        s.query_tokens = query_tokens;
        s.query_len    = static_cast<int>(query_tokens.size());
        s.max_tokens   = max_tokens;
        sessions_.push_back(std::move(s));
    }

    // Select which sessions to process in the next forward pass.
    // Fills `out` with pointers to selected sessions. Returns count.
    // max_tokens_in_batch: maximum total tokens that fit in one forward pass.
    int getNextBatch(std::vector<Session*>& out, int max_tokens_in_batch) {
        out.clear();
        int total = 0;
        for (auto& s : sessions_) {
            if (s.status == SessionStatus::COMPLETED) continue;

            int need = 0;
            if (s.processed_length < s.query_len) {
                // Prefill: remaining prompt tokens (capped to fit).
                need = std::min(s.query_len - s.processed_length,
                                max_tokens_in_batch - total);
                if (need <= 0) break;
            } else {
                // Decode: 1 token.
                if (total + 1 > max_tokens_in_batch) break;
                need = 1;
            }

            if (s.status == SessionStatus::WAITING)
                s.status = SessionStatus::RUNNING;

            out.push_back(&s);
            total += need;
        }
        return static_cast<int>(out.size());
    }

    // Update a session's processed length after a forward pass.
    void updateSession(const std::string& session_id, int new_tokens) {
        if (auto* s = getSession(session_id)) {
            s->processed_length += new_tokens;
        }
    }

    // Mark a session as completed (e.g., <EOS> generated).
    void completeSession(const std::string& session_id) {
        if (auto* s = getSession(session_id))
            s->status = SessionStatus::COMPLETED;
    }

    // Remove a completed or cancelled session.
    void removeSession(const std::string& session_id) {
        sessions_.erase(
            std::remove_if(sessions_.begin(), sessions_.end(),
                           [&](const Session& s) { return s.id == session_id; }),
            sessions_.end());
    }

    bool hasActiveSessions() const {
        for (const auto& s : sessions_)
            if (s.status != SessionStatus::COMPLETED) return true;
        return false;
    }

    Session* getSession(const std::string& id) {
        for (auto& s : sessions_)
            if (s.id == id) return &s;
        return nullptr;
    }

    std::vector<Session>& sessions() { return sessions_; }
    const std::vector<Session>& sessions() const { return sessions_; }

private:
    std::vector<Session> sessions_;
};

// ════════════════════════════════════════════════════════════════════════════
// 2. KV Cache Manager
//
// Handles multiple sessions' KV cache histories within a single contiguous
// buffer. Since inputs are concatenated (not stacked), the KV cache is laid
// out as a flat sequence across all active sessions.
//
//   |<--- Session A --->|<--- Session B --->|<--- free --->|
//
// ════════════════════════════════════════════════════════════════════════════

struct KVCacheSegment {
    std::string session_id;
    int         start_pos = 0;
    int         length    = 0;
};

class KVCacheManager {
public:
    // Allocate a KV cache segment for a session. Returns start position.
    int allocate(const std::string& session_id, int length) {
        int start = 0;
        if (!segments_.empty()) {
            const auto& last = segments_.back();
            start = last.start_pos + last.length;
        }
        segments_.push_back({session_id, start, length});
        return start;
    }

    // Extend an existing session's KV cache after processing new tokens.
    void extend(const std::string& session_id, int additional_length) {
        if (auto* seg = getSegment(session_id))
            seg->length += additional_length;
    }

    // Free the KV cache segment for a completed session.
    void release(const std::string& session_id) {
        for (auto it = segments_.begin(); it != segments_.end(); ++it) {
            if (it->session_id == session_id) {
                segments_.erase(it);
                return;
            }
        }
    }

    // Defragment: shift segments left to remove gaps.
    // Returns a list of (src_start, dst_start, length) moves that the caller
    // must apply to the actual KV buffer.
    struct MoveOp { int src; int dst; int len; };
    std::vector<MoveOp> compact() {
        std::vector<MoveOp> moves;
        int next_pos = 0;
        for (auto& seg : segments_) {
            if (seg.start_pos != next_pos && seg.length > 0) {
                moves.push_back({seg.start_pos, next_pos, seg.length});
                seg.start_pos = next_pos;
            }
            next_pos = seg.start_pos + seg.length;
        }
        return moves;
    }

    // Compute right-shifts needed before a step so that each session has room
    // to grow by `growth[session_id]` tokens. Returns a list of buffer moves
    // that the caller must apply. Also updates all segment start_pos values.
    //
    // For segments with existing data (length > 0), a MoveOp is emitted.
    // For zero-length segments (newly allocated), only start_pos is updated
    // so subsequent KV writes land at the correct position.
    std::vector<MoveOp> shiftForGrowth(
        const std::vector<std::pair<std::string, int>>& growth_list)
    {
        // Build lookup: session_id -> growth amount.
        std::unordered_map<std::string, int> growth;
        for (const auto& [id, g] : growth_list)
            growth[id] = g;

        // Compute cumulative shift for each segment.
        // Segment i must shift right by the sum of growth of segments [0..i-1].
        std::vector<int> shift(segments_.size(), 0);
        int cumul = 0;
        for (size_t i = 0; i < segments_.size(); ++i) {
            shift[i] = cumul;
            auto it = growth.find(segments_[i].session_id);
            if (it != growth.end()) cumul += it->second;
        }

        // Apply shifts right-to-left. Emit MoveOps for non-empty segments.
        std::vector<MoveOp> moves;
        for (int i = static_cast<int>(segments_.size()) - 1; i >= 0; --i) {
            auto& seg = segments_[i];
            if (shift[i] > 0) {
                if (seg.length > 0)
                    moves.push_back({seg.start_pos, seg.start_pos + shift[i], seg.length});
                seg.start_pos += shift[i];
            }
        }

        // Position zero-length segments (new sessions) at the correct place.
        // They should sit right after the preceding segment's (start + length + growth).
        for (size_t i = 0; i < segments_.size(); ++i) {
            if (segments_[i].length == 0) {
                int pos = 0;
                if (i > 0) {
                    auto it = growth.find(segments_[i - 1].session_id);
                    int prev_grow = (it != growth.end()) ? it->second : 0;
                    pos = segments_[i - 1].start_pos + segments_[i - 1].length + prev_grow;
                }
                segments_[i].start_pos = pos;
            }
        }

        return moves;
    }

    // Build position IDs for concatenated input across sessions.
    // Each session's tokens get positions [kv_length, kv_length + input_len).
    // out_pos_ids must have size >= padded_len. Returns total input length.
    static int getPositionIds(const std::vector<KVCacheSegment>& segs,
                              const std::vector<std::pair<int, int>>& in_segs,
                              int padded_len,
                              std::vector<int32_t>& out_pos_ids) {
        out_pos_ids.assign(padded_len, 0);
        int total = 0;
        for (size_t i = 0; i < segs.size(); ++i) {
            const auto [in_start, in_len] = in_segs[i];
            for (int j = 0; j < in_len; ++j)
                out_pos_ids[in_start + j] = segs[i].length + j;
            total += in_len;
        }
        return total;
    }

    // Build block-diagonal attention mask so sessions don't attend to each other.
    // Shape: flat [seq_len * (kv_len + seq_len)].
    static void getAttentionMask(
        const std::vector<KVCacheSegment>& kv_segs,
        const std::vector<std::pair<int, int>>& in_segs,
        int seq_len, int kv_len,
        std::vector<float>& out_mask)
    {
        const int W = kv_len + seq_len;
        out_mask.assign(static_cast<size_t>(seq_len) * W, -1e9f);

        for (size_t si = 0; si < kv_segs.size(); ++si) {
            const auto& kv = kv_segs[si];
            const auto [in_s, in_l] = in_segs[si];

            for (int r = 0; r < in_l; ++r) {
                float* row = out_mask.data() + (in_s + r) * W;

                // Past KV positions belonging to this session.
                for (int c = kv.start_pos; c < kv.start_pos + kv.length; ++c)
                    row[c] = 0.f;

                // Causal mask within this session's current input tokens.
                for (int c = 0; c <= r; ++c)
                    row[kv_len + in_s + c] = 0.f;
            }
        }
    }

    KVCacheSegment* getSegment(const std::string& session_id) {
        for (auto& seg : segments_)
            if (seg.session_id == session_id) return &seg;
        return nullptr;
    }

    const std::vector<KVCacheSegment>& segments() const { return segments_; }

    // Total occupied KV positions.
    int totalUsed() const {
        int end = 0;
        for (const auto& seg : segments_)
            end = std::max(end, seg.start_pos + seg.length);
        return end;
    }

private:
    std::vector<KVCacheSegment> segments_;
};

// ════════════════════════════════════════════════════════════════════════════
// 3. Next Token Extraction
//
// After a forward pass on the concatenated input, the output logits contain
// predictions for all sessions interleaved. This function extracts the last
// token logit for each session and samples from it.
// ════════════════════════════════════════════════════════════════════════════

struct TokenResult {
    std::string session_id;
    int32_t     next_token_id;
};

// Extract the next token for each session from concatenated logits.
//
// logits:       [total_seq_len][vocab_size] (flat)
// sessions:     array of active sessions in this batch
// seg_lengths:  per-session input segment length for this step
// num_sessions: count
// vocab_size:   vocabulary size
//
// For each session, picks the logit at the last position of that
// session's segment and does greedy argmax sampling.
inline std::vector<TokenResult> extractNextTokens(
    const float* logits,
    const std::vector<Session*>& sessions,
    const std::vector<int>& seg_lengths,
    int vocab_size)
{
    std::vector<TokenResult> results;
    results.reserve(sessions.size());
    int offset = 0;
    for (size_t i = 0; i < sessions.size(); ++i) {
        const float* last_logit = logits + static_cast<size_t>(offset + seg_lengths[i] - 1) * vocab_size;

        // Greedy argmax.
        int32_t best = 0;
        float best_val = last_logit[0];
        for (int v = 1; v < vocab_size; ++v) {
            if (last_logit[v] > best_val) {
                best_val = last_logit[v];
                best = v;
            }
        }

        results.push_back({sessions[i]->id, best});
        offset += seg_lengths[i];
    }
    return results;
}

// ════════════════════════════════════════════════════════════════════════════
// CBLLMModel
//
// Subclass of LLMModel that implements continuous batching by concatenating
// inputs from multiple sessions along the sequence dimension.
//
// Always uses the prefill graph (phase=0, seq_len=128). During decode, each
// active session contributes 1 token to the concatenated input. The attention
// mask is block-diagonal so sessions never attend to each other.
// ════════════════════════════════════════════════════════════════════════════

class CBLLMModel : public LLMModel {
public:
    explicit CBLLMModel(LLMSpec spec) : LLMModel(std::move(spec)) {}

    // ── Multi-session generation ───────────────────────────────────────────
    // Runs all sessions in the scheduler until every one hits EOS or max_tokens.
    void generateBatch(
        Scheduler& scheduler,
        KVCacheManager& kv_mgr,
        std::function<void(const std::string& session_id, int32_t token)> token_callback = nullptr)
    {
        const int seq_len = static_cast<int>(spec_.seq_len_prefill);
        const int vocab   = static_cast<int>(spec_.vocab_size);

        while (scheduler.hasActiveSessions()) {
            // ── 1. Scheduler: select sessions for this step ─────────────────
            std::vector<Session*> batch;
            int count = scheduler.getNextBatch(batch, seq_len);
            if (count == 0) break;

            // Build per-session input tokens and segment lengths.
            std::vector<std::vector<int32_t>> per_session_tokens(batch.size());
            std::vector<int> seg_lengths(batch.size());

            for (size_t i = 0; i < batch.size(); ++i) {
                Session& s = *batch[i];
                if (s.processed_length < s.query_len) {
                    // Prefill: next chunk of unprocessed prompt.
                    int rem = s.query_len - s.processed_length;
                    int already_used = 0;
                    for (size_t j = 0; j < i; ++j) already_used += seg_lengths[j];
                    int chunk = std::min(rem, seq_len - already_used);
                    per_session_tokens[i].assign(
                        s.query_tokens.begin() + s.processed_length,
                        s.query_tokens.begin() + s.processed_length + chunk);
                    seg_lengths[i] = chunk;
                } else {
                    // Decode: feed back last sampled token.
                    per_session_tokens[i] = {s.pending_token};
                    seg_lengths[i] = 1;
                }
            }

            // ── 2. KV Cache Manager: allocate new sessions, shift for growth ─
            // Ensure new sessions have a segment allocated.
            for (size_t i = 0; i < batch.size(); ++i) {
                if (!kv_mgr.getSegment(batch[i]->id))
                    kv_mgr.allocate(batch[i]->id, 0);
            }

            // Compute total KV needed after this step for CL selection.
            int total_growth = 0;
            std::vector<std::pair<std::string, int>> growth_list;
            for (size_t i = 0; i < batch.size(); ++i) {
                growth_list.push_back({batch[i]->id, seg_lengths[i]});
                total_growth += seg_lengths[i];
            }

            int kv_used = kv_mgr.totalUsed();
            int required_kv = kv_used + total_growth;
            selectCL(required_kv, seq_len, kv_used);

            int kv_len = static_cast<int>(spec_.context_lengths[active_cl_idx_]) - seq_len;

            // Shift existing segments right to make room for per-session growth.
            auto shift_moves = kv_mgr.shiftForGrowth(growth_list);
            for (const auto& m : shift_moves)
                moveKVBuffer(m.src, m.dst, m.len);

            // ── 3. Build concatenated inputs ────────────────────────────────
            std::vector<int32_t> concat_tokens;
            concat_tokens.reserve(seq_len);

            std::vector<std::pair<int, int>> in_segs;   // (start, len) in input
            int off = 0;
            for (size_t i = 0; i < batch.size(); ++i) {
                in_segs.push_back({off, seg_lengths[i]});
                concat_tokens.insert(concat_tokens.end(),
                                     per_session_tokens[i].begin(),
                                     per_session_tokens[i].end());
                off += seg_lengths[i];
            }
            concat_tokens.resize(seq_len, 0);  // zero-pad to graph input size

            // Gather KV segments for participating sessions (snapshot before extend).
            std::vector<KVCacheSegment> batch_kv_segs;
            for (auto* s : batch)
                batch_kv_segs.push_back(*kv_mgr.getSegment(s->id));

            // Attention mask (block-diagonal causal).
            std::vector<float> mask;
            KVCacheManager::getAttentionMask(batch_kv_segs, in_segs, seq_len, kv_len, mask);

            // Position IDs and RoPE.
            std::vector<int32_t> pos_ids;
            KVCacheManager::getPositionIds(batch_kv_segs, in_segs, seq_len, pos_ids);

            RotaryEmbedding rope(kHeadDim, kRopeTheta);
            auto [cos_vec, sin_vec] = rope.forward(pos_ids);

            // ── 4. Run all shards ───────────────────────────────────────────
            for (size_t s = 0; s < shard_count_; ++s) {
                Graph& g = graph(graphIndex(/*phase=*/0, s, active_cl_idx_));

                if (g.hasInput(spec_.attention_mask_name))
                    g.write(spec_.attention_mask_name, mask.data(), mask.size());
                if (g.hasInput("input_ids"))
                    g.write("input_ids", concat_tokens.data(), concat_tokens.size());
                if (g.hasInput("position_ids_cos"))
                    g.write("position_ids_cos", cos_vec.data(), cos_vec.size());
                if (g.hasInput("position_ids_sin"))
                    g.write("position_ids_sin", sin_vec.data(), sin_vec.size());

                TimeLog tl;
                if (!g.execute(tl)) {
                    throw std::runtime_error(
                        "CB graph execute failed: shard=" + std::to_string(s) +
                        " cl_idx=" + std::to_string(active_cl_idx_));
                }

                // Copy each session's new KV slice from output → input cache.
                copyNewKV(s, batch, batch_kv_segs, in_segs, seg_lengths);

                if (s + 1 < shard_count_)
                    applyConnections({shard_hidden_state_[active_cl_idx_][s]});
            }

            // ── 5. Update scheduler & KV manager after forward pass ─────────
            for (size_t i = 0; i < batch.size(); ++i) {
                scheduler.updateSession(batch[i]->id, seg_lengths[i]);
                kv_mgr.extend(batch[i]->id, seg_lengths[i]);
            }

            // ── 6. Next token extraction ────────────────────────────────────
            // For each session that has finished prefill, extract the next
            // token from the logit row at its last input position.
            // Uses sampleNextToken() which handles output dequantization.
            int logit_off = 0;
            for (size_t i = 0; i < batch.size(); ++i) {
                Session& s = *batch[i];
                bool done_prefill = (s.processed_length >= s.query_len);

                if (done_prefill) {
                    int last_pos = logit_off + seg_lengths[i] - 1;
                    int32_t tok = sampleNextToken(/*phase=*/0,
                                                  static_cast<size_t>(last_pos));

                    bool eos = false;
                    for (int32_t e : spec_.eos_token_ids)
                        if (tok == e) { eos = true; break; }

                    if (eos || s.generated_len >= s.max_tokens) {
                        scheduler.completeSession(s.id);
                    } else {
                        s.generated_tokens.push_back(tok);
                        s.generated_len++;
                        s.pending_token = tok;
                        if (token_callback) token_callback(s.id, tok);
                    }
                }

                logit_off += seg_lengths[i];
            }

            // ── 7. KV compaction: release completed sessions ────────────────
            for (auto& s : scheduler.sessions()) {
                if (s.status == SessionStatus::COMPLETED && kv_mgr.getSegment(s.id)) {
                    kv_mgr.release(s.id);
                }
            }
            auto compact_moves = kv_mgr.compact();
            for (const auto& m : compact_moves)
                moveKVBuffer(m.src, m.dst, m.len);
        }
    }

    // ── Single-session generate (delegates to batch) ───────────────────────
    std::vector<int32_t> generate(
        const std::vector<int32_t>& prompt_tokens,
        const GenerationConfig& gen_cfg = {},
        std::function<void(int32_t)> token_callback = nullptr) override
    {
        Scheduler sched;
        sched.addSession("s0", prompt_tokens, gen_cfg.max_tokens);
        KVCacheManager kv_mgr;
        generateBatch(sched, kv_mgr, [&](const std::string&, int32_t t) {
            if (token_callback) token_callback(t);
        });
        auto* s = sched.getSession("s0");
        return s ? s->generated_tokens : std::vector<int32_t>{};
    }

    void resetKVCache() override {
        LLMModel::resetKVCache();
    }

private:
    // ── CL selection ───────────────────────────────────────────────────────
    void selectCL(int required_kv, int seq_len, int n_valid) {
        if (num_cl_ <= 1) return;

        size_t new_cl = active_cl_idx_;
        while (new_cl + 1 < num_cl_ &&
               static_cast<int>(spec_.context_lengths[new_cl]) - seq_len < required_kv)
            ++new_cl;

        if (new_cl > active_cl_idx_) {
            GENIEX_LOG_INFO("CB: upgrading CL {} -> {} (required_kv={})",
                            active_cl_idx_, new_cl, required_kv);
            const size_t old_kv = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_prefill;
            const size_t new_kv = spec_.context_lengths[new_cl]          - spec_.seq_len_prefill;
            for (size_t s = 0; s < shard_count_; ++s)
                reshapeKV(s, old_kv, new_kv, static_cast<size_t>(n_valid));
            active_cl_idx_ = new_cl;
        }
    }

    // ── Move KV data within the shared buffer (memmove-safe) ───────────────
    void moveKVBuffer(int src_start, int dst_start, int length) {
        if (src_start == dst_start || length == 0) return;

        const auto& kv_block = requireKVStateBlock();
        for (size_t s = 0; s < shard_count_; ++s) {
            const auto& ranges = kv_block.shard_layer_ranges[s];
            if (ranges.empty()) continue;
            Graph& g = graph(graphIndex(0, s, active_cl_idx_));

            for (const auto& lr : ranges) {
                for (size_t l = lr.begin; l <= lr.end; ++l) {
                    // Key  [num_kv_heads, 1, head_dim, kv_len]
                    {
                        auto  name = fmtPattern(kv_block.key_in_pattern, l);
                        auto* buf  = static_cast<uint8_t*>(g.inputPtr(name));
                        const auto& sp = g.inputSpec(name);
                        const size_t es     = sp.elementSize();
                        const size_t n_rows = spec_.num_kv_heads * spec_.head_dim;
                        const size_t stride = sp.shape[3];
                        for (size_t r = 0; r < n_rows; ++r)
                            std::memmove(buf + (r * stride + dst_start) * es,
                                         buf + (r * stride + src_start) * es,
                                         static_cast<size_t>(length) * es);
                    }
                    // Value [num_kv_heads, 1, kv_len, head_dim]
                    {
                        auto  name = fmtPattern(kv_block.value_in_pattern, l);
                        auto* buf  = static_cast<uint8_t*>(g.inputPtr(name));
                        const auto& sp = g.inputSpec(name);
                        const size_t es     = sp.elementSize();
                        const size_t n_heads = spec_.num_kv_heads;
                        const size_t stride  = sp.shape[2];
                        const size_t tok_sz  = spec_.head_dim * es;
                        for (size_t h = 0; h < n_heads; ++h)
                            std::memmove(buf + (h * stride + dst_start) * tok_sz,
                                         buf + (h * stride + src_start) * tok_sz,
                                         static_cast<size_t>(length) * tok_sz);
                    }
                }
            }
        }
    }

    // ── Per-session KV update after graph execution ────────────────────────
    void copyNewKV(size_t shard,
                   const std::vector<Session*>& batch,
                   const std::vector<KVCacheSegment>& batch_kv_segs,
                   const std::vector<std::pair<int, int>>& in_segs,
                   const std::vector<int>& seg_lengths)
    {
        const auto& kv_block = requireKVStateBlock();
        const auto& ranges = kv_block.shard_layer_ranges[shard];
        if (ranges.empty()) return;
        Graph& g = graph(graphIndex(0, shard, active_cl_idx_));

        for (size_t si = 0; si < batch.size(); ++si) {
            const auto [in_s, in_l] = in_segs[si];
            // Destination: end of this session's current KV segment.
            size_t dst = static_cast<size_t>(batch_kv_segs[si].start_pos + batch_kv_segs[si].length);

            for (const auto& lr : ranges) {
                for (size_t l = lr.begin; l <= lr.end; ++l) {
                    copyKV(g, fmtPattern(kv_block.key_out_pattern, l), true,
                           g, fmtPattern(kv_block.key_in_pattern, l),
                           static_cast<size_t>(in_s), dst,
                           static_cast<size_t>(in_l), true);
                    copyKV(g, fmtPattern(kv_block.value_out_pattern, l), true,
                           g, fmtPattern(kv_block.value_in_pattern, l),
                           static_cast<size_t>(in_s), dst,
                           static_cast<size_t>(in_l), false);
                }
            }
        }
    }
};

// ── Model spec (same as qwen3_4b_instruct_2507_aihub) ─────────────────────

inline LLMSpec makeSpec() {
    return LLMSpec{
        .shards = {
            {"input_ids",
             "_model_model_embed_tokens_Gather_output_0"},
            {"_model_model_embed_tokens_Gather_output_0",
             "_model_model_layers_11_Add_1_output_0"},
            {"_model_model_layers_11_Add_1_output_0",
             "_model_model_layers_23_Add_1_output_0"},
            {"_model_model_layers_23_Add_1_output_0",
             "logits"},
        },
        .state_blocks = {
            makeKVOnlyStateBlock({std::nullopt,
                                  LayerRange{0, 11},
                                  LayerRange{12, 23},
                                  LayerRange{24, 35}}),
        },

        .seq_len_prefill = 128,
        .seq_len_decode  = 1,

        .hidden_size   = 2560,
        .num_heads     = 32,
        .num_kv_heads  = 8,
        .head_dim      = kHeadDim,
        .vocab_size    = 151936,

        .context_lengths = {512, 1024, 2048, 3072, 4096},

        .graph_name_pattern = "{phase}_ar{ar}_cl{cl}_{shard}_of_{total}",

        .eos_token_ids = {151645},
    };
}

inline CBLLMModel makeModel() {
    CBLLMModel m(makeSpec());
    m.addInputProvider(std::make_unique<TokenIdInputProvider>("input_ids", 151645));
    m.addInputProvider(std::make_unique<RoPEInputProvider>(kHeadDim, kRopeTheta));
    return m;
}

inline ChatTemplateFunc chatTemplate = chatMLTemplate;

} // namespace qwen3_cb
} // namespace geniex
