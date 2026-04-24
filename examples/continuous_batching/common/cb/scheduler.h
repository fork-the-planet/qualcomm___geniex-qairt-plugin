#pragma once

#include "cb/session.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace geniex {
namespace cb {

// ────────────────────────────────────────────────────────────────────────────
// Scheduler
//
// Model-agnostic session bookkeeping for continuous batching. Selects which
// sessions participate in each forward pass and advances their processed
// length after the pass completes.
//
// Packing policy: sessions are visited in insertion order. Each session
// either contributes its unprocessed prefill tokens (capped so the total
// does not exceed `max_tokens_in_batch`) or a single decode token. Packing
// stops as soon as the next session cannot fit.
// ────────────────────────────────────────────────────────────────────────────
class Scheduler {
public:
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

    // Fills `out` with pointers to sessions that should run this step.
    // Returns the number of selected sessions.
    int getNextBatch(std::vector<Session*>& out, int max_tokens_in_batch) {
        out.clear();
        int total = 0;
        for (auto& s : sessions_) {
            if (s.status == SessionStatus::COMPLETED) continue;

            int need = 0;
            if (s.processed_length < s.query_len) {
                need = std::min(s.query_len - s.processed_length,
                                max_tokens_in_batch - total);
                if (need <= 0) break;
            } else {
                if (total + 1 > max_tokens_in_batch) break;
                need = 1;
            }

            if (s.status == SessionStatus::WAITING) s.status = SessionStatus::RUNNING;

            out.push_back(&s);
            total += need;
        }
        return static_cast<int>(out.size());
    }

    void updateSession(const std::string& session_id, int new_tokens) {
        if (auto* s = getSession(session_id)) s->processed_length += new_tokens;
    }

    void completeSession(const std::string& session_id) {
        if (auto* s = getSession(session_id)) s->status = SessionStatus::COMPLETED;
    }

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

    std::vector<Session>&       sessions()       { return sessions_; }
    const std::vector<Session>& sessions() const { return sessions_; }

private:
    std::vector<Session> sessions_;
};

}  // namespace cb
}  // namespace geniex
