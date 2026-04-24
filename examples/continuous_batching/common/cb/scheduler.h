#pragma once

#include "cb/session.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace geniex {
namespace cb {

// Selects which sessions run in each forward pass and advances their
// processed length afterward.
//
// Packing policy: insertion order, first-fit. A session contributes either
// its next prefill chunk (capped by the remaining sequence budget) or a
// single decode token; packing stops as soon as the next session would
// exceed `max_tokens_in_batch`.
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

    // Fills `out` with sessions to run this step and returns the count.
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
