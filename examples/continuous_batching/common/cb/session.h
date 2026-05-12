// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace geniex {
namespace cb {

enum class SessionStatus { WAITING, RUNNING, COMPLETED };

// One concurrent user request. Holds both generation config and running
// state so Scheduler and CBLLMModel share a single source of truth.
struct Session {
    std::string          id;
    std::vector<int32_t> query_tokens;
    int                  query_len        = 0;
    int                  processed_length = 0;
    SessionStatus        status           = SessionStatus::WAITING;
    std::vector<int32_t> generated_tokens;
    int                  generated_len = 0;

    int     max_tokens    = 512;
    int32_t pending_token = 0;  // last sampled token, fed back in decode
};

}  // namespace cb
}  // namespace geniex
