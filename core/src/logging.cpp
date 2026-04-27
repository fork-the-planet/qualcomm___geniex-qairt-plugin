// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "logging.h"

#include <iostream>

namespace geniex {

// ANSI color codes work on Windows 10+ with VT processing enabled.
static void default_log_handler(LogLevel level, const char* message) {
    const char* color = "";
    const char* label = "";
    const char* reset = "\033[0m";

    switch (level) {
        case LogLevel::Trace: color = "\033[90m"; label = "TRACE"; break;  // dark gray
        case LogLevel::Debug: color = "\033[34m"; label = "DEBUG"; break;  // blue
        case LogLevel::Info:  color = "\033[32m"; label = "INFO"; break;   // green
        case LogLevel::Warn:  color = "\033[33m"; label = "WARN"; break;   // yellow
        case LogLevel::Error: color = "\033[31m"; label = "ERROR"; break;  // red
    }

    // Info/Warn labels are 4 chars vs. 5 for others; pad so log columns align.
    const bool needs_pad = (level == LogLevel::Info || level == LogLevel::Warn);
    std::cerr << color << "[" << label << (needs_pad ? "] " : "]") << reset << " " << message << "\n";
}

LogCallback geniex_log_callback = default_log_handler;

void geniex_set_log_callback(LogCallback cb) {
    geniex_log_callback = (cb != nullptr) ? cb : default_log_handler;
}

}  // namespace geniex
