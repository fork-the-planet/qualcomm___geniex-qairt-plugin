#include "logging.h"

#include <iostream>

namespace geniex {

// ── Default colorized stderr handler ─────────────────────────────────────────

static void default_log_handler(LogLevel level, const char* message) {
    // ANSI color codes (work on Windows 10+ with VT processing enabled)
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

    // Format: [LEVEL] message  — fixed-width by padding shorter labels with a trailing space
    // TRACE/DEBUG/ERROR = 5 chars, INFO/WARN = 4 chars → pad to 5 for alignment
    const bool needs_pad = (level == LogLevel::Info || level == LogLevel::Warn);
    std::cerr << color << "[" << label << (needs_pad ? "] " : "]") << reset << " " << message << "\n";
}

// ── Global callback ───────────────────────────────────────────────────────────

LogCallback geniex_log_callback = default_log_handler;

void geniex_set_log_callback(LogCallback cb) {
    geniex_log_callback = (cb != nullptr) ? cb : default_log_handler;
}

}  // namespace geniex
