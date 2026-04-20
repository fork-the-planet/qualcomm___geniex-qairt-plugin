#pragma once

#include <string>

#include "geniex_export.h"

namespace geniex {

// Signature for all chat-template formatters.
// Parameters: user_message, system_prompt, first_turn, enable_thinking.
using ChatTemplateFunc = std::string(*)(
    const std::string& user_message,
    const std::string& system_prompt,
    bool first_turn,
    bool enable_thinking);

// ── Common chat templates (used by multiple model families) ──────────────────

// ChatML format — used by Qwen3, Qwen2.5, and other ChatML-based models.
GENIEX_API std::string chatMLTemplate(
    const std::string& user_message,
    const std::string& system_prompt,
    bool first_turn, bool enable_thinking);

// Phi format — used by Phi3.5 and Phi4.
GENIEX_API std::string phiChatTemplate(
    const std::string& user_message,
    const std::string& system_prompt,
    bool first_turn, bool enable_thinking);

} // namespace geniex
