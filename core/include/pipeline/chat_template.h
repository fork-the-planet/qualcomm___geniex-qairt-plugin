// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>
#include <vector>

#include "geniex-proc/types.h"  // ChatMessage, Role, ChatTool, ChatTools
#include "geniex_export.h"

namespace geniex {

// Signature for all chat-template formatters.
//
// Message-list contract:
//   - At most one Role::System message; if present it must be the first
//     entry. Formatters MUST NOT inject a hard-coded default — callers that
//     want one prepend it on the first turn only, otherwise it would be
//     re-emitted into the cached KV prefix on every later turn.
//   - Trailing message is expected to be Role::User; formatters end the
//     output with the assistant generation header.
//
// Currently dropped: Role::Tool, ChatMessage::tool_calls, tool_call_id,
// name, reasoning_content, mm_content. mm_content is VLM territory.
//
// `tools` is rendered into the system block by formatters that support it;
// others ignore it.
using ChatTemplateFunc = std::string (*)(
    const std::vector<ChatMessage>& messages, const ChatTools& tools, bool enable_thinking);

// ChatML format — used by Qwen3, Qwen2.5, and other ChatML-based models.
GENIEX_API std::string chatMLTemplate(
    const std::vector<ChatMessage>& messages, const ChatTools& tools, bool enable_thinking);

// Phi format — used by Phi3.5 and Phi4.
GENIEX_API std::string phiChatTemplate(
    const std::vector<ChatMessage>& messages, const ChatTools& tools, bool enable_thinking);

// Llama3 header-id format — used by Llama 3, Llama 3.1, and Llama 3.2 model families.
GENIEX_API std::string llama3ChatTemplate(
    const std::vector<ChatMessage>& messages, const ChatTools& tools, bool enable_thinking);

}  // namespace geniex
