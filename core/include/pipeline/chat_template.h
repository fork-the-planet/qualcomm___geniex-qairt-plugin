// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>

#include "geniex_export.h"

namespace geniex {

// Signature for all chat-template formatters.
// Parameters: user_message, system_prompt, enable_thinking.
//
// NOTE: system_prompt is emitted unconditionally on every call. It is the
// caller's responsibility to pass a non-empty system_prompt only on the first
// turn of a conversation; passing it on subsequent turns will inject a second
// system block into the KV-cache context.
using ChatTemplateFunc = std::string (*)(
    const std::string& user_message, const std::string& system_prompt, bool enable_thinking);

// ChatML format — used by Qwen3, Qwen2.5, and other ChatML-based models.
GENIEX_API std::string chatMLTemplate(
    const std::string& user_message, const std::string& system_prompt, bool enable_thinking);

// Phi format — used by Phi3.5 and Phi4.
GENIEX_API std::string phiChatTemplate(
    const std::string& user_message, const std::string& system_prompt, bool enable_thinking);

}  // namespace geniex
