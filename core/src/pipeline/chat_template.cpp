// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "pipeline/chat_template.h"

namespace geniex {

std::string chatMLTemplate(const std::string& user_message,
                           const std::string& system_prompt,
                           bool enable_thinking) {
    std::string result;
    if (!system_prompt.empty())
        result += "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
    result += "<|im_start|>user\n" + user_message + "<|im_end|>\n";
    result += "<|im_start|>assistant\n";
    if (!enable_thinking)
        result += "<think>\n\n</think>\n\n";
    return result;
}

std::string phiChatTemplate(const std::string& user_message,
                            const std::string& system_prompt,
                            bool /*enable_thinking*/) {
    std::string result;
    std::string sys = system_prompt.empty()
                          ? "You are a helpful assistant."
                          : system_prompt;
    result += "<|system|>" + sys + "<|end|>\n";
    result += "<|user|>\n" + user_message + "<|end|>\n";
    result += "<|assistant|>\n";
    return result;
}

} // namespace geniex
