// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Pure-string unit tests for the LLM chat-template formatters. No NPU /
// model bundle required; runs as a CTest target on any host.

#include "pipeline/chat_template.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "geniex-proc/types.h"

namespace {

int g_failures = 0;

void expectEq(const std::string& label, const std::string& got, const std::string& want) {
    if (got == want) {
        std::cout << "[PASS] " << label << "\n";
        return;
    }
    ++g_failures;
    std::cerr << "[FAIL] " << label << "\n"
              << "  got:  " << got << "\n"
              << "  want: " << want << "\n";
}

// ─── chatMLTemplate ──────────────────────────────────────────────────────────

void chatml_singleTurn_systemAndUser_matchesLegacy() {
    // Shape produced by the SDK plugin on the first turn after
    // default-system injection.
    const std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::System, "You are a helpful AI assistant."},
        {geniex::Role::User, "Hello"},
    };
    const std::string got = geniex::chatMLTemplate(msgs, /*tools=*/{}, /*enable_thinking=*/true);
    const std::string want =
        "<|im_start|>system\n"
        "You are a helpful AI assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "Hello<|im_end|>\n"
        "<|im_start|>assistant\n";
    expectEq("chatml/single-turn parity", got, want);
}

void chatml_singleTurn_thinkingDisabled_appendsEmptyThinkBlock() {
    const std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::User, "Hi"},
    };
    const std::string got = geniex::chatMLTemplate(msgs, {}, /*enable_thinking=*/false);
    const std::string want =
        "<|im_start|>user\n"
        "Hi<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n";
    expectEq("chatml/no-thinking suffix", got, want);
}

void chatml_multiTurn_assistantAndUser_emittedInOrder() {
    const std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::System, "sys"},
        {geniex::Role::User, "q1"},
        {geniex::Role::Assistant, "a1"},
        {geniex::Role::User, "q2"},
    };
    const std::string got = geniex::chatMLTemplate(msgs, {}, true);
    const std::string want =
        "<|im_start|>system\n"
        "sys<|im_end|>\n"
        "<|im_start|>user\n"
        "q1<|im_end|>\n"
        "<|im_start|>assistant\n"
        "a1<|im_end|>\n"
        "<|im_start|>user\n"
        "q2<|im_end|>\n"
        "<|im_start|>assistant\n";
    expectEq("chatml/multi-turn", got, want);
}

void chatml_toolsBlock_withTools() {
    const std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::System, "sys"},
        {geniex::Role::User, "q"},
    };
    const geniex::ChatTools tools = {
        {"get_weather", "Returns weather", R"({"type":"object","properties":{}})"},
    };
    const std::string got = geniex::chatMLTemplate(msgs, tools, true);
    // Anchor head/signature/tail rather than the full body — the merged
    // tools-system text is verbose and only cosmetic to this test.
    const std::string head =
        "<|im_start|>system\n"
        "sys\n\n"
        "# Tools\n\n";
    const bool head_ok = got.rfind(head, 0) == 0;
    expectEq("chatml/tools head", head_ok ? "ok" : got.substr(0, head.size()), "ok");
    const bool has_sig = got.find(R"("name": "get_weather")") != std::string::npos &&
                         got.find(R"("parameters": {"type":"object","properties":{}})") != std::string::npos;
    expectEq("chatml/tools signature spliced", has_sig ? "ok" : "missing", "ok");
    const std::string tail =
        "<|im_start|>user\n"
        "q<|im_end|>\n"
        "<|im_start|>assistant\n";
    const bool tail_ok = got.size() >= tail.size() && got.compare(got.size() - tail.size(), tail.size(), tail) == 0;
    expectEq("chatml/tools tail", tail_ok ? "ok" : "tail-mismatch", "ok");
}

// ─── phiChatTemplate ─────────────────────────────────────────────────────────

void phi_singleTurn_systemAndUser_matchesLegacy() {
    const std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::System, "sys"},
        {geniex::Role::User, "Hello"},
    };
    const std::string got = geniex::phiChatTemplate(msgs, {}, true);
    const std::string want =
        "<|system|>sys<|end|>\n"
        "<|user|>\n"
        "Hello<|end|>\n"
        "<|assistant|>";
    expectEq("phi/single-turn parity", got, want);
}

void phi_multiTurn_assistantAndUser() {
    const std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::User, "q1"},
        {geniex::Role::Assistant, "a1"},
        {geniex::Role::User, "q2"},
    };
    const std::string got = geniex::phiChatTemplate(msgs, {}, true);
    const std::string want =
        "<|user|>\n"
        "q1<|end|>\n"
        "<|assistant|>a1<|end|>\n"
        "<|user|>\n"
        "q2<|end|>\n"
        "<|assistant|>";
    expectEq("phi/multi-turn", got, want);
}

// ─── llama3ChatTemplate ──────────────────────────────────────────────────────

void llama3_singleTurn_systemAndUser_matchesLegacy() {
    const std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::System, "sys"},
        {geniex::Role::User, "Hello"},
    };
    const std::string got = geniex::llama3ChatTemplate(msgs, {}, true);
    const std::string want =
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "Cutting Knowledge Date: December 2023\n"
        "sys\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Hello<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n";
    expectEq("llama3/single-turn parity", got, want);
}

void llama3_multiTurn_toolsInjectedIntoFirstUserOnly() {
    const std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::System, "sys"},
        {geniex::Role::User, "q1"},
        {geniex::Role::Assistant, "a1"},
        {geniex::Role::User, "q2"},
    };
    const geniex::ChatTools tools = {
        {"get_weather", "Returns weather", R"({"type":"object"})"},
    };
    const std::string got = geniex::llama3ChatTemplate(msgs, tools, true);
    // Tools preamble must be emitted exactly once — in the first user turn.
    const std::string preamble  = "Given the following functions, please respond";
    const auto        first_pos = got.find(preamble);
    const auto second_pos = first_pos == std::string::npos ? std::string::npos : got.find(preamble, first_pos + 1);
    expectEq("llama3/tools preamble injected once",
        first_pos != std::string::npos && second_pos == std::string::npos ? "once" : "wrong",
        "once");
    const auto q1 = got.find("\n\nq1<|eot_id|>");
    const auto q2 = got.find("\n\nq2<|eot_id|>");
    expectEq("llama3/multi-turn user content present",
        q1 != std::string::npos && q2 != std::string::npos && q1 < q2 ? "ok" : "bad-order",
        "ok");
    const std::string tail = "<|start_header_id|>assistant<|end_header_id|>\n\n";
    const bool tail_ok     = got.size() >= tail.size() && got.compare(got.size() - tail.size(), tail.size(), tail) == 0;
    expectEq("llama3/multi-turn tail", tail_ok ? "ok" : "tail-mismatch", "ok");
}

// ─── default-system-injection invariants (formatter side) ────────────────────

void noLeadingSystem_emitsNoSystemBlock() {
    // The SDK plugin is the sole owner of default-system injection; the
    // formatter must not silently emit one.
    const std::vector<geniex::ChatMessage> msgs = {
        {geniex::Role::User, "Hi"},
    };
    const std::string got = geniex::chatMLTemplate(msgs, {}, true);
    const bool        absent =
        got.find("helpful AI assistant") == std::string::npos && got.find("<|im_start|>system") == std::string::npos;
    expectEq("invariant/no implicit system", absent ? "ok" : "leaked", "ok");
}

}  // namespace

int main() {
    chatml_singleTurn_systemAndUser_matchesLegacy();
    chatml_singleTurn_thinkingDisabled_appendsEmptyThinkBlock();
    chatml_multiTurn_assistantAndUser_emittedInOrder();
    chatml_toolsBlock_withTools();

    phi_singleTurn_systemAndUser_matchesLegacy();
    phi_multiTurn_assistantAndUser();

    llama3_singleTurn_systemAndUser_matchesLegacy();
    llama3_multiTurn_toolsInjectedIntoFirstUserOnly();

    noLeadingSystem_emitsNoSystemBlock();

    if (g_failures != 0) {
        std::cerr << g_failures << " failure(s).\n";
        return 1;
    }
    std::cout << "All chat-template tests passed.\n";
    return 0;
}
