// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "geniex-proc/types.h"  // ChatMessage
#include "geniex_export.h"
#include "llm/llm_model.h"
#include "pipeline/chat_template.h"
#include "types.h"

namespace geniex {

struct GenerateResult {
    std::string full_text;
    int64_t     prompt_tokens     = 0;
    int64_t     generated_tokens  = 0;
    double      ttft_ms           = 0.0;  // time-to-first-token
    double      decode_ms         = 0.0;  // decode phase wall time
    double      tokens_per_second = 0.0;
    std::string stop_reason;  // "eos" | "length" | "user" | "context_length"
};

// High-level API: tokenizer + chat template + streaming generation over an LLMModel.
class GENIEX_API LLMPipeline {
   public:
    LLMPipeline();
    ~LLMPipeline();

    LLMPipeline(LLMPipeline&&) noexcept;
    LLMPipeline& operator=(LLMPipeline&&) noexcept;
    LLMPipeline(const LLMPipeline&)            = delete;
    LLMPipeline& operator=(const LLMPipeline&) = delete;

    // Takes ownership of `model` and initializes it. Returns false on failure.
    bool create(ChatTemplateFunc chat_template, LLMModel model, const QnnRuntimeConfig& runtime_cfg,
        const ModelConfig& model_cfg);

    // Polymorphic overload: accepts any LLMModel subclass without slicing.
    template <typename ModelT,
        std::enable_if_t<std::is_base_of_v<LLMModel, ModelT> && !std::is_same_v<LLMModel, ModelT>, int> = 0>
    bool create(ChatTemplateFunc chat_template, ModelT model, const QnnRuntimeConfig& runtime_cfg,
        const ModelConfig& model_cfg) {
        return createImpl(chat_template, std::make_unique<ModelT>(std::move(model)), runtime_cfg, model_cfg);
    }

    bool isReady() const;

    // Clears KV state and resets to the start of a new conversation.
    void reset();

    // Stateless: renders `messages` through the configured chat-template
    // formatter. See chat_template.h for the message-list contract; in
    // particular, the caller (not this method) is responsible for
    // first-turn default-system injection and for not re-emitting a
    // system block on later turns — doing so would invalidate the cached
    // KV prefix.
    std::string applyChatTemplate(
        const std::vector<ChatMessage>& messages, const ChatTools& tools = {}, bool enable_thinking = true) const;

    // on_token is called with each decoded text piece; return false to stop early.
    GenerateResult generate(const std::string& prompt_utf8, const GenerationConfig& gen_cfg = {},
        std::function<bool(const char*)> on_token = nullptr);

    void saveKVCache(const std::string& path) const;
    void loadKVCache(const std::string& path);

    size_t nPast() const;

    bool createImpl(ChatTemplateFunc chat_template, std::unique_ptr<LLMModel> model,
        const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg);

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace geniex