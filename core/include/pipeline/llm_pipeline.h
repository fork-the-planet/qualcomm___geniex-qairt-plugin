#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>

#include "geniex_export.h"
#include "llm/llm_model.h"
#include "pipeline/chat_template.h"
#include "types.h"

namespace geniex {

// ── Data types ───────────────────────────────────────────────────────────────

struct GenerateResult {
    std::string full_text;
    int64_t     prompt_tokens     = 0;
    int64_t     generated_tokens  = 0;
    double      ttft_ms           = 0.0;   // time-to-first-token
    double      decode_ms         = 0.0;   // decode phase wall time
    double      tokens_per_second = 0.0;
    std::string stop_reason;               // "eos", "length", "user"
};


// ── LLMPipeline ──────────────────────────────────────────────────────────────
//
// High-level wrapper that combines LLMModel + tokenizer into a single
// object with a chat-template / generate / reset workflow.
//
// Usage (C++) — manual create:
//   LLMPipeline pipe;
//   if (!pipe.create(chatMLTemplate, qwen3_4b::makeModel(), runtime_cfg, model_cfg))
//       return 1;
//   pipe.setSystemPrompt("You are a helpful assistant.");
//   auto result = pipe.generate(pipe.applyChatTemplate("Hi"), gen_cfg, on_token);
//
// Usage (C++) — per-model factory (returns std::nullopt on failure):
//   auto pipe = qwen3_4b::makePipeline(runtime_cfg, model_cfg);
//   if (!pipe) return 1;
//   pipe->generate(pipe->applyChatTemplate("Hi"), gen_cfg, on_token);
//
class GENIEX_API LLMPipeline {
public:
    LLMPipeline();
    ~LLMPipeline();

    LLMPipeline(LLMPipeline&&) noexcept;
    LLMPipeline& operator=(LLMPipeline&&) noexcept;
    LLMPipeline(const LLMPipeline&)            = delete;
    LLMPipeline& operator=(const LLMPipeline&) = delete;

    // Takes ownership of `model` (via move) and initialises it.
    // `chat_template` is the formatting function to use (e.g. chatMLTemplate,
    // phiChatTemplate, or a model-specific inline template).
    // Returns false if QNN initialisation or tokenizer loading fails.
    bool create(ChatTemplateFunc chat_template,
                LLMModel model,
                const QnnRuntimeConfig& runtime_cfg,
                const ModelConfig& model_cfg);

    // Polymorphic overload: accepts any LLMModel subclass (e.g. SSDModel)
    // without slicing. Prefer this when using model subclasses.
    template <typename ModelT,
              std::enable_if_t<std::is_base_of_v<LLMModel, ModelT> &&
                               !std::is_same_v<LLMModel, ModelT>, int> = 0>
    bool create(ChatTemplateFunc chat_template,
                ModelT model,
                const QnnRuntimeConfig& runtime_cfg,
                const ModelConfig& model_cfg) {
        return createImpl(chat_template,
                          std::make_unique<ModelT>(std::move(model)),
                          runtime_cfg, model_cfg);
    }

    // True after a successful create(), false otherwise.
    bool isReady() const;

    // Reset KV cache and conversation state (first_turn flag).
    void reset();

    // ── System prompt ────────────────────────────────────────────────────────
    void setSystemPrompt(const std::string& prompt);

    // ── Chat template ────────────────────────────────────────────────────────
    // Formats a single user message using the selected chat template.
    // The system prompt (set via setSystemPrompt) is emitted only on the
    // first call after create() / reset().
    std::string applyChatTemplate(
        const std::string& user_message,
        bool enable_thinking = true);

    // ── Generation ───────────────────────────────────────────────────────────
    // Generate from a UTF-8 prompt string (tokenises internally).
    // `on_token` is called with each decoded piece; return false to stop.
    GenerateResult generate(
        const std::string& prompt_utf8,
        const GenerationConfig& gen_cfg = {},
        std::function<bool(const char*)> on_token = nullptr);


    // ── KV-cache persistence ─────────────────────────────────────────────────
    void saveKVCache(const std::string& path) const;
    void loadKVCache(const std::string& path);

    size_t nPast() const;

    // Implementation helper for polymorphic model storage.
    bool createImpl(ChatTemplateFunc chat_template,
                    std::unique_ptr<LLMModel> model,
                    const QnnRuntimeConfig& runtime_cfg,
                    const ModelConfig& model_cfg);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace geniex