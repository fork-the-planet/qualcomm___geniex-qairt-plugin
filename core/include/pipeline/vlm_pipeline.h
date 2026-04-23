#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "geniex_export.h"
#include "pipeline/llm_pipeline.h"   // GenerateResult
#include "types.h"
#include "vlm/vlm_model.h"
#include "vlm/vlm_types.h"

namespace geniex {
class VisionProcessor;
class Tokenizer;
}

namespace geniex {

// High-level wrapper that pairs a VLMModel with a VisionProcessor (which owns
// the tokenizer and chat template) and exposes a generate / reset workflow
// analogous to LLMPipeline.
class GENIEX_VLM_API VLMPipeline {
public:
    VLMPipeline();
    ~VLMPipeline();

    VLMPipeline(VLMPipeline&&) noexcept;
    VLMPipeline& operator=(VLMPipeline&&) noexcept;
    VLMPipeline(const VLMPipeline&)            = delete;
    VLMPipeline& operator=(const VLMPipeline&) = delete;

    bool create(std::unique_ptr<VLMModel>        model,
                std::unique_ptr<VisionProcessor> processor,
                Tokenizer&                       tokenizer);

    // Polymorphic overload: accepts any VLMModel subclass without slicing.
    template <typename ModelT,
              std::enable_if_t<std::is_base_of_v<VLMModel, ModelT>, int> = 0>
    bool create(std::unique_ptr<ModelT>          model,
                std::unique_ptr<VisionProcessor> processor,
                Tokenizer&                       tokenizer) {
        return create(std::unique_ptr<VLMModel>(model.release()),
                      std::move(processor), tokenizer);
    }

    bool isReady() const;

    // Clears KV state and resets to the start of a new conversation.
    void reset();

    void setSystemPrompt(const std::string& prompt);

    // Formats one turn into model-ready tokens and populates `vlm_input` with
    // pixel_values / image_grid_thw.
    std::vector<int32_t> applyChatTemplate(
        const std::string&              user_message,
        const std::vector<std::string>& image_paths,
        VLMInput&                       vlm_input) const;

    // on_token is called with each decoded text piece; return false to stop early.
    GenerateResult generate(
        const std::string&              user_message,
        const std::vector<std::string>& image_paths = {},
        const GenerationConfig&         gen_cfg     = {},
        std::function<bool(const char*)> on_token   = nullptr);

    // Text-only convenience overload.
    GenerateResult generate(
        const std::string&              user_message,
        const GenerationConfig&         gen_cfg,
        std::function<bool(const char*)> on_token   = nullptr);

    void saveKVCache(const std::string& path) const;
    void loadKVCache(const std::string& path);

    size_t nPast() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace geniex
