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

// Forward declarations — avoid pulling geniex-proc headers into geniex_vlm's
// public ABI surface. The implementation includes them directly.
namespace geniex {
class VisionProcessor;
class Tokenizer;
}

namespace geniex {

// ── VLMPipeline ──────────────────────────────────────────────────────────────
//
// High-level wrapper that combines a VLMModel + a VisionProcessor (which owns
// the tokenizer and chat template) into a single object with a
// generate / reset workflow analogous to LLMPipeline.
//
// The VisionProcessor produces a BatchFeatures (formatted prompt, input_ids,
// pixel_values, image_grid_thw) from a conversation + image paths. Every
// turn — text-only or with images — is formatted by the processor, so the
// chat template lives entirely in geniex-proc, not in this pipeline.
//
// Multi-turn bridging (closing the previous assistant reply) is currently
// hardcoded to the ChatML end-of-turn marker "<|im_end|>\n". This matches
// every VLM family supported today (Qwen2-VL, Qwen2.5-VL, InternVL2+, …).
// If a non-ChatML VLM is ever added, the bridge handling should move into
// VisionProcessor (e.g. via a `conversation_open` flag on
// VisionProcessorInput) rather than being parameterised here.
//
// Usage (C++):
//   auto model = geniex::qwen2_5_vl_7b::makeModel(runtime_cfg, cfg);
//   auto proc  = geniex::qwen2vl::Qwen2VLProcessor::create(tokenizer_path, pcfg);
//   auto& tok  = proc->tokenizer();
//
//   VLMPipeline pipe;
//   pipe.create(std::move(model), std::move(proc), tok);
//   pipe.setSystemPrompt("You are a helpful AI assistant.");
//
//   auto r = pipe.generate("describe this image", {"cat.jpg"}, gen_cfg, on_tok);
//
class GENIEX_VLM_API VLMPipeline {
public:
    VLMPipeline();
    ~VLMPipeline();

    VLMPipeline(VLMPipeline&&) noexcept;
    VLMPipeline& operator=(VLMPipeline&&) noexcept;
    VLMPipeline(const VLMPipeline&)            = delete;
    VLMPipeline& operator=(const VLMPipeline&) = delete;

    // Takes ownership of an already-initialized VLMModel (e.g. the output of
    // a per-model makeModel() factory) and a VisionProcessor configured for
    // that model family. `tokenizer` must outlive the pipeline — typically
    // it is owned by the processor (e.g. Qwen2VLProcessor::tokenizer()) and
    // the caller passes a reference to it.
    // Returns false if `model` or `processor` is null.
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

    // True after a successful create(), false otherwise.
    bool isReady() const;

    // Reset KV cache and conversation state (first_turn flag).
    void reset();

    // ── System prompt ────────────────────────────────────────────────────────
    void setSystemPrompt(const std::string& prompt);

    // ── Chat template ────────────────────────────────────────────────────────
    // Formats one conversation turn into model-ready tokens and populates
    // `vlm_input` with pixel_values / image_grid_thw.
    //
    //   * first turn  : the system prompt (if set) is prepended before the
    //                   user message.
    //   * later turns : the previous assistant turn is closed with
    //                   "<|im_end|>\n" before the new user turn.
    //
    // Unlike LLMPipeline::applyChatTemplate (which returns a std::string),
    // this returns tokens directly because image placeholder token IDs
    // (<|vision_start|>, N × <|image_pad|>, <|vision_end|>) are interleaved
    // with encoded text and cannot be losslessly represented as a string.
    //
    // Does NOT modify conversation state — generate() advances first_turn.
    std::vector<int32_t> applyChatTemplate(
        const std::string&              user_message,
        const std::vector<std::string>& image_paths,
        VLMInput&                       vlm_input) const;

    // ── Generation ───────────────────────────────────────────────────────────
    // Generate a response to a user message with optional image paths.
    // Equivalent to applyChatTemplate() + VLMModel::generate() with streaming
    // decode and timing instrumentation.
    //
    // `on_token` is called with each decoded piece; return false to stop.
    GenerateResult generate(
        const std::string&              user_message,
        const std::vector<std::string>& image_paths = {},
        const GenerationConfig&         gen_cfg     = {},
        std::function<bool(const char*)> on_token   = nullptr);

    // Convenience overload: text-only turn.
    GenerateResult generate(
        const std::string&              user_message,
        const GenerationConfig&         gen_cfg,
        std::function<bool(const char*)> on_token   = nullptr);

    // ── KV-cache persistence ─────────────────────────────────────────────────
    void saveKVCache(const std::string& path) const;
    void loadKVCache(const std::string& path);

    size_t nPast() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace geniex
