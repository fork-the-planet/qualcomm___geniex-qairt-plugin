// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "geniex_export.h"
#include "llm/llm_model.h"
#include "vlm/vision_encoder.h"
#include "vlm/vlm_input_provider.h"
#include "vlm/vlm_types.h"

namespace geniex {

// LLMModel subclass that encodes multimodal inputs and injects their embeddings
// before calling the LLM generate loop.
class GENIEX_VLM_API VLMModel : public LLMModel {
   public:
    explicit VLMModel(LLMSpec spec);

    // Return false from the callback to stop generation early.
    std::vector<int32_t> generate(const std::vector<int32_t>& prompt_tokens, const VLMInput& vlm_input,
        const GenerationConfig& gen_cfg = {}, std::function<bool(int32_t)> token_callback = nullptr);

   protected:
    bool onInitialized() override;

    virtual std::vector<float> encodeVision(const PixelData& pixel_data) = 0;

    // Called after embedding injection, before LLMModel::generate().
    virtual void preparePositions(const std::vector<int32_t>& input_ids, const VLMInput& vlm_input, size_t n_past);

    // Called after LLMModel::generate() returns to reset position provider state.
    virtual void clearPositions();

    // Overwrites rows in input_embeds where input_ids == target_token_id
    // with consecutive rows from multimodal_embeds.
    static void maskedScatter(std::vector<float>& input_embeds, const std::vector<float>& multimodal_embeds,
        const std::vector<int32_t>& input_ids, int32_t target_token_id, size_t hidden_size);

    // Must be called before initialize().
    void setEmbeddingProvider(std::unique_ptr<PrecomputedEmbeddingProvider> provider);

    std::unique_ptr<VisionEncoder> vision_encoder_;

    int32_t image_token_id_ = 0;

   private:
    PrecomputedEmbeddingProvider* emb_provider_ = nullptr;  // non-owning; owned by input_providers_
};

}  // namespace geniex
