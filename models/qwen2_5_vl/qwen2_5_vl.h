// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llm/input_provider.h"
#include "llm/llm_model.h"
#include "llm/llm_types.h"
#include "pipeline/chat_template.h"
#include "types.h"
#include "vlm/vlm_input_provider.h"
#include "vlm/vlm_model.h"
#include "vlm/vision_encoder.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace geniex {
namespace qwen2_5_vl_7b {

static constexpr size_t kHiddenSize = 3584;
static constexpr size_t kNumHeads   = 28;
static constexpr size_t kNumKVHeads = 4;
static constexpr size_t kHeadDim    = 128;      // hidden_size / num_heads
static constexpr size_t kNumLayers  = 28;
static constexpr size_t kVocabSize  = 152064;
static constexpr float  kRopeTheta  = 1000000.0f;

static constexpr int kImageHeight       = 336;
static constexpr int kImageWidth        = 504;
static constexpr int kPatchSize         = 14;
static constexpr int kTemporalPatchSize = 2;
static constexpr int kSpatialMergeSize  = 2;

static constexpr int kGridT            = 1;
static constexpr int kGridH            = kImageHeight / kPatchSize;      // 24
static constexpr int kGridW            = kImageWidth  / kPatchSize;      // 36
static constexpr int kNumPatches       = kGridT * kGridH * kGridW;       // 864
static constexpr int kNumImageTokens   = kNumPatches /
    (kSpatialMergeSize * kSpatialMergeSize);                             // 216
static constexpr int kPatchFeatureSize = 3 * kTemporalPatchSize *
    kPatchSize * kPatchSize;                                             // 1176

// Qwen2.5-VL special tokens (same family as Qwen2-VL / Qwen2-Omni).
static constexpr int32_t kVisionStartTokenId = 151652;
static constexpr int32_t kVisionEndTokenId   = 151653;
static constexpr int32_t kImageTokenId       = 151655;  // <|image_pad|>

// MRoPE section (from config.json rope_scaling.mrope_section).
// Sum = 64 = head_dim/2. Qwen2.5-VL uses BLOCK interleaving.
inline const std::vector<int>& mRoPESection() {
    static const std::vector<int> kSec = {16, 24, 24};
    return kSec;
}

struct Qwen25VLConfig {
    // Text backbone: 5 shards (part1_of_5.bin … part5_of_5.bin).
    ModelConfig llm_config;

    // Vision encoder: single graph (vision_encoder.bin).
    ModelConfig vision_config;
};

// Single-graph QNN vision encoder.
//
// Inputs:
//   pixel_values          [864, 1176]   — flattened patches
//   position_ids_cos/sin  [864, 40]     — per-patch 2D rotary
//   window_attention_mask [1, 864, 864] — windowed attention pattern
//   full_attention_mask   [1, 864, 864] — full attention pattern
// Output:
//   image_features        [216, 3584]   — merged image tokens, same hidden as LLM
//
// The graph is compiled for a fixed 24×36 patch grid (336×504 pixels).
// Multi-image prompts are handled by looping and calling execute() per image;
// dynamic shapes are not supported.
class Qwen25VLVisionEncoder : public QnnVisionEncoder {
public:
    std::vector<float> encode(const PixelData& pixel_data) override;
};

class Qwen25VLModel : public VLMModel {
public:
    Qwen25VLModel();

    void setVisionEncoder(std::unique_ptr<Qwen25VLVisionEncoder> vis);
    void setMRoPEProvider(std::unique_ptr<MRoPEInputProvider> provider);

protected:
    std::vector<float> encodeVision(const PixelData& pixel_data) override;

    // Computes 3D MRoPE position IDs accounting for image grid extents and accumulated
    // mrope_deltas, then pushes them to the MRoPE provider.
    void preparePositions(const std::vector<int32_t>& input_ids,
                          const VLMInput&             vlm_input,
                          size_t                      n_past) override;

    void clearPositions() override;

private:
    MRoPEInputProvider*  mrope_provider_ = nullptr;  // non-owning
    std::vector<int32_t> mrope_deltas_   = {0, 0, 0};
};

// Shard layout (28 transformer layers across 5 shards, no on-device embed):
//   shard 1 : layers  0 –  5   inputs_embeds → add_13335
//   shard 2 : layers  6 – 11   add_13335     → add_25971
//   shard 3 : layers 12 – 17   add_25971     → add_38607
//   shard 4 : layers 18 – 23   add_38607     → add_51243
//   shard 5 : layers 24 – 27   add_51243     → logits
inline LLMSpec makeSpec() {
    return LLMSpec{
        .shards = {
            {"inputs_embeds", "add_13335"},
            {"add_13335",     "add_25971"},
            {"add_25971",     "add_38607"},
            {"add_38607",     "add_51243"},
            {"add_51243",     "logits"},
        },
        .state_blocks = {
            makeKVOnlyStateBlock({
                LayerRange{0,  5},
                LayerRange{6,  11},
                LayerRange{12, 17},
                LayerRange{18, 23},
                LayerRange{24, 27},
            }),
        },

        .seq_len_prefill = 128,
        .seq_len_decode  = 1,

        .hidden_size  = kHiddenSize,
        .num_heads    = kNumHeads,
        .num_kv_heads = kNumKVHeads,
        .head_dim     = kHeadDim,
        .vocab_size   = kVocabSize,

        .context_lengths = {2048},

        .graph_name_pattern = "{phase}_ar{ar}_cl{cl}_{shard}_of_{total}",

        .eos_token_ids = {151643, 151645},
    };
}

// Full Qwen2.5-VL-7B stack (vision encoder + LLM). Returns nullptr on failure.
std::unique_ptr<Qwen25VLModel> makeModel(const QnnRuntimeConfig& runtime_cfg,
                                         const Qwen25VLConfig&   config);

} // namespace qwen2_5_vl_7b
} // namespace geniex
