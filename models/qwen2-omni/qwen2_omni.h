#pragma once

#include "types.h"
#include "vlm/vlm_model.h"
#include "vlm/vlm_input_provider.h"
#include "vlm/vision_encoder.h"
#include "vlm/audio_encoder.h"
#include "vlm/audio_utils.h"
#include "vlm/vit_utils.h"
#include "vlm/vlm_utils.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace geniex {
namespace qwen2_omni {

struct Qwen2OmniConfig {
    // LLM: model_paths = [shard_0.bin, shard_1.bin]
    ModelConfig llm_config;

    // Vision encoder: model_paths = [patch_embed.bin, vit.bin]
    ModelConfig vision_config;

    // Audio encoder: model_paths = [helper0.bin, encoder.bin, helper1.bin]
    ModelConfig audio_config;
};

// ── Vision encoder ────────────────────────────────────────────────────────────

// Two-stage QNN vision encoder: patch_embed (graph 0) → vit_model (graph 1).
// CPU windowing/reordering is applied between the two graph executions.
class Qwen2OmniVisionEncoder : public QnnVisionEncoder {
public:
    // Returns flat [num_image_tokens * 2048] embeddings.
    // num_image_tokens = sum over images of T * (H / 2) * (W / 2).
    std::vector<float> encode(const PixelData& pixel_data) override;

private:
    static constexpr int kSpatialMergeSize = 2;
    static constexpr int kPatchSize        = 14;
    static constexpr int kWindowSize       = 112;
    static constexpr int kPatchChannels    = 6;
    static constexpr int kEmbedDim         = 1280;
    static constexpr int kOutputDim        = 2048;
};

// ── Audio encoder ─────────────────────────────────────────────────────────────

// Three-stage QNN audio encoder: helper0 (graph 0) → encoder (graph 1) → helper1 (graph 2).
// helper0 processes chunks of [num_mel_bins, 200]; encoder processes chunks of [1280, 100, 1];
// helper1 projects chunks of [100, 1280] → [100, 2048].
class Qwen2OmniAudioEncoder : public QnnAudioEncoder {
public:
    explicit Qwen2OmniAudioEncoder();

    // Returns flat [num_audio_tokens * 2048] embeddings.
    std::vector<float> encode(const AudioData& audio_data) override;

private:
    static constexpr int kNWindow             = 100;
    static constexpr int kEmbedDim            = 1280;
    static constexpr int kOutputDim           = 2048;
    static constexpr int kNumMelBins          = 128;
    static constexpr int kMaxSourcePositions  = 1500;

    std::vector<float> positional_embedding_;  // [kMaxSourcePositions * kEmbedDim]
};

// ── VLM model ─────────────────────────────────────────────────────────────────

class Qwen2OmniModel : public VLMModel {
public:
    explicit Qwen2OmniModel();

    // Called by makeModel() after encoder ownership is transferred.
    void setEncoders(std::unique_ptr<Qwen2OmniVisionEncoder> vis,
                     std::unique_ptr<Qwen2OmniAudioEncoder>  aud);

    // Called by makeModel() to register the MRoPE provider.
    void setMRoPEProvider(std::unique_ptr<MRoPEInputProvider> provider);

protected:
    std::vector<float> encodeVision(const PixelData& pixel_data) override;
    std::vector<float> encodeAudio(const AudioData& audio_data) override;

    // Computes 3D position IDs from input_ids and image/audio grids,
    // applies n_past offset + mrope_deltas, and sets them on mrope_provider_.
    void preparePositions(const std::vector<int32_t>& input_ids,
                          const VLMInput&             vlm_input,
                          size_t                      n_past) override;

    void clearPositions() override;

private:
    // Computes the number of LLM audio tokens from a raw frame count.
    // This formula is Qwen2-Omni-specific (two CNN layers + avg-pool by 2).
    static int32_t audioFramesToLLMTokens(int32_t num_frames);

    MRoPEInputProvider* mrope_provider_ = nullptr;  // non-owning; owned by input_providers_
    std::vector<int32_t> mrope_deltas_  = {0, 0, 0};

    static constexpr int32_t kImageTokenId       = 151655;
    static constexpr int32_t kAudioTokenId       = 151646;
    static constexpr int32_t kVisionStartTokenId = 151652;
    static constexpr int32_t kAudioStartTokenId  = 151647;
    static constexpr int      kSpatialMergeSize   = 2;
};

// ── Factory ───────────────────────────────────────────────────────────────────

inline LLMSpec makeSpec() {
    return LLMSpec{
        .shards = {
            {"input_embeds",         "_Add_88_Add_output_0"},
            {"_Add_88_Add_output_0", "logits"},
        },
        .state_blocks = {
            makeKVOnlyStateBlock({LayerRange{0, 17}, LayerRange{18, 35}}),
        },

        .seq_len_prefill = 128,
        .seq_len_decode  = 1,

        .hidden_size  = 2048,
        .num_heads    = 16,
        .num_kv_heads = 2,   // QNN binary: 2 KV groups (GQA), shape[0]=2
        .head_dim     = 128,
        .vocab_size   = 151936,

        .context_lengths = {4096},

        .eos_token_ids = {151645},
    };
}

// Creates, configures, and initializes all components of the Qwen2Omni model.
// Returns nullptr if any initialization step fails.
std::unique_ptr<Qwen2OmniModel> makeModel(const QnnRuntimeConfig& runtime_cfg,
                                          const Qwen2OmniConfig&  config);

} // namespace qwen2_omni
} // namespace geniex
