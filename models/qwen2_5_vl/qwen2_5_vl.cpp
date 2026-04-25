// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: Apache-2.0

#include "qwen2_5_vl.h"

#include "utils.h"
#include "vlm/vit_utils.h"
#include "vlm/vlm_utils.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace geniex {
namespace qwen2_5_vl_7b {

namespace {

// hidden=1280, num_heads=16 → head_dim=80; rotary dim = head_dim/2 = 40.
constexpr int kVitWindowSize  = 112;             // in pixels
constexpr int kVitRopeDim     = 40;              // half_dim*2 = 40 = cos/sin emb_dim
constexpr float kVitRopeTheta = 10000.0f;

} // namespace

std::vector<float> Qwen25VLVisionEncoder::encode(const PixelData& pixel_data) {
    if (pixel_data.image_grid_thw.empty()) {
        throw std::runtime_error("Qwen25VLVisionEncoder: empty image_grid_thw");
    }
    for (const auto& thw : pixel_data.image_grid_thw) {
        if (thw[0] != kGridT || thw[1] != kGridH || thw[2] != kGridW) {
            throw std::runtime_error(
                "Qwen25VLVisionEncoder: grid_thw = (" +
                std::to_string(thw[0]) + "," + std::to_string(thw[1]) + "," +
                std::to_string(thw[2]) + "), expected (" +
                std::to_string(kGridT) + "," + std::to_string(kGridH) + "," +
                std::to_string(kGridW) + ") for every image");
        }
    }

    const size_t n_images          = pixel_data.image_grid_thw.size();
    const size_t N                 = static_cast<size_t>(kNumPatches);
    const size_t sm_unit           = static_cast<size_t>(kSpatialMergeSize * kSpatialMergeSize);  // 4
    const size_t n_groups          = N / sm_unit;                                                  // 216
    const size_t per_image_pixels  = N * kPatchFeatureSize;                                        // 864 * 1176
    const size_t per_image_tokens  = static_cast<size_t>(kNumImageTokens) * kHiddenSize;           // 216 * 3584

    Graph& g = graph(0);

    // Permutations driven by the window layout.
    const auto window_index = qwen_vit::computeWindowIndex(
        kGridT, kGridH, kGridW, kSpatialMergeSize, kVitWindowSize, kPatchSize);
    const auto reverse_index = qwen_vit::reverseWindowIndex(window_index);
    const auto cu_window    = qwen_vit::computeCuWindowSeqlens(
        kGridT, kGridH, kGridW, kSpatialMergeSize, kVitWindowSize, kPatchSize);

    const auto inv_freq = qwen_vit::makeInvFreq(kVitRopeDim, kVitRopeTheta);
    auto [rope_cos_nat, rope_sin_nat] =
        qwen_vit::computePatchRoPE(kGridT, kGridH, kGridW, kSpatialMergeSize, inv_freq);
    const auto rope_cos = qwen_vit::windowReorder(
        rope_cos_nat, n_groups, sm_unit, kVitRopeDim, window_index);
    const auto rope_sin = qwen_vit::windowReorder(
        rope_sin_nat, n_groups, sm_unit, kVitRopeDim, window_index);

    constexpr float kAllowed = 0.0f;
    constexpr float kBlocked = -1e9f;
    const auto window_mask = qwen_vit::buildBlockAttentionMask(N, cu_window, kAllowed, kBlocked);
    const auto full_mask   = qwen_vit::buildBlockAttentionMask(
        N, { 0, static_cast<int64_t>(N) }, kAllowed, kBlocked);

    if (pixel_data.pixel_values.size() != n_images * per_image_pixels) {
        throw std::runtime_error(
            "Qwen25VLVisionEncoder: pixel_values has " +
            std::to_string(pixel_data.pixel_values.size()) +
            " floats, expected " +
            std::to_string(n_images * per_image_pixels) + " (= " +
            std::to_string(n_images) + " images * " +
            std::to_string(per_image_pixels) + ")");
    }

    std::vector<float> image_features(n_images * per_image_tokens);
    TimeLog tl;

    for (size_t img = 0; img < n_images; ++img) {
        const float* pixels_natural =
            pixel_data.pixel_values.data() + img * per_image_pixels;

        g.write("pixel_values",          pixels_natural,      per_image_pixels);
        g.write("position_ids_cos",      rope_cos.data(),     rope_cos.size());
        g.write("position_ids_sin",      rope_sin.data(),     rope_sin.size());
        g.write("window_attention_mask", window_mask.data(),  window_mask.size());
        g.write("full_attention_mask",   full_mask.data(),    full_mask.size());

        if (!g.execute(tl)) {
            throw std::runtime_error(
                "Qwen25VLVisionEncoder: vision_encoder graph execute failed on image " +
                std::to_string(img));
        }

        // Read window-ordered merged tokens, then un-permute into the per-image
        // slot of the concatenated output buffer.
        std::vector<float> tokens_reordered(per_image_tokens);
        g.read("image_features", tokens_reordered.data(), tokens_reordered.size());

        float* out_slot = image_features.data() + img * per_image_tokens;
        for (size_t i = 0; i < reverse_index.size(); ++i) {
            const size_t src = static_cast<size_t>(reverse_index[i]);
            std::memcpy(out_slot + i   * kHiddenSize,
                        tokens_reordered.data() + src * kHiddenSize,
                        kHiddenSize * sizeof(float));
        }
    }

    return image_features;
}

Qwen25VLModel::Qwen25VLModel() : VLMModel(makeSpec()) {
    image_token_id_ = kImageTokenId;
    setEmbeddingProvider(
        std::make_unique<PrecomputedEmbeddingProvider>("inputs_embeds"));
}

void Qwen25VLModel::setVisionEncoder(std::unique_ptr<Qwen25VLVisionEncoder> vis) {
    vision_encoder_ = std::move(vis);
}

void Qwen25VLModel::setMRoPEProvider(std::unique_ptr<MRoPEInputProvider> provider) {
    mrope_provider_ = provider.get();
    addInputProvider(std::move(provider));
}

std::vector<float> Qwen25VLModel::encodeVision(const PixelData& pixel_data) {
    return vision_encoder_->encode(pixel_data);
}

void Qwen25VLModel::preparePositions(const std::vector<int32_t>& input_ids,
                                     const VLMInput&             vlm_input,
                                     size_t                      n_past) {
    if (!mrope_provider_) return;

    // Convert image_grid_thw (std::array<int32_t,3>) to ImageGrid (alias of same type).
    std::vector<ImageGrid> image_grids(vlm_input.pixel_data.image_grid_thw.begin(),
                                       vlm_input.pixel_data.image_grid_thw.end());

    auto mrope = computeMRoPEPositions(
        input_ids, image_grids, /*audio_segments=*/{},
        kSpatialMergeSize,
        kVisionStartTokenId, kImageTokenId,
        /*audio_start=*/0, /*audio_token=*/0);

    const size_t seq_len = input_ids.size();

    for (size_t dim = 0; dim < 3; ++dim)
        for (size_t t = 0; t < seq_len; ++t)
            mrope.position_ids[dim * seq_len + t] +=
                static_cast<int32_t>(n_past) + mrope_deltas_[dim];

    mrope_provider_->setPositionIds(mrope.position_ids, seq_len, n_past);

    for (int d = 0; d < 3; ++d) mrope_deltas_[d] += mrope.deltas[d];
    mrope_provider_->setMropeDeltas(mrope_deltas_);
}

void Qwen25VLModel::clearPositions() {
    if (mrope_provider_) mrope_provider_->clearPositionIds();
}

std::unique_ptr<Qwen25VLModel> makeModel(const QnnRuntimeConfig& runtime_cfg,
                                         const Qwen25VLConfig&   config) {
    auto vis_enc = std::make_unique<Qwen25VLVisionEncoder>();
    if (!vis_enc->initialize(runtime_cfg, config.vision_config)) return nullptr;

    auto model = std::make_unique<Qwen25VLModel>();
    model->setVisionEncoder(std::move(vis_enc));
    model->setMRoPEProvider(std::make_unique<MRoPEInputProvider>(
        mRoPESection(), kRopeTheta, MRoPEInterleaving::BLOCK));

    if (!model->initialize(runtime_cfg, config.llm_config)) return nullptr;

    return model;
}

} // namespace qwen2_5_vl_7b
} // namespace geniex
