// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "geniex_export.h"

namespace geniex {

// Utilities for the Qwen-family Vision Transformer (Qwen2-VL, Qwen2-Omni, Qwen3-VL).
namespace qwen_vit {

// inv_freq[i] = 1 / theta^(i*2 / dim), for i in [0, dim/2).
GENIEX_VLM_API std::vector<float> makeInvFreq(int dim = 40, float theta = 10000.0f);

// Returns the window reorder permutation for sparse windowed attention.
// window_index[i] = original group index for window position i.
// T, H, W: temporal/height/width of the patch grid.
GENIEX_VLM_API std::vector<int64_t> computeWindowIndex(
    int T, int H, int W, int spatial_merge_size, int window_size, int patch_size);

GENIEX_VLM_API std::vector<int64_t> reverseWindowIndex(const std::vector<int64_t>& window_index);

// Returns {cos, sin}, each flat [seq_len * emb_dim] after window reordering.
// seq_len = T*H*W; emb_dim = inv_freq.size()*2 (h-freq concat w-freq).
// Expanded by spatial_merge_size^2 so each merged group shares the same encoding.
GENIEX_VLM_API std::pair<std::vector<float>, std::vector<float>> computeSpatialCosSin(int T, int H, int W,
    int spatial_merge_size, const std::vector<float>& inv_freq, const std::vector<int64_t>& window_index);

// Reorders hidden states [n_groups * sm_unit * embed_dim] by window_index at the group level.
GENIEX_VLM_API std::vector<float> windowReorder(const std::vector<float>& hidden, size_t n_groups, size_t sm_unit,
    size_t embed_dim, const std::vector<int64_t>& window_index);

// Returns {cos, sin}, each flat [T*H*W, 2*inv_freq.size()] in natural patch order.
// Mirrors HF Qwen2_5_VisionTransformerPretrainedModel.rot_pos_emb().
GENIEX_VLM_API std::pair<std::vector<float>, std::vector<float>> computePatchRoPE(
    int T, int H, int W, int spatial_merge_size, const std::vector<float>& inv_freq);

// cu[i+1] - cu[i] = number of patches in window i. Length = num_windows * sm_unit + 1.
GENIEX_VLM_API std::vector<int64_t> computeCuWindowSeqlens(
    int T, int H, int W, int spatial_merge_size, int window_size, int patch_size);

// Builds a block-diagonal additive attention mask of shape [N, N].
// boundaries[0]=0 and boundaries.back()=N; positions in the same segment attend freely.
GENIEX_VLM_API std::vector<float> buildBlockAttentionMask(
    size_t N, const std::vector<int64_t>& boundaries, float allowed = 0.0f, float blocked = -1e9f);

}  // namespace qwen_vit
}  // namespace geniex
