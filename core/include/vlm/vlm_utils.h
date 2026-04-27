// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "geniex_export.h"

#include <array>
#include <cstdint>
#include <vector>

namespace geniex {

// One image's patch grid for MRoPE: T=temporal, H=height, W=width (before spatial merge).
using ImageGrid = std::array<int32_t, 3>;

// One audio segment's LLM token count for MRoPE.
// The caller pre-computes num_llm_tokens because the formula is model-specific.
struct AudioSegmentInfo {
    int32_t num_llm_tokens;
};

struct MRoPEPositions {
    std::vector<int32_t> position_ids;  // flat [3 * seq_len]: temporal, height, width
    std::vector<int32_t> deltas;        // [3]: per-dimension position delta for this turn
};

// Computes 3D MRoPE position IDs for a sequence with interleaved text, image, and audio tokens.
// Generic algorithm shared by Qwen2-VL, Qwen2-Omni, Qwen3-VL: text tokens get sequential
// positions in all 3 dims; image tokens get grid positions; audio tokens get sequential positions.
GENIEX_VLM_API MRoPEPositions computeMRoPEPositions(
    const std::vector<int32_t>&          input_ids,
    const std::vector<ImageGrid>&        image_grids,
    const std::vector<AudioSegmentInfo>& audio_segments,
    int                                  spatial_merge_size,
    int32_t                              vision_start_token_id,
    int32_t                              image_token_id,
    int32_t                              audio_start_token_id,
    int32_t                              audio_token_id);

} // namespace geniex
