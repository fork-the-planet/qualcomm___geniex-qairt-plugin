#pragma once

#include "geniex_export.h"

#include <array>
#include <cstdint>
#include <vector>

/// VLM-wide utilities shared across model families.
/// This is the home for logic that operates at the VLM level (e.g. position ID
/// computation) but is not specific to any single model or modality.

namespace geniex {

/// Describes one image's patch grid for MRoPE position computation.
/// T = temporal frames, H = height patches, W = width patches (before spatial merge).
using ImageGrid = std::array<int32_t, 3>;

/// Describes one audio segment's contribution to MRoPE positions.
/// The caller pre-computes num_llm_tokens because the formula is model-specific
/// (e.g. Qwen2-Omni: ((raw_len - 1) / 2 + 1 - 2) / 2 + 1 / 2).
struct AudioSegmentInfo {
    int32_t num_llm_tokens;
};

/// Result of computing MRoPE position IDs for a single input sequence.
struct MRoPEPositions {
    std::vector<int32_t> position_ids;  ///< flat [3 * seq_len]: temporal, height, width
    std::vector<int32_t> deltas;        ///< [3]: per-dimension delta for this turn
};

/// Computes 3D MRoPE position IDs for an input sequence containing
/// interleaved text, image, and audio tokens.
///
/// This is the generic algorithm shared by Qwen2-VL, Qwen2-Omni, Qwen3-VL:
///   - Text tokens get sequential positions in all 3 dims.
///   - Image tokens get (temporal, height, width) grid positions.
///   - Audio tokens get sequential positions in all 3 dims.
///
/// Token IDs marking modality boundaries are model-specific and passed as parameters.
/// Audio segment token counts are pre-computed by the caller.
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
