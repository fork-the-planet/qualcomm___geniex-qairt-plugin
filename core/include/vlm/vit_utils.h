#pragma once

#include "geniex_export.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace geniex {

/// Utilities specific to the Qwen-family Vision Transformer (Qwen2-VL, Qwen2-Omni, Qwen3-VL).
/// Other ViT families can add their own namespaces in this file.
namespace qwen_vit {

/// Computes inverse frequencies for spatial RoPE inside the Qwen ViT.
/// inv_freq[i] = 1 / theta^(i*2 / dim), for i in [0, dim/2).
/// Default: dim=40, theta=10000 (standard Qwen ViT RoPE).
GENIEX_VLM_API std::vector<float> makeInvFreq(int dim = 40, float theta = 10000.0f);

/// Computes the window reorder permutation for sparse windowed attention.
/// Returns indices such that window_index[i] = original group index for window position i.
///
/// Parameters match the Qwen ViT config:
///   T, H, W          — temporal, height, width of the patch grid
///   spatial_merge_size — how many patches are merged per LLM token (typically 2)
///   window_size        — attention window size in pixels (typically 112)
///   patch_size         — patch size in pixels (typically 14)
GENIEX_VLM_API std::vector<int64_t> computeWindowIndex(int T, int H, int W,
                                                        int spatial_merge_size,
                                                        int window_size,
                                                        int patch_size);

/// Computes the inverse permutation (argsort) of window_index.
/// reverse[window_index[i]] = i.
GENIEX_VLM_API std::vector<int64_t> reverseWindowIndex(const std::vector<int64_t>& window_index);

/// Computes spatial rotary cos/sin for the ViT after window reordering.
/// Returns {cos, sin}, each flat [seq_len * emb_dim] where:
///   seq_len = T * H * W  (total patch count)
///   emb_dim = inv_freq.size() * 2  (h-freq concat w-freq)
///
/// The cos/sin are expanded by spatial_merge_size^2 (sm_unit) so each
/// merged group of patches shares the same positional encoding.
GENIEX_VLM_API std::pair<std::vector<float>, std::vector<float>>
computeSpatialCosSin(int T, int H, int W,
                     int spatial_merge_size,
                     const std::vector<float>& inv_freq,
                     const std::vector<int64_t>& window_index);

/// Reorders hidden states by window_index at the group level.
/// Input:  flat [n_groups * sm_unit * embed_dim]
/// Output: flat [n_groups * sm_unit * embed_dim], reordered so that
///         out[i * sm_unit * embed_dim ..] = hidden[window_index[i] * sm_unit * embed_dim ..].
GENIEX_VLM_API std::vector<float> windowReorder(const std::vector<float>& hidden,
                                                 size_t n_groups,
                                                 size_t sm_unit,
                                                 size_t embed_dim,
                                                 const std::vector<int64_t>& window_index);

} // namespace qwen_vit
} // namespace geniex
