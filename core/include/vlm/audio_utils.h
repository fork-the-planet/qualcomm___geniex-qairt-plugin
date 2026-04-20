#pragma once

#include "geniex_export.h"

#include <cstddef>
#include <vector>

namespace geniex {

/// Sinusoidal positional embedding table [max_positions, embed_dim].
/// Used by Whisper-style audio encoders.
/// Layout: table[pos * embed_dim + i] = sin(pos * inv_t_i)          for i < embed_dim/2
///         table[pos * embed_dim + embed_dim/2 + i] = cos(pos * inv_t_i)  for i < embed_dim/2
GENIEX_VLM_API std::vector<float> makeSinusoidalPositionalEmbedding(int max_positions,
                                                                     int embed_dim);

/// Average pool along axis 0 with non-overlapping windows:
///   [seq_len, dim] → [seq_len / stride, dim]
/// Each output row is the mean of `stride` consecutive input rows.
GENIEX_VLM_API std::vector<float> avgPool1d(const std::vector<float>& input,
                                             size_t seq_len,
                                             size_t dim,
                                             size_t stride = 2);

} // namespace geniex
