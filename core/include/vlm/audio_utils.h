#pragma once

#include "geniex_export.h"

#include <cstddef>
#include <vector>

namespace geniex {

// Sinusoidal positional embedding table [max_positions, embed_dim].
// Layout: table[pos * embed_dim + i] = sin(pos * inv_t_i) for i < embed_dim/2,
//         then cos for the upper half. Used by Whisper-style audio encoders.
GENIEX_VLM_API std::vector<float> makeSinusoidalPositionalEmbedding(int max_positions,
                                                                     int embed_dim);

GENIEX_VLM_API std::vector<float> avgPool1d(const std::vector<float>& input,
                                             size_t seq_len,
                                             size_t dim,
                                             size_t stride = 2);

} // namespace geniex
