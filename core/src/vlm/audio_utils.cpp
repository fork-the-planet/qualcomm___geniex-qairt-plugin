#include "vlm/audio_utils.h"

#include <cmath>
#include <cstddef>

namespace geniex {

std::vector<float> makeSinusoidalPositionalEmbedding(int max_positions, int embed_dim) {
    const float log_scale = std::log(1e4f) / (float(embed_dim / 2) - 1.0f);
    std::vector<float> table(static_cast<size_t>(max_positions) * embed_dim);

    for (int pos = 0; pos < max_positions; ++pos) {
        for (int i = 0; i < embed_dim / 2; ++i) {
            float inv_t = std::exp(-log_scale * float(i));
            float angle = float(pos) * inv_t;
            table[pos * embed_dim + i]                 = std::sin(angle);
            table[pos * embed_dim + embed_dim / 2 + i] = std::cos(angle);
        }
    }
    return table;
}

std::vector<float> avgPool1d(const std::vector<float>& input,
                              size_t seq_len,
                              size_t dim,
                              size_t stride) {
    const size_t out_len = seq_len / stride;
    const float  scale   = 1.0f / float(stride);
    std::vector<float> out(out_len * dim, 0.0f);

    for (size_t i = 0; i < out_len; ++i)
        for (size_t s = 0; s < stride; ++s)
            for (size_t d = 0; d < dim; ++d)
                out[i * dim + d] += scale * input[(i * stride + s) * dim + d];

    return out;
}

} // namespace geniex
