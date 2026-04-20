#include "llm/position_encoder.h"

namespace geniex {

void RoPEEncoder::write(Graph& g, size_t n_past, size_t curr_len) const {
    auto [cos_vec, sin_vec] = rope_.forward(get_position_ids(n_past, curr_len));

    if (g.hasInput("position_ids_cos")) {
        g.write("position_ids_cos", cos_vec.data(), cos_vec.size());
    }
    if (g.hasInput("position_ids_sin")) {
        g.write("position_ids_sin", sin_vec.data(), sin_vec.size());
    }
}

std::unique_ptr<PositionEncoder> makePositionEncoder(PositionEncodingType type,
                                                      size_t head_dim,
                                                      float  rope_theta) {
    switch (type) {
        case PositionEncodingType::ROPE:
            return std::make_unique<RoPEEncoder>(head_dim, rope_theta);
        case PositionEncodingType::NONE:
            return std::make_unique<NullEncoder>();
    }
    return std::make_unique<NullEncoder>(); // unreachable; satisfy compiler
}

} // namespace geniex
