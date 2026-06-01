// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "llm/position_encoder.h"

#include <algorithm>

#include "graph.h"
#include "llm/llm_utils.h"

namespace geniex {

namespace {

// Pad position IDs from `curr_len` up to the graph's RoPE-tensor row capacity
// with zero, so trailing rows produce identity rotation (cos=1.0, sin=0.0).
std::pair<std::vector<int32_t>, size_t> paddedPositionIdsFor(
    const Graph& g, const std::string& name, size_t half_dim, size_t n_past, size_t curr_len) {
    size_t cap_rows = curr_len;
    if (g.hasInput(name) && half_dim > 0) {
        const auto& spec = g.inputSpec(name);
        size_t      cap  = 1;
        for (auto d : spec.shape) cap *= d;
        cap_rows = cap / half_dim;
    }
    const size_t         total = std::max(curr_len, cap_rows);
    std::vector<int32_t> ids(total, 0);
    for (size_t i = 0; i < curr_len && i < total; ++i) ids[i] = static_cast<int32_t>(n_past + i);
    return {std::move(ids), cap_rows};
}

}  // namespace

void RoPEEncoder::write(Graph& g, size_t n_past, size_t curr_len) const {
    const bool has_cos = g.hasInput("position_ids_cos");
    const bool has_sin = g.hasInput("position_ids_sin");
    if (!has_cos && !has_sin) return;

    const size_t half_dim = rope_.halfDim();
    auto [ids, _] =
        paddedPositionIdsFor(g, has_cos ? "position_ids_cos" : "position_ids_sin", half_dim, n_past, curr_len);

    auto [cos_vec, sin_vec] = rope_.forward(ids);
    if (has_cos) g.write("position_ids_cos", cos_vec.data(), cos_vec.size());
    if (has_sin) g.write("position_ids_sin", sin_vec.data(), sin_vec.size());
}

std::unique_ptr<PositionEncoder> makePositionEncoder(PositionEncodingType type, size_t head_dim, float rope_theta) {
    switch (type) {
        case PositionEncodingType::ROPE:
            return std::make_unique<RoPEEncoder>(head_dim, rope_theta);
        case PositionEncodingType::NONE:
            return std::make_unique<NullEncoder>();
    }
    return std::make_unique<NullEncoder>();  // unreachable; satisfy compiler
}

}  // namespace geniex
