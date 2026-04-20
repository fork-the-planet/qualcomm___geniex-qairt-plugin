#pragma once

#include "Graph.h"
#include "llm/llm_types.h"
#include "llm/llm_utils.h"
#include "geniex_export.h"

#include <cstddef>
#include <memory>

namespace geniex {

// Abstract strategy for writing position-related input tensors into a Graph.
// LLMModel::runShard() calls pos_encoder_->write(graph, n_past, curr_len)
// without knowing which encoding scheme is in use.
class GENIEX_API PositionEncoder {
public:
    virtual ~PositionEncoder() = default;

    // Write position tensors for the window [n_past, n_past + curr_len) into g.
    virtual void write(Graph& g, size_t n_past, size_t curr_len) const = 0;
};

// Writes position_ids_cos / position_ids_sin computed from Rotary Position Embedding.
class GENIEX_API RoPEEncoder : public PositionEncoder {
public:
    RoPEEncoder(size_t head_dim, float theta)
        : rope_(head_dim, theta) {}

    void write(Graph& g, size_t n_past, size_t curr_len) const override;

private:
    RotaryEmbedding rope_;
};

// No-op encoder for models that compute positional encoding inside the graph.
class GENIEX_API NullEncoder : public PositionEncoder {
public:
    void write(Graph&, size_t, size_t) const override {}
};

// Factory: constructs the appropriate PositionEncoder from LLMSpec fields.
// Declared here to avoid a circular include; defined in position_encoder.cpp.
GENIEX_API std::unique_ptr<PositionEncoder> makePositionEncoder(PositionEncodingType type,
                                                      size_t head_dim,
                                                      float  rope_theta);

} // namespace geniex
