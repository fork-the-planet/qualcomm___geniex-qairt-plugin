#pragma once

#include <memory>
#include <vector>

#include "IOTensor.hpp"
#include "QnnApi.hpp"
#include "graph.h"
#include "types.h"
#include "geniex_export.h"

namespace geniex {

class GENIEX_API Model {
public:
    virtual ~Model() = default;

    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) noexcept        = default;
    Model& operator=(Model&&) noexcept = default;

    // Constructs QnnApi + IOTensor, calls initializeHtp(), sets the perf
    // profile, then constructs and sets up all Graph objects. Must be called
    // before any subclass forward() invocation.
    bool initialize(const QnnRuntimeConfig& runtime_cfg,
                    const ModelConfig&      model_cfg);

    bool isInitialized() const;

    // ── Graph access ─────────────────────────────────────────────────────────
    size_t       graphCount() const;
    Graph&       graph(size_t idx);
    const Graph& graph(size_t idx) const;

    // ── Sub-model access ─────────────────────────────────────────────────────
    void         addSubModel(std::shared_ptr<Model> sub_model);
    Model&       subModel(size_t idx);
    const Model& subModel(size_t idx) const;

protected:
    // Prevent direct instantiation; only concrete subclasses should be created.
    Model() = default;

    // Called by initialize() after all Graph objects have been set up.
    // Override to build connection lists, allocate KV buffers, etc.
    virtual bool onInitialized() { return true; }

    // Executes a set of inter-graph tensor transfers using raw byte copies.
    // Each Connection routes one graph's output buffer into another graph's
    // input buffer, with size derived from the source TensorSpec::byteCount().
    void applyConnections(const std::vector<Connection>& connections);

    // ── Owned backend objects ─────────────────────────────────────────────────
    // Declaration order matters: graphs_ holds non-owning pointers into api_
    // and io_tensor_, so graphs_ must be destroyed first (it is listed last).
    ModelConfig                         model_cfg_;
    std::unique_ptr<QnnApi>             api_;
    std::shared_ptr<IOTensor>   io_tensor_;

    std::vector<Graph>                  graphs_;
    std::vector<std::shared_ptr<Model>> sub_models_;

    bool initialized_ = false;
};

} // namespace geniex
