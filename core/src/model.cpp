// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

// CPU-reachable Model members. Device bring-up lives in model_init.cpp so this
// TU can be included in the CPU coverage surface independently.

#include "model.h"

#include <algorithm>

namespace geniex {

bool Model::isInitialized() const { return initialized_; }

size_t Model::graphCount() const { return graphs_.size(); }

Graph& Model::graph(size_t idx) { return graphs_.at(idx); }

const Graph& Model::graph(size_t idx) const { return graphs_.at(idx); }

void Model::addSubModel(std::shared_ptr<Model> sub_model) { sub_models_.push_back(std::move(sub_model)); }

Model& Model::subModel(size_t idx) { return *sub_models_.at(idx); }

const Model& Model::subModel(size_t idx) const { return *sub_models_.at(idx); }

void Model::applyConnections(const std::vector<Connection>& connections) {
    for (const auto& conn : connections) {
        try {
            const void* src = graphs_[conn.src_graph_idx].outputPtr(conn.src_tensor_name);
            void*       dst = graphs_[conn.dst_graph_idx].inputPtr(conn.dst_tensor_name);

            if (src == dst) continue;  // shared buffer — no copy needed
            const auto& spec = graphs_[conn.src_graph_idx].outputSpec(conn.src_tensor_name);
            graphs_[conn.dst_graph_idx].write(conn.dst_tensor_name, src, spec.byteCount());
        } catch (const std::exception& e) {
            throw std::runtime_error("applyConnections failed: graph[" + std::to_string(conn.src_graph_idx) + "]." +
                                     conn.src_tensor_name + " -> graph[" + std::to_string(conn.dst_graph_idx) + "]." +
                                     conn.dst_tensor_name + ": " + e.what());
        }
    }
}

}  // namespace geniex
