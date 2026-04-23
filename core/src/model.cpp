#include "model.h"
#include "logging.h"
#include "runtime.h"

#include <algorithm>
#include <cstdarg>

#include "QnnConfig.hpp"
#include "qnn-utils.hpp"

namespace geniex {

// Bridges QNN's va_list callback into geniex logging. Prefixes "[QNN] " to distinguish
// QNN-internal messages from library-originated ones.
static void qnnLogCallback(const char* fmt, uint32_t level, uint64_t /*timestamp*/, va_list args) {
    char buf[1024];
    vsnprintf(buf, sizeof(buf), fmt, args);

    // QNN numeric levels: 1=ERROR, 2=WARN, 3=INFO, 4=VERBOSE, 5=DEBUG.
    LogLevel mapped;
    switch (level) {
        case 1:  mapped = LogLevel::Error; break;
        case 2:  mapped = LogLevel::Warn;  break;
        case 3:  mapped = LogLevel::Info;  break;
        case 4:  mapped = LogLevel::Trace; break;
        default: mapped = LogLevel::Debug; break;
    }

    if (geniex_log_callback) {
        auto msg = fmt::format("[QNN] {}", buf);
        geniex_log_callback(mapped, msg.c_str());
    }
}

bool Model::isInitialized() const { return initialized_; }

size_t Model::graphCount() const { return graphs_.size(); }

Graph& Model::graph(size_t idx) { return graphs_.at(idx); }

const Graph& Model::graph(size_t idx) const { return graphs_.at(idx); }

void Model::addSubModel(std::shared_ptr<Model> sub_model) {
    sub_models_.push_back(std::move(sub_model));
}

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
            throw std::runtime_error(
                "applyConnections failed: graph[" + std::to_string(conn.src_graph_idx) + "]." +
                conn.src_tensor_name + " -> graph[" + std::to_string(conn.dst_graph_idx) + "]." +
                conn.dst_tensor_name + ": " + e.what());
        }
    }
}

bool Model::initialize(const QnnRuntimeConfig& runtime_cfg,
                       const ModelConfig&      model_cfg) {
    if (initialized_) return true;
    model_cfg_ = model_cfg;

    QnnRuntimeConfig resolved_cfg = runtime_cfg;
    resolveHtpPaths(resolved_cfg);

    api_ = std::make_unique<QnnApi>();

    io_tensor_ = std::make_shared<IOTensor>(BufferAlloc::SHARED_BUFFER,
                                            api_->getQnnInterfaceVer());
    api_->setIOTensorBufferMgr(io_tensor_.get());

    // extensions_path value_or("") preserves the convention where empty string disables the library.
    BackendExtensionsConfigs ext_cfg(resolved_cfg.extensions_path.value_or(""),
                                    model_cfg.htp_config_path);

    const bool ok = api_->initializeHtp(
        resolved_cfg.backend_path.value(),
        model_cfg.model_paths,
        ext_cfg,
        qnn::tools::netrun::PerfProfile::BURST,
        {},
        true,
        resolved_cfg.system_lib_path.value_or(""),
        resolved_cfg.debug,
        0,
        0,
        false,
        false,
        0,
        true,
        false,
        {},
        false,
        false,
        static_cast<uint32_t>(resolved_cfg.log_level),
        qnnLogCallback);

    if (!ok) {
        return false;
    }

    auto quallaPerf = qualla::QnnUtils::qnnToQuallaPerformanceProfile(model_cfg.perf_profile);
    api_->setPerfProfile(quallaPerf);

    qnn_wrapper_api::GraphInfo_t** graphs_info = api_->getGraphsInfo();
    const uint32_t count = api_->getGraphsCount();

    graphs_.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        qnn_wrapper_api::GraphInfo_t* g = graphs_info[i];
        Qnn_ContextHandle_t ctx = api_->getContexts(g);
        graphs_.emplace_back(g, api_.get(), io_tensor_.get());
        if (!graphs_.back().setup(ctx)) {
            return false;
        }
    }

    if (!onInitialized()) {
        return false;
    }

    initialized_ = true;
    return true;
}

} // namespace geniex
