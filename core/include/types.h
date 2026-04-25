// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "QnnLog.h"
#include "QnnTypes.h"
#include "IBackend.hpp"  // for qnn::tools::netrun::PerfProfile

namespace geniex {

// QNN backend settings shared across all models.
//
// The three path fields are optional. Leave them as std::nullopt (the default)
// to have geniex_core auto-detect the correct HTP runtime folder based on the
// device's HTP architecture version (see runtime_resolver.h). Set them
// explicitly to override the auto-detected paths.
struct QnnRuntimeConfig {
    // Path to QnnHtp.dll / libQnnHtp.so.
    // std::nullopt = auto-detect from htp-files/ next to geniex_core.
    std::optional<std::string> backend_path;

    // Path to QnnSystem.dll / libQnnSystem.so.
    // std::nullopt = auto-detect (same folder as backend_path).
    std::optional<std::string> system_lib_path;

    // Path to QnnHtpNetRunExtensions.dll / libQnnHtpNetRunExtensions.so.
    // std::nullopt = auto-detect (same folder as backend_path).
    std::optional<std::string> extensions_path;

    QnnLog_Level_t log_level = QNN_LOG_LEVEL_ERROR;
    bool           debug     = false;
};

// Per-model configuration: everything needed to load and run a QNN graph model.
struct ModelConfig {
    std::vector<std::string>   model_paths;    // .bin shards in order
    std::string                tokenizer_path;
    std::string                embedding_path; // empty if token embeddings are computed by the model graph
    std::string                htp_config_path; // HTP JSON config (empty = default)
    qnn::tools::netrun::PerfProfile perf_profile = qnn::tools::netrun::PerfProfile::BURST;
};

// Generation-time parameters passed to LLMModel::generate() / VLMModel::generate().
struct GenerationConfig {
    int32_t max_tokens    = 512;
    float   temperature   = 1.0f;
    float   top_p         = 1.0f;
    bool    thinking_mode = false;
};

// Static description of a single graph tensor, populated from GraphInfo_t.
struct TensorSpec {
    std::string           name;
    Qnn_DataType_t        dtype        = QNN_DATATYPE_FLOAT_32;
    std::vector<uint32_t> shape;
    float                 quant_scale  = 1.0f;
    int32_t               quant_offset = 0;

    size_t elementSize() const {
        switch (dtype) {
            case QNN_DATATYPE_FLOAT_32:
            case QNN_DATATYPE_INT_32:
            case QNN_DATATYPE_UINT_32:
            case QNN_DATATYPE_SFIXED_POINT_32:
            case QNN_DATATYPE_UFIXED_POINT_32:
                return 4;
            case QNN_DATATYPE_FLOAT_16:
            case QNN_DATATYPE_INT_16:
            case QNN_DATATYPE_UINT_16:
            case QNN_DATATYPE_SFIXED_POINT_16:
            case QNN_DATATYPE_UFIXED_POINT_16:
                return 2;
            case QNN_DATATYPE_INT_8:
            case QNN_DATATYPE_UINT_8:
            case QNN_DATATYPE_SFIXED_POINT_8:
            case QNN_DATATYPE_UFIXED_POINT_8:
            case QNN_DATATYPE_BOOL_8:
                return 1;
            case QNN_DATATYPE_INT_64:
            case QNN_DATATYPE_UINT_64:
            case QNN_DATATYPE_FLOAT_64:
                return 8;
            default:
                return 0;
        }
    }

    size_t elementCount() const {
        size_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }

    size_t byteCount() const { return elementSize() * elementCount(); }
};

// Wires one graph's output tensor to another graph's input tensor.
struct Connection {
    int         src_graph_idx;
    std::string src_tensor_name;
    int         dst_graph_idx;
    std::string dst_tensor_name;
};

// Modality-native input for VisionEncoder::encode().
// Decoupled from VLMInput so VisionEncoder can be used outside a VLM context.
struct PixelData {
    std::vector<float>                  pixel_values;    // flat [total_patches * C * H * W]
    std::vector<std::array<int32_t, 3>> image_grid_thw;  // [{T, H, W}] per image
};

// Modality-native input for AudioEncoder::encode().
// Decoupled from VLMInput so AudioEncoder can be used outside a VLM context.
struct AudioData {
    std::vector<float>   audio_features;        // flat [num_frames * num_mel_bins]
    std::vector<int32_t> audio_attention_mask;  // [num_frames], 1 = valid, 0 = padding
    int32_t              num_frames   = 0;
    int32_t              num_mel_bins = 0;
};

} // namespace geniex

