// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "vlm/vision_encoder.h"

namespace geniex {

bool QnnVisionEncoder::initialize(const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg) {
    return Model::initialize(runtime_cfg, model_cfg);
}

}  // namespace geniex
