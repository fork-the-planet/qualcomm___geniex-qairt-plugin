// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: Apache-2.0

#include "vlm/vision_encoder.h"

namespace geniex {

bool QnnVisionEncoder::initialize(const QnnRuntimeConfig& runtime_cfg,
                                   const ModelConfig&      model_cfg) {
    return Model::initialize(runtime_cfg, model_cfg);
}

} // namespace geniex
