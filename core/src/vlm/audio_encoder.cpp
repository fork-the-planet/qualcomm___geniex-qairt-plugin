// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "vlm/audio_encoder.h"

namespace geniex {

bool QnnAudioEncoder::initialize(const QnnRuntimeConfig& runtime_cfg,
                                  const ModelConfig&      model_cfg) {
    return Model::initialize(runtime_cfg, model_cfg);
}

} // namespace geniex
