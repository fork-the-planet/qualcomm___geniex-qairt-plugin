// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "model.h"
#include "types.h"
#include "geniex_export.h"

#include <vector>

namespace geniex {

// Abstract audio encoder interface.
class GENIEX_VLM_API AudioEncoder {
public:
    virtual ~AudioEncoder() = default;

    virtual bool initialize(const QnnRuntimeConfig& runtime_cfg,
                            const ModelConfig&      model_cfg) = 0;

    // Returns flat [num_audio_tokens * hidden_size] embeddings.
    virtual std::vector<float> encode(const AudioData& audio_data) = 0;
};

// QNN-backed audio encoder base. Owns QNN graphs via Model.
// Subclasses implement encode() for model-specific inference logic.
class GENIEX_VLM_API QnnAudioEncoder : public AudioEncoder, public Model {
public:
    bool initialize(const QnnRuntimeConfig& runtime_cfg,
                    const ModelConfig&      model_cfg) override;

    std::vector<float> encode(const AudioData& audio_data) override = 0;
};

} // namespace geniex
