// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "types.h"

namespace geniex {

// Generic multimodal input for VLMModel::generate().
// Subclass to carry additional modality fields.
struct VLMInput {
    PixelData pixel_data;
    virtual ~VLMInput() = default;
};

}  // namespace geniex
