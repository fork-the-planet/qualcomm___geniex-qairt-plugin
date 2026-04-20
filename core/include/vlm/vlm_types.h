#pragma once

#include "types.h"

namespace geniex {

// Generic multimodal input for VLMModel::generate().
// VLMModel::generate() unpacks this into PixelData / AudioData
// before calling the respective encoders — the encoders never see VLMInput.
// Subclass to carry additional modality fields (e.g. audio).
struct VLMInput {
    PixelData pixel_data;
    virtual ~VLMInput() = default;
};

// Extended input for models that also carry an audio modality (e.g. OmniNeural).
struct AudioVLMInput : public VLMInput {
    AudioData audio_data;
};

} // namespace geniex
