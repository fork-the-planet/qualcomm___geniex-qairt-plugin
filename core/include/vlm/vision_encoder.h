#pragma once

#include "model.h"
#include "types.h"
#include "geniex_export.h"

#include <vector>

namespace geniex {

// Abstract vision encoder interface.
class GENIEX_VLM_API VisionEncoder {
public:
    virtual ~VisionEncoder() = default;

    virtual bool initialize(const QnnRuntimeConfig& runtime_cfg,
                            const ModelConfig&      model_cfg) = 0;

    // Returns flat [num_image_tokens * hidden_size] embeddings.
    virtual std::vector<float> encode(const PixelData& pixel_data) = 0;
};

// QNN-backed vision encoder base. Owns QNN graphs via Model.
// Subclasses implement encode() for model-specific inference logic.
class GENIEX_VLM_API QnnVisionEncoder : public VisionEncoder, public Model {
public:
    bool initialize(const QnnRuntimeConfig& runtime_cfg,
                    const ModelConfig&      model_cfg) override;

    std::vector<float> encode(const PixelData& pixel_data) override = 0;
};

} // namespace geniex
