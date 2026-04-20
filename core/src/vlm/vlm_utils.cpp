#include "vlm/vlm_utils.h"

#include <algorithm>

namespace geniex {

MRoPEPositions computeMRoPEPositions(
    const std::vector<int32_t>&          input_ids,
    const std::vector<ImageGrid>&        image_grids,
    const std::vector<AudioSegmentInfo>& audio_segments,
    int                                  spatial_merge_size,
    int32_t                              vision_start_token_id,
    int32_t                              image_token_id,
    int32_t                              audio_start_token_id,
    int32_t                              audio_token_id) {

    const size_t seq_len = input_ids.size();

    // Output: flat [3 * seq_len], layout [temporal..., height..., width...]
    std::vector<int32_t> pos(3 * seq_len, 0);
    int32_t* temporal = pos.data();
    int32_t* height   = pos.data() + seq_len;
    int32_t* width    = pos.data() + 2 * seq_len;

    int image_idx = 0, audio_idx = 0;
    int32_t st = 0;  // running sequential counter

    size_t i = 0;
    while (i < seq_len) {
        const int32_t tok = input_ids[i];

        if (tok == vision_start_token_id &&
            image_idx < static_cast<int>(image_grids.size())) {
            // vision_start token: sequential position in all 3 dims
            temporal[i] = height[i] = width[i] = st++;
            ++i;

            // Consume all image tokens for this image
            const auto& thw = image_grids[image_idx];
            const int T     = thw[0], H = thw[1], W = thw[2];
            const int llm_h = H / spatial_merge_size;
            const int llm_w = W / spatial_merge_size;

            const int32_t img_start = st;

            for (int t = 0; t < T && i < seq_len; ++t)
                for (int h = 0; h < llm_h && i < seq_len; ++h)
                    for (int w = 0; w < llm_w && i < seq_len; ++w) {
                        if (input_ids[i] != image_token_id) goto image_done;
                        temporal[i] = img_start + t;
                        height[i]   = img_start + h;
                        width[i]    = img_start + w;
                        ++i;
                    }
            image_done:
            st = img_start + std::max({T, llm_h, llm_w});
            ++image_idx;

        } else if (tok == audio_start_token_id &&
                   audio_idx < static_cast<int>(audio_segments.size())) {
            // audio_start token: sequential position
            temporal[i] = height[i] = width[i] = st++;
            ++i;

            const int32_t num_atok  = audio_segments[audio_idx].num_llm_tokens;
            const int32_t aud_start = st;
            for (int32_t a = 0; a < num_atok && i < seq_len; ++a) {
                if (input_ids[i] != audio_token_id) break;
                temporal[i] = height[i] = width[i] = aud_start + a;
                ++i;
            }
            st = aud_start + num_atok;
            ++audio_idx;

        } else {
            // Regular text token: same sequential position in all 3 dims
            temporal[i] = height[i] = width[i] = st++;
            ++i;
        }
    }

    // mrope_deltas: difference between final position and seq_len
    const int32_t max_pos = (seq_len > 0)
        ? *std::max_element(temporal, temporal + seq_len) : 0;
    const int32_t delta = max_pos + 1 - static_cast<int32_t>(seq_len);

    return MRoPEPositions{pos, {delta, delta, delta}};
}

} // namespace geniex
