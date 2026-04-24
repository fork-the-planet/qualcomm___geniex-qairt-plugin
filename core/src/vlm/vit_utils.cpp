#include "vlm/vit_utils.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace geniex {
namespace qwen_vit {

std::vector<float> makeInvFreq(int dim, float theta) {
    const int half = dim / 2;
    std::vector<float> inv(half);
    for (int i = 0; i < half; ++i)
        inv[i] = 1.0f / std::pow(theta, float(i * 2) / float(dim));
    return inv;
}

std::vector<int64_t> computeWindowIndex(int T, int H, int W,
                                         int spatial_merge_size,
                                         int window_size,
                                         int patch_size) {
    const int vit_ws  = window_size / spatial_merge_size / patch_size;
    const int llm_h   = H / spatial_merge_size;
    const int llm_w   = W / spatial_merge_size;
    const int n_tok   = T * llm_h * llm_w;

    const int pad_h = (vit_ws - (llm_h % vit_ws)) % vit_ws;
    const int pad_w = (vit_ws - (llm_w % vit_ws)) % vit_ws;
    const int nw_h  = (llm_h + pad_h) / vit_ws;
    const int nw_w  = (llm_w + pad_w) / vit_ws;

    // -100 marks padding cells that should be excluded from the window index.
    const int pH = llm_h + pad_h, pW = llm_w + pad_w;
    std::vector<int64_t> idx_pad(T * pH * pW, -100);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < llm_h; ++h)
            for (int w = 0; w < llm_w; ++w)
                idx_pad[t * pH * pW + h * pW + w] =
                    static_cast<int64_t>(t * llm_h * llm_w + h * llm_w + w);

    std::vector<int64_t> window_index;
    window_index.reserve(n_tok);

    for (int t = 0; t < T; ++t)
        for (int wh = 0; wh < nw_h; ++wh)
            for (int ww = 0; ww < nw_w; ++ww)
                for (int ih = 0; ih < vit_ws; ++ih)
                    for (int iw = 0; iw < vit_ws; ++iw) {
                        int h = wh * vit_ws + ih, w = ww * vit_ws + iw;
                        if (h >= pH || w >= pW) continue;
                        int64_t v = idx_pad[t * pH * pW + h * pW + w];
                        if (v != -100) window_index.push_back(v);
                    }

    return window_index;
}

std::vector<int64_t> reverseWindowIndex(const std::vector<int64_t>& window_index) {
    std::vector<int64_t> rev(window_index.size());
    for (size_t i = 0; i < window_index.size(); ++i)
        rev[static_cast<size_t>(window_index[i])] = static_cast<int64_t>(i);
    return rev;
}

std::pair<std::vector<float>, std::vector<float>>
computeSpatialCosSin(int T, int H, int W,
                     int spatial_merge_size,
                     const std::vector<float>& inv_freq,
                     const std::vector<int64_t>& window_index) {
    const int llm_h    = H / spatial_merge_size;
    const int llm_w    = W / spatial_merge_size;
    const int sm_unit  = spatial_merge_size * spatial_merge_size;
    const int n_groups = T * llm_h * llm_w;
    const int half     = static_cast<int>(inv_freq.size());
    const int emb_dim  = half * 2;  // h_freq concat w_freq

    std::vector<float> rot_emb(n_groups * emb_dim);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < llm_h; ++h)
            for (int w = 0; w < llm_w; ++w) {
                int row = t * llm_h * llm_w + h * llm_w + w;
                float* e = rot_emb.data() + row * emb_dim;
                for (int i = 0; i < half; ++i) {
                    e[i]        = float(h) * inv_freq[i];
                    e[half + i] = float(w) * inv_freq[i];
                }
            }

    const int seq_len = n_groups * sm_unit;
    std::vector<float> cos_out(seq_len * emb_dim), sin_out(seq_len * emb_dim);

    for (size_t i = 0; i < window_index.size(); ++i) {
        const float* src = rot_emb.data() + window_index[i] * emb_dim;
        for (int j = 0; j < sm_unit; ++j) {
            float* c = cos_out.data() + (i * sm_unit + j) * emb_dim;
            float* s = sin_out.data() + (i * sm_unit + j) * emb_dim;
            for (int d = 0; d < emb_dim; ++d) {
                c[d] = std::cos(src[d]);
                s[d] = std::sin(src[d]);
            }
        }
    }

    return {cos_out, sin_out};
}

std::vector<float> windowReorder(const std::vector<float>& hidden,
                                  size_t n_groups,
                                  size_t sm_unit,
                                  size_t embed_dim,
                                  const std::vector<int64_t>& window_index) {
    const size_t row_stride = sm_unit * embed_dim;
    std::vector<float> out(n_groups * row_stride);

    for (size_t i = 0; i < window_index.size(); ++i) {
        size_t src = static_cast<size_t>(window_index[i]);
        std::memcpy(out.data() + i * row_stride,
                    hidden.data() + src * row_stride,
                    row_stride * sizeof(float));
    }
    return out;
}

std::pair<std::vector<float>, std::vector<float>>
computePatchRoPE(int T, int H, int W,
                 int spatial_merge_size,
                 const std::vector<float>& inv_freq) {
    const int sm    = spatial_merge_size;
    const int half  = static_cast<int>(inv_freq.size());   // e.g. 20
    const int embed = half * 2;                            // e.g. 40

    const int llm_h = H / sm;
    const int llm_w = W / sm;

    auto make_ids = [&](bool is_h) {
        std::vector<int> ids(H * W);
        size_t k = 0;
        for (int br = 0; br < llm_h; ++br)
            for (int bc = 0; bc < llm_w; ++bc)
                for (int ir = 0; ir < sm; ++ir)
                    for (int ic = 0; ic < sm; ++ic) {
                        const int h_coord = br * sm + ir;
                        const int w_coord = bc * sm + ic;
                        ids[k++] = is_h ? h_coord : w_coord;
                    }
        return ids;
    };

    const auto hpos_ids = make_ids(/*is_h=*/true);
    const auto wpos_ids = make_ids(/*is_h=*/false);

    const size_t N = static_cast<size_t>(T) * H * W;
    std::vector<float> cos_out(N * embed), sin_out(N * embed);

    for (int t = 0; t < T; ++t) {
        for (size_t p = 0; p < static_cast<size_t>(H * W); ++p) {
            const size_t row = static_cast<size_t>(t) * H * W + p;
            const float h = static_cast<float>(hpos_ids[p]);
            const float w = static_cast<float>(wpos_ids[p]);
            float* c = cos_out.data() + row * embed;
            float* s = sin_out.data() + row * embed;
            for (int i = 0; i < half; ++i) {
                const float ht = h * inv_freq[i];
                const float wt = w * inv_freq[i];
                c[i]        = std::cos(ht);
                s[i]        = std::sin(ht);
                c[half + i] = std::cos(wt);
                s[half + i] = std::sin(wt);
            }
        }
    }
    return { std::move(cos_out), std::move(sin_out) };
}

std::vector<int64_t>
computeCuWindowSeqlens(int T, int H, int W,
                       int spatial_merge_size,
                       int window_size,
                       int patch_size) {
    const int sm     = spatial_merge_size;
    const int sm_u   = sm * sm;
    const int vit_ws = window_size / sm / patch_size;   // 112/2/14 = 4
    const int llm_h  = H / sm;
    const int llm_w  = W / sm;

    const int pad_h = (vit_ws - llm_h % vit_ws) % vit_ws;
    const int pad_w = (vit_ws - llm_w % vit_ws) % vit_ws;
    const int nw_h  = (llm_h + pad_h) / vit_ws;
    const int nw_w  = (llm_w + pad_w) / vit_ws;
    const int pH    = llm_h + pad_h;
    const int pW    = llm_w + pad_w;

    std::vector<int64_t> cu = {0};
    for (int t = 0; t < T; ++t) {
        for (int wh = 0; wh < nw_h; ++wh) {
            for (int ww = 0; ww < nw_w; ++ww) {
                int valid = 0;
                for (int ih = 0; ih < vit_ws; ++ih)
                    for (int iw = 0; iw < vit_ws; ++iw) {
                        const int h = wh * vit_ws + ih;
                        const int w = ww * vit_ws + iw;
                        if (h < llm_h && w < llm_w) ++valid;
                    }
                const int64_t next = cu.back() + static_cast<int64_t>(valid) * sm_u;
                if (next != cu.back()) cu.push_back(next);
            }
        }
    }
    (void)pH; (void)pW;    // silence -Wunused when NDEBUG
    return cu;
}

std::vector<float>
buildBlockAttentionMask(size_t N,
                        const std::vector<int64_t>& boundaries,
                        float allowed, float blocked) {
    if (boundaries.empty() || boundaries.front() != 0 ||
        static_cast<size_t>(boundaries.back()) != N) {
        throw std::runtime_error(
            "buildBlockAttentionMask: boundaries must start at 0 and end at N");
    }
    std::vector<int> row_block(N);
    for (size_t b = 0; b + 1 < boundaries.size(); ++b)
        for (int64_t i = boundaries[b]; i < boundaries[b + 1]; ++i)
            row_block[i] = static_cast<int>(b);

    std::vector<float> mask(N * N, blocked);
    for (size_t i = 0; i < N; ++i) {
        float* row = mask.data() + i * N;
        const int b = row_block[i];
        const int64_t lo = boundaries[b];
        const int64_t hi = boundaries[b + 1];
        for (int64_t j = lo; j < hi; ++j) row[j] = allowed;
    }
    return mask;
}

} // namespace qwen_vit
} // namespace geniex
