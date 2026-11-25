// SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
// SPDX-License-Identifier: MIT
// ik2::k::rms_f16 (RMS norm to FP16 and optionally F32) against a CPU reference, on aligned (vector) and misaligned (scalar) inputs.
#include "ik2_kernels.dp.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    std::mt19937 rng(9); std::normal_distribution<float> nd(0.f, 1.5f);
    const int C = 2304, T = 3; const float eps = 1e-6f; int bad = 0;
    for (int variant = 0; variant < 2; ++variant) {                 // 0: aligned, 1: x offset by one float (the scalar path)
        const int off = variant ? 1 : 0;
        std::vector<float> x((size_t) T * C + 4), w(C);
        for (auto &v : x) v = nd(rng); for (auto &v : w) v = 1.f + 0.2f * nd(rng);
        float *dx = sycl::malloc_device<float>(x.size(), q), *dw = sycl::malloc_device<float>(w.size(), q), *d32 = sycl::malloc_device<float>((size_t) T * C, q);
        uint16_t *d16 = sycl::malloc_device<uint16_t>((size_t) T * C, q);
        q.memcpy(dx, x.data(), x.size() * 4); q.memcpy(dw, w.data(), w.size() * 4).wait();
        ik2::k::rms_f16(dx + off, dw, T, C, eps, d16, d32, &q); q.wait_and_throw();
        std::vector<uint16_t> h((size_t) T * C); std::vector<float> f((size_t) T * C);
        q.memcpy(h.data(), d16, h.size() * 2); q.memcpy(f.data(), d32, f.size() * 4).wait();
        size_t nan16 = 0, inf16 = 0, nan32 = 0; double e16 = 0, e32 = 0;
        for (int t = 0; t < T; ++t) {
            double ss = 0; for (int i = 0; i < C; ++i) ss += (double) x[off + (size_t) t * C + i] * x[off + (size_t) t * C + i];
            const double rs = 1.0 / std::sqrt(ss / C + eps);
            for (int i = 0; i < C; ++i) {
                const double ref = x[off + (size_t) t * C + i] * rs * w[i];
                const float g16 = (float) sycl::bit_cast<sycl::half>(h[(size_t) t * C + i]), g32 = f[(size_t) t * C + i];
                if (g16 != g16) ++nan16; else if (std::isinf(g16)) ++inf16; else e16 = std::max(e16, std::fabs(ref - g16));
                if (g32 != g32) ++nan32; else e32 = std::max(e32, std::fabs(ref - g32));
            }
        }
        const bool ok = !nan16 && !inf16 && !nan32 && e16 < 5e-3 && e32 < 1e-4; bad += !ok;
        std::printf("%s input: f16 %zu NaN %zu inf, max err %.3e | f32 %zu NaN, max err %.3e  %s\n", variant ? "misaligned" : "aligned   ", nan16, inf16, e16, nan32, e32, ok ? "ok" : "WRONG");
        sycl::free(dx, q); sycl::free(dw, q); sycl::free(d32, q); sycl::free(d16, q);
    }
    return bad;
}
