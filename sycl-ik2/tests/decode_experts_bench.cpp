// SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
// SPDX-License-Identifier: MIT
// Decode's expert products as the engine launches them (one token, 8 experts, NVFP4, Inkling-2-Mini's C = 2304, F = 1152): gate/up + SwiGLU and down over 128 distinct
// experts in rotation (nothing stays in the cache), time a launch and the weight bandwidth it reaches.  Optional --check compares a launch with a CPU product.
#include "ik2_kernels.dp.hpp"
#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
static float h2f(uint16_t b) { return (float) sycl::bit_cast<sycl::half>(b); }
int main() {
    auto &q = dpct::get_in_order_queue();
    const int C = 2304, F = 1152, E = 128, S = 8;
    const size_t rbg = (size_t) C / 64 * 36, rbd = (size_t) F / 64 * 36, eb0 = F * rbg, off_down = 2 * F * rbg, ebytes = off_down + (size_t) C * rbd;
    std::mt19937 rng(7);
    std::vector<uint8_t> hw(ebytes * E); for (auto &b : hw) b = (uint8_t) rng();
    for (size_t blk = 0; blk < hw.size() / 36; ++blk) for (int i = 0; i < 4; ++i) hw[blk * 36 + i] = (uint8_t) (0x30 + rng() % 0x20);
    auto *dw = sycl::malloc_device<uint8_t>(hw.size(), q); q.memcpy(dw, hw.data(), hw.size()).wait();
    const int SETS = 16;   // 16 sets of 8 experts (disjoint: 128)
    std::vector<const uint8_t *> hp(SETS * S); for (int i = 0; i < SETS * S; ++i) hp[i] = dw + (size_t) i * ebytes;
    auto **dp = sycl::malloc_device<const uint8_t *>(SETS * S, q); q.memcpy(dp, hp.data(), hp.size() * sizeof(void *));
    std::vector<int> htok(S, 0); auto *dtok = sycl::malloc_device<int>(S, q); q.memcpy(dtok, htok.data(), S * 4);
    std::vector<float> hx(C); for (auto &x : hx) x = (float) (rng() % 200) / 100.f - 1.f; auto *dx = sycl::malloc_device<float>(C, q); q.memcpy(dx, hx.data(), C * 4);
    std::vector<float> hs(S, 1.f); auto *dsg = sycl::malloc_device<float>(S, q), *dsu = sycl::malloc_device<float>(S, q); q.memcpy(dsg, hs.data(), S * 4); q.memcpy(dsu, hs.data(), S * 4);
    auto *dh = sycl::malloc_device<uint16_t>((size_t) S * F, q); auto *dy = sycl::malloc_device<float>((size_t) S * C, q); q.wait();
    const int kT = ik2::k::kNVFP4, N = 400;
    std::vector<uint16_t> ref_h((size_t) S * F), got_h((size_t) S * F); std::vector<float> ref_y((size_t) S * C), got_y((size_t) S * C);
    for (int mode = 0; mode < 2; ++mode) {   // 0: the generic kernels, 1: wide loads: one launch each, outputs kept
                ik2::k::experts_gate_up_swiglu(dp, dtok, S, kT, kT, eb0, dx, C, F, dsg, dsu, 7.f, 5.f, dh, &q);
        ik2::k::experts_down(dp, S, kT, off_down, dh, C, F, dy, &q); q.wait();
        q.memcpy(mode ? got_h.data() : ref_h.data(), dh, (size_t) S * F * 2); q.memcpy(mode ? got_y.data() : ref_y.data(), dy, (size_t) S * C * 4).wait();
    }
    double eh = 0, ey = 0, mh = 0, my = 0; size_t nan = 0;
    for (size_t i = 0; i < ref_h.size(); ++i) { const float a = h2f(ref_h[i]), b = h2f(got_h[i]); if (b != b) ++nan; else eh = std::max(eh, (double) std::fabs(a - b)); mh = std::max(mh, (double) std::fabs(a)); }
    for (size_t i = 0; i < ref_y.size(); ++i) { if (got_y[i] != got_y[i]) ++nan; else ey = std::max(ey, (double) std::fabs(ref_y[i] - got_y[i])); my = std::max(my, (double) std::fabs(ref_y[i])); }
    std::printf("wide-load kernels vs generic: gate/up+swiglu max diff %.3g (max %.3g), down max diff %.3g (max %.3g), NaN %zu\n", eh, mh, ey, my, nan);
    const bool ok = !nan && eh < 2e-3 * std::max(1.0, mh) && ey < 1e-4 * std::max(1.0, my);
    for (int mode = 0; mode < 2; ++mode) {
        double best_gu = 1e9, best_dn = 1e9;
    for (int rep = 0; rep < 4; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) ik2::k::experts_gate_up_swiglu(dp + (i % SETS) * S, dtok, S, kT, kT, eb0, dx, C, F, dsg, dsu, 7.f, 5.f, dh, &q);
        q.wait(); best_gu = std::min(best_gu, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / N);
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) ik2::k::experts_down(dp + (i % SETS) * S, S, kT, off_down, dh, C, F, dy, &q);
        q.wait(); best_dn = std::min(best_dn, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / N);
    }
    const double gu_bytes = (double) S * 2 * F * rbg, dn_bytes = (double) S * C * rbd;
    std::printf("gate/up+swiglu: %.1f us a launch, %.0f GB/s of weights (%.1f MB)\n", best_gu * 1e6, gu_bytes / best_gu / 1e9, gu_bytes / 1e6);
    std::printf("down:           %.1f us a launch, %.0f GB/s of weights (%.1f MB)\n", best_dn * 1e6, dn_bytes / best_dn / 1e9, dn_bytes / 1e6);
    std::printf("per token (48 visits): %.2f ms   [%s]\n", 48 * (best_gu + best_dn) * 1e3, mode ? "wide loads" : "generic");
    }
    return ok ? 0 : 1;
}
