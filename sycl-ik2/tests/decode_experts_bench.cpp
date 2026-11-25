// SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
// SPDX-License-Identifier: MIT
// Decode's expert products as the engine launches them (one token, 8 experts, NVFP4, Inkling-2-Mini's C = 2304, F = 1152) over 128 distinct experts in rotation (nothing stays in the cache):
// the generic kernels (experts_gate_up_swiglu / experts_down) and the int8 limb path (expert_limb_mv, each call with its activation-packing launch, as the default engine runs them).
// Prints microseconds a launch, the weight bandwidth reached, and the limb path's difference from the generic one.
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
    const int C = 2304, F = 1152, E = 128, S = 8, SETS = E / S, N = 400, kT = ik2::k::kNVFP4;
    const size_t rbg = (size_t) C / 64 * 36, rbd = (size_t) F / 64 * 36, eb0 = F * rbg, off_down = 2 * F * rbg, ebytes = off_down + (size_t) C * rbd;
    std::mt19937 rng(7);
    std::vector<uint8_t> hw(ebytes * E); for (auto &b : hw) b = (uint8_t) rng();
    for (size_t blk = 0; blk < hw.size() / 36; ++blk) for (int i = 0; i < 4; ++i) hw[blk * 36 + i] = (uint8_t) (0x30 + rng() % 0x20);
    auto *dw = sycl::malloc_device<uint8_t>(hw.size(), q); q.memcpy(dw, hw.data(), hw.size()).wait();
    std::vector<const uint8_t *> hp(E); for (int i = 0; i < E; ++i) hp[i] = dw + (size_t) i * ebytes;
    auto **dp = sycl::malloc_device<const uint8_t *>(E, q); q.memcpy(dp, hp.data(), hp.size() * sizeof(void *));
    std::vector<int> htok(S, 0); auto *dtok = sycl::malloc_device<int>(S, q); q.memcpy(dtok, htok.data(), S * 4);
    std::vector<float> hx(C); for (auto &x : hx) x = (float) (rng() % 200) / 100.f - 1.f; auto *dx = sycl::malloc_device<float>(C, q); q.memcpy(dx, hx.data(), C * 4);
    std::vector<float> hs(S, 1.f); auto *dsg = sycl::malloc_device<float>(S, q), *dsu = sycl::malloc_device<float>(S, q); q.memcpy(dsg, hs.data(), S * 4); q.memcpy(dsu, hs.data(), S * 4);
    auto *dh = sycl::malloc_device<uint16_t>((size_t) S * F, q); auto *dy = sycl::malloc_device<float>((size_t) S * C, q);
    auto *dscr = sycl::malloc_device<uint8_t>((size_t) std::max(C / 16, S * F / 16) * 36 + 256, q); q.wait();
    const double gu_bytes = (double) S * 2 * F * rbg, dn_bytes = (double) S * C * rbd;

    // accuracy: the generic kernels' outputs against the limb path's, on the same input (down reads the generic gate's hidden in both)
    std::vector<uint16_t> gh((size_t) S * F), lh((size_t) S * F); std::vector<float> gy((size_t) S * C), ly((size_t) S * C);
    ik2::k::experts_gate_up_swiglu(dp, dtok, S, kT, kT, eb0, dx, C, F, dsg, dsu, 7.f, 5.f, dh, &q);
    q.memcpy(gh.data(), dh, (size_t) S * F * 2).wait();
    ik2::k::experts_down(dp, S, kT, off_down, dh, C, F, dy, &q); q.wait(); q.memcpy(gy.data(), dy, (size_t) S * C * 4).wait();
    ik2::k::expert_limb_mv(false, dp, off_down, C, F, S, dx, dh, dy, dsg, dsu, 7.f, 5.f, dscr, &q, false, false); q.wait(); q.memcpy(ly.data(), dy, (size_t) S * C * 4).wait();
    ik2::k::expert_limb_mv(true, dp, eb0, C, F, S, dx, dh, dy, dsg, dsu, 7.f, 5.f, dscr, &q, false, false); q.wait(); q.memcpy(lh.data(), dh, (size_t) S * F * 2).wait();
    double e1 = 0, m1 = 0, e2 = 0, m2 = 0; size_t nan = 0;
    for (size_t i = 0; i < gh.size(); ++i) { const float b = h2f(lh[i]); if (b != b) ++nan; else e1 = std::max(e1, (double) std::fabs(h2f(gh[i]) - b)); m1 = std::max(m1, (double) std::fabs(h2f(gh[i]))); }
    for (size_t i = 0; i < gy.size(); ++i) { if (ly[i] != ly[i]) ++nan; else e2 = std::max(e2, (double) std::fabs(gy[i] - ly[i])); m2 = std::max(m2, (double) std::fabs(gy[i])); }
    std::printf("limb vs generic: gate/up+swiglu max diff %.3g (max %.3g), down max diff %.3g (max %.3g), NaN %zu\n", e1, m1, e2, m2, nan);
    const bool ok = !nan && e1 < 2e-3 * std::max(1.0, m1) && e2 < 2e-3 * std::max(1.0, m2);

    double g_gu = 1e9, g_dn = 1e9, l_gu = 1e9, l_dn = 1e9;
    for (int rep = 0; rep < 4; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) ik2::k::experts_gate_up_swiglu(dp + (i % SETS) * S, dtok, S, kT, kT, eb0, dx, C, F, dsg, dsu, 7.f, 5.f, dh, &q);
        q.wait(); g_gu = std::min(g_gu, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / N);
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) ik2::k::experts_down(dp + (i % SETS) * S, S, kT, off_down, dh, C, F, dy, &q);
        q.wait(); g_dn = std::min(g_dn, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / N);
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) ik2::k::expert_limb_mv(true, dp + (i % SETS) * S, eb0, C, F, S, dx, dh, dy, dsg, dsu, 7.f, 5.f, dscr, &q, false, false);
        q.wait(); l_gu = std::min(l_gu, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / N);
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) ik2::k::expert_limb_mv(false, dp + (i % SETS) * S, off_down, C, F, S, dx, dh, dy, dsg, dsu, 7.f, 5.f, dscr, &q, false, false);
        q.wait(); l_dn = std::min(l_dn, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / N);
    }
    std::printf("generic gate/up+swiglu: %6.1f us, %3.0f GB/s | generic down: %6.1f us, %3.0f GB/s | per token (48 visits) %.2f ms\n", g_gu * 1e6, gu_bytes / g_gu / 1e9, g_dn * 1e6, dn_bytes / g_dn / 1e9, 48 * (g_gu + g_dn) * 1e3);
    std::printf("limb    gate/up+swiglu: %6.1f us, %3.0f GB/s | limb    down: %6.1f us, %3.0f GB/s | per token (48 visits) %.2f ms   (each with its pack launch)\n", l_gu * 1e6, gu_bytes / l_gu / 1e9, l_dn * 1e6, dn_bytes / l_dn / 1e9, 48 * (l_gu + l_dn) * 1e3);
    return ok ? 0 : 1;
}
