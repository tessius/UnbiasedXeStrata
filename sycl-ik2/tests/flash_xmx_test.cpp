// SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
// SPDX-License-Identifier: MIT
// The XMX prefill attention against the CUDA kernel's (emulated) path: random q / K / V, causal, sliding window, relative bias, GQA, splits, ragged Tq.
#include "ik2_kernels.dp.hpp"
#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

static uint16_t h16(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }

static void perf(int H, int KV, int Tq, int qpos0, int window, int extent, int n_split) {
    auto &q = dpct::get_in_order_queue();
    std::mt19937 rng(3); std::normal_distribution<float> nd(0.f, 1.f);
    const int D = 128, KD = KV * D, q_ld = H * D, nk = qpos0 + Tq;
    std::vector<uint16_t> hq((size_t) Tq * q_ld), hk((size_t) nk * KD), hv((size_t) nk * KD);
    for (auto &x : hq) x = h16(nd(rng) * 0.6f); for (auto &x : hk) x = h16(nd(rng) * 0.6f); for (auto &x : hv) x = h16(nd(rng));
    std::vector<float> hb((size_t) Tq * H * std::max(1, extent)), hs((size_t) Tq * H, 0.f); for (auto &x : hb) x = nd(rng) * 0.5f;
    auto *dq = sycl::malloc_device<uint16_t>(hq.size(), q), *dk = sycl::malloc_device<uint16_t>(hk.size(), q), *dv = sycl::malloc_device<uint16_t>(hv.size(), q);
    auto *db = sycl::malloc_device<float>(hb.size(), q), *dsink = sycl::malloc_device<float>(hs.size(), q);
    const size_t scratch = (size_t) n_split * Tq * H * (D + 2) + 16; auto *dsc = sycl::malloc_device<float>(scratch, q); auto *dout = sycl::malloc_device<float>((size_t) Tq * q_ld, q);
    q.memcpy(dq, hq.data(), hq.size() * 2); q.memcpy(dk, hk.data(), hk.size() * 2); q.memcpy(dv, hv.data(), hv.size() * 2); q.memcpy(db, hb.data(), hb.size() * 4); q.memcpy(dsink, hs.data(), hs.size() * 4).wait();
    ik2::k::set_flash_xmx(true);
    double best = 1e9;
    for (int it = 0; it < 6; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        ik2::k::attn_flash_f16(dq, q_ld, dk, dv, 0, H, KV, D, Tq, qpos0, 1.f / std::sqrt((float) D), extent ? db : nullptr, extent, window, dsink, dout, q_ld, dsc, scratch, n_split, &q);
        q.wait_and_throw(); best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    const double keys = window < nk ? std::min<double>(window, qpos0 + Tq / 2.0) : qpos0 + Tq / 2.0;   // (average keys a query sees)
    std::printf("perf H%d KV%d Tq%d qpos0 %d window %s extent %d split %d: %.2f ms  (%.2f TFLOP/s of QK+PV)\n", H, KV, Tq, qpos0, window > 1000000 ? "full" : std::to_string(window).c_str(), extent, n_split,
                best * 1e3, 4.0 * H * D * Tq * keys / best * 1e-12);
    sycl::free(dq, q); sycl::free(dk, q); sycl::free(dv, q); sycl::free(db, q); sycl::free(dsink, q); sycl::free(dsc, q); sycl::free(dout, q);
}

int main(int argc, char **argv) {
    if (argc > 1 && std::string(argv[1]) == "perf") {   // Inkling-2-Mini's shapes, a chunk of 1024 queries into a 4000-token context
        perf(24, 8, 4096, 0, 1024, 1024, 1); perf(24, 8, 4096, 0, 1 << 30, 1024, 1); perf(24, 8, 4096, 4096, 1024, 1024, 1); perf(24, 8, 4096, 4096, 1 << 30, 1024, 1);   // (chunks of the engine's 4096 default: sliding / full, first / second chunk)
        if (argc > 2) { perf(24, 8, 1024, 3000, 1024, 0, 1); perf(24, 8, 1024, 3000, 1 << 30, 0, 1); }   // (no bias: what the relative bias costs)
        return 0;
    }
    auto &q = dpct::get_in_order_queue();
    std::mt19937 rng(5); std::normal_distribution<float> nd(0.f, 1.f);
    struct Cfg { int H, KV, Tq, qpos0, window, extent, n_split, kv_cap; } cfgs[] = {
        {4, 2, 100, 37, 1 << 30, 0, 1, 0}, {4, 2, 100, 37, 64, 0, 1, 0}, {4, 2, 100, 37, 1 << 30, 48, 1, 0},
        {6, 2, 70, 0, 128, 40, 2, 0}, {6, 2, 33, 200, 1 << 30, 0, 3, 0}, {2, 2, 8, 0, 1 << 30, 0, 1, 0}, {4, 2, 150, 500, 96, 64, 2, 1024}};
    int bad = 0;
    for (auto &c : cfgs) {
        const int D = 128, KD = c.KV * D, q_ld = c.H * D, nk = c.qpos0 + c.Tq, cap = c.kv_cap ? c.kv_cap : nk;
        std::vector<uint16_t> hq((size_t) c.Tq * q_ld), hk((size_t) cap * KD), hv((size_t) cap * KD);
        for (auto &x : hq) x = h16(nd(rng) * 0.6f); for (auto &x : hk) x = h16(nd(rng) * 0.6f); for (auto &x : hv) x = h16(nd(rng));
        std::vector<float> hb((size_t) c.Tq * c.H * std::max(1, c.extent)), hs((size_t) c.Tq * c.H);
        for (auto &x : hb) x = nd(rng) * 0.5f; for (auto &x : hs) x = nd(rng) * 0.3f;
        auto *dq = sycl::malloc_device<uint16_t>(hq.size(), q), *dk = sycl::malloc_device<uint16_t>(hk.size(), q), *dv = sycl::malloc_device<uint16_t>(hv.size(), q);
        auto *db = sycl::malloc_device<float>(hb.size(), q), *dsink = sycl::malloc_device<float>(hs.size(), q);
        const size_t scratch = (size_t) c.n_split * c.Tq * c.H * (D + 2) + 16;
        auto *dsc = sycl::malloc_device<float>(scratch, q); auto *dout = sycl::malloc_device<float>((size_t) c.Tq * q_ld, q);
        q.memcpy(dq, hq.data(), hq.size() * 2); q.memcpy(dk, hk.data(), hk.size() * 2); q.memcpy(dv, hv.data(), hv.size() * 2);
        q.memcpy(db, hb.data(), hb.size() * 4); q.memcpy(dsink, hs.data(), hs.size() * 4).wait();
        std::vector<float> out[2]; double secs[2];
        for (int mode = 0; mode < 2; ++mode) {   // 0: the CUDA kernel's path, 1: XMX
            ik2::k::set_flash_xmx(mode == 1);
            q.memset(dout, 0, (size_t) c.Tq * q_ld * 4).wait();
            auto t0 = std::chrono::steady_clock::now();
            ik2::k::attn_flash_f16(dq, q_ld, dk, dv, c.kv_cap, c.H, c.KV, D, c.Tq, c.qpos0, 1.f / std::sqrt((float) D), c.extent ? db : nullptr, c.extent,
                                   c.window, dsink, dout, q_ld, dsc, scratch, c.n_split, &q);
            q.wait_and_throw(); secs[mode] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            out[mode].resize((size_t) c.Tq * q_ld); q.memcpy(out[mode].data(), dout, out[mode].size() * 4).wait();
        }
        double maxd = 0, maxv = 0; size_t nan = 0;
        for (size_t i = 0; i < out[0].size(); ++i) { if (out[1][i] != out[1][i]) ++nan; else maxd = std::max(maxd, (double) std::fabs(out[0][i] - out[1][i])); maxv = std::max(maxv, (double) std::fabs(out[0][i])); }
        const bool ok = !nan && maxd < 3e-3 * std::max(1.0, maxv);
        bad += !ok;
        std::printf("H%d KV%d Tq%-3d q0 %-3d win %-10d ext %-2d split %d cap %-4d: max|diff| %.2e (max|out| %.2f) NaN %zu | emulated %.1f ms, xmx %.1f ms  %s\n", c.H, c.KV, c.Tq, c.qpos0,
                    c.window, c.extent, c.n_split, c.kv_cap, maxd, maxv, nan, secs[0] * 1e3, secs[1] * 1e3, ok ? "ok" : "WRONG");
        sycl::free(dq, q); sycl::free(dk, q); sycl::free(dv, q); sycl::free(db, q); sycl::free(dsink, q); sycl::free(dsc, q); sycl::free(dout, q);
    }
    return bad;
}
