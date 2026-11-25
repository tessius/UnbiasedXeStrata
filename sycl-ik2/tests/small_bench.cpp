// SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
// SPDX-License-Identifier: MIT
// Decode's small kernels one at a time, generic against rewritten, on synthetic data of Inkling-2-Mini's shapes: the largest relative difference of the outputs and the time a launch (back to back).
#include "ik2_kernels.dp.hpp"
#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <functional>
#include <vector>
static double bench(sycl::queue &q, int n, const std::function<void()> &f) {
    double best = 1e9;
    for (int rep = 0; rep < 4; ++rep) { auto t0 = std::chrono::steady_clock::now(); for (int i = 0; i < n; ++i) f(); q.wait(); best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / n); }
    return best * 1e6;
}
int main() {
    auto &q = dpct::get_in_order_queue();
    std::mt19937 rng(3); std::normal_distribution<float> nd(0.f, 1.f);
    int bad = 0;
    {   // mv_f16: the router (128 x 2304) and a small one
        for (auto shape : {std::pair<int, int>{128, 2304}, std::pair<int, int>{24, 2304}, std::pair<int, int>{96, 2304}}) {
            const int N = shape.first, K = shape.second;
            std::vector<uint16_t> hw((size_t) N * K); std::vector<float> hx(K);
            for (auto &w : hw) w = sycl::bit_cast<uint16_t>(sycl::half(nd(rng) * 0.05f)); for (auto &v : hx) v = nd(rng);
            auto *dw = sycl::malloc_device<uint16_t>(hw.size(), q); auto *dx = sycl::malloc_device<float>(K, q); auto *dy = sycl::malloc_device<float>(N, q);
            q.memcpy(dw, hw.data(), hw.size() * 2); q.memcpy(dx, hx.data(), K * 4).wait();
            std::vector<float> y[2]; double t[2];
            for (int mode = 0; mode < 2; ++mode) {
                ik2::k::set_small_wide(mode == 1);
                ik2::k::mv_f16(dw, N, K, dx, dy, &q); q.wait(); y[mode].resize(N); q.memcpy(y[mode].data(), dy, N * 4).wait();
                t[mode] = bench(q, 300, [&] { ik2::k::mv_f16(dw, N, K, dx, dy, &q); });
            }
            double md = 0, mv = 0; for (int i = 0; i < N; ++i) { md = std::max(md, (double) std::fabs(y[0][i] - y[1][i])); mv = std::max(mv, (double) std::fabs(y[0][i])); }
            const bool ok = md < 1e-4 * std::max(1.0, mv); bad += !ok;
            std::printf("mv_f16 %dx%d: generic %.1f us, wide %.1f us, max diff %.2e (max |y| %.2f)  %s\n", N, K, t[0], t[1], md, mv, ok ? "ok" : "WRONG");
            sycl::free(dw, q); sycl::free(dx, q); sycl::free(dy, q);
        }
    }
    return bad;
}
