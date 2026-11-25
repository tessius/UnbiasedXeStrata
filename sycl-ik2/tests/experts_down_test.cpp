// SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
// SPDX-License-Identifier: MIT
// experts_down_g (the grouped expert down product) on synthetic data, over quant type and tokens a group: did it finish, and (with --check)
// is it the CPU's dot product?  The first test of any kernel on a new card: a hang here is a hang in the engine.
//   ik2_experts_down_test [--type 40|8] [--ng N] [--rows C] [--cols F]
#include "ik2_kernels.dp.hpp"
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    int type = 40, ng = 1, C = 2304, F = 1152, G = 1, iters = 1; bool all = true;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--type") && i + 1 < argc) type = std::atoi(argv[++i]), all = false;
        else if (!std::strcmp(argv[i], "--ng") && i + 1 < argc) ng = std::atoi(argv[++i]), all = false;
        else if (!std::strcmp(argv[i], "--groups") && i + 1 < argc) G = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--rows") && i + 1 < argc) C = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--cols") && i + 1 < argc) F = std::atoi(argv[++i]);
    }
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    std::mt19937 rng(5);
    const int types[2] = {40, 8};
    const int ngs[4] = {1, 2, 4, 8};
    int bad = 0;
    for (int ti = 0; ti < 2; ++ti) for (int gi = 0; gi < 4; ++gi) {
        const int ty = all ? types[ti] : type, n = all ? ngs[gi] : ng;
        if (!all && (ti || gi)) break;
        const size_t rb = ty == 40 ? (size_t) F / 64 * 36 : (size_t) F / 32 * 34;   // NVFP4 blocks of 64 values (36 bytes), Q8_0 of 32 (34)
        std::vector<uint8_t> w(rb * C + 64); for (auto &b : w) b = (uint8_t) rng();
        if (ty == 40) for (size_t i = 0; i + 36 <= w.size(); i += 36) for (int j = 0; j < 4; ++j) w[i + j] = 0x38 + (rng() & 7);   // sane block scales
        std::vector<uint16_t> h((size_t) 64 * F); for (auto &v : h) v = sycl::bit_cast<uint16_t>(sycl::half((float) (rng() % 200) / 100.f - 1.f));
        uint8_t *dw = sycl::malloc_device<uint8_t>(w.size(), q); uint16_t *dh = sycl::malloc_device<uint16_t>(h.size(), q);
        float *dy = sycl::malloc_device<float>((size_t) 64 * C, q);
        const uint8_t **dblob = sycl::malloc_device<const uint8_t *>(G, q); int *dn = sycl::malloc_device<int>(G, q), *drow = sycl::malloc_device<int>((size_t) G * ik2::k::kGroupMax, q);
        q.memcpy(dw, w.data(), w.size()); q.memcpy(dh, h.data(), h.size() * 2);
        std::vector<const uint8_t *> bps(G, dw); std::vector<int> ns(G, n), rows((size_t) G * ik2::k::kGroupMax);
        for (int g = 0; g < G; ++g) { ns[g] = 1 + (g % n); for (int i = 0; i < ik2::k::kGroupMax; ++i) rows[(size_t) g * ik2::k::kGroupMax + i] = (g * 3 + i) % 24; }
        q.memcpy(dblob, bps.data(), G * sizeof(void *)); q.memcpy(dn, ns.data(), G * 4); q.memcpy(drow, rows.data(), rows.size() * 4);
        q.memset(dy, 0, (size_t) 64 * C * 4).wait();
        const auto t0 = std::chrono::steady_clock::now();
        bool ok = true;
        try { for (int it = 0; it < iters; ++it) { ik2::k::experts_down_g(dblob, dn, drow, G, ty, 0, dh, C, F, dy, &q); q.wait_and_throw(); } }
        catch (const sycl::exception &e) { ok = false; std::printf("type %2d ng %d: FAILED: %s\n", ty, n, e.what()); ++bad; break; }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("type %2d, %d groups of 1..%d tokens: %d launches finished in %.2f ms\n", ty, G, n, iters, ms);
        sycl::free(dw, q); sycl::free(dh, q); sycl::free(dy, q); sycl::free(dblob, q); sycl::free(dn, q); sycl::free(drow, q);
        if (!ok) break;
    }
    return bad ? 1 : 0;
}
