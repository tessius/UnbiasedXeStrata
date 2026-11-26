// SPDX-License-Identifier: MIT
// Bench and accuracy check for docs/experiments/iq_mmq_f16_tile.inc.cpp (build it into iq_mmq.cpp first): int8 path against the half-precision path, Qwen3.6 expert shapes.
#include "iq_mmq.hpp"
#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using namespace strata::kernels;
int main(int argc, char** argv) {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    std::printf("path %s, f16 usable %d\n", iq_mmq_path(q) ? iq_mmq_path(q) : "none", (int) iq_mmq_f16_usable(q));
    const int ty = argc > 1 ? atoi(argv[1]) : 12; const int ROWS = argc > 2 ? atoi(argv[2]) : 64; const int NE = argc > 3 ? atoi(argv[3]) : 64;
    struct Shape { const char* name; int64_t R, C; } shapes[] = {{"gate+up", 1024, 2048}, {"down", 2048, 512}};
    int bad = 0;
    for (auto sh : shapes) {
        const int64_t total = (int64_t) NE * ROWS;
        const size_t rb = iq_mmq_row_bytes(ty, sh.C), eb = rb * sh.R;
        std::mt19937 rng(5);
        std::vector<uint8_t> w(eb * NE); for (auto& b : w) b = (uint8_t) rng();
        const uint16_t half = 0x2400; const size_t bs = ty == 12 ? 144 : 176;
        for (size_t blk = 0; blk < w.size(); blk += bs) { std::memcpy(&w[blk], &half, 2); std::memcpy(&w[blk + 2], &half, 2); }
        std::vector<float> x((size_t) total * sh.C); for (auto& v : x) v = (float) ((int) (rng() % 2001) - 1000) / 1000.f;
        std::vector<int32_t> bnd(NE + 1), ids(total); for (int i = 0; i <= NE; ++i) bnd[i] = i * ROWS; for (int i = 0; i < total; ++i) ids[i] = i;
        auto* dw = sycl::malloc_device<uint8_t>(w.size(), q); q.memcpy(dw, w.data(), w.size());
        auto* dx = sycl::malloc_device<float>(x.size(), q); q.memcpy(dx, x.data(), x.size() * 4);
        auto* dxq = sycl::malloc_device<uint8_t>(iq_mmq_act_bytes(total, sh.C) + 4096, q);
        auto* dxh = sycl::malloc_device<uint8_t>(iq_mmq_f16_act_bytes(total, sh.C), q);
        auto* db = sycl::malloc_device<int32_t>(NE + 1, q); q.memcpy(db, bnd.data(), (NE + 1) * 4);
        auto* di = sycl::malloc_device<int32_t>(total, q); q.memcpy(di, ids.data(), total * 4);
        auto* d8 = sycl::malloc_device<float>((size_t) total * sh.R, q); auto* d16 = sycl::malloc_device<float>((size_t) total * sh.R, q); q.wait();
        iq_mmq_quantize(q, dx, nullptr, dxq, sh.C, sh.C, total, true, false);
        iq_mmq_quantize_f16(q, dx, nullptr, dxh, sh.C, sh.C, total);
        iq_mmq_grouped(q, ty, dw, eb, sh.R, sh.C, NE, dxq, total, db, ROWS, di, d8, sh.R, nullptr, 0, total).wait();
        iq_mmq_grouped_f16(q, ty, dw, eb, sh.R, sh.C, NE, dxh, db, ROWS, di, d16, sh.R, nullptr, 0).wait();
        std::vector<float> o8((size_t) total * sh.R), o16((size_t) total * sh.R);
        q.memcpy(o8.data(), d8, o8.size() * 4).wait(); q.memcpy(o16.data(), d16, o16.size() * 4).wait();
        double dm = 0, rm = 0, nan = 0; for (size_t i = 0; i < o8.size(); ++i) { if (o16[i] != o16[i]) nan++; dm = std::max(dm, (double) std::fabs(o8[i] - o16[i])); rm = std::max(rm, (double) std::fabs(o8[i])); }
        auto time = [&](auto&& f) { f(); q.wait(); double best = 1e18; for (int rep = 0; rep < 3; ++rep) { auto t0 = std::chrono::steady_clock::now(); for (int i = 0; i < 20; ++i) f(); q.wait(); best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 20 * 1e6); } return best; };
        const double t8 = time([&] { iq_mmq_grouped(q, ty, dw, eb, sh.R, sh.C, NE, dxq, total, db, ROWS, di, d8, sh.R, nullptr, 0, total); });
        const double t16 = time([&] { iq_mmq_grouped_f16(q, ty, dw, eb, sh.R, sh.C, NE, dxh, db, ROWS, di, d16, sh.R, nullptr, 0); });
        const double ops = 2.0 * sh.R * sh.C * total;
        std::printf("  %-8s int8 %7.1f us (%5.1f TOPS) | f16 %7.1f us (%5.1f TOPS) | f16 vs int8: max diff %.3g of %.4g (rel %.2e), NaN %.0f\n", sh.name, t8, ops / (t8 * 1e-6) / 1e12, t16, ops / (t16 * 1e-6) / 1e12, dm, rm, dm / rm, nan);
        if (!(dm / rm < 2e-2) || nan) bad = 1;
        sycl::free(dw, q); sycl::free(dx, q); sycl::free(dxq, q); sycl::free(dxh, q); sycl::free(db, q); sycl::free(di, q); sycl::free(d8, q); sycl::free(d16, q);
    }
    return bad;
}
