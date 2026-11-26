// SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
// SPDX-License-Identifier: MIT
// The experts' prompt products (XeStrata's iq_mmq_grouped) on the matrix engine against the same products in dp4a, on synthetic weights of five ggml types.
// The path is chosen once a process, so the test runs itself twice (STRATA_NO_XMX=0 and 1) and compares the outputs.  It must report "path: XMX" for the
// first run on a card that has XMX, else it fails: a probe that rejects the matrix engine would otherwise pass this test by comparing dp4a with itself.
// (It caught icpx 2026.0 ignoring the sub-group size property under -fsycl-default-sub-group-size=32: every product but 16 columns wrong.)
#include "iq_mmq.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
using namespace strata::kernels;
static const int kTypes[] = {12 /*Q4_K*/, 13 /*Q5_K*/, 14 /*Q6_K*/, 8 /*Q8_0*/, 23 /*IQ4_XS*/};
static const char* kNames[] = {"Q4_K", "Q5_K", "Q6_K", "Q8_0", "IQ4_XS"};
static const int64_t R = 256, C = 1024, TOTAL = 205, MAXR = 130;
static const int NE = 3;
static int run(const char* file, bool want_xmx) {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    const char* path = iq_mmq_path(q);
    std::printf("path: %s\n", path ? path : "none");
    if (want_xmx && !(path && std::strcmp(path, "XMX") == 0)) { std::printf("FAIL: expected the XMX path\n"); return 1; }
    const int32_t bounds_h[NE + 1] = {0, 70, 75, 205};
    FILE* f = std::fopen(file, "wb");
    for (int ty : kTypes) {
        const bool sums = iq_mmq_type_sums(ty), dup = iq_mmq_type_dup(ty);
        const int64_t wc = dup ? 2 * C : C;
        const size_t rb = iq_mmq_row_bytes(ty, C), eb = rb * R;
        std::mt19937 rng(100 + ty);
        std::vector<uint8_t> w(eb * NE);
        for (auto& b : w) b = (uint8_t) rng();
        const uint16_t half = 0x2400;   // sane fp16 scales (random ones give inf / NaN)
        size_t bs = 0; int d0 = 0, d1 = -1;
        if (ty == 12) { bs = 144; d1 = 2; } else if (ty == 13) { bs = 176; d1 = 2; } else if (ty == 14) { bs = 210; d0 = 208; }
        else if (ty == 8) { bs = 34; } else { bs = 136; }
        for (size_t blk = 0; blk < w.size(); blk += bs) { std::memcpy(&w[blk + d0], &half, 2); if (d1 >= 0) std::memcpy(&w[blk + d1], &half, 2); }
        std::vector<float> x((size_t) TOTAL * C);
        for (auto& v : x) v = (float) ((int) (rng() % 2001) - 1000) / 1000.f;
        std::vector<int32_t> ids(TOTAL);
        for (int i = 0; i < TOTAL; ++i) ids[i] = i;
        auto* dw = sycl::malloc_device<uint8_t>(w.size(), q); q.memcpy(dw, w.data(), w.size());
        auto* dx = sycl::malloc_device<float>(x.size(), q); q.memcpy(dx, x.data(), x.size() * 4);
        auto* dxq = sycl::malloc_device<uint8_t>(iq_mmq_act_bytes(TOTAL, wc) + 4096, q);
        auto* db = sycl::malloc_device<int32_t>(NE + 1, q); q.memcpy(db, bounds_h, sizeof bounds_h);
        auto* di = sycl::malloc_device<int32_t>(TOTAL, q); q.memcpy(di, ids.data(), TOTAL * 4);
        auto* dd = sycl::malloc_device<float>((size_t) TOTAL * R, q); q.memset(dd, 0, (size_t) TOTAL * R * 4).wait();
        iq_mmq_quantize(q, dx, nullptr, dxq, C, C, TOTAL, sums, dup);
        iq_mmq_grouped(q, ty, dw, eb, R, wc, NE, dxq, TOTAL, db, MAXR, di, dd, R, nullptr, 0, TOTAL).wait();
        std::vector<float> out((size_t) TOTAL * R);
        q.memcpy(out.data(), dd, out.size() * 4).wait();
        std::fwrite(out.data(), 4, out.size(), f);
        sycl::free(dw, q); sycl::free(dx, q); sycl::free(dxq, q); sycl::free(db, q); sycl::free(di, q); sycl::free(dd, q);
    }
    std::fclose(f);
    return 0;
}
int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "run") return run(argv[2], std::getenv("STRATA_NO_XMX") == nullptr || std::string(std::getenv("STRATA_NO_XMX")) == "0");
    const std::string self = argv[0], a = "poly_mmq_xmx_a.tmp", b = "poly_mmq_xmx_b.tmp";
    if (std::system(("STRATA_NO_XMX=0 " + self + " run " + a).c_str()) != 0) return 1;
    if (std::system(("STRATA_NO_XMX=1 " + self + " run " + b).c_str()) != 0) return 1;
    int bad = 0;
    FILE* fa = std::fopen(a.c_str(), "rb"); FILE* fb = std::fopen(b.c_str(), "rb");
    const size_t n = (size_t) TOTAL * R;
    std::vector<float> va(n), vb(n);
    for (size_t t = 0; t < sizeof kTypes / sizeof *kTypes; ++t) {
        if (std::fread(va.data(), 4, n, fa) != n || std::fread(vb.data(), 4, n, fb) != n) { std::printf("short read\n"); return 1; }
        double dmax = 0, rmax = 0;
        for (size_t i = 0; i < n; ++i) { dmax = std::max(dmax, (double) std::fabs(va[i] - vb[i])); rmax = std::max(rmax, (double) std::fabs(vb[i])); }
        const bool ok = rmax > 0 && dmax <= 1e-5 * rmax;
        std::printf("%-7s XMX vs dp4a: max diff %.3g (max |ref| %.4g, rel %.2g)  %s\n", kNames[t], dmax, rmax, dmax / rmax, ok ? "ok" : "WRONG");
        bad += !ok;
    }
    std::fclose(fa); std::fclose(fb); std::remove(a.c_str()); std::remove(b.c_str());
    return bad ? 1 : 0;
}
