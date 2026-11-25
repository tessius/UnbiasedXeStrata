// SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
// SPDX-License-Identifier: MIT
// The emulated mma.sync.m16n8k16 and ldmatrix.x4 (src/ik2_mma_emul.dp.inc) against a CPU matrix multiply and a CPU transpose,
// with the operands laid out in per-lane registers as PTX ISA says.  Exit 0 when all agree.
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

namespace t {
#define __dpct_inline__ inline
#include "../src/ik2_mma_emul.dp.inc"
}

static uint16_t hbits(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }
static float hval(uint16_t b) { return (float) sycl::bit_cast<sycl::half>(b); }

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> ud(-1.f, 1.f);
    int bad = 0;
    for (int trial = 0; trial < 20; ++trial) {
        float A[16][16], B[16][8], C[16][8];
        for (auto& r : A) for (auto& v : r) v = hval(hbits(ud(rng)));
        for (auto& r : B) for (auto& v : r) v = hval(hbits(ud(rng)));
        for (auto& r : C) for (auto& v : r) v = ud(rng);
        // registers per lane, by the PTX layout
        std::vector<uint32_t> ra(32 * 4), rb(32 * 2); std::vector<float> rc(32 * 4);
        auto pack = [](float lo, float hi) { return (uint32_t) hbits(lo) | ((uint32_t) hbits(hi) << 16); };
        for (int l = 0; l < 32; ++l) {
            const int g = l >> 2, tt = l & 3;
            ra[l * 4 + 0] = pack(A[g][2 * tt], A[g][2 * tt + 1]);
            ra[l * 4 + 1] = pack(A[g + 8][2 * tt], A[g + 8][2 * tt + 1]);
            ra[l * 4 + 2] = pack(A[g][2 * tt + 8], A[g][2 * tt + 9]);
            ra[l * 4 + 3] = pack(A[g + 8][2 * tt + 8], A[g + 8][2 * tt + 9]);
            rb[l * 2 + 0] = pack(B[2 * tt][g], B[2 * tt + 1][g]);
            rb[l * 2 + 1] = pack(B[2 * tt + 8][g], B[2 * tt + 9][g]);
            rc[l * 4 + 0] = C[g][2 * tt]; rc[l * 4 + 1] = C[g][2 * tt + 1];
            rc[l * 4 + 2] = C[g + 8][2 * tt]; rc[l * 4 + 3] = C[g + 8][2 * tt + 1];
        }
        uint32_t *dA = sycl::malloc_device<uint32_t>(ra.size(), q), *dB = sycl::malloc_device<uint32_t>(rb.size(), q);
        float *dC = sycl::malloc_device<float>(rc.size(), q);
        q.memcpy(dA, ra.data(), ra.size() * 4); q.memcpy(dB, rb.data(), rb.size() * 4); q.memcpy(dC, rc.data(), rc.size() * 4).wait();
        q.parallel_for(sycl::nd_range<1>(32, 32), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int l = (int) it.get_local_id(0); float c[4] = {dC[l * 4], dC[l * 4 + 1], dC[l * 4 + 2], dC[l * 4 + 3]};
            uint32_t a[4] = {dA[l * 4], dA[l * 4 + 1], dA[l * 4 + 2], dA[l * 4 + 3]};
            t::mma16816(c, a, dB[l * 2], dB[l * 2 + 1]);
            for (int i = 0; i < 4; ++i) dC[l * 4 + i] = c[i];
        }).wait();
        q.memcpy(rc.data(), dC, rc.size() * 4).wait();
        double err = 0;
        for (int l = 0; l < 32; ++l) {
            const int g = l >> 2, tt = l & 3;
            const int rows[4] = {g, g, g + 8, g + 8}, cols[4] = {2 * tt, 2 * tt + 1, 2 * tt, 2 * tt + 1};
            for (int i = 0; i < 4; ++i) {
                double ref = C[rows[i]][cols[i]];
                for (int k = 0; k < 16; ++k) ref += (double) A[rows[i]][k] * B[k][cols[i]];
                err = std::max(err, std::fabs(ref - rc[l * 4 + i]));
            }
        }
        if (err > 1e-4) { ++bad; std::printf("mma trial %d: max error %.3e  WRONG\n", trial, err); }
        sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);
    }
    std::printf("mma16816: %s\n", bad ? "FAIL" : "ok (20 random trials, max error < 1e-4)");
    // ldmatrix: four 8x8 matrices of halves in memory, row stride 24 halves (as in padded shared tiles)
    int bad_ld = 0;
    for (int trans = 0; trans < 2; ++trans) {
        std::vector<uint16_t> mem(4 * 8 * 24); for (auto& v : mem) v = (uint16_t) rng();
        uint16_t *dm = sycl::malloc_device<uint16_t>(mem.size(), q); uint32_t *out = sycl::malloc_device<uint32_t>(32 * 4, q);
        q.memcpy(dm, mem.data(), mem.size() * 2).wait();
        q.parallel_for(sycl::nd_range<1>(32, 32), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int l = (int) it.get_local_id(0);
            const void *p = dm + (size_t) ((l >> 3) * 8 + (l & 7)) * 24;          // row (l & 7) of matrix (l >> 3)
            uint32_t r0, r1, r2, r3;
            if (trans) t::ldsm_x4_t(r0, r1, r2, r3, p); else t::ldsm_x4(r0, r1, r2, r3, p);
            out[l * 4] = r0; out[l * 4 + 1] = r1; out[l * 4 + 2] = r2; out[l * 4 + 3] = r3;
        }).wait();
        std::vector<uint32_t> got(32 * 4); q.memcpy(got.data(), out, got.size() * 4).wait();
        for (int l = 0; l < 32; ++l) for (int m = 0; m < 4; ++m) {
            const int g = l >> 2, tt = l & 3; auto el = [&](int row, int col) { return mem[(size_t) (m * 8 + row) * 24 + col]; };
            const uint32_t want = trans ? ((uint32_t) el(2 * tt, g) | ((uint32_t) el(2 * tt + 1, g) << 16))
                                        : ((uint32_t) el(g, 2 * tt) | ((uint32_t) el(g, 2 * tt + 1) << 16));
            if (got[l * 4 + m] != want) ++bad_ld;
        }
        sycl::free(dm, q); sycl::free(out, q);
    }
    std::printf("ldmatrix x4 (plain and .trans): %s\n", bad_ld ? "FAIL" : "ok");
    return (bad || bad_ld) ? 1 : 0;
}
