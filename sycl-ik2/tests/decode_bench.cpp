// SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
// SPDX-License-Identifier: MIT
// NVFP4 -> FP16 weight decode (what prefill does per expert): throughput of ik2::k::decode_f16, and against a plain write of the same output.
#include "ik2_kernels.dp.hpp"
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
int main() {
    auto &q = dpct::get_in_order_queue();
    const int C = 2304, F = 1152, E = 32;
    const size_t n_gu = (size_t) 2 * F * C, bytes_gu = n_gu / 64 * 36;   // one expert's gate+up: 5.3M weights
    std::vector<uint8_t> hw(bytes_gu * E); std::mt19937 rng(1); for (auto &b : hw) b = (uint8_t) rng();
    for (size_t blk = 0; blk < hw.size() / 36; ++blk) for (int i = 0; i < 4; ++i) hw[blk * 36 + i] = (uint8_t) (0x20 + rng() % 0x30);
    auto *dw = sycl::malloc_device<uint8_t>(hw.size(), q); auto *out = sycl::malloc_device<uint16_t>(n_gu, q);
    q.memcpy(dw, hw.data(), hw.size()).wait();
    double bw = 1e9, t[2] = {1e9, 1e9};
    std::vector<uint16_t> ref(n_gu), got(n_gu); int bad = 0;
    for (int mode = 0; mode < 2; ++mode) {   // 0: the original kernel, 1: the byte table
        ik2::k::set_decode_lut(mode == 1);
        for (int it = 0; it < 5; ++it) {
            auto t0 = std::chrono::steady_clock::now();
            for (int e = 0; e < E; ++e) ik2::k::decode_f16(ik2::k::kNVFP4, dw + e * bytes_gu, n_gu, out, &q);
            q.wait(); t[mode] = std::min(t[mode], std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / E);
        }
        for (int e : {0, E - 1}) {   // (bit-exact against the original kernel)
            ik2::k::set_decode_lut(false); ik2::k::decode_f16(ik2::k::kNVFP4, dw + e * bytes_gu, n_gu, out, &q); q.memcpy(ref.data(), out, n_gu * 2).wait();
            ik2::k::set_decode_lut(mode == 1); ik2::k::decode_f16(ik2::k::kNVFP4, dw + e * bytes_gu, n_gu, out, &q); q.memcpy(got.data(), out, n_gu * 2).wait();
            for (size_t i = 0; i < n_gu; ++i) bad += ref[i] != got[i];
        }
    }
    for (int it = 0; it < 5; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        for (int e = 0; e < E; ++e) q.memset(out, e, n_gu * 2);
        q.wait(); bw = std::min(bw, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / E);
    }
    std::printf("decode_f16 NVFP4 gate+up (%.1fM weights, %.1f MB out): original %.1f us (%.0f GB/s), byte table %.1f us (%.0f GB/s) | plain memset %.1f us (%.0f GB/s) | mismatching halves %d\n",
                n_gu / 1e6, n_gu * 2 / 1e6, t[0] * 1e6, n_gu * 2 / t[0] / 1e9, t[1] * 1e6, n_gu * 2 / t[1] / 1e9, bw * 1e6, n_gu * 2 / bw / 1e9, bad);
    return bad != 0;
}
