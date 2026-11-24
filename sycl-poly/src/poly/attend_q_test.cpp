// SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
// SPDX-License-Identifier: MIT
// attend() for Qwen3.6's shape (head 256, 16 heads, 2 KV heads) against a CPU reference, at several context lengths.
//   attend_q_test   (exit 0: every length within 2e-3 of the reference)
#include "strata/poly/kernels.hpp"

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace strata::poly;

static uint16_t to_half(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }
static float from_half(uint16_t h) { return (float) sycl::bit_cast<sycl::half>(h); }

int main() {
    const int dim = 256, n_head = 16, kv = 2, per = n_head / kv;
    const int lens[] = {16, 33, 64, 90, 93, 94, 100, 128, 200, 300, 600};
    int worst_len = -1;
    double worst = 0;
    sycl::queue& q = dpct::get_in_order_queue();
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int L : lens) {
        std::vector<uint16_t> k((size_t) L * kv * dim), v(k.size());
        std::vector<float> qq((size_t) n_head * dim), ref((size_t) n_head * dim), out(ref.size());
        for (auto& x : k) x = to_half(nd(rng));
        for (auto& x : v) x = to_half(nd(rng));
        for (auto& x : qq) x = nd(rng);
        const float scale = 1.f / std::sqrt((float) dim);
        const int pos = L - 1;   // one token at position L-1 attends rows 0..L-1
        for (int h = 0; h < n_head; ++h) {
            const int g = h / per;
            std::vector<double> s(L);
            double mx = -1e30;
            for (int j = 0; j < L; ++j) {
                double d = 0;
                for (int i = 0; i < dim; ++i) d += (double) qq[(size_t) h * dim + i] * from_half(k[((size_t) j * kv + g) * dim + i]);
                s[j] = d * scale;
                mx = std::max(mx, s[j]);
            }
            double sum = 0;
            for (int j = 0; j < L; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
            for (int i = 0; i < dim; ++i) {
                double o = 0;
                for (int j = 0; j < L; ++j) o += s[j] / sum * from_half(v[((size_t) j * kv + g) * dim + i]);
                ref[(size_t) h * dim + i] = (float) o;
            }
        }
        auto* dk = sycl::malloc_device<uint16_t>(k.size(), q);
        auto* dv = sycl::malloc_device<uint16_t>(v.size(), q);
        auto* dq = sycl::malloc_device<float>(qq.size(), q);
        auto* dout = sycl::malloc_device<float>(out.size(), q);
        const size_t score_n = (size_t) n_head * (L + 64) * 4;
        auto* ds = sycl::malloc_device<float>(score_n, q);
        q.memcpy(dk, k.data(), k.size() * 2); q.memcpy(dv, v.data(), v.size() * 2); q.memcpy(dq, qq.data(), qq.size() * 4);
        q.wait();
        attend(dq, dk, dv, pos, 1, ds, 1, dout, dim, n_head, kv, scale, &q, 0, 0, false, nullptr, 0, 0, {}, {});
        q.memcpy(out.data(), dout, out.size() * 4).wait();
        double err = 0, nrm = 0;
        for (size_t i = 0; i < out.size(); ++i) { err = std::max(err, (double) std::fabs(out[i] - ref[i])); nrm = std::max(nrm, (double) std::fabs(ref[i])); }
        std::printf("L=%4d max|err|=%.3e (max|ref|=%.3f)%s\n", L, err, nrm, err > 2e-3 ? "   <-- WRONG" : "");
        if (err > worst) { worst = err; worst_len = L; }
        sycl::free(dk, q); sycl::free(dv, q); sycl::free(dq, q); sycl::free(dout, q); sycl::free(ds, q);
    }
    std::printf(worst > 2e-3 ? "FAIL (worst at L=%d)\n" : "ok (worst at L=%d)\n", worst_len);
    return worst > 2e-3;
}
