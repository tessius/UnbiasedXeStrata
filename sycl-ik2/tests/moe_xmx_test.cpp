// SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
// SPDX-License-Identifier: MIT
// The XMX prefill MoE products (NVFP4 experts dequantized into SLM) against decode_f16 + a CPU F32 product: ragged expert token counts, gate/up + SwiGLU, down.
#include "ik2_kernels.dp.hpp"
#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static float h2f(uint16_t b) { return (float) sycl::bit_cast<sycl::half>(b); }
static int pcol(int c) { const int j = c & 15; return (c & ~15) + (j < 8 ? 2 * j : 2 * (j - 8) + 1); }
static uint16_t f2hh(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }

int main() {
    auto &q = dpct::get_in_order_queue();
    std::mt19937 rng(11); std::normal_distribution<float> nd(0.f, 1.f);
    int bad = 0;
    struct Cfg { int C, F; std::vector<int> cnt; } cfgs[] = {{384, 192, {5, 64, 130, 1, 70}}, {2304, 1152, {64, 200, 9}}, {2304, 1152, std::vector<int>(128, 64)}};
    for (auto &cf : cfgs) {
        const int C = cf.C, F = cf.F, E = (int) cf.cnt.size();
        const size_t rbg = (size_t) C / 64 * 36, rbd = (size_t) F / 64 * 36, eb_up = F * rbg, off_down = 2 * F * rbg, ebytes = off_down + (size_t) C * rbd;
        int S = 0; for (int c : cf.cnt) S += c;
        std::vector<uint8_t> hw(ebytes * E);
        for (auto &b : hw) b = (uint8_t) rng();
        for (size_t blk = 0; blk < hw.size() / 36; ++blk) for (int i = 0; i < 4; ++i) hw[blk * 36 + i] = (uint8_t) (0x20 + rng() % 0x30);   // sane scales (no NaN code)
        std::vector<uint16_t> hA((size_t) (S + 16) * C), hH((size_t) (S + 16) * F);
        for (auto &x : hA) x = f2hh(nd(rng) * 0.5f); for (auto &x : hH) x = f2hh(nd(rng) * 0.5f);
        std::vector<float> hsg(S), hsu(S); for (auto &x : hsg) x = 0.5f + (rng() % 100) / 100.f; for (auto &x : hsu) x = 0.5f + (rng() % 100) / 100.f;
        auto *dw = sycl::malloc_device<uint8_t>(hw.size(), q); auto *dA = sycl::malloc_device<uint16_t>(hA.size(), q), *dH = sycl::malloc_device<uint16_t>((size_t) (S + 16) * F, q);
        auto *dsg = sycl::malloc_device<float>(S, q), *dsu = sycl::malloc_device<float>(S, q); auto *dy = sycl::malloc_device<float>((size_t) S * C, q);
        auto *dhout = sycl::malloc_device<uint16_t>((size_t) S * F, q);
        std::vector<uint16_t> pA(hA.size()), pH(hH.size());   // (pair order)
        for (size_t r = 0; r < hA.size() / C; ++r) for (int c = 0; c < C; ++c) pA[r * C + pcol(c)] = hA[r * C + c];
        for (size_t r = 0; r < hH.size() / F; ++r) for (int c = 0; c < F; ++c) pH[r * F + pcol(c)] = hH[r * F + c];
        q.memcpy(dw, hw.data(), hw.size()); q.memcpy(dA, pA.data(), pA.size() * 2); q.memcpy(dH, pH.data(), pH.size() * 2);
        q.memcpy(dsg, hsg.data(), S * 4); q.memcpy(dsu, hsu.data(), S * 4).wait();
        std::vector<ik2::k::MoeXmxExp> tab; int st = 0, mx = 0;
        for (int e = 0; e < E; ++e) { tab.push_back({dw + e * ebytes, st, cf.cnt[e]}); st += cf.cnt[e]; mx = std::max(mx, cf.cnt[e]); }
        auto *dtab = sycl::malloc_device<ik2::k::MoeXmxExp>(E, q); q.memcpy(dtab, tab.data(), E * sizeof(tab[0])).wait();
        const float gcap = 7.f, ucap = 5.f;
        auto t0 = std::chrono::steady_clock::now();
        ik2::k::moe_gate_up_swiglu_xmx(dtab, E, mx, dA, C, F, eb_up, dsg, dsu, gcap, ucap, dhout, &q);
        ik2::k::moe_down_xmx(dtab, E, mx, dH, C, F, off_down, dy, &q); q.wait_and_throw();
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        t0 = std::chrono::steady_clock::now();
        ik2::k::moe_gate_up_swiglu_xmx(dtab, E, mx, dA, C, F, eb_up, dsg, dsu, gcap, ucap, dhout, &q);
        ik2::k::moe_down_xmx(dtab, E, mx, dH, C, F, off_down, dy, &q); q.wait_and_throw();
        const double secs2 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::vector<uint16_t> gh((size_t) S * F); std::vector<float> gy((size_t) S * C);
        q.memcpy(gh.data(), dhout, gh.size() * 2); q.memcpy(gy.data(), dy, gy.size() * 4).wait();
        // reference weights through decode_f16
        std::vector<uint16_t> wg((size_t) 2 * F * C), wd((size_t) C * F);
        auto *dwg = sycl::malloc_device<uint16_t>(wg.size(), q), *dwd = sycl::malloc_device<uint16_t>(wd.size(), q);
        double eh = 0, ey = 0, mh = 0, my = 0; size_t nan = 0; st = 0;
        const bool timing_only = E > 16;
        if (timing_only) {   // (a layer visit's worth: 128 experts)
            double best = 1e9, bg = 1e9;
            for (int it = 0; it < 5; ++it) {
                auto a = std::chrono::steady_clock::now();
                ik2::k::moe_gate_up_swiglu_xmx(dtab, E, mx, dA, C, F, eb_up, dsg, dsu, gcap, ucap, dhout, &q); q.wait();
                auto b = std::chrono::steady_clock::now();
                ik2::k::moe_down_xmx(dtab, E, mx, dH, C, F, off_down, dy, &q); q.wait();
                auto c2 = std::chrono::steady_clock::now();
                bg = std::min(bg, std::chrono::duration<double>(b - a).count()); best = std::min(best, std::chrono::duration<double>(c2 - b).count());
            }
            std::printf("layer visit (128 experts x 64 tokens): gate/up+swiglu %.2f ms, down %.2f ms\n", bg * 1e3, best * 1e3);
            sycl::free(dw, q); sycl::free(dA, q); sycl::free(dH, q); sycl::free(dsg, q); sycl::free(dsu, q); sycl::free(dy, q); sycl::free(dhout, q); sycl::free(dtab, q); sycl::free(dwg, q); sycl::free(dwd, q);
            continue;
        }
        for (int e = 0; e < E; ++e) {
            ik2::k::decode_f16(ik2::k::kNVFP4, dw + e * ebytes, (size_t) 2 * F * C, dwg, &q);
            ik2::k::decode_f16(ik2::k::kNVFP4, dw + e * ebytes + off_down, (size_t) C * F, dwd, &q); q.wait();
            q.memcpy(wg.data(), dwg, wg.size() * 2); q.memcpy(wd.data(), dwd, wd.size() * 2).wait();
            for (int m = 0; m < cf.cnt[e]; ++m) {
                const int r = st + m;
                for (int j = 0; j < F; ++j) {
                    double g = 0, u = 0;
                    for (int k = 0; k < C; ++k) { const float a = h2f(hA[(size_t) r * C + k]); g += a * h2f(wg[(size_t) j * C + k]); u += a * h2f(wg[(size_t) (F + j) * C + k]); }
                    g *= hsg[r]; u *= hsu[r];
                    const double ref = gcap * std::tanh(g / gcap) * (1.0 / (1.0 + std::exp(-g))) * ucap * std::tanh(u / ucap), got = h2f(gh[(size_t) r * F + pcol(j)]);
                    if (got != got) ++nan; else eh = std::max(eh, std::fabs(ref - got)); mh = std::max(mh, std::fabs(ref));
                }
                for (int n = 0; n < C; ++n) {
                    double y = 0; for (int k = 0; k < F; ++k) y += h2f(hH[(size_t) r * F + k]) * h2f(wd[(size_t) n * F + k]);
                    const double got = gy[(size_t) r * C + n];
                    if (got != got) ++nan; else ey = std::max(ey, std::fabs(y - got)); my = std::max(my, std::fabs(y));
                }
            }
            st += cf.cnt[e];
        }
        const bool ok = !nan && eh < 4e-3 * std::max(1.0, mh) && ey < 2e-3 * std::max(1.0, my);
        bad += !ok;
        std::printf("C %d F %d experts %d tokens %d: gate/up+swiglu max err %.2e (max %.2f), down max err %.2e (max %.2f), NaN %zu | %.2f ms (warm %.2f ms)  %s\n", C, F, E, S, eh, mh, ey, my, nan, secs * 1e3, secs2 * 1e3, ok ? "ok" : "WRONG");
        sycl::free(dw, q); sycl::free(dA, q); sycl::free(dH, q); sycl::free(dsg, q); sycl::free(dsu, q); sycl::free(dy, q); sycl::free(dhout, q); sycl::free(dtab, q); sycl::free(dwg, q); sycl::free(dwd, q);
    }
    return bad;
}
