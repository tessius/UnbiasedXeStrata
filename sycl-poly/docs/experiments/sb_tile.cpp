// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT  (a derivative of src/xe/iq_mmq.cpp, XeStrata's tiles: see ../../src/xe/README.md)
// EXPERIMENT, NOT BUILT, NOT USED (2026-11-26, DEVLOG section 41): a Q4_K XMX tile with the epilogue once per 256 of K (the block scale split into two 3-bit halves folded into the weights, the activations with a scale per 256 as ggml's Q8_K,
// the minimum term as one half-precision product of the block sums).  Correct (rel. 5e-3 against the int8 tile) but 0.75-0.8x its speed.  Self-contained: it includes iq_mmq.cpp for its internals; build as the int8 bench in DEVLOG 33.
#include <chrono>
#include <cmath>
#include <random>
#include <vector>
namespace strata::kernels {
namespace {
constexpr int SB_WM = 64, SB_WN = 128, SB_KC = 128;   // tile; K a step = 4 blocks of 32 (half a superblock)
// local memory (bytes): A tile-major 8 KB | A'' 2 KB | B1 16 KB | B0 16 KB | B'' 4 KB | wd 512 | wdm 512 | xsr 256
constexpr int SB_A = 0, SB_A2 = 8192, SB_B1 = 10240, SB_B0 = 26624, SB_BM = 43008, SB_WD = 47104, SB_WDM = 47616, SB_XSR = 48128, SB_LOCAL = 48384;
struct KernelSB {
    const uint8_t* w; const uint8_t* const* wtab; size_t woff, expert_bytes, row_bytes;
    const int8_t* xq; const float* xs; const sycl::half* sh; int64_t K; const int32_t* bounds; const int32_t* ids; float* dst; int64_t ld_dst, tiles_m, tiles_n;
    sycl::local_accessor<uint8_t, 1> lm;
    [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> it) const {
        const int64_t g = (int64_t) it.get_group(0), per = tiles_m * tiles_n, e = g / per, r = g % per;
        const int64_t m0 = r % tiles_m * SB_WM, n0 = r / tiles_m * SB_WN;
        const int64_t row0 = bounds[e], rows = bounds[e + 1] - row0;
        if (m0 >= rows) return;
        const auto sg = it.get_sub_group();
        const int sgid = (int) sg.get_group_linear_id(), lid = (int) it.get_local_id(0), lane = (int) sg.get_local_linear_id();
        const int cs = sgid % 8 * 16, rs = sgid / 8 * 32;
        const uint8_t* wx = wtab != nullptr ? wtab[e] + woff : w + (size_t) e * expert_bytes;
        const auto base = lm.template get_multi_ptr<sycl::access::decorated::no>();
        const auto pA = local_as<int8_t>(base + SB_A); int8_t* sA = pA.get();
        const auto pA2 = local_as<sycl::half>(base + SB_A2); sycl::half* sA2 = pA2.get();
        uint32_t* sB1 = local_as<uint32_t>(base + SB_B1).get();
        uint32_t* sB0 = local_as<uint32_t>(base + SB_B0).get();
        sycl::half* sBM = local_as<sycl::half>(base + SB_BM).get();
        float* sWD = local_as<float>(base + SB_WD).get(); float* sWDM = local_as<float>(base + SB_WDM).get(); float* sXSR = local_as<float>(base + SB_XSR).get();
        const int nsb = (int) (K / 256);
        const int nt = (int) std::min<int64_t>(4, (rows - m0 - rs + 7) / 8);

        mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 8, 16> acc[4], mf[4];
        mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, 8, 16> i1[4], i0[4];
        for (int t = 0; t < 4; ++t) { mx::joint_matrix_fill(sg, acc[t], 0.0f); mx::joint_matrix_fill(sg, i1[t], 0); mx::joint_matrix_fill(sg, i0[t], 0); }
        for (int i = lid; i < 1024; i += 256) reinterpret_cast<uint32_t*>(sBM)[i] = 0u;   // B'' zero (rows 4..7 stay zero)
        sycl::group_barrier(it.get_group());

        const int nn = lid / 2 * 1, bl2 = 0; (void) nn; (void) bl2;
        for (int step = 0; step < nsb * 2; ++step) {
            const int sbi = step / 2, half_ = step % 2;
            sycl::group_barrier(it.get_group());   // the last step's tiles are read
            // ---- stage the activations' tile: 64 rows x 128 bytes = 512 chunks of 16 B (2 a thread), tile-major
            for (int v = lid; v < SB_WM * 8; v += 256) {
                const int rr = v / 8, c = v % 8 * 16;
                const int64_t row = m0 + rr;
                u32x4 val = {0u, 0u, 0u, 0u};
                if (row < rows) val = *reinterpret_cast<const u32x4*>(xq + (row0 + row) * K + step * SB_KC + c);
                *reinterpret_cast<u32x4*>(sA + (((c / 32) * (SB_WM / 8) + rr / 8) * 8 + rr % 8) * 32 + c % 32) = val;
            }
            if (half_ == 0) {   // the superblock's S halves (16 a row: 32 bytes) and the rows' scales
                for (int v = lid; v < SB_WM * 2; v += 256) {
                    const int rr = v / 2, c = v % 2 * 8;
                    const int64_t row = m0 + rr;
                    u32x4 val = {0u, 0u, 0u, 0u};
                    if (row < rows) val = *reinterpret_cast<const u32x4*>(sh + ((row0 + row) * nsb + sbi) * 16 + c);
                    *reinterpret_cast<u32x4*>(sA2 + ((rr / 8) * 8 + rr % 8) * 16 + c) = val;
                }
                if (lid < SB_WM) { const int64_t row = m0 + lid; sXSR[lid] = row < rows ? xs[(row0 + row) * nsb + sbi] : 0.0f; }
            }
            // ---- decode the weights: 128 outputs x 4 blocks, 2 items a thread
            for (int v = lid; v < SB_WN * 4; v += 256) {
                const int nn_ = v / 4, bl = v % 4, j = half_ * 4 + bl;   // output, block of the step, block of the superblock
                const block_q4_K* blk = reinterpret_cast<const block_q4_K*>(wx + (size_t) (n0 + nn_) * row_bytes) + sbi;
                uint32_t q[8];
                for (int k = 0; k < 8; ++k) q[k] = ((uint32_t) get_int_b4(blk->qs, 8 * (j / 2) + k) >> (4 * (j & 1))) & 0x0F0F0F0Fu;
                int sc, mn;
                scale_min_k4(j, blk->scales, sc, mn);
                const uint32_t hi = (uint32_t) (sc >> 3), lo = (uint32_t) (sc & 7);
                for (int k = 0; k < 8; ++k) {
                    const int at = ((bl * (SB_WN / 16) + nn_ / 16) * 8 + k) * 16 + nn_ % 16;
                    sB1[at] = q[k] * hi;   // bytes <= 15 * 7 = 105: no carries
                    sB0[at] = q[k] * lo;
                }
                // B'': K = block j of the superblock (halves, VNNI pairs): dword row j / 2, half j % 2, column nn_
                sBM[(((nn_ / 16) * 8 + j / 2) * 16 + nn_ % 16) * 2 + (j & 1)] = (sycl::half) (float) mn;
                if (bl == 0 && half_ == 0) { sWD[nn_] = (float) blk->dm[0]; sWDM[nn_] = (float) blk->dm[1]; }
            }
            sycl::group_barrier(it.get_group());
            // ---- products: 4 blocks x 4 tiles, two int8 products each
            for (int bl = 0; bl < 4; ++bl) {
                mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, 32, 16, mx::layout::ext_intel_packed> b1, b0;
                mx::joint_matrix_load(sg, b1, local_as<int8_t>(base + SB_B1 + (bl * (SB_WN / 16) + cs / 16) * 512), 64);
                mx::joint_matrix_load(sg, b0, local_as<int8_t>(base + SB_B0 + (bl * (SB_WN / 16) + cs / 16) * 512), 64);
                for (int t = 0; t < 4; ++t) {
                    if (t < nt) {
                        mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, 8, 32, mx::layout::row_major> a;
                        mx::joint_matrix_load(sg, a, pA + ((bl * (SB_WM / 8) + (rs / 8 + t)) * 256), 32);
                        mx::joint_matrix_mad(sg, i1[t], a, b1, i1[t]);
                        mx::joint_matrix_mad(sg, i0[t], a, b0, i0[t]);
                    }
                }
            }
            if (half_ == 1) {   // the superblock is done: the epilogue, once
                mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, 16, 16, mx::layout::ext_intel_packed> bm;
                mx::joint_matrix_load(sg, bm, local_as<sycl::half>(base + SB_BM + (cs / 16) * 512), 32);
                const float dd = sWD[cs + lane], dmm = sWDM[cs + lane];
                for (int t = 0; t < 4; ++t) {
                    if (t < nt) {
                        mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, 8, 16, mx::layout::row_major> a2;
                        mx::joint_matrix_load(sg, a2, pA2 + (rs / 8 + t) * 8 * 16, 16);
                        mx::joint_matrix_fill(sg, mf[t], 0.0f);
                        mx::joint_matrix_mad(sg, mf[t], a2, bm, mf[t]);
                        float f1[8], f0[8], fm[8];
                        int i = 0; mx::joint_matrix_apply(sg, i1[t], [&](int32_t& v) { f1[i++] = (float) v; });
                        i = 0; mx::joint_matrix_apply(sg, i0[t], [&](int32_t& v) { f0[i++] = (float) v; });
                        i = 0; mx::joint_matrix_apply(sg, mf[t], [&](float& v) { fm[i++] = v; });
                        i = 0;
                        mx::joint_matrix_apply(sg, acc[t], [&](float& v) {
                            v = sycl::fma(sXSR[rs + t * 8 + i], dd * sycl::fma(8.0f, f1[i], f0[i]) - dmm * fm[i], v);
                            ++i;
                        });
                        mx::joint_matrix_fill(sg, i1[t], 0); mx::joint_matrix_fill(sg, i0[t], 0);
                    }
                }
            }
        }
        for (int t = 0; t < 4; ++t) {
            if (t < nt) {
                int i = 0;
                mx::joint_matrix_apply(sg, acc[t], [&](float& v) {
                    const int64_t row = m0 + rs + (int64_t) t * 8 + i;
                    if (row < rows) dst[(int64_t) ids[row0 + row] * ld_dst + n0 + cs + lane] = v;
                    ++i;
                });
            }
        }
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<16>, sycl::ext::intel::experimental::grf_size<256>}; }
};
}  // namespace
// activations with one scale a 256 (ggml's Q8_K): int8 [row][K] | float scales [row][K/256] | half block sums [row][K/256][16] (8 sums, 8 zeros)
size_t sb_act_bytes(int64_t rows, int64_t cols) { return (size_t) (rows + 8) * (size_t) (cols + cols / 256 * 4 + cols / 256 * 32); }
sycl::event sb_quantize(sycl::queue& q, const float* x, void* xq, int64_t cols, int64_t rows) {
    auto* x8 = static_cast<int8_t*>(xq); auto* xs = reinterpret_cast<float*>(x8 + rows * cols); const int64_t nsb = cols / 256;
    auto* sh = reinterpret_cast<sycl::half*>(xs + rows * nsb);
    return q.parallel_for(sycl::range<1>((size_t) (rows * nsb)), [=](sycl::id<1> i) {
        const int64_t r = (int64_t) i[0] / nsb, s = (int64_t) i[0] % nsb;
        const float* xr = x + r * cols + s * 256;
        float amax = 0.0f; for (int j = 0; j < 256; ++j) amax = sycl::fmax(amax, sycl::fabs(xr[j]));
        const float d = amax / 127.0f, id = amax > 0.0f ? 1.0f / d : 0.0f;
        for (int b = 0; b < 8; ++b) {
            int sum = 0;
            for (int j = 0; j < 32; ++j) { const int qv = (int) sycl::round(xr[b * 32 + j] * id); sum += qv; x8[r * cols + s * 256 + b * 32 + j] = (int8_t) qv; }
            sh[(r * nsb + s) * 16 + b] = (sycl::half) (float) sum; sh[(r * nsb + s) * 16 + 8 + b] = (sycl::half) 0.0f;
        }
        xs[r * nsb + s] = d;
    });
}
sycl::event sb_grouped(sycl::queue& q, const void* w, size_t expert_bytes, int64_t w_rows, int64_t w_cols, int n, const void* xq, int64_t total_rows, const int32_t* bounds,
                       int64_t max_rows, const int32_t* ids, float* dst, int64_t ld_dst) {
    const int64_t tiles_m = (max_rows + SB_WM - 1) / SB_WM, tiles_n = w_rows / SB_WN, nsb = w_cols / 256;
    const size_t row_bytes = (size_t) nsb * sizeof(block_q4_K);
    const auto* x8 = static_cast<const int8_t*>(xq); const auto* xs = reinterpret_cast<const float*>(x8 + total_rows * w_cols);
    const auto* sh = reinterpret_cast<const sycl::half*>(xs + total_rows * nsb);
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 1> lm(sycl::range<1>((size_t) SB_LOCAL), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n * (size_t) (tiles_m * tiles_n) * 256, 256),
                       KernelSB{static_cast<const uint8_t*>(w), nullptr, 0, expert_bytes, row_bytes, x8, xs, sh, w_cols, bounds, ids, dst, ld_dst, tiles_m, tiles_n, lm});
    });
}
}  // namespace strata::kernels
using namespace strata::kernels;
int main(int argc, char** argv) {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    const int ROWS = argc > 1 ? atoi(argv[1]) : 64; const int NE = argc > 2 ? atoi(argv[2]) : 64; const int ty = 12;
    struct Shape { const char* name; int64_t R, C; } shapes[] = {{"gate+up", 1024, 2048}, {"down(Q4_K here)", 2048, 512}};
    for (auto sh : shapes) {
        const int64_t total = (int64_t) NE * ROWS; const size_t eb = iq_mmq_row_bytes(ty, sh.C) * sh.R;
        std::mt19937 rng(5);
        std::vector<uint8_t> w(eb * NE); for (auto& b : w) b = (uint8_t) rng();
        const uint16_t half = 0x2400; for (size_t blk = 0; blk < w.size(); blk += 144) { std::memcpy(&w[blk], &half, 2); std::memcpy(&w[blk + 2], &half, 2); }
        std::vector<float> x((size_t) total * sh.C); for (auto& v : x) v = (float) ((int) (rng() % 2001) - 1000) / 1000.f;
        std::vector<int32_t> bnd(NE + 1), ids(total); for (int i = 0; i <= NE; ++i) bnd[i] = i * ROWS; for (int i = 0; i < total; ++i) ids[i] = i;
        auto* dw = sycl::malloc_device<uint8_t>(w.size(), q); q.memcpy(dw, w.data(), w.size());
        auto* dx = sycl::malloc_device<float>(x.size(), q); q.memcpy(dx, x.data(), x.size() * 4);
        auto* dxq = sycl::malloc_device<uint8_t>(iq_mmq_act_bytes(total, sh.C) + 4096, q);
        auto* dxs = sycl::malloc_device<uint8_t>(sb_act_bytes(total, sh.C), q);
        auto* db = sycl::malloc_device<int32_t>(NE + 1, q); q.memcpy(db, bnd.data(), (NE + 1) * 4);
        auto* di = sycl::malloc_device<int32_t>(total, q); q.memcpy(di, ids.data(), total * 4);
        auto* d8 = sycl::malloc_device<float>((size_t) total * sh.R, q); auto* ds = sycl::malloc_device<float>((size_t) total * sh.R, q); q.wait();
        iq_mmq_quantize(q, dx, nullptr, dxq, sh.C, sh.C, total, true, false); sb_quantize(q, dx, dxs, sh.C, total);
        iq_mmq_grouped(q, ty, dw, eb, sh.R, sh.C, NE, dxq, total, db, ROWS, di, d8, sh.R, nullptr, 0, total).wait();
        sb_grouped(q, dw, eb, sh.R, sh.C, NE, dxs, total, db, ROWS, di, ds, sh.R).wait();
        std::vector<float> o8((size_t) total * sh.R), os((size_t) total * sh.R);
        q.memcpy(o8.data(), d8, o8.size() * 4).wait(); q.memcpy(os.data(), ds, os.size() * 4).wait();
        double dm = 0, rm = 0, nan = 0; for (size_t i = 0; i < o8.size(); ++i) { if (os[i] != os[i]) nan++; dm = std::max(dm, (double) std::fabs(o8[i] - os[i])); rm = std::max(rm, (double) std::fabs(o8[i])); }
        auto time = [&](auto&& f) { f(); q.wait(); double best = 1e18; for (int rep = 0; rep < 4; ++rep) { auto t0 = std::chrono::steady_clock::now(); for (int i = 0; i < 20; ++i) f(); q.wait(); best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 20 * 1e6); } return best; };
        const double t8 = time([&] { iq_mmq_grouped(q, ty, dw, eb, sh.R, sh.C, NE, dxq, total, db, ROWS, di, d8, sh.R, nullptr, 0, total); });
        const double ts = time([&] { sb_grouped(q, dw, eb, sh.R, sh.C, NE, dxs, total, db, ROWS, di, ds, sh.R); });
        std::printf("  %-16s current %7.1f us | superblock %7.1f us (%.2fx) | max diff %.3g of %.4g (rel %.2e) NaN %.0f\n", sh.name, t8, ts, t8 / ts, dm, rm, dm / rm, nan);
    }
}
