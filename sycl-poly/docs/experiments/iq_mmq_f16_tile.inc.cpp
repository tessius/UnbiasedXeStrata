// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT  (a derivative of src/xe/iq_mmq.cpp, XeStrata's tiles: see ../../src/xe/README.md)
// EXPERIMENT, NOT BUILT, NOT USED (2026-11-26, DEVLOG section 39): the half-precision XMX tile for Q4_K / Q5_K.  Correct (rel. 4e-3 against the int8 tile, i.e. its activation
// quantization), but 1.46x SLOWER than the int8 tile on the Arc Pro B60 (gate/up 1,045 against 716 us at 64 rows an expert).  Goes into iq_mmq.cpp's anonymous namespace beside Kernel
// (it uses Dec<TY>, Sub, local_as, mx, syclex, half_scales); the entry points at the end go into namespace strata::kernels with declarations in iq_mmq.hpp.
// docs/experiments/f16bench.cpp compares it with the int8 path and times both.
// ---------------------------------------------------------------- the half-precision tile (Intel XMX, 8 x 16 x 16 half)
// For the K-quants with one scale a block of 32 (Q4_K, Q5_K): the weights are decoded to scaled FP16 values (q * d sc - dmin m, in half2) into local memory and
// multiplied by FP16 activations, one accumulator per 8 x 16 tile across the whole of K: there is no per-block scale epilogue, which was ~60% of the int8 tile's
// instructions (DEVLOG 37-39).  The activations are FP16 rows with the elements of every four in the order 0 2 1 3 (a source word's bytes q0 q1 q2 q3 become
// the pairs (q0, q2) and (q1, q3) of the matrix engine's VNNI words by two masks, and the order of K does not matter to a product if both sides share it).
// Work-group = 16 sub-groups of 16 lanes (as the int8 tile's 64 x 128): a 64-row x 128-output tile, K in steps of 64 (two blocks), the weights' tile
// double-buffered in local memory, one barrier a step.  The activations are read from global memory by the matrix loads.
constexpr int F16_WM = 64, F16_WN = 128, F16_KC = 64, F16_LDW = F16_WN + 4;   // a pair row of local memory: 128 outputs and a pad of 4 words
constexpr int F16_LDA = F16_KC + 8;   // the activations' tile in local memory: a row is 64 halves and a pad of 8 (16-byte rows apart in the banks)
constexpr int F16_BUF_WORDS = (F16_KC / 2) * F16_LDW, F16_ABUF_HALVES = F16_WM * F16_LDA;
constexpr int F16_BUF_BYTES = F16_BUF_WORDS * 4 + F16_ABUF_HALVES * 2;   // one step's weights tile and activations tile
template<int TY>
struct Kernel16 {
    static_assert(!half_scales<TY>(), "one scale a block of 32");
    static constexpr int NSG16 = 16, LANES = NSG16 * 16;   // sub-groups of 16, work-items in the work-group
    static constexpr int LOCAL = 2 * F16_BUF_BYTES;
    const uint8_t* w;
    const uint8_t* const* wtab;
    size_t woff, expert_bytes, row_bytes;
    const sycl::half* xh;   // the activations: [row][K] half, in the order above
    int64_t K;
    const int32_t* bounds;
    const int32_t* ids;
    float* dst;
    int64_t ld_dst, tiles_m, tiles_n;
    sycl::local_accessor<uint8_t, 1> lm;

    [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> it) const {
#if STRATA_IQ_MMQ_INTEL
        const int64_t g = (int64_t) it.get_group(0), per = tiles_m * tiles_n, e = g / per, r = g % per;
        const int64_t m0 = r % tiles_m * F16_WM, n0 = r / tiles_m * F16_WN;
        const int64_t row0 = bounds[e], rows = bounds[e + 1] - row0;
        if (m0 >= rows) return;   // the whole work-group
        const auto sg = it.get_sub_group();
        const int sgid = (int) sg.get_group_linear_id(), lid = (int) it.get_local_id(0), lane = (int) sg.get_local_linear_id();
        const int cs = sgid % 8 * 16, rs = sgid / 8 * 32;   // the sub-group's first output column, and first row (4 tiles of 8 rows)
        const uint8_t* wx = wtab != nullptr ? wtab[e] + woff : w + (size_t) e * expert_bytes;
        const auto base = lm.template get_multi_ptr<sycl::access::decorated::no>();
        const sycl::half* ab = xh + (row0 + m0) * K;
        const int nt = (int) std::min<int64_t>(4, (rows - m0 - rs + 7) / 8);   // the 8-row tiles of this sub-group that have rows (it may be negative: none)

        mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 8, 16> acc[4];
        for (int t = 0; t < 4; ++t) mx::joint_matrix_fill(sg, acc[t], 0.0f);

        // thread lid's item of a step: output nn of the work-group's 128, block b of the step's 2 (neighbouring lanes: the blocks of an output, then the next output)
        const int nn = lid / 2, b = lid % 2;
        const uint8_t* wrow = wx + (size_t) (n0 + nn) * row_bytes;
        // and its two 16-byte pieces of the activations' tile (64 rows x 64 halves = 8 KB a step: 2 a thread)
        const int ar0 = lid / 8, ac = lid % 8 * 8;   // rows ar0 and ar0 + 32, halves ac .. ac + 7
        const int64_t nsteps = K / F16_KC;
        using u4 = uint32_t __attribute__((ext_vector_type(4)));
        auto load_a = [&](int64_t step, u4* r) {
            for (int h = 0; h < 2; ++h) {
                const int64_t row = m0 + ar0 + h * 32;
                r[h] = row < rows ? *reinterpret_cast<const u4*>(ab + (int64_t) (ar0 + h * 32) * K + step * F16_KC + ac) : u4{0u, 0u, 0u, 0u};
            }
        };
        auto decode = [&](int64_t step, uint32_t* Bk) {
            const Sub d = Dec<TY>::get(wrow, (int) (step * 2) + b, row_bytes);
            const sycl::half2 s2((sycl::half) d.s0, (sycl::half) d.s0), nm2((sycl::half) -d.m, (sycl::half) -d.m);
            const sycl::half2 k1024 = sycl::bit_cast<sycl::half2>(0x64006400u);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t x = d.v[j];
                const sycl::half2 lo = sycl::bit_cast<sycl::half2>((x & 0x00FF00FFu) | 0x64006400u) - k1024;          // (q0, q2)
                const sycl::half2 hi = sycl::bit_cast<sycl::half2>(((x >> 8) & 0x00FF00FFu) | 0x64006400u) - k1024;   // (q1, q3)
                Bk[(b * 16 + 2 * j) * F16_LDW + nn] = sycl::bit_cast<uint32_t>(sycl::fma(lo, s2, nm2));
                Bk[(b * 16 + 2 * j + 1) * F16_LDW + nn] = sycl::bit_cast<uint32_t>(sycl::fma(hi, s2, nm2));
            }
        };
        auto buf_b = [&](int64_t step) { return local_as<uint32_t>(base + (int) ((step & 1) * F16_BUF_BYTES)).get(); };
        auto buf_a = [&](int64_t step) { return local_as<sycl::half>(base + (int) ((step & 1) * F16_BUF_BYTES + F16_BUF_WORDS * 4)).get(); };
        auto store_a = [&](int64_t step, const u4* r) {
            sycl::half* A = buf_a(step);
            for (int h = 0; h < 2; ++h) *reinterpret_cast<u4*>(A + (ar0 + h * 32) * F16_LDA + ac) = r[h];
        };
        {
            u4 ra[2];
            load_a(0, ra);
            decode(0, buf_b(0));
            store_a(0, ra);
        }
        for (int64_t step = 0; step < nsteps; ++step) {
            sycl::group_barrier(it.get_group());   // this step's tiles are written, and the step before's (the other buffer) are read
            const bool more = step + 1 < nsteps;
            u4 ra[2];
            if (more) {
                load_a(step + 1, ra);   // in flight while this step's products run
                decode(step + 1, buf_b(step + 1));
            }
            const auto bptr = local_as<sycl::half>(base + (int) ((step & 1) * F16_BUF_BYTES));
            const auto aptr = local_as<sycl::half>(base + (int) ((step & 1) * F16_BUF_BYTES + F16_BUF_WORDS * 4));
            for (int kk = 0; kk < 4; ++kk) {
                mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, 16, 16, mx::layout::ext_intel_packed> bm;
                mx::joint_matrix_load(sg, bm, bptr + (int) (kk * 8 * F16_LDW * 2 + cs * 2), 2 * F16_LDW);
                for (int t = 0; t < 4; ++t) {
                    if (t < nt) {
                        mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, 8, 16, mx::layout::row_major> a;
                        mx::joint_matrix_load(sg, a, aptr + (int) ((rs + t * 8) * F16_LDA + kk * 16), F16_LDA);
                        mx::joint_matrix_mad(sg, acc[t], a, bm, acc[t]);
                    }
                }
            }
            if (more) store_a(step + 1, ra);
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
#else
        (void) it;
#endif
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<16>, sycl::ext::intel::experimental::grf_size<128>}; }
};

template<int TY>
sycl::event launch16(sycl::queue& q, const void* w, const uint8_t* const* wtab, size_t woff, size_t expert_bytes, int64_t w_rows, int64_t w_cols, int n,
                     const void* xh, const int32_t* bounds, int64_t max_rows, const int32_t* ids, float* dst, int64_t ld_dst) {
    using KT = Kernel16<TY>;
    const int64_t tiles_m = (max_rows + F16_WM - 1) / F16_WM, tiles_n = w_rows / F16_WN;
    const size_t row_bytes = (size_t) (w_cols / Dec<TY>::QK) * (size_t) Dec<TY>::BSZ;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 1> lm(sycl::range<1>((size_t) KT::LOCAL), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n * (size_t) (tiles_m * tiles_n) * KT::LANES, KT::LANES),
                       KT{static_cast<const uint8_t*>(w), wtab, woff, expert_bytes, row_bytes, static_cast<const sycl::half*>(xh), w_cols, bounds, ids, dst, ld_dst,
                          tiles_m, tiles_n, lm});
    });
}


// ---- the half-precision products (Intel XMX only; Q4_K, Q5_K): see Kernel16
bool iq_mmq_f16_type(int t) { return t == 12 || t == 13; }
size_t iq_mmq_f16_act_bytes(int64_t rows, int64_t cols) { return (size_t) (rows + 8) * (size_t) cols * sizeof(sycl::half); }   // 8 rows of slack: the last tile reads whole 8-row tiles
bool iq_mmq_f16_usable(sycl::queue& q) {
    static const bool on = [] { const char* v = std::getenv("STRATA_MMQ_F16"); return v == nullptr || std::strtol(v, nullptr, 10) != 0; }();
    return on && intel_shape_of(q) == 1;
}
// FP16 rows of x (row ids[i], or i), every four elements in the order 0 2 1 3
sycl::event iq_mmq_quantize_f16(sycl::queue& q, const float* x, const int32_t* ids, void* xq, int64_t cols, int64_t ld, int64_t rows) {
    auto* o = static_cast<sycl::half*>(xq);
    const int64_t q4 = cols / 4;
    return q.parallel_for(sycl::range<1>((size_t) (rows * q4)), [=](sycl::id<1> i) {
        const int64_t r = (int64_t) i[0] / q4, g = (int64_t) i[0] % q4;
        const float* xr = x + (int64_t) (ids ? ids[r] : r) * ld + g * 4;
        sycl::half* d = o + r * cols + g * 4;
        d[0] = (sycl::half) xr[0]; d[1] = (sycl::half) xr[2]; d[2] = (sycl::half) xr[1]; d[3] = (sycl::half) xr[3];
    });
}
sycl::event iq_mmq_grouped_f16(sycl::queue& q, int t, const void* w, size_t expert_bytes, int64_t w_rows, int64_t w_cols, int n, const void* xh,
                               const int32_t* bounds, int64_t max_rows, const int32_t* ids, float* dst, int64_t ld_dst, const uint8_t* const* wtab, size_t woff) {
    switch (t) {
        case 12: return launch16<12>(q, w, wtab, woff, expert_bytes, w_rows, w_cols, n, xh, bounds, max_rows, ids, dst, ld_dst);
        case 13: return launch16<13>(q, w, wtab, woff, expert_bytes, w_rows, w_cols, n, xh, bounds, max_rows, ids, dst, ld_dst);
        default: return {};
    }
}

