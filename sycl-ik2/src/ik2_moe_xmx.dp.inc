// SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
// SPDX-License-Identifier: MIT
// Prefill MoE products on the Intel matrix engine (XMX, SYCL joint_matrix) straight from the NVFP4 expert weights: all hit experts of a layer in ONE launch
// per product, the weights dequantized on the fly into shared local memory (no FP16 copy of the weights in global memory, no per-expert launches).
// Replaces, per expert, decode_f16 (NVFP4 -> FP16) + oneMKL gemm16 (+ swiglu_cap_f32).  No include guard / includes: #included in the kernels' namespace
// after ik2_flash_xmx.dp.inc (xmx_local, xmx:: alias, ik2_e2m1, ue4m3).
//
// The activations are in the "pair" column order (see mx_perm_col): within each 16 columns, position 2q holds column q and 2q+1 holds column q + 8, so one
// NVFP4 byte (elements q, q + 8) is one VNNI pair and the dequantization is one table lookup per byte (+ one half2 multiply by the sub-block scale).
//
// Work-group = 4 sub-groups (16 lanes), tile = 64 tokens (rows of the expert's sorted input) x 64 weight rows, K in steps of one NVFP4 block (64 values):
//   thread t dequantizes weight row t's block into B (VNNI, k pairs) in SLM; each sub-group takes 16 of the 64 columns: acc[8 row blocks of 8 tokens] +=
//   A (global, f16, [token][K]) x B.  Gate/up: the sub-groups 0,1 hold 32 gate columns and 2,3 the same 32 up columns, so the SwiGLU is in the epilogue.
template <bool> class ik2_moe_xmx_kernel_name;   // (MoeXmxExp: ik2_kernels.dp.hpp)
constexpr int kMxBsLd = 68, kMxCsLd = 68;   // (words / floats a row: a pad against bank conflicts, 16-byte aligned rows)
constexpr int kMxBsWords = 32 * kMxBsLd;                       // (one dequantized tile: 32 k-pair rows)
constexpr int kMxSmem = 64 * kMxCsLd * 4 + 256 * 4;               // (two tiles = 17408 B = the output staging, which reuses them; then the byte table)

// column c of a row of activations -> its position in the pair order (16-column groups: (q, q + 8) adjacent)
__dpct_inline__ int mx_perm_col(int c) { const int j = c & 15; return (c & ~15) + (j < 8 ? 2 * j : 2 * (j - 8) + 1); }
// the byte table: byte b = (nibble lo, nibble hi) -> half2 bits (lo, hi) of the DOUBLED E2M1 values {0, 1, 2, 3, 4, 6, 8, 12} with their signs
__dpct_inline__ uint32_t mx_lut_entry(uint32_t b) {
    auto h = [](uint32_t n) -> uint32_t {
        const uint32_t m = n & 7u, mag = m == 0 ? 0u : m == 1 ? 0x3C00u : (m << 9) + 0x3C00u;   // (m >= 2: (2 + (m & 1)) << (m >> 1 - 1) as FP16 bits)
        return mag | ((n & 8u) << 12);
    };
    return h(b & 15u) | (h(b >> 4) << 16);
}
// thread t's weight row, one 64-value block (36 bytes: 4 sub-block scales, 4 x 8 bytes of nibbles) -> B[k/2][t] (k pair = (q, q + 8) of the sub-block)
__dpct_inline__ void mx_dequant_block(const uint32_t *w, bool ok, uint32_t *Bs, int t, const uint32_t *lut) {   // (w: the block's 9 words, loaded a step ahead)
    const uint32_t sc = ok ? w[0] : 0u;
    for (int s = 0; s < 4; ++s) {
        const uint16_t dh = sycl::bit_cast<uint16_t>((sycl::half) (ok ? ue4m3((uint8_t) (sc >> (8 * s))) : 0.0f));   // (exact in FP16)
        const sycl::half2 d2 = sycl::bit_cast<sycl::half2>((uint32_t) dh | ((uint32_t) dh << 16));
        const uint32_t q0 = w[1 + 2 * s], q1 = w[2 + 2 * s];
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const uint32_t byte = ((q < 4 ? q0 : q1) >> (8 * (q & 3))) & 0xFFu;
            Bs[(s * 8 + q) * kMxBsLd + t] = ok ? sycl::bit_cast<uint32_t>(sycl::bit_cast<sycl::half2>(lut[byte]) * d2) : 0u;
        }
    }
}

template <bool GATEUP>
__dpct_inline__ void moe_xmx_kernel(const MoeXmxExp *__restrict__ tab, const uint16_t *__restrict__ A, int K, int N, int F, size_t eb_up, size_t off_w,
                                    const float *__restrict__ sg, const float *__restrict__ su, float gcap, float ucap, void *out, uint8_t *smem,
                                    sycl::nd_item<3> it) {
    auto sg_ = it.get_sub_group();
    const int tid = (int) it.get_local_id(2), s = tid >> 4, lane = tid & 15;
    const MoeXmxExp ex = tab[it.get_group(0)];
    const int m0 = (int) it.get_group(1) * 64;
    if (m0 >= ex.cnt) return;                 // (the whole work-group: no barrier is skipped)
    const int n0 = (int) it.get_group(2) * (GATEUP ? 32 : 64);
    uint32_t *Bs = (uint32_t *) smem;
    float *Cs = (float *) smem;   // (after the last K step: Bs's room)
    uint32_t *lut = (uint32_t *) (smem + 64 * kMxCsLd * 4);
    for (int i = tid; i < 256; i += 64) lut[i] = mx_lut_entry((uint32_t) i);
    const size_t rb = (size_t) (K / 64) * 36;   // bytes a weight row
    // this thread's weight row
    const int row = GATEUP ? n0 + (tid & 31) : n0 + tid;
    const bool rok = row < N;
    const uint8_t *wrow = ex.blob + off_w + (GATEUP && tid >= 32 ? eb_up : 0) + (size_t) row * rb;
    const int nrb = sycl::min(8, (ex.cnt - m0 + 7) / 8);   // the 8-token row blocks with tokens
    const uint16_t *Ab = A + (size_t) (ex.start + m0) * K;

    xmx::joint_matrix<sycl::sub_group, float, xmx::use::accumulator, 8, 16> acc[8];
    for (int r = 0; r < 8; ++r) xmx::joint_matrix_fill(sg_, acc[r], 0.0f);

    sycl::group_barrier(it.get_group());   // (the byte table)
    for (int kb = 0; kb < K / 64; ++kb) {
        uint32_t *Bk = Bs + (kb & 1) * kMxBsWords;   // (double buffered: one barrier a step - when a thread dequantizes step kb + 2 into this buffer, everyone passed the barrier of kb + 1, i.e. finished step kb)
        {
            uint32_t w9[9];
            for (int i = 0; i < 9; ++i) w9[i] = rok ? ik2_ldg((const uint32_t *) (wrow + (size_t) kb * 36) + i) : 0u;
            mx_dequant_block(w9, rok, Bk, tid, lut);
        }
        sycl::group_barrier(it.get_group());
        for (int kk = 0; kk < 4; ++kk) {
            xmx::joint_matrix<sycl::sub_group, sycl::half, xmx::use::b, 16, 16, xmx::layout::ext_intel_packed> b;
            xmx::joint_matrix_load(sg_, b, xmx_local((sycl::half *) (Bk + kk * 8 * kMxBsLd + 16 * s)), 2 * kMxBsLd);
            for (int r = 0; r < 8; ++r) {
                if (r < nrb) {
                    xmx::joint_matrix<sycl::sub_group, sycl::half, xmx::use::a, 8, 16, xmx::layout::row_major> a;
                    xmx::joint_matrix_load(sg_, a,
                                           sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                               (sycl::half *) (Ab + (size_t) (r * 8) * K + kb * 64 + kk * 16)), K);
                    xmx::joint_matrix_mad(sg_, acc[r], a, b, acc[r]);
                }
            }
        }
    }
    sycl::group_barrier(it.get_group());   // (everyone is done with the tiles: the staging overlays them)
    for (int r = 0; r < nrb; ++r) xmx::joint_matrix_store(sg_, acc[r], xmx_local(Cs + r * 8 * kMxCsLd + 16 * s), kMxCsLd, xmx::layout::row_major);
    sycl::group_barrier(it.get_group());
    const int rows = sycl::min(64, ex.cnt - m0);
    if constexpr (GATEUP) {   // h = swiglu(gate * sg, up * su) -> F16 [token][F]
        uint16_t *h = (uint16_t *) out;
        for (int idx = tid; idx < rows * 32; idx += 64) {
            const int m = idx >> 5, c = idx & 31, col = n0 + c;
            if (col >= F) continue;
            const size_t r = (size_t) ex.start + m0 + m;
            const float g = Cs[m * kMxCsLd + c] * sg[r], u = Cs[m * kMxCsLd + 32 + c] * su[r];
            h[r * F + mx_perm_col(col)] = f2h(gcap * sycl::tanh(g / gcap) * (1.0f / (1.0f + sycl::native::exp(-g))) * ucap * sycl::tanh(u / ucap));
        }
    } else {                  // y = h W_down^T -> F32 [token][N]
        float *y = (float *) out;
        for (int idx = tid; idx < rows * 64; idx += 64) {
            const int m = idx >> 6, c = idx & 63, col = n0 + c;
            if (col < N) y[((size_t) ex.start + m0 + m) * N + col] = Cs[m * kMxCsLd + c];
        }
    }
}

template <bool GATEUP>
static void launch_moe_xmx(const MoeXmxExp *tab, int n_exp, int max_cnt, const uint16_t *A, int K, int N, int F, size_t eb_up, size_t off_w, const float *sg,
                           const float *su, float gcap, float ucap, void *out, dpct::queue_ptr s) {
    if (n_exp <= 0 || max_cnt <= 0) return;
    const unsigned ntile = (unsigned) ((N + (GATEUP ? 32 : 64) - 1) / (GATEUP ? 32 : 64));
    s->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> sm(sycl::range(kMxSmem), cgh);
        cgh.parallel_for<class ik2_moe_xmx_kernel_name<GATEUP>>(
            sycl::nd_range<3>(sycl::range(n_exp, (max_cnt + 63) / 64, ntile) * sycl::range(1, 1, 64), sycl::range(1, 1, 64)),
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
                moe_xmx_kernel<GATEUP>(tab, A, K, N, F, eb_up, off_w, sg, su, gcap, ucap, out, sm.get_multi_ptr<sycl::access::decorated::no>().get(), it);
            });
    });
}
// gate/up + SwiGLU for every hit expert: A [S][C] sorted input, h [S][F] F16 out.  The up matrix follows the gate's rows at eb_up bytes.
void moe_gate_up_swiglu_xmx(const MoeXmxExp *tab, int n_exp, int max_cnt, const uint16_t *A, int C, int F, size_t eb_up, const float *sg, const float *su,
                            float gcap, float ucap, uint16_t *h, dpct::queue_ptr s) {
    launch_moe_xmx<true>(tab, n_exp, max_cnt, A, C, F, F, eb_up, 0, sg, su, gcap, ucap, h, s);
    check("moe_gate_up_swiglu_xmx");
}
// down for every hit expert: A [S][F] the hidden, y [S][C] F32 out; the down matrix at off_down bytes in the expert's blob.
void moe_down_xmx(const MoeXmxExp *tab, int n_exp, int max_cnt, const uint16_t *A, int C, int F, size_t off_down, float *y, dpct::queue_ptr s) {
    launch_moe_xmx<false>(tab, n_exp, max_cnt, A, F, C, F, 0, off_down, nullptr, nullptr, 0.f, 0.f, y, s);
    check("moe_down_xmx");
}
