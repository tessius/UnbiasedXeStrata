// SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
// SPDX-License-Identifier: MIT
// Prefill attention (FP16 KV cache, head dim 128) on the Intel matrix engine (XMX, SYCL joint_matrix), a drop-in for attn_flash's FP16 path:
// the same arguments, the same o_part / ml_part (unnormalised output and (max in log2 units, sum) of a split), so attn_flash_combine is unchanged.
// No include guard and no includes: #included in the kernels' namespace after attn_flash_launch.
//
// A work-group = kXNSG sub-groups (16 lanes) = 8 * kXNSG queries of one head against one split of its keys, in tiles of 64 keys.  The tile of K and V
// is staged once per work-group in shared local memory in the VNNI form the matrix engine wants (K: pairs of dims interleaved, V: pairs of keys).
// Each sub-group, for its 8 queries:  S[8x64] = Q K^T (4 x 8 mads of 8x16x16, F32 accumulate) -> stored in SLM -> online softmax by the lanes (2 lanes a
// row, log2 domain, mask / window / relative bias exactly as the CUDA kernel) -> P (FP16) in SLM -> O[8x128] += P V (4 x 8 mads), O rescaled in place.
namespace xmx = sycl::ext::oneapi::experimental::matrix;
constexpr int kXNSG = 27, kXBM = 8 * kXNSG, kXBN = 64, kXNT = 16 * kXNSG;   // (sub-groups a work-group, queries, keys a tile, threads)
constexpr int kXKpLd = 68, kXVpLd = 132;   // (words a packed row: a pad against bank conflicts, and 64-byte aligned blocks)
constexpr int kXSsLd = 68, kXPsLd = 72;   // (floats / halves a row of the logits / probabilities: a pad against bank conflicts in the softmax, 16-byte aligned rows)
constexpr int kXWork = 8 * kXSsLd * 4 + 8 * kXPsLd * 2 + 8 * 4;   // Ss (float) + Ps (half) + cs (float)
constexpr int kXPerSg = kXWork > 8 * 128 * 2 ? kXWork : 8 * 128 * 2;   // (Qs, 2 KB, is dead once its fragments are loaded: the logits / probabilities use its room)
constexpr int kXSmem = 64 * kXKpLd * 4 + 32 * kXVpLd * 4 + kXNSG * kXPerSg;
bool g_flash_xmx = !(std::getenv("IK2_FLASH_XMX") && !std::atoi(std::getenv("IK2_FLASH_XMX")));   // IK2_FLASH_XMX=0 (or set_flash_xmx(false)): the CUDA kernel's emulated path

template <class T>
__dpct_inline__ auto xmx_local(T *p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
}

__dpct_inline__ void attn_flash_xmx_kernel(
    const uint16_t *__restrict__ q16, int q_ld, const uint16_t *__restrict__ kc, const uint16_t *__restrict__ vc, int kv_cap, int H, int KV, int Tq,
    int qpos0, float qk_scale, const float *__restrict__ bias, int extent, int window, int n_split, float *__restrict__ o_part,
    sycl::float2 *__restrict__ ml_part, uint8_t *smem, sycl::nd_item<3> it) {
    auto sg = it.get_sub_group();
    const int tid = (int) it.get_local_id(2), sgi = tid >> 4, lane = tid & 15;
    const int h = (int) it.get_group(1), split = (int) it.get_group(0), g = h / (H / KV), KD = KV * 128;
    uint32_t *Kp = (uint32_t *) smem;
    uint32_t *Vp = Kp + 64 * kXKpLd;
    uint8_t *mine = (uint8_t *) (Vp + 32 * kXVpLd) + sgi * kXPerSg;
    sycl::half *Qs = (sycl::half *) mine;
    float *Ss = (float *) mine;
    sycl::half *Ps = (sycl::half *) (mine + 8 * kXSsLd * 4);
    float *cs = (float *) (mine + 8 * kXSsLd * 4 + 8 * kXPsLd * 2);

    const int i0 = (int) it.get_group(2) * kXBM;
    const int i_last = sycl::min(Tq, i0 + kXBM) - 1;
    const int k_hi = qpos0 + i_last + 1;
    const int k_lo = sycl::max(0, qpos0 + i0 - window + 1);
    const int n_tiles = (k_hi - k_lo + kXBN - 1) / kXBN;
    const int per = (n_tiles + n_split - 1) / n_split;
    const int t0 = sycl::min(n_tiles, split * per), t1 = sycl::min(n_tiles, t0 + per);

    const int wi = i0 + sgi * 8;                 // this sub-group's first query
    const int r = lane & 7, hh = lane >> 3;      // the row this lane works on in the softmax, and its half of the 64 columns
    const int ir = wi + r, qp = qpos0 + ir, lo = qp - window + 1;
    const float sc = qk_scale * kLog2e;

    // Q (zero past Tq) into SLM, then the eight A fragments (kept for every tile)
    for (int e = 0; e < 32; ++e) {
        const int idx = lane + 16 * e, rr = idx >> 6, c2 = (idx & 63) * 2, i = wi + rr;
        *(uint32_t *) (Qs + rr * 128 + c2) = i < Tq ? *(const uint32_t *) (q16 + (size_t) i * q_ld + (size_t) h * 128 + c2) : 0u;
    }
    sycl::group_barrier(sg);
    xmx::joint_matrix<sycl::sub_group, sycl::half, xmx::use::a, 8, 16, xmx::layout::row_major> qa[8];
    for (int kk = 0; kk < 8; ++kk) xmx::joint_matrix_load(sg, qa[kk], xmx_local(Qs + kk * 16), 128);
    sycl::group_barrier(sg);   // (the fragments are in registers: Qs's room is the logits' from here)
    xmx::joint_matrix<sycl::sub_group, float, xmx::use::accumulator, 8, 16> o[8];
    for (int j = 0; j < 8; ++j) xmx::joint_matrix_fill(sg, o[j], 0.0f);
    float m = -INFINITY, l = 0.0f;   // this lane's row: running max (log2 units), and the sum of its half of the columns

    for (int t = t0; t < t1; ++t) {
        const int kt = k_lo + t * kXBN;
        sycl::group_barrier(it.get_group());   // (the last tile's readers are done)
        // K: pairs of dims interleaved per key (VNNI); V: pairs of keys interleaved per dim.  16-byte global loads: a thread takes 8 dims of a key (K: lanes
        // are consecutive keys, so the four SLM words it writes are one bank each across the lanes) or of two keys (V: lanes are consecutive 8-dim groups).
        for (int c = tid; c < 64 * 16; c += kXNT) {
            const int key = c & 63, q4 = c >> 6, kp = kt + key;
            const bool ok = kp < k_hi;
            const size_t pr = ok ? (kv_cap > 0 ? kp % kv_cap : kp) : 0;
            const sycl::uint4 v = ok ? *(const sycl::uint4 *) (kc + pr * KD + (size_t) g * 128 + q4 * 8) : sycl::uint4(0, 0, 0, 0);
            Kp[(q4 * 4 + 0) * kXKpLd + key] = v.x();
            Kp[(q4 * 4 + 1) * kXKpLd + key] = v.y();
            Kp[(q4 * 4 + 2) * kXKpLd + key] = v.z();
            Kp[(q4 * 4 + 3) * kXKpLd + key] = v.w();
        }
        for (int c = tid; c < 32 * 16; c += kXNT) {
            const int e4 = c & 15, pk = c >> 4, ka = kt + 2 * pk, kb = ka + 1;
            const bool oa = ka < k_hi, ob = kb < k_hi;
            const size_t pa = oa ? (kv_cap > 0 ? ka % kv_cap : ka) : 0, pb = ob ? (kv_cap > 0 ? kb % kv_cap : kb) : 0;
            const sycl::uint4 ua = oa ? *(const sycl::uint4 *) (vc + pa * KD + (size_t) g * 128 + e4 * 8) : sycl::uint4(0, 0, 0, 0);
            const sycl::uint4 ub = ob ? *(const sycl::uint4 *) (vc + pb * KD + (size_t) g * 128 + e4 * 8) : sycl::uint4(0, 0, 0, 0);
            const uint32_t wa[4] = {ua.x(), ua.y(), ua.z(), ua.w()}, wb[4] = {ub.x(), ub.y(), ub.z(), ub.w()};
            uint32_t *dst = Vp + pk * kXVpLd + e4 * 8;
            for (int j = 0; j < 4; ++j) {
                dst[2 * j] = (wa[j] & 0xFFFFu) | (wb[j] << 16);
                dst[2 * j + 1] = (wa[j] >> 16) | (wb[j] & 0xFFFF0000u);
            }
        }
        sycl::group_barrier(it.get_group());

        // a tile none of this sub-group's 8 queries can see (past its last query, or before its first one's window) costs nothing but the barriers: the work-group's key
        // range is set by all its queries, so in a sliding-window layer a sub-group would otherwise run ~20% more tiles than it needs (fully masked: the same result)
        const int q_first = qpos0 + wi, q_last = qpos0 + sycl::min(wi + 7, Tq - 1);
        if (wi < Tq && kt <= q_last && kt + kXBN - 1 >= q_first - window + 1) {
        // S = Q K^T
        for (int jn = 0; jn < 4; ++jn) {
            xmx::joint_matrix<sycl::sub_group, float, xmx::use::accumulator, 8, 16> s;
            xmx::joint_matrix_fill(sg, s, 0.0f);
            for (int kk = 0; kk < 8; ++kk) {
                xmx::joint_matrix<sycl::sub_group, sycl::half, xmx::use::b, 16, 16, xmx::layout::ext_intel_packed> b;
                xmx::joint_matrix_load(sg, b, xmx_local((sycl::half *) (Kp + kk * 8 * kXKpLd + jn * 16)), 2 * kXKpLd);
                xmx::joint_matrix_mad(sg, s, qa[kk], b, s);
            }
            xmx::joint_matrix_store(sg, s, xmx_local(Ss + jn * 16), kXSsLd, xmx::layout::row_major);
        }
        sycl::group_barrier(sg);

        // the softmax: this lane's row r, columns [32 hh, 32 hh + 32)
        float v[32], mx = -INFINITY;
        {
            const int d0 = qp - (kt + hh * 32);   // the distance of this lane's first column; column c is d0 - c away
            // the run of 32 columns is wholly visible (causal, inside the window) and, with a bias, wholly inside its extent: no masks, the bias as vector loads
            const bool fast = ir < Tq && d0 - 31 >= 0 && d0 <= window - 1 && (extent == 0 || d0 < extent);
            if (fast) {
                if (extent > 0) {
                    const float *brow = bias + ((size_t) ir * H + h) * extent + (d0 - 31);   // brow[i] is column 31 - i's bias
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        sycl::vec<float, 4> bv;
                        bv.load(j, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(brow));
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int c = 31 - (4 * j + i);
                            v[c] = Ss[r * kXSsLd + hh * 32 + c] * sc + bv[i] * kLog2e;
                        }
                    }
                } else {
#pragma unroll
                    for (int c = 0; c < 32; ++c) v[c] = Ss[r * kXSsLd + hh * 32 + c] * sc;
                }
#pragma unroll
                for (int c = 0; c < 32; ++c) mx = sycl::fmax(mx, v[c]);
            } else {
#pragma unroll
                for (int c = 0; c < 32; ++c) {
                    const int col = hh * 32 + c, kp = kt + col, dist = qp - kp;
                    float x = Ss[r * kXSsLd + col] * sc;
                    if (extent > 0 && dist >= 0 && dist < extent && ir < Tq) x += bias[((size_t) ir * H + h) * extent + dist] * kLog2e;
                    x = (dist < 0 || kp < lo || ir >= Tq) ? -INFINITY : x;
                    v[c] = x;
                    mx = sycl::fmax(mx, x);
                }
            }
        }
        mx = sycl::fmax(mx, sycl::select_from_group(sg, mx, lane ^ 8));
        const float mn = sycl::fmax(m, mx);
        const float cr = mn == -INFINITY ? 1.0f : sycl::exp2(m - mn);   // (m = -inf: exp2(-inf) = 0)
        const float u = mn == -INFINITY ? 0.0f : mn;
        m = mn;
        l *= cr;
#pragma unroll
        for (int c = 0; c < 32; ++c) {
            const float p = sycl::exp2(v[c] - u);
            l += p;
            Ps[r * kXPsLd + hh * 32 + c] = (sycl::half) p;
        }
        if (hh == 0) cs[r] = cr;
        sycl::group_barrier(sg);

        // O = O * c + P V
        for (int j = 0; j < 8; ++j)
            { int row = 0; xmx::joint_matrix_apply(sg, o[j], [&](float &x) { x *= cs[row++]; }); }   // (an 8x16 accumulator: a lane holds column `lane`, rows 0..7 in order)
        for (int kk = 0; kk < 4; ++kk) {
            xmx::joint_matrix<sycl::sub_group, sycl::half, xmx::use::a, 8, 16, xmx::layout::row_major> pa;
            xmx::joint_matrix_load(sg, pa, xmx_local(Ps + kk * 16), kXPsLd);
            for (int j = 0; j < 8; ++j) {
                xmx::joint_matrix<sycl::sub_group, sycl::half, xmx::use::b, 16, 16, xmx::layout::ext_intel_packed> b;
                xmx::joint_matrix_load(sg, b, xmx_local((sycl::half *) (Vp + kk * 8 * kXVpLd + j * 16)), 2 * kXVpLd);
                xmx::joint_matrix_mad(sg, o[j], pa, b, o[j]);
            }
        }
        sycl::group_barrier(sg);   // (Ss / Ps / cs are rewritten next tile)
        }
    }

    // the split's partials: o_part [split][Tq][H][D], ml_part [split][Tq][H]
    l += sycl::select_from_group(sg, l, lane ^ 8);
    for (int j = 0; j < 8; ++j) {
        xmx::joint_matrix_store(sg, o[j], xmx_local(Ss), 16, xmx::layout::row_major);   // (8 x 16 floats)
        sycl::group_barrier(sg);
        for (int e = 0; e < 8; ++e) {
            const int idx = lane + 16 * e, rr = idx >> 4, cc = idx & 15, i = wi + rr;
            if (i < Tq) o_part[(((size_t) split * Tq + i) * H + h) * 128 + j * 16 + cc] = Ss[rr * 16 + cc];
        }
        sycl::group_barrier(sg);
    }
    if (hh == 0 && ir < Tq) ml_part[((size_t) split * Tq + ir) * H + h] = sycl::float2(m, l);
}

void attn_flash_xmx_launch(const uint16_t *q16, int q_ld, const uint16_t *kc, const uint16_t *vc, int kv_cap, int H, int KV, int Tq, int qpos0,
                           float qk_scale, const float *bias, int extent, int window, int n_split, float *o_part, sycl::float2 *ml_part,
                           dpct::queue_ptr s) {
    const unsigned nqb = (unsigned) ((Tq + kXBM - 1) / kXBM);
    s->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> sm(sycl::range(kXSmem), cgh);
        cgh.parallel_for<class ik2_flash_xmx_kernel_name>(
            sycl::nd_range<3>(sycl::range(n_split, H, nqb) * sycl::range(1, 1, 16 * kXNSG), sycl::range(1, 1, 16 * kXNSG)),
            sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},   // (the Q fragments, 8 accumulators and the logits spill at 128 registers: 2x slower; SLM limits a core to < 4 threads an EU anyway)
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
                attn_flash_xmx_kernel(q16, q_ld, kc, vc, kv_cap, H, KV, Tq, qpos0, qk_scale, bias, extent, window, n_split, o_part, ml_part,
                                      sm.get_multi_ptr<sycl::access::decorated::no>().get(), it);
            });
    });
    check("attn_flash_xmx");
}
void set_flash_xmx(bool on) { g_flash_xmx = on; }
