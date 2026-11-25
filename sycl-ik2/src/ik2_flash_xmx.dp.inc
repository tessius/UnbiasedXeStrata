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
constexpr int kXNSG = 4, kXBM = 8 * kXNSG, kXBN = 64;
constexpr int kXKpLd = 68, kXVpLd = 132;   // (words a packed row: a pad against bank conflicts, and 64-byte aligned blocks)
constexpr int kXPerSg = 8 * 128 * 2 + 8 * 64 * 4 + 8 * 64 * 2 + 8 * 4;   // Qs (half) + Ss (float) + Ps (half) + cs (float)
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
    float *Ss = (float *) (mine + 8 * 128 * 2);
    sycl::half *Ps = (sycl::half *) (mine + 8 * 128 * 2 + 8 * 64 * 4);
    float *cs = (float *) (mine + 8 * 128 * 2 + 8 * 64 * 4 + 8 * 64 * 2);

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
    xmx::joint_matrix<sycl::sub_group, float, xmx::use::accumulator, 8, 16> o[8];
    for (int j = 0; j < 8; ++j) xmx::joint_matrix_fill(sg, o[j], 0.0f);
    float m = -INFINITY, l = 0.0f;   // this lane's row: running max (log2 units), and the sum of its half of the columns

    for (int t = t0; t < t1; ++t) {
        const int kt = k_lo + t * kXBN;
        sycl::group_barrier(it.get_group());   // (the last tile's readers are done)
        // K: pairs of dims interleaved per key (VNNI); V: pairs of keys interleaved per dim
        for (int c = tid; c < 64 * 64; c += 64) {
            const int key = c >> 6, dp = c & 63, kp = kt + key;
            const bool ok = kp < k_hi;
            const size_t pr = ok ? (kv_cap > 0 ? kp % kv_cap : kp) : 0;
            Kp[dp * kXKpLd + key] = ok ? *(const uint32_t *) (kc + pr * KD + (size_t) g * 128 + dp * 2) : 0u;
        }
        for (int c = tid; c < 32 * 64; c += 64) {
            const int pk = c >> 6, e = c & 63, ka = kt + 2 * pk, kb = ka + 1;
            const bool oa = ka < k_hi, ob = kb < k_hi;
            const size_t pa = oa ? (kv_cap > 0 ? ka % kv_cap : ka) : 0, pb = ob ? (kv_cap > 0 ? kb % kv_cap : kb) : 0;
            const uint32_t u0 = oa ? *(const uint32_t *) (vc + pa * KD + (size_t) g * 128 + e * 2) : 0u;
            const uint32_t u1 = ob ? *(const uint32_t *) (vc + pb * KD + (size_t) g * 128 + e * 2) : 0u;
            Vp[pk * kXVpLd + 2 * e] = (u0 & 0xFFFFu) | (u1 << 16);
            Vp[pk * kXVpLd + 2 * e + 1] = (u0 >> 16) | (u1 & 0xFFFF0000u);
        }
        sycl::group_barrier(it.get_group());

        // S = Q K^T
        for (int jn = 0; jn < 4; ++jn) {
            xmx::joint_matrix<sycl::sub_group, float, xmx::use::accumulator, 8, 16> s;
            xmx::joint_matrix_fill(sg, s, 0.0f);
            for (int kk = 0; kk < 8; ++kk) {
                xmx::joint_matrix<sycl::sub_group, sycl::half, xmx::use::b, 16, 16, xmx::layout::ext_intel_packed> b;
                xmx::joint_matrix_load(sg, b, xmx_local((sycl::half *) (Kp + kk * 8 * kXKpLd + jn * 16)), 2 * kXKpLd);
                xmx::joint_matrix_mad(sg, s, qa[kk], b, s);
            }
            xmx::joint_matrix_store(sg, s, xmx_local(Ss + jn * 16), 64, xmx::layout::row_major);
        }
        sycl::group_barrier(sg);

        // the softmax: this lane's row r, columns [32 hh, 32 hh + 32)
        float v[32], mx = -INFINITY;
        for (int c = 0; c < 32; ++c) {
            const int col = hh * 32 + c, kp = kt + col, dist = qp - kp;
            float x = Ss[r * 64 + col] * sc;
            if (extent > 0 && dist >= 0 && dist < extent && ir < Tq) x += bias[((size_t) ir * H + h) * extent + dist] * kLog2e;
            x = (dist < 0 || kp < lo || ir >= Tq) ? -INFINITY : x;
            v[c] = x;
            mx = sycl::fmax(mx, x);
        }
        mx = sycl::fmax(mx, sycl::select_from_group(sg, mx, lane ^ 8));
        const float mn = sycl::fmax(m, mx);
        const float cr = mn == -INFINITY ? 1.0f : sycl::exp2(m - mn);   // (m = -inf: exp2(-inf) = 0)
        const float u = mn == -INFINITY ? 0.0f : mn;
        m = mn;
        l *= cr;
        for (int c = 0; c < 32; ++c) {
            const float p = sycl::exp2(v[c] - u);
            l += p;
            Ps[r * 64 + hh * 32 + c] = (sycl::half) p;
        }
        if (hh == 0) cs[r] = cr;
        sycl::group_barrier(sg);

        // O = O * c + P V
        for (int j = 0; j < 8; ++j)
            { int row = 0; xmx::joint_matrix_apply(sg, o[j], [&](float &x) { x *= cs[row++]; }); }   // (an 8x16 accumulator: a lane holds column `lane`, rows 0..7 in order)
        for (int kk = 0; kk < 4; ++kk) {
            xmx::joint_matrix<sycl::sub_group, sycl::half, xmx::use::a, 8, 16, xmx::layout::row_major> pa;
            xmx::joint_matrix_load(sg, pa, xmx_local(Ps + kk * 16), 64);
            for (int j = 0; j < 8; ++j) {
                xmx::joint_matrix<sycl::sub_group, sycl::half, xmx::use::b, 16, 16, xmx::layout::ext_intel_packed> b;
                xmx::joint_matrix_load(sg, b, xmx_local((sycl::half *) (Vp + kk * 8 * kXVpLd + j * 16)), 2 * kXVpLd);
                xmx::joint_matrix_mad(sg, o[j], pa, b, o[j]);
            }
        }
        sycl::group_barrier(sg);   // (Ss / Ps / cs are rewritten next tile)
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
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
                attn_flash_xmx_kernel(q16, q_ld, kc, vc, kv_cap, H, KV, Tq, qpos0, qk_scale, bias, extent, window, n_split, o_part, ml_part,
                                      sm.get_multi_ptr<sycl::access::decorated::no>().get(), it);
            });
    });
    check("attn_flash_xmx");
}
void set_flash_xmx(bool on) { g_flash_xmx = on; }
