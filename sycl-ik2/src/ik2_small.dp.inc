// SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
// SPDX-License-Identifier: MIT
// Decode's small kernels, rewritten for latency (Intel).  No include guard / includes: #included in the kernels' namespace before mv_f16.
//
// mv_f16: y [N] = W [N][K] FP16 . x [K], a warp a row.  The generic kernel makes K / 32 strided scalar loads a lane (72 for the router, K = 2304): a long serial chain.  Here a lane takes
// 8-value chunks (one 16-byte weight load, two float4 activation loads) and does K / 256 of them (9): 8x fewer load instructions.  The additions are in another order (a chunk at a time),
// so sums differ from the generic kernel's in the last bits.
template <int WARPS>
__dpct_inline__ void mv_f16_wide_kernel(const uint16_t *__restrict__ W, int N, int K, const float *__restrict__ x, float *__restrict__ y) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = (int) it.get_group(2) * WARPS + ((int) it.get_local_id(2) >> 5), lane = (int) it.get_local_id(2) & 31;
    if (row >= N) return;
    const sycl::uint4 *w4 = (const sycl::uint4 *) (W + (size_t) row * K);
    const sycl::float4 *x4 = (const sycl::float4 *) x;
    float a = 0.0f;
    for (int c = lane; c < K / 8; c += 32) {
        const sycl::uint4 wv = ik2_ldg(w4 + c);
        const sycl::float4 xa = ik2_ldg(x4 + 2 * c), xb = ik2_ldg(x4 + 2 * c + 1);
        const sycl::float2 p0 = u32_to_f2(wv.x()), p1 = u32_to_f2(wv.y()), p2 = u32_to_f2(wv.z()), p3 = u32_to_f2(wv.w());
        a += (p0.x() * xa.x() + p0.y() * xa.y()) + (p1.x() * xa.z() + p1.y() * xa.w()) + (p2.x() * xb.x() + p2.y() * xb.y()) + (p3.x() * xb.z() + p3.y() * xb.w());
    }
    a = wsum(a);
    if (lane == 0) y[row] = a;
}
bool g_small_wide = !(std::getenv("IK2_SMALL_WIDE") && !std::atoi(std::getenv("IK2_SMALL_WIDE")));   // IK2_SMALL_WIDE=0: the generic kernels
void set_small_wide(bool on) { g_small_wide = on; }
bool mv_f16_wide_ok(const uint16_t *W, int K, const float *x) { return g_small_wide && K % 256 == 0 && (((uintptr_t) W & 15) == 0) && (((uintptr_t) x & 15) == 0); }
void mv_f16_wide(const uint16_t *W, int N, int K, const float *x, float *y, dpct::queue_ptr s) {
    s->parallel_for<class mv_f16_wide_kernel_name>(
        sycl::nd_range<3>(sycl::range(1, 1, (unsigned) ((N + 3) / 4)) * sycl::range(1, 1, 128), sycl::range(1, 1, 128)),
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] { mv_f16_wide_kernel<4>(W, N, K, x, y); });
    check("mv_f16");
}
