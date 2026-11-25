// SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
// SPDX-License-Identifier: MIT
// The NVFP4 limb kernel with RP row pairs a warp (Intel).  No include guard / includes: #included in the kernels' namespace after nv_limb_mv.
//
// nv_limb_mv gives a warp one row pair (gate row + up row, or two down rows), so each lane re-loads the activation record of a sub-block (9 words) for just two dot products.  With RP pairs the record
// is loaded once for 2 * RP dots (and 6 * 2 * RP weight words are in flight).  The sums per row are the same as nv_limb_mv's (a lane's sub-blocks in the same order), so the results are bit-identical.
template <bool, int> class nv_limb_rp_kernel_name;
template <bool Gate, int RP>
__dpct_inline__ void nv_limb_mv_rp(const uint8_t *const *blobs, const NvInput *packed, size_t offset, int C, int F, const float *sg, const float *su, float gcap, float ucap,
                                   uint16_t *hidden, float *y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &lut = *sycl::ext::oneapi::group_local_memory_for_overwrite<uint16_t[256]>(sycl::ext::oneapi::this_work_item::get_work_group<3>());
    for (int i = (int) item_ct1.get_local_id(2); i < 256; i += (int) item_ct1.get_local_range(2)) lut[i] = nv_limb_lut_entry(i);
    item_ct1.barrier();
    const int lane = (int) item_ct1.get_local_id(2) % 32, warp = (int) item_ct1.get_local_id(2) / 32, slot = (int) item_ct1.get_group(1);
    const int K = Gate ? C : F, N = Gate ? F : C;
    constexpr int NP = Gate ? RP : 2 * RP;   // rows a warp (Gate: pairs of (gate, up) rows; down: consecutive rows) / streams = 2 RP either way
    const int r0 = ((int) item_ct1.get_group(2) * ((int) item_ct1.get_local_range(2) / 32) + warp) * NP;
    if (r0 >= N) return;
    const int stride = K / 64 * 36;
    const uint8_t *base = blobs[slot] + (Gate ? 0 : offset);
    const NvInput *xp = packed + (Gate ? 0 : slot * K / 16);
    const uint8_t *row[2 * RP];
#pragma unroll
    for (int t = 0; t < RP; ++t) {
        if (Gate) {
            const int rr = sycl::min(r0 + t, N - 1);
            row[2 * t] = base + (size_t) rr * stride;
            row[2 * t + 1] = row[2 * t] + offset;
        } else {
            row[2 * t] = base + (size_t) sycl::min(r0 + 2 * t, N - 1) * stride;
            row[2 * t + 1] = base + (size_t) sycl::min(r0 + 2 * t + 1, N - 1) * stride;
        }
    }
    float acc[2 * RP];
#pragma unroll
    for (int i = 0; i < 2 * RP; ++i) acc[i] = 0.0f;
    for (int s = lane; s < K / 16; s += 32) {
        const NvInput z = xp[s];
#pragma unroll
        for (int i = 0; i < 2 * RP; ++i) acc[i] += nv_limb_dot(row[i], s, z, lut);
    }
#pragma unroll
    for (int i = 0; i < 2 * RP; ++i) acc[i] = wsum(acc[i]);
    if (lane) return;
    if (Gate) {
#pragma unroll
        for (int t = 0; t < RP; ++t) {
            const int r = r0 + t;
            if (r >= N) break;
            const float g = acc[2 * t] * (sg ? sg[slot] : 1.f), u = acc[2 * t + 1] * (su ? su[slot] : 1.f);
            hidden[(size_t) slot * F + r] = f2h(gcap * sycl::tanh(g / gcap) * (1.f / (1.f + sycl::native::exp(-g))) * ucap * sycl::tanh(u / ucap));
        }
    } else {
#pragma unroll
        for (int i = 0; i < 2 * RP; ++i)
            if (r0 + i < N) y[(size_t) slot * C + r0 + i] = acc[i];
    }
}
template <bool Gate, int RP>
void launch_nv_limb_rp(const uint8_t *const *blob, const NvInput *p, size_t offset, int C, int F, int S, const float *sg, const float *su, float gcap, float ucap, uint16_t *h, float *y,
                       dpct::queue_ptr stream) {
    constexpr int W = 4, NP = Gate ? RP : 2 * RP;   // 4 warps a work-group, as nv_limb_mv
    const int N = Gate ? F : C;
    const unsigned gx = (unsigned) ((N + NP * W - 1) / (NP * W));
    stream->parallel_for<class nv_limb_rp_kernel_name<Gate, RP>>(
        sycl::nd_range<3>(sycl::range(1, S, gx) * sycl::range(1, 1, W * 32), sycl::range(1, 1, W * 32)),
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] { nv_limb_mv_rp<Gate, RP>(blob, p, offset, C, F, sg, su, gcap, ucap, h, y); });
}
