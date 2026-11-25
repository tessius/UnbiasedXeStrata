// SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
// SPDX-License-Identifier: MIT
// NVFP4 -> FP16 weight decode, store-coalesced (Intel): prefill decodes every hit expert's weights each layer visit, so this kernel's speed is prefill's.
// No include guard / includes: #included in the kernels' namespace before decode_f16.
//
// decode_f16_nvfp4_kernel gives a thread a whole 16-weight sub-block: two 16-byte stores 32 bytes apart per lane, so each store instruction writes half of
// every cache line it touches (a plain memset of the same output is 2x faster).  Here a thread takes HALF a sub-block (8 weights, ONE 16-byte store): 32
// lanes write 512 contiguous bytes an instruction.  The two threads of a sub-block read the same 8 bytes (an L1 hit) and decode the low or the high nibbles.
// The conversion is decode_f16_nvfp4_kernel's, so the output is identical.
__dpct_inline__ void decode_f16_nvfp4_half_kernel(const uint8_t *blocks, size_t nhalf, uint16_t *out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t t = (size_t) item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    if (t >= nhalf) return;
    const size_t sb = t >> 1;
    const bool hi = (t & 1) != 0;                 // 0: weights 0..7 of the sub-block (low nibbles), 1: 8..15 (high nibbles)
    const uint8_t *b = blocks + (sb >> 2) * 36;
    const int sub = (int) (sb & 3);
    const float d = ue4m3((uint8_t) (*(const uint32_t *) b >> (8 * sub)));
    const uint32_t q0 = *(const uint32_t *) (b + 4 + sub * 8), q1 = *(const uint32_t *) (b + 8 + sub * 8);
    const sycl::float2 w0 = hi ? e2m1_pair_hi(q0) : e2m1_pair_lo(q0), w1 = hi ? e2m1_pair_hi(q0 >> 16) : e2m1_pair_lo(q0 >> 16);
    const sycl::float2 w2 = hi ? e2m1_pair_hi(q1) : e2m1_pair_lo(q1), w3 = hi ? e2m1_pair_hi(q1 >> 16) : e2m1_pair_lo(q1 >> 16);
    *(sycl::uint4 *) (out + t * 8) = sycl::uint4(f2_to_u32(w0.x() * d, w0.y() * d), f2_to_u32(w1.x() * d, w1.y() * d),
                                                 f2_to_u32(w2.x() * d, w2.y() * d), f2_to_u32(w3.x() * d, w3.y() * d));
}
void decode_f16_nvfp4_half(const uint8_t *blocks, size_t n, uint16_t *out, dpct::queue_ptr s) {
    const size_t nhalf = n / 8;
    s->submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class decode_f16_nvfp4_half_kernel_name>(
            sycl::nd_range<3>(sycl::range(1, 1, (nhalf + 255) / 256) * sycl::range(1, 1, 256), sycl::range(1, 1, 256)),
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(32)]] { decode_f16_nvfp4_half_kernel(blocks, nhalf, out); });
    });
    check("decode_f16_nvfp4");
}
bool g_decode_lut = std::getenv("IK2_DECODE_LUT") && std::atoi(std::getenv("IK2_DECODE_LUT"));   // opt-in (IK2_DECODE_LUT=1): ~25% faster in a loop, 21-36 us run to run, no gain end to end
void set_decode_lut(bool on) { g_decode_lut = on; }
