// SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
// SPDX-License-Identifier: MIT
// NVIDIA's mma.sync.m16n8k16 (f16 in, f32 accumulate) and ldmatrix.x4 emulated with sub-group shuffles.  No include guard and no
// includes: it is #included inside the kernels' namespace (and by tests/mma_emul_test.cpp).  The register layouts are PTX ISA's:
// lane l, g = l >> 2, t = l & 3.
//   A (16x16 row-major):  a0 = (row g,   k 2t,2t+1)   a1 = (row g+8, k 2t,2t+1)   a2 = (row g,   k 2t+8,2t+9)   a3 = (row g+8, k 2t+8,2t+9)
//   B (16x8, "col"):      b0 = (k 2t,2t+1, col g)     b1 = (k 2t+8,2t+9, col g)
//   C/D (16x8):           c0,c1 = (row g,   cols 2t,2t+1)      c2,c3 = (row g+8, cols 2t,2t+1)
// ldmatrix.x4: lane l gives the address of row (l & 7) of matrix (l >> 3); register m comes back as matrix m's row g, columns 2t,2t+1
// (or, with .trans, the transposed: the original's rows 2t,2t+1 at column g).
__dpct_inline__ float ik2_half_lo(uint32_t w) { return (float) sycl::bit_cast<sycl::half>((uint16_t) (w & 0xFFFFu)); }
__dpct_inline__ float ik2_half_hi(uint32_t w) { return (float) sycl::bit_cast<sycl::half>((uint16_t) (w >> 16)); }
// SYCL: emulation of mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 (no 16x8x16 shape on Intel's matrix engine): each lane
// gathers the A row and B columns of its four outputs from the lanes that hold them (32 shuffles) and sums in F32.
__dpct_inline__ void mma16816(float *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    const int l = (int) sg.get_local_linear_id(), g = l >> 2, t = l & 3;
    float acc[4] = {c[0], c[1], c[2], c[3]};
    for (int p = 0; p < 4; ++p) {            // the k pair p (k = 2p, 2p+1) and p+4 (k = 8 + 2p, 8 + 2p + 1)
        const int src_a = g * 4 + p;         // the lane holding row g (and g+8), k pair p
        uint32_t ar[4];
        for (int r = 0; r < 4; ++r) ar[r] = sycl::select_from_group(sg, a[r], src_a);
        // B's columns 2t and 2t+1 (held by lanes (2t) * 4 + p and (2t + 1) * 4 + p)
        uint32_t bc[2][2];
        for (int j = 0; j < 2; ++j) {
            bc[j][0] = sycl::select_from_group(sg, b0, (2 * t + j) * 4 + p);
            bc[j][1] = sycl::select_from_group(sg, b1, (2 * t + j) * 4 + p);
        }
        for (int j = 0; j < 2; ++j) {
            // rows g: a0 (k 2p,2p+1), a2 (k 2p+8,2p+9);  rows g+8: a1, a3
            acc[0 + j] += ik2_half_lo(ar[0]) * ik2_half_lo(bc[j][0]) + ik2_half_hi(ar[0]) * ik2_half_hi(bc[j][0])
                        + ik2_half_lo(ar[2]) * ik2_half_lo(bc[j][1]) + ik2_half_hi(ar[2]) * ik2_half_hi(bc[j][1]);
            acc[2 + j] += ik2_half_lo(ar[1]) * ik2_half_lo(bc[j][0]) + ik2_half_hi(ar[1]) * ik2_half_hi(bc[j][0])
                        + ik2_half_lo(ar[3]) * ik2_half_lo(bc[j][1]) + ik2_half_hi(ar[3]) * ik2_half_hi(bc[j][1]);
        }
    }
    c[0] = acc[0]; c[1] = acc[1]; c[2] = acc[2]; c[3] = acc[3];
}
// SYCL: emulation of ldmatrix.sync.aligned.m8n8.x4[.trans].shared.b16
__dpct_inline__ void ik2_ldsm_x4(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, const void *p, bool trans) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    const int l = (int) sg.get_local_linear_id(), g = l >> 2, t = l & 3;
    const uint64_t me = (uint64_t) (uintptr_t) p;
    uint32_t out[4];
    for (int m = 0; m < 4; ++m) {
        if (!trans) {
            const uint64_t row = sycl::select_from_group(sg, me, m * 8 + g);                       // row g of matrix m
            out[m] = *reinterpret_cast<const uint32_t *>((uintptr_t) row + 4 * t);                 // columns 2t, 2t+1
        } else {
            const uint64_t row0 = sycl::select_from_group(sg, me, m * 8 + 2 * t);                  // original rows 2t and 2t+1
            const uint64_t row1 = sycl::select_from_group(sg, me, m * 8 + 2 * t + 1);
            const uint16_t lo = *reinterpret_cast<const uint16_t *>((uintptr_t) row0 + 2 * g);     // ... at column g
            const uint16_t hi = *reinterpret_cast<const uint16_t *>((uintptr_t) row1 + 2 * g);
            out[m] = (uint32_t) lo | ((uint32_t) hi << 16);
        }
    }
    r0 = out[0]; r1 = out[1]; r2 = out[2]; r3 = out[3];
}
__dpct_inline__ void ldsm_x4_t(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, const void *p) { ik2_ldsm_x4(r0, r1, r2, r3, p, true); }
__dpct_inline__ void ldsm_x4(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, const void *p) { ik2_ldsm_x4(r0, r1, r2, r3, p, false); }
