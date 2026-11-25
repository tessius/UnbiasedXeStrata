// src/ik2/ik2_kernels.cu - maya-ik2's kernels (the interface and the conventions: ik2_kernels.cuh).
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "ik2_kernels.dp.hpp"
#include "ik2_iq_tables.h"

#include <algorithm>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <chrono>
#include <map>
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ik2::k {
namespace {

struct SyncStats {
    std::map<std::string, std::pair<double, long>> t;
    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
    ~SyncStats() {
        std::vector<std::pair<double, std::string>> v;
        double all = 0; for (auto &kv : t) { v.push_back({kv.second.first, kv.first}); all += kv.second.first; }
        std::sort(v.rbegin(), v.rend());
        if (v.empty()) return;
        std::fprintf(stderr, "[ik2 time per launch name, %.3f s in all]\n", all);
        for (size_t i = 0; i < v.size() && i < 24; ++i) std::fprintf(stderr, "  %7.1f ms %5.1f%% %6ld x  %s\n", v[i].first * 1e3, 100 * v[i].first / all, t[v[i].second].second, v[i].second.c_str());
    }
};
void check(const char* what) {
    static const char *mode = std::getenv("IK2_SYNC_CHECK");
    static SyncStats stats;
    if (mode) {
        try {
            dpct::get_current_device().queues_wait_and_throw();
            if (mode[0] == '2') { const auto now = std::chrono::steady_clock::now(); auto &e = stats.t[what]; e.first += std::chrono::duration<double>(now - stats.last).count(); ++e.second; stats.last = now; }
            else std::fprintf(stderr, "[ik2 done] %s\n", what);
        } catch (const sycl::exception &ex) { std::fprintf(stderr, "[ik2 FAILED] %s: %s\n", what, ex.what()); std::exit(2); }
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (e != 0) {
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        std::fprintf(stderr, "ik2::k::%s: launch failed: %s\n", what,
                     dpct::get_error_string_dummy(e));
        /*
        DPCT1001: The statement could not be removed.
        */
        std::abort();
    }
}

__dpct_inline__ float h2f(uint16_t h) {
    return sycl::vec<sycl::half, 1>(
               sycl::bit_cast<sycl::half, unsigned short>(h))
        .convert<float, sycl::rounding_mode::automatic>()[0];
}
__dpct_inline__ uint16_t f2h(float f) {
    return sycl::bit_cast<unsigned short, sycl::half>(
               sycl::vec<float, 1>(f)
                   .convert<sycl::half, sycl::rounding_mode::rte>()[0]);
}
__dpct_inline__ sycl::float2 u32_to_f2(uint32_t u) {
    const sycl::half2 h2 = sycl::half2(
        sycl::bit_cast<sycl::half, unsigned short>((uint16_t)(u & 0xFFFFu)),
        sycl::bit_cast<sycl::half, unsigned short>((uint16_t)(u >> 16)));
    return h2.template convert<float, sycl::rounding_mode::automatic>();
}
__dpct_inline__ float bits_to_float(uint32_t u) {

    return sycl::bit_cast<float>(u);
}
__dpct_inline__ float bits_to_float_host_ct6(uint32_t u) {

    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
__dpct_inline__ uint32_t float_to_bits(float f) {

    return sycl::bit_cast<unsigned int>(f);
}
__dpct_inline__ uint32_t float_to_bits_host_ct7(float f) {

    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
inline float half_bits_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
    uint32_t u;
    if (e == 0) {
        if (m == 0) {
            u = sign;
        } else {   // subnormal
            int sh = 0;
            uint32_t mm = m;
            while (!(mm & 1024)) mm <<= 1, ++sh;
            u = sign | ((uint32_t) (113 - sh) << 23) | ((mm & 1023) << 13);
        }
    } else if (e == 31) {
        u = sign | 0x7f800000u | (m << 13);
    } else {
        u = sign | ((e + 112) << 23) | (m << 13);
    }
    return bits_to_float(u);
}
__dpct_inline__ uint8_t f32_to_e4m3fn(float v) {
#if 0   // SYCL: no sm_89 FP8 hardware conversions; the software path is the one
    /*
    DPCT1007: Migration of __nv_cvt_float_to_fp8 is not supported.
    */
    return (uint8_t)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
#else
    const uint32_t u = float_to_bits_host_ct7(v);
    const uint8_t sign = (uint8_t) ((u >> 24) & 0x80u);
    const uint32_t ax = u & 0x7FFFFFFFu;
    if (ax > 0x7F800000u) return sign | 0x7Fu;
    if (ax >= 0x43E00000u) return sign | 0x7Eu;
    if (ax < 0x3C800000u) {
#if defined(__CUDA_ARCH__)
        const uint32_t sub = float_to_bits(__fadd_rn(bits_to_float(ax), 16384.0f)) - 0x46800000u;
#else
        const uint32_t sub =
            float_to_bits_host_ct7(bits_to_float_host_ct6(ax) + 16384.0f) -
            0x46800000u;
#endif
        return sign | (uint8_t) sub;
    }
    const uint32_t lsb = (ax >> 20) & 1u;
    return sign | (uint8_t) (((ax + 0x7FFFFu + lsb) >> 20) - (120u << 3));
#endif
}
inline __dpct_inline__ uint8_t f32_to_e4m3fn_host_ct4(float v) {
#if 0   // SYCL: the host copy of the function: its `__CUDA_ARCH__ >= 890` is false, so the software else-branch (dpct mangled the test)
    return (uint8_t)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
#else
    const uint32_t u = float_to_bits_host_ct7(v);
    const uint8_t sign = (uint8_t)((u >> 24) & 0x80u);
    const uint32_t ax = u & 0x7FFFFFFFu;
    if (ax > 0x7F800000u) return sign | 0x7Fu;
    if (ax >= 0x43E00000u) return sign | 0x7Eu;
    if (ax < 0x3C800000u) {

        const uint32_t sub =
            float_to_bits_host_ct7(bits_to_float_host_ct6(ax) + 16384.0f) -
            0x46800000u;

        return sign | (uint8_t)sub;
    }
    const uint32_t lsb = (ax >> 20) & 1u;
    return sign | (uint8_t)(((ax + 0x7FFFFu + lsb) >> 20) - (120u << 3));
#endif
}
__dpct_inline__ uint16_t f32x2_to_e4m3fnx2(float a, float b) {
#if 0   // SYCL: no sm_89 FP8 hardware conversions; the software path is the one
    /*
    DPCT1007: Migration of __nv_cvt_float2_to_fp8x2 is not supported.
    */
    return (uint16_t)__nv_cvt_float2_to_fp8x2(sycl::float2(a, b),
                                              __NV_SATFINITE, __NV_E4M3);
#else
    return (uint16_t) f32_to_e4m3fn(a) | ((uint16_t) f32_to_e4m3fn(b) << 8);
#endif
}
__dpct_inline__ uint16_t e4m3fn_to_f16(uint8_t q) {
#if 0   // SYCL: no sm_89 FP8 hardware conversions; the software path is the one
    /*
    DPCT1007: Migration of __nv_cvt_fp8_to_halfraw is not supported.
    */
    return sycl::bit_cast<unsigned short, sycl::half>(
        sycl::half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)q, __NV_E4M3)));
#else
    const uint16_t sign = (uint16_t) (q & 0x80u) << 8;
    const uint32_t mag = q & 0x7Fu;
    if (mag == 0x7Fu) return sign | 0x7FFFu;
    if (mag < 8u) {
#if defined(__CUDA_ARCH__)
        return sign | (uint16_t) (dpct::byte_level_permute(0x1E1C1800u, 0x23222120u, mag << 4) & 0xFF00u);
#else
        static constexpr uint16_t kSub[8] = {0x0000u, 0x1800u, 0x1C00u, 0x1E00u, 0x2000u, 0x2100u, 0x2200u, 0x2300u};
        return sign | kSub[mag];
#endif
    }
    return sign | (uint16_t) ((mag << 7) + 0x2000u);
#endif
}
inline __dpct_inline__ uint16_t e4m3fn_to_f16_host_ct3(uint8_t q) {
#if 0   // SYCL: the host copy of the function: its `__CUDA_ARCH__ >= 890` is false, so the software else-branch (dpct mangled the test)
    return __half_as_ushort(
        __half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)q, __NV_E4M3)));
#else
    const uint16_t sign = (uint16_t)(q & 0x80u) << 8;
    const uint32_t mag = q & 0x7Fu;
    if (mag == 0x7Fu) return sign | 0x7FFFu;
    if (mag < 8u) {

        static constexpr uint16_t kSub[8] = {0x0000u, 0x1800u, 0x1C00u,
                                             0x1E00u, 0x2000u, 0x2100u,
                                             0x2200u, 0x2300u};
        return sign | kSub[mag];
    }
    return sign | (uint16_t)((mag << 7) + 0x2000u);
#endif
}
__dpct_inline__ uint32_t e4m3fnx2_to_h2_u32(uint16_t p2) {
#if 0   // SYCL: no sm_89 FP8 hardware conversions; the software path is the one
    /*
    DPCT1007: Migration of __nv_cvt_fp8x2_to_halfraw2 is not supported.
    */
    const sycl::half2 h2 = sycl::half2(
        __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)p2, __NV_E4M3));
    return (uint32_t)sycl::bit_cast<unsigned short, sycl::half>(h2[0]) |
           ((uint32_t)sycl::bit_cast<unsigned short, sycl::half>(h2[1]) << 16);
#else
    return (uint32_t) e4m3fn_to_f16((uint8_t) (p2 & 0xFFu)) |
           ((uint32_t) e4m3fn_to_f16((uint8_t) (p2 >> 8)) << 16);
#endif
}
__dpct_inline__ float e4m3fn_to_f32(uint8_t q) {

    return h2f(e4m3fn_to_f16(q));
}
inline __dpct_inline__ float e4m3fn_to_f32_host_ct2(uint8_t q) {

    return half_bits_to_float(e4m3fn_to_f16_host_ct3(q));
}
__dpct_inline__ sycl::float2 e4m3fnx2_to_f2(uint16_t p2) {
#if 0   // SYCL: no sm_89 FP8 hardware conversions; the software path is the one
    /*
    DPCT1007: Migration of __nv_cvt_fp8x2_to_halfraw2 is not supported.
    */
    return sycl::half2(
               __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)p2, __NV_E4M3))
        .template convert<float, sycl::rounding_mode::automatic>();
#else
    return u32_to_f2(e4m3fnx2_to_h2_u32(p2));
#endif
}

// the block's sum / max of v (blockDim.x a power of two, sh [blockDim.x]); every thread gets it
inline float bsum(float v, float *sh) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_local_id(2);
    sh[t] = v;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#pragma unroll
    for (int o = item_ct1.get_local_range(2) >> 1; o > 0; o >>= 1) {
        if (t < o) sh[t] += sh[t + o];
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    const float r = sh[0];
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    return r;
}
inline float bmax(float v, float *sh) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_local_id(2);
    sh[t] = v;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#pragma unroll
    for (int o = item_ct1.get_local_range(2) >> 1; o > 0; o >>= 1) {
        if (t < o) sh[t] = sycl::fmax(sh[t], sh[t + o]);
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    const float r = sh[0];
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    return r;
}

constexpr int kBlk = 256;

__dpct_inline__ uint32_t f2_to_u32(float a, float b) {
    const sycl::half2 h2 =
        sycl::float2(a, b)
            .template convert<sycl::half, sycl::rounding_mode::rte>();
    return (uint32_t)sycl::bit_cast<unsigned short, sycl::half>(h2[0]) |
           ((uint32_t)sycl::bit_cast<unsigned short, sycl::half>(h2[1]) << 16);
}

template <bool WRITE_F16, bool WRITE_F32>
/*
DPCT1110: The total declared local variable size in device function
rms_impl_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void rms_impl_kernel(const float *x, const float *w, int D,
                                     float eps, uint16_t *y16, float *y32) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sh = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[kBlk]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const float *xr = x + (size_t)item_ct1.get_group(2) * D;
    uint16_t *y16r =
        WRITE_F16 ? y16 + (size_t)item_ct1.get_group(2) * D : nullptr;
    float *y32r = WRITE_F32 ? y32 + (size_t)item_ct1.get_group(2) * D : nullptr;
    const bool vec4 = (((uintptr_t) xr & 15) == 0) && (!w || ((uintptr_t) w & 15) == 0) &&
                      (!WRITE_F32 || ((uintptr_t) y32r & 15) == 0) &&
                      (!WRITE_F16 || ((uintptr_t) y16r & 7) == 0) && ((D & 3) == 0);
    float a = 0.0f;
    if (vec4) {
        const int n4 = D >> 2;
        const sycl::float4 *x4 = (const sycl::float4 *)xr;
        for (int d4 = item_ct1.get_local_id(2); d4 < n4; d4 += kBlk) {
            const sycl::float4 v = ::ik2_ldg(x4 + d4);
            a += v.x() * v.x() + v.y() * v.y() + v.z() * v.z() + v.w() * v.w();
        }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        const float rs = sycl::rsqrt(bsum(a, sh) / (float)D + eps);
        const sycl::float4 *w4 = (const sycl::float4 *)w;
        sycl::float4 *y32_4 = (sycl::float4 *)y32r;
        sycl::uint2 *y16_4 = (sycl::uint2 *)y16r;
        for (int d4 = item_ct1.get_local_id(2); d4 < n4; d4 += kBlk) {
            const sycl::float4 v = ::ik2_ldg(x4 + d4);
            const sycl::float4 ww =
                w4 ? ::ik2_ldg(w4 + d4) : sycl::float4(1.0f, 1.0f, 1.0f, 1.0f);
            const float r0 = v.x() * rs * ww.x(), r1 = v.y() * rs * ww.y(),
                        r2 = v.z() * rs * ww.z(), r3 = v.w() * rs * ww.w();
            if (WRITE_F32) y32_4[d4] = sycl::float4(r0, r1, r2, r3);
            if (WRITE_F16) y16_4[d4] =
                sycl::uint2(f2_to_u32(r0, r1), f2_to_u32(r2, r3));
        }
    } else {
#pragma unroll
        for (int d = item_ct1.get_local_id(2); d < D; d += kBlk) a +=
            xr[d] * xr[d];
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        const float rs = sycl::rsqrt(bsum(a, sh) / (float)D + eps);
        for (int d = item_ct1.get_local_id(2); d < D; d += kBlk) {
            const float r = xr[d] * rs * (w ? w[d] : 1.0f);
            if (WRITE_F32) y32r[d] = r;
            if (WRITE_F16) y16r[d] = f2h(r);
        }
    }
}
// rms of up to three vectors, each with its own weights, in one launch (a block a vector; rms_impl_kernel's sums)
/*
DPCT1110: The total declared local variable size in device function
rms_multi_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void rms_multi_kernel(RmsJobs jobs, int D, float eps) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sh = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[kBlk]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const float *xr = jobs.x[item_ct1.get_group(2)];
    const float *w = jobs.w[item_ct1.get_group(2)];
    float *y32r = jobs.y[item_ct1.get_group(2)];
    const bool vec4 = (((uintptr_t) xr & 15) == 0) && (!w || ((uintptr_t) w & 15) == 0) &&
                      (((uintptr_t) y32r & 15) == 0) && ((D & 3) == 0);
    float a = 0.0f;
    if (vec4) {
        const int n4 = D >> 2;
        const sycl::float4 *x4 = (const sycl::float4 *)xr;
        for (int d4 = item_ct1.get_local_id(2); d4 < n4; d4 += kBlk) {
            const sycl::float4 v = ::ik2_ldg(x4 + d4);
            a += v.x() * v.x() + v.y() * v.y() + v.z() * v.z() + v.w() * v.w();
        }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        const float rs = sycl::rsqrt(bsum(a, sh) / (float)D + eps);
        const sycl::float4 *w4 = (const sycl::float4 *)w;
        sycl::float4 *y32_4 = (sycl::float4 *)y32r;
        for (int d4 = item_ct1.get_local_id(2); d4 < n4; d4 += kBlk) {
            const sycl::float4 v = ::ik2_ldg(x4 + d4);
            const sycl::float4 ww =
                w4 ? ::ik2_ldg(w4 + d4) : sycl::float4(1.0f, 1.0f, 1.0f, 1.0f);
            const float r0 = v.x() * rs * ww.x(), r1 = v.y() * rs * ww.y(),
                        r2 = v.z() * rs * ww.z(), r3 = v.w() * rs * ww.w();
            y32_4[d4] = sycl::float4(r0, r1, r2, r3);
        }
    } else {
#pragma unroll
        for (int d = item_ct1.get_local_id(2); d < D; d += kBlk) a +=
            xr[d] * xr[d];
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        const float rs = sycl::rsqrt(bsum(a, sh) / (float)D + eps);
#pragma unroll
        for (int d = item_ct1.get_local_id(2); d < D; d += kBlk)
            y32r[d] = xr[d] * rs * (w ? w[d] : 1.0f);
    }
}

__dpct_inline__ float wsum(float v) {
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) v +=
        dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), v,
            o);
    return v;
}
__dpct_inline__ float wmax(float v) {
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) v = sycl::fmax(
        v, dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(),
               v, o));
    return v;
}

__dpct_inline__ void l2clip_warp(const float *src, const float *sc,
                                 float *dst32, uint16_t *dst16, int D, float th,
                                 float tau) {
    const int lane =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) &
        31;
    if (D == 128 && (((uintptr_t) src & 15) == 0) && (!sc || ((uintptr_t) sc & 15) == 0) &&
        (!dst32 || ((uintptr_t) dst32 & 15) == 0) && (!dst16 || ((uintptr_t) dst16 & 7) == 0)) {
        const sycl::float4 xv = ((const sycl::float4 *)src)[lane];
        const sycl::float4 sv = sc ? ::ik2_ldg((const sycl::float4 *)sc + lane)
                                   : sycl::float4(1.0f, 1.0f, 1.0f, 1.0f);
        const float v0 = xv.x() * sv.x(), v1 = xv.y() * sv.y(),
                    v2 = xv.z() * sv.z(), v3 = xv.w() * sv.w();
        const float nrm =
            sycl::sqrt(wsum(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3));
        const float f = sycl::fmin(1.0f, th / (nrm + 1e-6f)) * tau;
        const float o0 = v0 * f, o1 = v1 * f, o2 = v2 * f, o3 = v3 * f;
        if (dst32)((sycl::float4 *)dst32)[lane] = sycl::float4(o0, o1, o2, o3);
        if (dst16)((sycl::uint2 *)dst16)[lane] =
            sycl::uint2(f2_to_u32(o0, o1), f2_to_u32(o2, o3));
        return;
    }
    float a = 0.0f;
    for (int d = lane; d < D; d += 32) {
        const float v = src[d] * (sc ? sc[d] : 1.0f);
        a += v * v;
    }
    const float nrm = sycl::sqrt(wsum(a));
    const float f = sycl::fmin(1.0f, th / (nrm + 1e-6f)) * tau;
    for (int d = lane; d < D; d += 32) {
        const float o = src[d] * (sc ? sc[d] : 1.0f) * f;
        if (dst32) dst32[d] = o;
        if (dst16) dst16[d] = f2h(o);
    }
}

__dpct_inline__ void rms_f16_warp(const float *src, uint16_t *dst16, int D,
                                  float eps) {
    const int lane =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) &
        31;
    if (D == 128 && (((uintptr_t) src & 15) == 0) && (((uintptr_t) dst16 & 7) == 0)) {
        const sycl::float4 xv = ((const sycl::float4 *)src)[lane];
        const float rs = sycl::rsqrt(wsum(xv.x() * xv.x() + xv.y() * xv.y() +
                                          xv.z() * xv.z() + xv.w() * xv.w()) /
                                         (float)D +
                                     eps);
        ((sycl::uint2 *)dst16)[lane] =
            sycl::uint2(f2_to_u32(xv.x() * rs, xv.y() * rs),
                        f2_to_u32(xv.z() * rs, xv.w() * rs));
        return;
    }
    float a = 0.0f;
    for (int d = lane; d < D; d += 32) {
        const float v = src[d];
        a += v * v;
    }
    const float rs = sycl::rsqrt(wsum(a) / (float)D + eps);
#pragma unroll
    for (int d = lane; d < D; d += 32) dst16[d] = f2h(src[d] * rs);
}

__dpct_inline__ void l2clip_fp8_warp(const float *src, uint8_t *dst8, int D,
                                     float th, float k_scale) {
    const int lane =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) &
        31;
    if (D == 128 && (((uintptr_t) src & 15) == 0) && (((uintptr_t) dst8 & 3) == 0)) {
        const sycl::float4 xv = ((const sycl::float4 *)src)[lane];
        const float nrm = sycl::sqrt(wsum(xv.x() * xv.x() + xv.y() * xv.y() +
                                          xv.z() * xv.z() + xv.w() * xv.w()));
        const float f = sycl::fmin(1.0f, th / (nrm + 1e-6f)) * k_scale;
        const uint32_t p0 = f32x2_to_e4m3fnx2(xv.x() * f, xv.y() * f);
        const uint32_t p1 = f32x2_to_e4m3fnx2(xv.z() * f, xv.w() * f);
        ((uint32_t*) dst8)[lane] = p0 | (p1 << 16);
        return;
    }
    float a = 0.0f;
    for (int d = lane; d < D; d += 32) {
        const float v = src[d];
        a += v * v;
    }
    const float nrm = sycl::sqrt(wsum(a));
    const float f = sycl::fmin(1.0f, th / (nrm + 1e-6f)) * k_scale;
#pragma unroll
    for (int d = lane; d < D; d += 32) dst8[d] = f32_to_e4m3fn(src[d] * f);
}

__dpct_inline__ void rms_fp8_warp(const float *src, uint8_t *dst8, int D,
                                  float eps, float v_scale) {
    const int lane =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) &
        31;
    if (D == 128 && (((uintptr_t) src & 15) == 0) && (((uintptr_t) dst8 & 3) == 0)) {
        const sycl::float4 xv = ((const sycl::float4 *)src)[lane];
        const float rs = sycl::rsqrt(wsum(xv.x() * xv.x() + xv.y() * xv.y() +
                                          xv.z() * xv.z() + xv.w() * xv.w()) /
                                         (float)D +
                                     eps) *
                         v_scale;
        const uint32_t p0 = f32x2_to_e4m3fnx2(xv.x() * rs, xv.y() * rs);
        const uint32_t p1 = f32x2_to_e4m3fnx2(xv.z() * rs, xv.w() * rs);
        ((uint32_t*) dst8)[lane] = p0 | (p1 << 16);
        return;
    }
    float a = 0.0f;
    for (int d = lane; d < D; d += 32) {
        const float v = src[d];
        a += v * v;
    }
    const float rs = sycl::rsqrt(wsum(a) / (float)D + eps) * v_scale;
#pragma unroll
    for (int d = lane; d < D; d += 32) dst8[d] = f32_to_e4m3fn(src[d] * rs);
}

__dpct_inline__ void store_k_warp(const float *kp, uint16_t *kdst, int D,
                                  float qk_clip, float) {
    l2clip_warp(kp, nullptr, nullptr, kdst, D, qk_clip, 1.0f);
}
__dpct_inline__ void store_k_warp(const float *kp, uint8_t *kdst, int D,
                                  float qk_clip, float k_scale) {
    l2clip_fp8_warp(kp, kdst, D, qk_clip, k_scale);
}
__dpct_inline__ void store_v_warp(const float *vp, uint16_t *vdst, int D,
                                  float eps, float) {
    rms_f16_warp(vp, vdst, D, eps);
}
__dpct_inline__ void store_v_warp(const float *vp, uint8_t *vdst, int D,
                                  float eps, float v_scale) {
    rms_fp8_warp(vp, vdst, D, eps, v_scale);
}

// 1 warp (32 threads) per group across all q, k, v, r groups of T tokens
template <class KV>
/*
DPCT1110: The total declared local variable size in device function
prep_qkv_r_impl_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
prep_qkv_r_impl_kernel(float *q, const float *q_clip, const float *k_in, KV *kc,
                       const float *v_in, KV *vc, float *r, const float *r_clip,
                       int T, int H, int KVh, int D, int d_rel, float qk_clip,
                       float rel_clip, float eps, const int *pos, float alpha,
                       float n_floor, int pos0, int kv_cap, float k_scale,
                       float v_scale, const int *pos0_d, uint16_t *q16,
                       const uint16_t *sink_w, const float *sink_x, int sink_k,
                       float *sink_y, int q_ld = 0, int r_ld = 0) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (pos0_d) pos0 =
        *pos0_d; // (a token of a replayed graph: its position read here)
    const int gid = item_ct1.get_group(2);
    const int per_tok = 2 * H + 2 * KVh;
    if (q_ld <= 0) q_ld = H * D;   // (q's and r's rows: a token's [H][D] and [H][d_rel] unless given)
    if (r_ld <= 0) r_ld = H * d_rel;
    if (gid >= T * per_tok) {
        // (the tokens' sinks, sink_y [T][H] = sink_w [H][sink_k] . sink_x [T][sink_k] - mv_f16_kernel's sums, a
        //  launch less)
        const int i = gid - T * per_tok, t = i / H, h = i % H,
                  lane = item_ct1.get_local_id(2);
        if (!sink_w || t >= T) return;
        const uint16_t* w = sink_w + (size_t) h * sink_k;
        const float* sx = sink_x + (size_t) t * sink_k;
        float a = 0.0f;
#pragma unroll
        for (int j = lane; j < sink_k; j += 32) a += h2f(w[j]) * sx[j];
        a = wsum(a);
        if (lane == 0) sink_y[(size_t) t * H + h] = a;
        return;
    }
    const int t = gid / per_tok, rem = gid % per_tok;
    const float tau =
        pos ? (1.0f + alpha * sycl::log(sycl::fmax(
                                  1.0f, ((float)pos[t] + 1.0f) / n_floor)))
            : 1.0f;
    if (rem < H) {
        const int h = rem;
        float* qp = q + (size_t) t * q_ld + (size_t) h * D;
        l2clip_warp(qp, q_clip ? q_clip + (size_t) h * D : nullptr, qp, q16 ? q16 + ((size_t) t * H + h) * D : nullptr, D,
                    qk_clip, tau);
    } else if (rem < H + KVh) {
        const int kv = rem - H;
        const int slot = kv_cap > 0 ? (pos0 + t) % kv_cap : (pos0 + t);
        const float* kp = k_in + ((size_t) t * KVh + kv) * D;
        KV* kdst = kc + ((size_t) slot * KVh + kv) * D;
        store_k_warp(kp, kdst, D, qk_clip, k_scale);
    } else if (rem < H + 2 * KVh) {
        const int kv = rem - H - KVh;
        const int slot = kv_cap > 0 ? (pos0 + t) % kv_cap : (pos0 + t);
        const float* vp = v_in + ((size_t) t * KVh + kv) * D;
        KV* vdst = vc + ((size_t) slot * KVh + kv) * D;
        store_v_warp(vp, vdst, D, eps, v_scale);
    } else {
        const int h = rem - H - 2 * KVh;
        float* rp = r + (size_t) t * r_ld + (size_t) h * d_rel;
        l2clip_warp(rp, r_clip ? r_clip + (size_t) h * d_rel : nullptr, rp, nullptr, d_rel, rel_clip, tau);
    }
}

__dpct_inline__ void l2clip_kernel(const float *x, const float *scale, int G,
                                   int D, float th, const int *pos, float alpha,
                                   float n_floor, float *y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sh = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int row = item_ct1.get_group(2) / G, g = item_ct1.get_group(2) % G;
    const float *xr = x + (size_t)item_ct1.get_group(2) * D;
    float *yr = y + (size_t)item_ct1.get_group(2) * D;
    const float* sc = scale ? scale + (size_t) g * D : nullptr;
    float a = 0.0f;
    for (int d = item_ct1.get_local_id(2); d < D; d += 128) {
        const float v = xr[d] * (sc ? sc[d] : 1.0f);
        a += v * v;
    }
    const float nrm = sycl::sqrt(bsum(a, sh));
    float f = sycl::fmin(1.0f, th / (nrm + 1e-6f));
    if (pos) f *=
        1.0f +
        alpha * sycl::log(sycl::fmax(1.0f, ((float)pos[row] + 1.0f) / n_floor));
#pragma unroll
    for (int d = item_ct1.get_local_id(2); d < D; d += 128)
        yr[d] = xr[d] * (sc ? sc[d] : 1.0f) * f;
}

constexpr int kConvFusedMaxT = 16;
constexpr int kConvTileT = 16;

// Single-launch channel-mapped conv4 for small T (T <= kConvFusedMaxT): y must not alias x.
__dpct_inline__ void conv4_fused_kernel(const float *x, float *hist,
                                        const float *w, const float *dyn,
                                        int heads, int T, int C,
                                        const float *residual, float *y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= C) return;
    float h0 = hist[c], h1 = hist[(size_t) C + c], h2 = hist[(size_t) 2 * C + c];
    const float* wc = w + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    const float* dc_base = nullptr;
    int d_stride = 0;
    if (dyn) {
        const int head = c / (C / heads);
        dc_base = dyn + (size_t) head * 4;
        d_stride = heads * 4;
    }
    for (int t = 0; t < T; ++t) {
        const size_t tc = (size_t) t * C + c;
        const float x_new = x[tc];
        float a = w0 * h0 + w1 * h1 + w2 * h2 + w3 * x_new;
        if (dc_base) {
            const float* dc = dc_base + (size_t) t * d_stride;
            a += dc[0] * x_new + dc[1] * h2 + dc[2] * h1 + dc[3] * h0;
        }
        if (residual) a += residual[tc];
        y[tc] = a;
        h0 = h1;
        h1 = h2;
        h2 = x_new;
    }
    hist[c] = h0;
    hist[(size_t) C + c] = h1;
    hist[(size_t) 2 * C + c] = h2;
}

// Channel-mapped tiled conv4 for large T (prefill): reads w and head once per tile, no hist writes.
__dpct_inline__ void conv4_tiled_kernel(const float *x, const float *hist,
                                        const float *w, const float *dyn,
                                        int heads, int T, int C,
                                        const float *residual, float *y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= C) return;
    const int t0 = item_ct1.get_group(1) * kConvTileT;
    const int t1 = sycl::min(T, t0 + kConvTileT);
    auto xp_at = [&](int r) -> float { return r < 3 ? hist[(size_t) r * C + c] : x[(size_t) (r - 3) * C + c]; };
    float h0 = xp_at(t0), h1 = xp_at(t0 + 1), h2 = xp_at(t0 + 2);
    const float* wc = w + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    const float* dc_base = nullptr;
    int d_stride = 0;
    if (dyn) {
        const int head = c / (C / heads);
        dc_base = dyn + (size_t) head * 4;
        d_stride = heads * 4;
    }
    for (int t = t0; t < t1; ++t) {
        const size_t tc = (size_t) t * C + c;
        const float x_new = x[tc];
        float a = w0 * h0 + w1 * h1 + w2 * h2 + w3 * x_new;
        if (dc_base) {
            const float* dc = dc_base + (size_t) t * d_stride;
            a += dc[0] * x_new + dc[1] * h2 + dc[2] * h1 + dc[3] * h0;
        }
        if (residual) a += residual[tc];
        y[tc] = a;
        h0 = h1;
        h1 = h2;
        h2 = x_new;
    }
}

__dpct_inline__ void conv4_hist_kernel(const float *x, float *hist, int T,
                                       int C) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= C) return;
    for (int j = 0; j < 3; ++j) {   // (row T + j of xp; in place: it reads rows past the ones written)
        const int r = T + j;
        hist[(size_t) j * C + c] = r < 3 ? hist[(size_t) r * C + c] : x[(size_t) (r - 3) * C + c];
    }
}

template <int NUM_WARPS>
__dpct_inline__ void attn_res_warp0_softmax(float sh_red[][kResMax * 4],
                                            float sc[3][kResMax], int n, int nq,
                                            int C, float res_eps, int lane) {
    float ss = 0.0f, d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
    if (lane < n) {
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            ss += sh_red[w][lane * 4 + 0];
            d0 += sh_red[w][lane * 4 + 1];
            d1 += sh_red[w][lane * 4 + 2];
            d2 += sh_red[w][lane * 4 + 3];
        }
    }
    const float r = (lane < n) ? sycl::rsqrt(ss / (float)C + res_eps) : 0.0f;
    const float inv_sqrt_c = sycl::rsqrt((float)C);
    const float raw[3] = {d0 * r * inv_sqrt_c, d1 * r * inv_sqrt_c, d2 * r * inv_sqrt_c};
#pragma unroll
    for (int qi = 0; qi < 3; ++qi) {
        if (qi >= nq) break;
        const float s_i = (lane < n) ? raw[qi] : -INFINITY;
        const float m = wmax(s_i);
        const float e_i = (lane < n) ? sycl::native::exp(s_i - m) : 0.0f;
        const float sum = wsum(e_i);
        if (lane < n) sc[qi][lane] = (e_i / sum) * r;
    }
}

constexpr int kResGroup = 4;
template <bool FUSED_RMS, int BLOCK_THREADS>
/*
DPCT1110: The total declared local variable size in device function
attn_res_vec4_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
attn_res_vec4_kernel(ResStates st, int n, const float *q, int nq, int T, int C,
                     float res_eps, float rms_eps, float *out, RmsOuts outs,
                     size_t q_step, size_t o_step) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    q += item_ct1.get_group(1) * q_step,
        out += item_ct1.get_group(1) *
               o_step; // (attn_res_jobs: a job a block of the launch)
    constexpr int NUM_WARPS = BLOCK_THREADS / 32;
    auto &sh_red = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[NUM_WARPS][kResMax * 4]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sc =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[3][kResMax]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &rs_rms =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[3]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_group(2), tid = item_ct1.get_local_id(2),
              lane = tid & 31, warp = tid >> 5;
    const int c4 = C >> 2, d4_0 = tid, d4_1 = tid + BLOCK_THREADS;
    sycl::float4 q0[3] = {}, q1[3] = {};
    if (d4_0 < c4) {
        q0[0] = ::ik2_ldg((const sycl::float4 *)q + d4_0);
        if (nq > 1) q0[1] = ::ik2_ldg((const sycl::float4 *)(q + C) + d4_0);
        if (nq > 2) q0[2] = ::ik2_ldg((const sycl::float4 *)(q + 2 * C) + d4_0);
    }
    if (d4_1 < c4) {
        q1[0] = ::ik2_ldg((const sycl::float4 *)q + d4_1);
        if (nq > 1) q1[1] = ::ik2_ldg((const sycl::float4 *)(q + C) + d4_1);
        if (nq > 2) q1[2] = ::ik2_ldg((const sycl::float4 *)(q + 2 * C) + d4_1);
    }
    // (kResGroup states' values asked for at once, then used in the states' order: a thread waits for a load before
    //  it issues the next one's, and a token's launch is those waits - the sums are the one-by-one loop's, bit for bit)
    const bool in0 = d4_0 < c4, in1 = d4_1 < c4;
    const sycl::float4 zero4 = sycl::float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i0 = 0; i0 < n; i0 += kResGroup) {
        sycl::float4 xa[kResGroup], xb[kResGroup];
#pragma unroll
        for (int j = 0; j < kResGroup; ++j) {
            const sycl::float4 *x4 =
                (const sycl::float4 *)(st.st[sycl::min(i0 + j, n - 1)] +
                                       (size_t)t * C);
            xa[j] = in0 ? ::ik2_ldg(x4 + d4_0) : zero4;
            xb[j] = in1 ? ::ik2_ldg(x4 + d4_1) : zero4;
        }
#pragma unroll
        for (int j = 0; j < kResGroup; ++j) {
            const int i = i0 + j;
            if (i >= n) break;
            float sum_sq = 0.0f, qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f;
            if (in0) {
                const sycl::float4 xv0 = xa[j];
                sum_sq += xv0.x() * xv0.x() + xv0.y() * xv0.y() +
                          xv0.z() * xv0.z() + xv0.w() * xv0.w();
                qd0 += xv0.x() * q0[0].x() + xv0.y() * q0[0].y() +
                       xv0.z() * q0[0].z() + xv0.w() * q0[0].w();
                if (nq > 1) qd1 += xv0.x() * q0[1].x() + xv0.y() * q0[1].y() +
                                   xv0.z() * q0[1].z() + xv0.w() * q0[1].w();
                if (nq > 2) qd2 += xv0.x() * q0[2].x() + xv0.y() * q0[2].y() +
                                   xv0.z() * q0[2].z() + xv0.w() * q0[2].w();
            }
            if (in1) {
                const sycl::float4 xv1 = xb[j];
                sum_sq += xv1.x() * xv1.x() + xv1.y() * xv1.y() +
                          xv1.z() * xv1.z() + xv1.w() * xv1.w();
                qd0 += xv1.x() * q1[0].x() + xv1.y() * q1[0].y() +
                       xv1.z() * q1[0].z() + xv1.w() * q1[0].w();
                if (nq > 1) qd1 += xv1.x() * q1[1].x() + xv1.y() * q1[1].y() +
                                   xv1.z() * q1[1].z() + xv1.w() * q1[1].w();
                if (nq > 2) qd2 += xv1.x() * q1[2].x() + xv1.y() * q1[2].y() +
                                   xv1.z() * q1[2].z() + xv1.w() * q1[2].w();
            }
            sum_sq = wsum(sum_sq);
            qd0 = wsum(qd0);
            if (nq > 1) qd1 = wsum(qd1);
            if (nq > 2) qd2 = wsum(qd2);
            if (lane == 0) {
                sh_red[warp][i * 4 + 0] = sum_sq;
                sh_red[warp][i * 4 + 1] = qd0;
                sh_red[warp][i * 4 + 2] = qd1;
                sh_red[warp][i * 4 + 3] = qd2;
            }
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (warp == 0) attn_res_warp0_softmax<NUM_WARPS>(sh_red, sc, n, nq, C, res_eps, lane);
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    sycl::float4 o0[3] = {}, o1[3] = {};
    for (int i0 = 0; i0 < n; i0 += kResGroup) {
        sycl::float4 xa[kResGroup], xb[kResGroup];
#pragma unroll
        for (int j = 0; j < kResGroup; ++j) {
            const sycl::float4 *x4 =
                (const sycl::float4 *)(st.st[sycl::min(i0 + j, n - 1)] +
                                       (size_t)t * C);
            xa[j] = in0 ? ::ik2_ldg(x4 + d4_0) : zero4;
            xb[j] = in1 ? ::ik2_ldg(x4 + d4_1) : zero4;
        }
#pragma unroll
        for (int j = 0; j < kResGroup; ++j) {
            const int i = i0 + j;
            if (i >= n) break;
#pragma unroll
            for (int qi = 0; qi < 3; ++qi) {
                if (qi >= nq) break;
                const float w_qi = sc[qi][i];
                if (in0) {
                    o0[qi].x() += w_qi * xa[j].x();
                    o0[qi].y() += w_qi * xa[j].y();
                    o0[qi].z() += w_qi * xa[j].z();
                    o0[qi].w() += w_qi * xa[j].w();
                }
                if (in1) {
                    o1[qi].x() += w_qi * xb[j].x();
                    o1[qi].y() += w_qi * xb[j].y();
                    o1[qi].z() += w_qi * xb[j].z();
                    o1[qi].w() += w_qi * xb[j].w();
                }
            }
        }
    }
    if constexpr (!FUSED_RMS) {
#pragma unroll
        for (int qi = 0; qi < 3; ++qi) {
            if (qi >= nq) break;
            sycl::float4 *y4 = (sycl::float4 *)(out + ((size_t)qi * T + t) * C);
            if (d4_0 < c4) y4[d4_0] = o0[qi];
            if (d4_1 < c4) y4[d4_1] = o1[qi];
        }
    } else {
#pragma unroll
        for (int qi = 0; qi < 3; ++qi) {
            if (qi >= nq) break;
            float sq = 0.0f;
            if (d4_0 < c4) sq +=
                o0[qi].x() * o0[qi].x() + o0[qi].y() * o0[qi].y() +
                o0[qi].z() * o0[qi].z() + o0[qi].w() * o0[qi].w();
            if (d4_1 < c4) sq +=
                o1[qi].x() * o1[qi].x() + o1[qi].y() * o1[qi].y() +
                o1[qi].z() * o1[qi].z() + o1[qi].w() * o1[qi].w();
            sq = wsum(sq);
            if (lane == 0) sh_red[warp][qi] = sq;
        }
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (warp == 0) {
#pragma unroll
            for (int qi = 0; qi < 3; ++qi) {
                if (qi >= nq) break;
                float s_tot = (lane < NUM_WARPS) ? sh_red[lane][qi] : 0.0f;
                s_tot = wsum(s_tot);
                if (lane == 0) rs_rms[qi] =
                    sycl::rsqrt(s_tot / (float)C + rms_eps);
            }
        }
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
#pragma unroll
        for (int qi = 0; qi < 3; ++qi) {
            if (qi >= nq) break;
            const float rs = rs_rms[qi];
            const sycl::float4 *w4 = (const sycl::float4 *)outs.w[qi];
            sycl::float4 *y32_4 =
                (sycl::float4 *)(outs.y32[qi] ? outs.y32[qi] + (size_t)t * C
                                              : nullptr);
            sycl::uint2 *y16_4 =
                (sycl::uint2 *)(outs.y16[qi] ? outs.y16[qi] + (size_t)t * C
                                             : nullptr);
            if (d4_0 < c4) {
                const sycl::float4 ww =
                    w4 ? ::ik2_ldg(w4 + d4_0)
                       : sycl::float4(1.0f, 1.0f, 1.0f, 1.0f);
                const float r0 = o0[qi].x() * rs * ww.x(),
                            r1 = o0[qi].y() * rs * ww.y();
                const float r2 = o0[qi].z() * rs * ww.z(),
                            r3 = o0[qi].w() * rs * ww.w();
                if (y32_4) y32_4[d4_0] = sycl::float4(r0, r1, r2, r3);
                if (y16_4) y16_4[d4_0] =
                    sycl::uint2(f2_to_u32(r0, r1), f2_to_u32(r2, r3));
            }
            if (d4_1 < c4) {
                const sycl::float4 ww =
                    w4 ? ::ik2_ldg(w4 + d4_1)
                       : sycl::float4(1.0f, 1.0f, 1.0f, 1.0f);
                const float r0 = o1[qi].x() * rs * ww.x(),
                            r1 = o1[qi].y() * rs * ww.y();
                const float r2 = o1[qi].z() * rs * ww.z(),
                            r3 = o1[qi].w() * rs * ww.w();
                if (y32_4) y32_4[d4_1] = sycl::float4(r0, r1, r2, r3);
                if (y16_4) y16_4[d4_1] =
                    sycl::uint2(f2_to_u32(r0, r1), f2_to_u32(r2, r3));
            }
        }
    }
}

template <bool FUSED_RMS>
/*
DPCT1110: The total declared local variable size in device function
attn_res_scalar_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
attn_res_scalar_kernel(ResStates st, int n, const float *q, int nq, int T,
                       int C, float res_eps, float rms_eps, float *out,
                       RmsOuts outs, size_t q_step, size_t o_step) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    q += item_ct1.get_group(1) * q_step, out += item_ct1.get_group(1) * o_step;
    constexpr int NUM_WARPS = kBlk / 32;
    auto &sh_red = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[NUM_WARPS][kResMax * 4]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sc =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[3][kResMax]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &rs_rms =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[3]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_group(2), tid = item_ct1.get_local_id(2),
              lane = tid & 31, warp = tid >> 5;
    for (int i = 0; i < n; ++i) {
        const float* x = st.st[i] + (size_t) t * C;
        float sum_sq = 0.0f, qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f;
        for (int d = tid; d < C; d += kBlk) {
            const float xv = x[d];
            sum_sq += xv * xv;
            qd0 += xv * q[d];
            if (nq > 1) qd1 += xv * q[C + d];
            if (nq > 2) qd2 += xv * q[2 * C + d];
        }
        sum_sq = wsum(sum_sq);
        qd0 = wsum(qd0);
        if (nq > 1) qd1 = wsum(qd1);
        if (nq > 2) qd2 = wsum(qd2);
        if (lane == 0) {
            sh_red[warp][i * 4 + 0] = sum_sq;
            sh_red[warp][i * 4 + 1] = qd0;
            sh_red[warp][i * 4 + 2] = qd1;
            sh_red[warp][i * 4 + 3] = qd2;
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (warp == 0) attn_res_warp0_softmax<NUM_WARPS>(sh_red, sc, n, nq, C, res_eps, lane);
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if constexpr (!FUSED_RMS) {
        for (int qi = 0; qi < nq; ++qi) {
            float* o = out + ((size_t) qi * T + t) * C;
            for (int d = tid; d < C; d += kBlk) {
                float a = 0.0f;
#pragma unroll
                for (int i = 0; i < n; ++i) a +=
                    sc[qi][i] * st.st[i][(size_t)t * C + d];
                o[d] = a;
            }
        }
    } else {
        for (int qi = 0; qi < nq; ++qi) {
            float sq = 0.0f;
            for (int d = tid; d < C; d += kBlk) {
                float a = 0.0f;
#pragma unroll
                for (int i = 0; i < n; ++i) a +=
                    sc[qi][i] * st.st[i][(size_t)t * C + d];
                sq += a * a;
            }
            sq = wsum(sq);
            if (lane == 0) sh_red[warp][qi] = sq;
        }
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (warp == 0) {
            for (int qi = 0; qi < nq; ++qi) {
                float s_tot = (lane < NUM_WARPS) ? sh_red[lane][qi] : 0.0f;
                s_tot = wsum(s_tot);
                if (lane == 0) rs_rms[qi] =
                    sycl::rsqrt(s_tot / (float)C + rms_eps);
            }
        }
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        for (int qi = 0; qi < nq; ++qi) {
            const float rs = rs_rms[qi];
            const float* wq = outs.w[qi];
            uint16_t* y16 = outs.y16[qi] ? outs.y16[qi] + (size_t) t * C : nullptr;
            float* y32 = outs.y32[qi] ? outs.y32[qi] + (size_t) t * C : nullptr;
            for (int d = tid; d < C; d += kBlk) {
                float a = 0.0f;
#pragma unroll
                for (int i = 0; i < n; ++i) a +=
                    sc[qi][i] * st.st[i][(size_t)t * C + d];
                const float r = a * rs * (wq ? wq[d] : 1.0f);
                if (y32) y32[d] = r;
                if (y16) y16[d] = f2h(r);
            }
        }
    }
}

__dpct_inline__ void attn_probs_kernel(float *lg, int nk, int qpos0, int kpos0,
                                       float qk_scale, const float *r, int r_ld,
                                       int d_rel, const float *proj, int extent,
                                       int window, const float *sink,
                                       int sink_ld) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sh = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[kBlk]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int i = item_ct1.get_group(2);
    float* row = lg + (size_t) i * nk;
    const float* ri = r + (size_t) i * r_ld;
    const int qp = qpos0 + i;
    float mx = -INFINITY;
    for (int j = item_ct1.get_local_id(2); j < nk; j += kBlk) {
        const int dist = qp - (kpos0 + j);
        float l = -INFINITY;
        if (dist >= 0 && dist < window) {
            l = row[j] * qk_scale;
            if (dist < extent) {
                const float* p = proj + (size_t) dist * d_rel;
                float b = 0.0f;
#pragma unroll
                for (int e = 0; e < d_rel; ++e) b += ri[e] * p[e];
                l += b;
            }
        }
        row[j] = l;
        mx = sycl::fmax(mx, l);
    }
    const float sk = sink[(size_t) i * sink_ld];
    const float m = sycl::fmax(bmax(mx, sh), sk);
    float a = 0.0f;
    for (int j = item_ct1.get_local_id(2); j < nk; j += kBlk) {
        const float e =
            sycl::native::exp(row[j] - m); // (a masked key: exp(-inf) = 0)
        row[j] = e;
        a += e;
    }
    const float inv = 1.0f / (bsum(a, sh) + sycl::native::exp(sk - m));
#pragma unroll
    for (int j = item_ct1.get_local_id(2); j < nk; j += kBlk) row[j] *= inv;
}

/*
DPCT1110: The total declared local variable size in device function
attn_probs_f16_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void attn_probs_f16_kernel(const float *lg, uint16_t *out16,
                                           int H, int Tq, int nk, int qpos0,
                                           int kpos0, float qk_scale,
                                           const float *r, int d_rel,
                                           const float *proj, int extent,
                                           int window, const float *sink) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sh = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[kBlk]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &rs = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[64]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int i = item_ct1.get_group(2), h = item_ct1.get_group(1),
              tid = item_ct1.get_local_id(2);
    const float* row = lg + ((size_t) h * Tq + i) * nk;
    uint16_t* row_out = out16 + ((size_t) h * Tq + i) * nk;
    const float* ri_g = r + ((size_t) i * H + h) * d_rel;
    if (d_rel <= 64) {
#pragma unroll
        for (int e = tid; e < d_rel; e += kBlk) rs[e] = ri_g[e];
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    const int qp = qpos0 + i;
    const int j_lo = sycl::min(nk, sycl::max(0, qp - kpos0 - window + 1));
    const int j_hi = sycl::max(j_lo, sycl::min(nk, qp - kpos0 + 1));
    float mx = -INFINITY, sum = 0.0f;
    for (int j = j_lo + tid; j < j_hi; j += kBlk) {
        const int dist = qp - (kpos0 + j);
        float l = ik2_ldg(row + j) * qk_scale;
        if (dist < extent) {
            const float* p = proj + (size_t) dist * d_rel;
            float b = 0.0f;
#pragma unroll
            for (int e = 0; e < d_rel; ++e) b +=
                (e < 64 ? rs[e] : ri_g[e]) * ik2_ldg(p + e);
            l += b;
        }
        if (l > mx) {
            sum = (mx == -INFINITY ? 0.0f : sum * sycl::native::exp(mx - l)) +
                  1.0f;
            mx = l;
        } else {
            sum += sycl::native::exp(l - mx);
        }
    }
    const float sk = sink[(size_t) i * H + h];
    const float m = sycl::fmax(bmax(mx, sh), sk);
    const float s_scaled =
        (mx == -INFINITY) ? 0.0f : sum * sycl::native::exp(mx - m);
    const float inv = 1.0f / (bsum(s_scaled, sh) + sycl::native::exp(sk - m));
#pragma unroll
    for (int j = tid; j < j_lo; j += kBlk) row_out[j] = 0;
    for (int j = j_lo + tid; j < j_hi; j += kBlk) {
        const int dist = qp - (kpos0 + j);
        float l = ik2_ldg(row + j) * qk_scale;
        if (dist < extent) {
            const float* p = proj + (size_t) dist * d_rel;
            float b = 0.0f;
#pragma unroll
            for (int e = 0; e < d_rel; ++e) b +=
                (e < 64 ? rs[e] : ri_g[e]) * ik2_ldg(p + e);
            l += b;
        }
        row_out[j] = f2h(sycl::native::exp(l - m) * inv);
    }
#pragma unroll
    for (int j = j_hi + tid; j < nk; j += kBlk) row_out[j] = 0;
}

// attn_probs_f16 with the relative bias already in a table: bias [Tq * H rows][extent] (row i * H + h, column the
// distance), so an element costs a load instead of a d_rel-long dot product (done twice) - the prefill's hot kernel.
__dpct_inline__ void attn_probs_bias_f16_kernel(const float *lg,
                                                uint16_t *out16, int H, int Tq,
                                                int nk, int qpos0, int kpos0,
                                                float qk_scale,
                                                const float *bias, int extent,
                                                int window, const float *sink) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sh = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[kBlk]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int i = item_ct1.get_group(2), h = item_ct1.get_group(1),
              tid = item_ct1.get_local_id(2);
    const float* row = lg + ((size_t) h * Tq + i) * nk;
    uint16_t* row_out = out16 + ((size_t) h * Tq + i) * nk;
    const float* brow = bias + ((size_t) i * H + h) * extent;
    const int qp = qpos0 + i;
    const int j_lo = sycl::min(nk, sycl::max(0, qp - kpos0 - window + 1));
    const int j_hi = sycl::max(j_lo, sycl::min(nk, qp - kpos0 + 1));
    const int j_b = sycl::max(
        j_lo, qp - kpos0 - extent + 1); // from here on the distance is < extent
    float mx = -INFINITY, sum = 0.0f;
    for (int j = j_lo + tid; j < j_hi; j += kBlk) {
        float l = ik2_ldg(row + j) * qk_scale;
        if (j >= j_b) l += ik2_ldg(brow + (qp - kpos0 - j));
        if (l > mx) {
            sum = (mx == -INFINITY ? 0.0f : sum * sycl::native::exp(mx - l)) +
                  1.0f;
            mx = l;
        } else {
            sum += sycl::native::exp(l - mx);
        }
    }
    const float sk = sink[(size_t) i * H + h];
    const float m = sycl::fmax(bmax(mx, sh), sk);
    const float s_scaled =
        (mx == -INFINITY) ? 0.0f : sum * sycl::native::exp(mx - m);
    const float inv = 1.0f / (bsum(s_scaled, sh) + sycl::native::exp(sk - m));
#pragma unroll
    for (int j = tid; j < j_lo; j += kBlk) row_out[j] = 0;
    for (int j = j_lo + tid; j < j_hi; j += kBlk) {
        float l = ik2_ldg(row + j) * qk_scale;
        if (j >= j_b) l += ik2_ldg(brow + (qp - kpos0 - j));
        row_out[j] = f2h(sycl::native::exp(l - m) * inv);
    }
#pragma unroll
    for (int j = j_hi + tid; j < nk; j += kBlk) row_out[j] = 0;
}

// Prefill attention in one pass (FlashAttention-2 style, tensor cores via mma.sync m16n8k16, FP16 in / F32
// accumulate): the logits never leave registers.  A block: 64 queries (4 warps x 16) of one head against one split of
// its keys, in tiles of 64; the online softmax in the log2 domain.  Each split writes its unnormalised output and its
// (max, sum); attn_flash_combine merges the splits and the sink.  The relative bias from the table [Tq * H][extent].
constexpr int kFaBN = 64, kFaD = 128, kFaLd = kFaD + 8;   // (row pad: conflict-free fragments/ldmatrix)
constexpr float kLog2e = 1.4426950408889634f;

#include "ik2_mma_emul.dp.inc"   // SYCL: the tensor-core primitives, emulated (see that file)
// SYCL: cp.async is NVIDIA's asynchronous global-to-shared copy.  Here it is a plain copy that finishes before it returns (the
// invalid case zero-fills, as cp.async's src-size 0 does), so commit/wait have nothing to do: every caller waits and then
// reaches a barrier before reading the tile.  (The overlap of the next tile's loads with this tile's compute is lost: a speed
// item, not a correctness one.)
// SYCL: cp.async is NVIDIA's asynchronous global-to-shared copy.  Here it is a plain copy that finishes before it returns (the
// invalid case zero-fills, as cp.async's src-size 0 does), so commit/wait have nothing to do: every caller waits and then
// reaches a barrier before reading the tile.  (The overlap of the next tile's loads with this tile's compute is lost: a speed
// item, not a correctness one.)
// SYCL: cp.async is NVIDIA's asynchronous global-to-shared copy.  Here it is a plain copy that finishes before it returns (the
// invalid case zero-fills, as cp.async's src-size 0 does), so commit/wait have nothing to do: every caller waits and then
// reaches a barrier before reading the tile.  (The overlap of the next tile's loads with this tile's compute is lost: a speed
// item, not a correctness one.)
// SYCL: cp.async is NVIDIA's asynchronous global-to-shared copy.  Here it is a plain copy that finishes before it returns (the
// invalid case zero-fills, as cp.async's src-size 0 does), so commit/wait have nothing to do: every caller waits and then
// reaches a barrier before reading the tile.  (The overlap of the next tile's loads with this tile's compute is lost: a speed
// item, not a correctness one.)
// SYCL: cp.async is NVIDIA's asynchronous global-to-shared copy.  Here it is a plain copy that finishes before it returns (the
// invalid case zero-fills, as cp.async's src-size 0 does), so commit/wait have nothing to do: every caller waits and then
// reaches a barrier before reading the tile.  (The overlap of the next tile's loads with this tile's compute is lost: a speed
// item, not a correctness one.)
// SYCL: cp.async is NVIDIA's asynchronous global-to-shared copy.  Here it is a plain copy that finishes before it returns (the
// invalid case zero-fills, as cp.async's src-size 0 does), so commit/wait have nothing to do: every caller waits and then
// reaches a barrier before reading the tile.  (The overlap of the next tile's loads with this tile's compute is lost: a speed
// item, not a correctness one.)
// SYCL: cp.async is NVIDIA's asynchronous global-to-shared copy.  Here it is a plain copy that finishes before it returns (the
// invalid case zero-fills, as cp.async's src-size 0 does), so commit/wait have nothing to do: every caller waits and then
// reaches a barrier before reading the tile.  (The overlap of the next tile's loads with this tile's compute is lost: a speed
// item, not a correctness one.)
// SYCL: cp.async is NVIDIA's asynchronous global-to-shared copy.  Here it is a plain copy that finishes before it returns (the
// invalid case zero-fills, as cp.async's src-size 0 does), so commit/wait have nothing to do: every caller waits and then
// reaches a barrier before reading the tile.  (The overlap of the next tile's loads with this tile's compute is lost: a speed
// item, not a correctness one.)
// SYCL: cp.async is NVIDIA's asynchronous global-to-shared copy.  Here it is a plain copy that finishes before it returns (the
// invalid case zero-fills, as cp.async's src-size 0 does), so commit/wait have nothing to do: every caller waits and then
// reaches a barrier before reading the tile.  (The overlap of the next tile's loads with this tile's compute is lost: a speed
// item, not a correctness one.)
// SYCL: cp.async is NVIDIA's asynchronous global-to-shared copy.  Here it is a plain copy that finishes before it returns (the
// invalid case zero-fills, as cp.async's src-size 0 does), so commit/wait have nothing to do: every caller waits and then
// reaches a barrier before reading the tile.  (The overlap of the next tile's loads with this tile's compute is lost: a speed
// item, not a correctness one.)
// SYCL: cp.async is NVIDIA's asynchronous global-to-shared copy.  Here it is a plain copy that finishes before it returns (the
// invalid case zero-fills, as cp.async's src-size 0 does), so commit/wait have nothing to do: every caller waits and then
// reaches a barrier before reading the tile.  (The overlap of the next tile's loads with this tile's compute is lost: a speed
// item, not a correctness one.)
// SYCL: cp.async is NVIDIA's asynchronous global-to-shared copy.  Here it is a plain copy that finishes before it returns (the
// invalid case zero-fills, as cp.async's src-size 0 does), so commit/wait have nothing to do: every caller waits and then
// reaches a barrier before reading the tile.  (The overlap of the next tile's loads with this tile's compute is lost: a speed
// item, not a correctness one.)
// SYCL: cp.async is NVIDIA's asynchronous global-to-shared copy.  Here it is a plain copy that finishes before it returns (the
// invalid case zero-fills, as cp.async's src-size 0 does), so commit/wait have nothing to do: every caller waits and then
// reaches a barrier before reading the tile.  (The overlap of the next tile's loads with this tile's compute is lost: a speed
// item, not a correctness one.)
// SYCL: cp.async is NVIDIA's asynchronous global-to-shared copy.  Here it is a plain copy that finishes before it returns (the
// invalid case zero-fills, as cp.async's src-size 0 does), so commit/wait have nothing to do: every caller waits and then
// reaches a barrier before reading the tile.  (The overlap of the next tile's loads with this tile's compute is lost: a speed
// item, not a correctness one.)
// SYCL: cp.async is NVIDIA's asynchronous global-to-shared copy.  Here it is a plain copy that finishes before it returns (the
// invalid case zero-fills, as cp.async's src-size 0 does), so commit/wait have nothing to do: every caller waits and then
// reaches a barrier before reading the tile.  (The overlap of the next tile's loads with this tile's compute is lost: a speed
// item, not a correctness one.)
// SYCL: cp.async is NVIDIA's asynchronous global-to-shared copy.  Here it is a plain copy that finishes before it returns (the
// invalid case zero-fills, as cp.async's src-size 0 does), so commit/wait have nothing to do: every caller waits and then
// reaches a barrier before reading the tile.  (The overlap of the next tile's loads with this tile's compute is lost: a speed
// item, not a correctness one.)
// SYCL: cp.async is NVIDIA's asynchronous global-to-shared copy.  Here it is a plain copy that finishes before it returns (the
// invalid case zero-fills, as cp.async's src-size 0 does), so commit/wait have nothing to do: every caller waits and then
// reaches a barrier before reading the tile.  (The overlap of the next tile's loads with this tile's compute is lost: a speed
// item, not a correctness one.)
// SYCL: cp.async is NVIDIA's asynchronous global-to-shared copy.  Here it is a plain copy that finishes before it returns (the
// invalid case zero-fills, as cp.async's src-size 0 does), so commit/wait have nothing to do: every caller waits and then
// reaches a barrier before reading the tile.  (The overlap of the next tile's loads with this tile's compute is lost: a speed
// item, not a correctness one.)
// SYCL: cp.async is NVIDIA's asynchronous global-to-shared copy.  Here it is a plain copy that finishes before it returns (the
// invalid case zero-fills, as cp.async's src-size 0 does), so commit/wait have nothing to do: every caller waits and then
// reaches a barrier before reading the tile.  (The overlap of the next tile's loads with this tile's compute is lost: a speed
// item, not a correctness one.)
__dpct_inline__ void cp_async16(void *dst, const void *src, bool valid) {
    if (valid) *reinterpret_cast<sycl::uint4 *>(dst) = *reinterpret_cast<const sycl::uint4 *>(src);
    else *reinterpret_cast<sycl::uint4 *>(dst) = sycl::uint4(0, 0, 0, 0);
}
__dpct_inline__ void cp_async4(void *dst, const void *src, bool valid) {
    *reinterpret_cast<uint32_t *>(dst) = valid ? *reinterpret_cast<const uint32_t *>(src) : 0u;
}
__dpct_inline__ void cp_async_commit() {}
template <int N> __dpct_inline__ void cp_async_wait() {}

// one 64-key tile of K and V (rows kt .. kt+63; rows at or past k_end zero-filled) into smem, NT threads
template <int NT>
__dpct_inline__ void fa_load_tile(
    uint16_t *Ks, uint16_t *Vs, const uint16_t *kc, const uint16_t *vc,
    int kv_cap, int KD, int g, int kt, int k_end, int tid) {
#pragma unroll
    for (int it = 0; it < (kFaBN * kFaD / 8) / NT; ++it) {
        const int c = it * NT + tid, row = c >> 4, col = (c & 15) * 8;
        const int kp = kt + row;
        const bool ok = kp < k_end;
        const int pr = ok ? (kv_cap > 0 ? kp % kv_cap : kp) : 0;
        const size_t off = (size_t) pr * KD + (size_t) g * kFaD + col;
        cp_async16(Ks + row * kFaLd + col, kc + off, ok);
        cp_async16(Vs + row * kFaLd + col, vc + off, ok);
    }
}

// ... from an FP8 (E4M3) cache: the tile's bytes are copied as they are (cp.async, a tile ahead, as the FP16 tile)
// and turned into FP16 in shared memory once they have arrived.  (Turning them as they were loaded had every load
// wait for its bytes - a branch between it and the next - and the kernel ran at a sixth of the memory's speed.)
constexpr int kFaRaw = 2 * kFaBN * kFaD;   // the bytes of a tile: K's [kFaBN][kFaD], then V's
template <int NT>
__dpct_inline__ void fa_load_tile_raw(uint8_t *Rs, const uint8_t *kc,
                                      const uint8_t *vc, int kv_cap, int KD,
                                      int g, int kt, int k_end, int tid) {
#pragma unroll
    for (int it = 0; it < (kFaBN * kFaD / 16) / NT; ++it) {
        const int c = it * NT + tid, row = c >> 3, col = (c & 7) * 16;
        const int kp = kt + row;
        const bool ok = kp < k_end;
        const int pr = ok ? (kv_cap > 0 ? kp % kv_cap : kp) : 0;
        const size_t off = (size_t) pr * KD + (size_t) g * kFaD + col;
        cp_async16(Rs + row * kFaD + col, kc + off, ok);
        cp_async16(Rs + kFaBN * kFaD + row * kFaD + col, vc + off, ok);
    }
}
// four E4M3 values as FP16: lo the first two (two halves a register), hi the other two.  A value is sign, 4 bits
// of exponent, 3 of mantissa, and as FP16 its magnitude is (mag << 7) + 0x2000: with a value the high byte of its
// half (x = sign 2^15 + mag 2^8) the FP16 is x - mag 2^7 + 0x2000.  A subnormal (exponent 0: under 1/64) is that
// less an amount by its mantissa, looked up for the four at once (a byte permutation) - no branch, the same cost
// whatever the values.  (0x7F, the NaN, comes out finite: f32_to_e4m3fn saturates, a cache has none.)
__dpct_inline__ void e4m3x4_h4(uint32_t w, uint32_t &lo, uint32_t &hi) {
#if 0   // SYCL: no sm_89 FP8 hardware conversions; the software path is the one
    lo = e4m3fnx2_to_h2_u32((uint16_t) w), hi = e4m3fnx2_to_h2_u32((uint16_t) (w >> 16));
#else
    const uint32_t t = w & 0x78787878u;
    uint32_t z = (t - 0x01010101u) & ~t & 0x80808080u;   // 0x80 a byte whose exponent is 0 ...
    z -= z >> 7;                                         // ... 0x7F there
    const uint32_t v = w & 0x07070707u, u = v | (v >> 4);
    // (the normal formula - the subnormal's FP16) / 0x80 by the mantissa: 64, 17, 10, 7, 4, 3, 2, 1
    const uint32_t dd = dpct::byte_level_permute(0x070A1140u, 0x01020304u, dpct::byte_level_permute(u, 0u, 0x4420u)) & z;
    const uint32_t x0 = dpct::byte_level_permute(w, 0u, 0x1404u), x1 = dpct::byte_level_permute(w, 0u, 0x3424u);
    const uint32_t y0 = dpct::byte_level_permute(dd, 0u, 0x1404u), y1 = dpct::byte_level_permute(dd, 0u, 0x3424u);
    lo = x0 - (((x0 & 0x7F007F00u) + y0) >> 1) + 0x20002000u;
    hi = x1 - (((x1 & 0x7F007F00u) + y1) >> 1) + 0x20002000u;
#endif
}
__dpct_inline__ sycl::uint4 e4m3x8_h8(sycl::uint2 p) {
    sycl::uint4 h;
    e4m3x4_h4(p.x(), h.x(), h.y());
    e4m3x4_h4(p.y(), h.z(), h.w());
    return h;
}
// the FP16 tile (Ks, Vs: [kFaBN][kFaLd]) of the bytes Rs, NT threads
template <int NT>
__dpct_inline__ void fa_tile_from_raw(uint16_t *Ks, uint16_t *Vs,
                                      const uint8_t *Rs, int tid) {
#pragma unroll
    for (int it = 0; it < (kFaBN * kFaD / 8) / NT; ++it) {
        const int c = it * NT + tid, row = c >> 4, col = (c & 15) * 8;
        const sycl::uint2 k8 = *(const sycl::uint2 *)(Rs + row * kFaD + col);
        const sycl::uint2 v8 =
            *(const sycl::uint2 *)(Rs + kFaBN * kFaD + row * kFaD + col);
        *(sycl::uint4 *)(Ks + row * kFaLd + col) = e4m3x8_h8(k8);
        *(sycl::uint4 *)(Vs + row * kFaLd + col) = e4m3x8_h8(v8);
    }
}

// The relative bias: a tile within the extent of a warp's queries has the warp copy its 16 rows x 64 distances of the
// table into shared memory (cp.async, coalesced, issued before the tile's Q K^T) - loading it element by element at
// the softmax cost ~3x the whole kernel in the sliding layers (every tile there is within the extent).
constexpr int kFaBld = 72;   // (a bias row in shared memory; 72: 2-way bank conflicts at most for the float2 reads)
constexpr int kFaNW = 4;     // warps a block (8: the same speed in a test, and past sm_89/sm_120's shared memory)
/*
DPCT1083: The size of local memory in the migrated code may be different
from the original code. Check that the allocated memory size in the migrated
code is correct.
*/
constexpr int kFaSmem = 4 * kFaBN * kFaLd * (int)sizeof(uint16_t) +
                        kFaNW * 16 * kFaBld * (int)sizeof(float);
// Hopper (sm_90a) warpgroup MMA: the four warps of a 128-thread block as one warpgroup, 64 rows.
// D[64 x 64] (+)= A[64 x 16] (registers, the mma.m16n8k16 A fragments of each warp's 16 rows) x B (smem
// descriptor; tnsp: B stored N-major), F16 in, F32 accumulate; d[j * 4 + i] as mma's C fragment of n8 block j
template <int TNSP>
__dpct_inline__ void wgmma_m64n64k16_rs(float *d, const uint32_t *a,
                                        uint64_t desc, int scale_d) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
        "{%32, %33, %34, %35}, %36, p, 1, 1, %38;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d), "n"(TNSP));
#else
    assert(0);
#endif
}

// D[64 x 128] (+)= A[64 x 16] (registers, the mma.m16n8k16 A fragments of each warp's 16 rows) x B (smem
// descriptor; tnsp: B stored N-major), F16 in, F32 accumulate; d[j * 4 + i] as mma's C fragment of n8 block j
template <int TNSP>
__dpct_inline__ void wgmma_m64n128k16_rs(float *d, const uint32_t *a,
                                         uint64_t desc, int scale_d) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
        "{%64, %65, %66, %67}, %68, p, 1, 1, %70;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d), "n"(TNSP));
#else
    assert(0);
#endif
}

__dpct_inline__ void wg_fence() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
__dpct_inline__ void wg_commit() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}
__dpct_inline__ void wg_wait0() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#endif
}
// (keeps the compiler from moving an accumulator's use across the asynchronous MMA's issue or wait)
/*
DPCT1053: Migration of device assembly code is not supported.
*/
__dpct_inline__ void wg_fence_operand(float &r) {
    (void) r;   // SYCL: no asm; the wgmma (sm_90a) path is not built
}
__dpct_inline__ void fence_proxy_async() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
}
// shared-memory matrix descriptor with the 128-byte swizzle: start, leading / stride byte offsets
__dpct_inline__ uint64_t wg_desc(const void *p, uint32_t lbo, uint32_t sbo) {
    const uint32_t a = (uint32_t) (uintptr_t)(p);
    return (uint64_t) ((a & 0x3FFFF) >> 4) | ((uint64_t) ((lbo & 0x3FFFF) >> 4) << 16) |
           ((uint64_t) ((sbo & 0x3FFFF) >> 4) << 32) | ((uint64_t) 1 << 62);
}
// A 64 x 128 tile (rows: keys, columns: dims), 1024-byte aligned, in the 128-byte-swizzle layout: two atoms of
// 64 rows x 64 columns (128 bytes a row), the 16-byte chunks of row r XOR-permuted by r % 8. K is then B K-major
// (SBO 1024 B between 8-row groups; a k-step 32 bytes further in the atom), V B MN-major (transposed: LBO 8192 B
// between the atoms along the dims, SBO 1024 B between 8-key groups).
constexpr int kWgTile = kFaBN * kFaD;   // halves a tile
__dpct_inline__ int wg_tile_off(int r, int c) {
    return (c >> 6) * (kFaBN * 64) + r * 64 + ((((c >> 3) & 7) ^ (r & 7)) << 3);
}
template <int NT>
__dpct_inline__ void fa_load_tile_wg(
    uint16_t *Ks, uint16_t *Vs, const uint16_t *kc, const uint16_t *vc,
    int kv_cap, int KD, int g, int kt, int k_end, int tid) {
#pragma unroll
    for (int it = 0; it < (kFaBN * kFaD / 8) / NT; ++it) {
        const int c = it * NT + tid, row = c >> 4, col = (c & 15) * 8;
        const int kp = kt + row;
        const bool ok = kp < k_end;
        const int pr = ok ? (kv_cap > 0 ? kp % kv_cap : kp) : 0;
        const size_t off = (size_t) pr * KD + (size_t) g * kFaD + col;
        cp_async16(Ks + wg_tile_off(row, col), kc + off, ok);
        cp_async16(Vs + wg_tile_off(row, col), vc + off, ok);
    }
}

template <
    int NW, bool WG,
    class KVT =
        uint16_t> // (KVT: the cache's values, FP16 or FP8 - an FP8 tile's bytes
                  // are
                  /*
                  DPCT1110: The total declared local variable size in device
                  function attn_flash_f16_kernel exceeds 128 bytes and may cause
                  high register pressure. Consult with your hardware vendor to
                  find the total register size available and adjust the code, or
                  use smaller sub-group size to avoid high register pressure.
                  */
                  __dpct_inline__ void //  made FP16 in shared memory; the
                                       //  warpgroup path is FP16's)
                                       attn_flash_f16_kernel(
                                           const uint16_t *__restrict__ q16,
                                           int q_ld, const KVT *__restrict__ kc,
                                           const KVT *__restrict__ vc,
                                           int kv_cap, int H, int KV, int Tq,
                                           int qpos0, float qk_scale,
                                           const float *__restrict__ bias,
                                           int extent, int window, int n_split,
                                           float *__restrict__ o_part,
                                           sycl::float2 *__restrict__ ml_part,
                                           uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto fa_sm = (uint16_t *)dpct_local;
    constexpr bool F8 = sizeof(KVT) == 1;        // (FP8: one FP16 tile, and the bytes of two [2][kFaRaw] after it)
    uint16_t* Ks = fa_sm;                        // [2][kFaBN][kFaLd]
    uint16_t* Vs = fa_sm + (F8 ? 1 : 2) * kFaBN * kFaLd;   // [2][kFaBN][kFaLd]
    uint8_t* Rs = (uint8_t*) (fa_sm + 2 * kFaBN * kFaLd);
    constexpr int kTileLd = WG ? kWgTile : F8 ? 0 : kFaBN * kFaLd;   // (halves from one buffer to the other)
    float *Bw = (float *)(fa_sm + 4 * kFaBN * kFaLd) +
                (item_ct1.get_local_id(2) >> 5) * 16 *
                    kFaBld; // this warp's [16][kFaBld]
    if constexpr (WG) {   // [2][kWgTile] each, 1024-byte aligned (the swizzle's atoms), inside the same 68 KB
        uint16_t* a = (uint16_t*) (((uintptr_t) fa_sm + 1023) & ~(uintptr_t) 1023);
        Ks = a, Vs = a + 2 * kWgTile;
    }
    constexpr int kFaBM = 16 * NW;
    const int tid = item_ct1.get_local_id(2), warp = tid >> 5, lane = tid & 31,
              gr = lane >> 2, tq = lane & 3;
    const int h = item_ct1.get_group(1), split = item_ct1.get_group(0),
              g = h / (H / KV), KD = KV * kFaD;
    const int i0 = item_ct1.get_group(2) *
                   kFaBM; // the block's first query (index in the launch)
    const int i_last = sycl::min(Tq, i0 + kFaBM) - 1;
    // the block's keys: [k_lo, k_hi), cut into tiles; this split's tiles [t0, t1)
    const int k_hi = qpos0 + i_last + 1;
    const int k_lo = sycl::max(0, qpos0 + i0 - window + 1);
    const int n_tiles = (k_hi - k_lo + kFaBN - 1) / kFaBN;
    const int per = (n_tiles + n_split - 1) / n_split;
    const int t0 = sycl::min(n_tiles, split * per),
              t1 = sycl::min(n_tiles, t0 + per);

    // this warp's 16 queries: rows r0 = gr, r1 = gr + 8
    const int wi = i0 + warp * 16;
    const int ia = wi + gr, ib = wi + gr + 8;
    const int qa = qpos0 + ia, qb = qpos0 + ib;
    uint32_t qf[kFaD / 16][4];
    {
        const uint16_t* qra = q16 + (size_t) ia * q_ld + (size_t) h * kFaD;
        const uint16_t* qrb = q16 + (size_t) ib * q_ld + (size_t) h * kFaD;
#pragma unroll
        for (int kk = 0; kk < kFaD / 16; ++kk) {
            const int d = kk * 16 + tq * 2;
            qf[kk][0] = ia < Tq ? *(const uint32_t*) (qra + d) : 0u;
            qf[kk][1] = ib < Tq ? *(const uint32_t*) (qrb + d) : 0u;
            qf[kk][2] = ia < Tq ? *(const uint32_t*) (qra + d + 8) : 0u;
            qf[kk][3] = ib < Tq ? *(const uint32_t*) (qrb + d + 8) : 0u;
        }
    }
    float o[kFaD / 8][4];
#pragma unroll
    for (int j = 0; j < kFaD / 8; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    float ma = -INFINITY, mb = -INFINITY, la = 0.0f, lb = 0.0f;
    const float sc = qk_scale * kLog2e;
    const int lo_a = qa - window + 1, lo_b = qb - window + 1;

    if (t0 < t1) {
        if constexpr (WG) fa_load_tile_wg<NW * 32>(Ks, Vs, kc, vc, kv_cap, KD, g, k_lo + t0 * kFaBN, k_hi, tid);
        else if constexpr (F8) fa_load_tile_raw<NW * 32>(Rs, kc, vc, kv_cap, KD, g, k_lo + t0 * kFaBN, k_hi, tid);
        else fa_load_tile<NW * 32>(Ks, Vs, kc, vc, kv_cap, KD, g, k_lo + t0 * kFaBN, k_hi, tid);
        cp_async_commit();
    }
    for (int t = t0; t < t1; ++t) {
        const int buf = (t - t0) & 1;
        const int kt = k_lo + t * kFaBN;
        // does this warp's rows see any key of the tile, and is any distance within the bias's extent?
        const bool any = !(kt > qpos0 + wi + 15 || kt + kFaBN - 1 < qpos0 + wi - window + 1) && wi < Tq;
        const bool near = qpos0 + wi - (kt + kFaBN - 1) < extent;
        // cp.async groups in flight here: K/V(t), then this tile's bias, then K/V(t+1) (empty groups keep the count)
        if (any && near) {
#pragma unroll 4
            for (int rr = 0; rr < 16; ++rr) {
                const int i = wi + rr, qp = qpos0 + i;
                const float* brow = bias + ((size_t) i * H + h) * extent;
#pragma unroll
                for (int hf = 0; hf < 2; ++hf) {
                    const int c = hf * 32 + lane, dist = qp - kt - c;
                    const bool ok = i < Tq && dist >= 0 && dist < extent;
                    cp_async4(Bw + rr * kFaBld + c, ok ? brow + dist : bias, ok);
                }
            }
        }
        cp_async_commit();
        if (t + 1 < t1) {
            if constexpr (WG)
                fa_load_tile_wg<NW * 32>(Ks + (buf ^ 1) * kTileLd, Vs + (buf ^ 1) * kTileLd, kc, vc, kv_cap,
                                         KD, g, k_lo + (t + 1) * kFaBN, k_hi, tid);
            else if constexpr (F8)
                fa_load_tile_raw<NW * 32>(Rs + (buf ^ 1) * kFaRaw, kc, vc, kv_cap, KD, g, k_lo + (t + 1) * kFaBN, k_hi,
                                          tid);
            else
                fa_load_tile<NW * 32>(Ks + (buf ^ 1) * kFaBN * kFaLd, Vs + (buf ^ 1) * kFaBN * kFaLd, kc, vc, kv_cap,
                                      KD, g, k_lo + (t + 1) * kFaBN, k_hi, tid);
        }
        cp_async_commit();
        cp_async_wait<2>();
        if constexpr (WG) fence_proxy_async();   // (the copies visible to the MMA's reads of shared memory)
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        if constexpr (F8) {   // (the tile's bytes are in: FP16 of them)
            fa_tile_from_raw<NW * 32>(Ks, Vs, Rs + buf * kFaRaw, tid);
            /*
            DPCT1118: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            item_ct1.barrier(sycl::access::fence_space::local_space);
        }
        const uint16_t* K = Ks + buf * kTileLd;
        const uint16_t* V = Vs + buf * kTileLd;
        if (any || WG) {   // (the warpgroup's MMA takes all four warps)
            float s[kFaBN / 8][4];
            if constexpr (WG) {
#pragma unroll
                for (int j = 0; j < kFaBN / 8; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
#pragma unroll
                for (int j = 0; j < kFaBN / 8; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) wg_fence_operand(s[j][e]);
                wg_fence();
#pragma unroll
                for (int kk = 0; kk < kFaD / 16; ++kk)
                    wgmma_m64n64k16_rs<0>(&s[0][0], qf[kk], wg_desc(K + (kk >> 2) * (kFaBN * 64) + (kk & 3) * 16, 16, 1024), 1);
                wg_commit();
                wg_wait0();
#pragma unroll
                for (int j = 0; j < kFaBN / 8; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) wg_fence_operand(s[j][e]);
            } else
#pragma unroll
            for (int j = 0; j < kFaBN / 8; ++j) {
                s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
                const uint16_t* kr = K + (j * 8 + (lane & 7)) * kFaLd + (lane >> 3) * 8;   // (two k-steps an ldmatrix)
#pragma unroll
                for (int kk = 0; kk < kFaD / 16; kk += 2) {
                    uint32_t b0, b1, b2, b3;
                    ldsm_x4(b0, b1, b2, b3, kr + kk * 16);
                    mma16816(s[j], qf[kk], b0, b1);
                    mma16816(s[j], qf[kk + 1], b2, b3);
                }
            }
            // logits (log2 units): scale, bias, mask - a tile inside every row's keys and past the bias's extent
            // needs only the scale
            cp_async_wait<1>();   // (this tile's bias)
            sycl::group_barrier(
                sycl::ext::oneapi::this_work_item::get_sub_group());
            const bool full = !near && kt + kFaBN - 1 <= qpos0 + wi && kt >= qpos0 + wi + 15 - window + 1 &&
                              wi + 15 < Tq;
            float mxa = -INFINITY, mxb = -INFINITY;
            if (full) {
#pragma unroll
                for (int j = 0; j < kFaBN / 8; ++j) {
                    s[j][0] *= sc, s[j][1] *= sc, s[j][2] *= sc, s[j][3] *= sc;
                    mxa = sycl::fmax(mxa, sycl::fmax(s[j][0], s[j][1]));
                    mxb = sycl::fmax(mxb, sycl::fmax(s[j][2], s[j][3]));
                }
            } else
#pragma unroll
            for (int j = 0; j < kFaBN / 8; ++j) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int c = j * 8 + tq * 2 + e, kp = kt + c;
                    const int da = qa - kp, db = qb - kp;
                    float va = s[j][e] * sc, vb = s[j][2 + e] * sc;
                    if (near) {   // (zero where out of the extent)
                        va += Bw[gr * kFaBld + c] * kLog2e;
                        vb += Bw[(gr + 8) * kFaBld + c] * kLog2e;
                    }
                    va = (da < 0 || kp < lo_a || ia >= Tq) ? -INFINITY : va;
                    vb = (db < 0 || kp < lo_b || ib >= Tq) ? -INFINITY : vb;
                    s[j][e] = va, s[j][2 + e] = vb;
                    mxa = sycl::fmax(mxa, va), mxb = sycl::fmax(mxb, vb);
                }
            }
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            mxa = sycl::fmax(
                mxa, dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(),
                         mxa, 1));
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            mxa = sycl::fmax(
                mxa, dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(),
                         mxa, 2));
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            mxb = sycl::fmax(
                mxb, dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(),
                         mxb, 1));
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            mxb = sycl::fmax(
                mxb, dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(),
                         mxb, 2));
            const float mna = sycl::fmax(ma, mxa), mnb = sycl::fmax(mb, mxb);
            const float ca =
                mna == -INFINITY
                    ? 1.0f
                    : sycl::exp2(ma - mna); // (ma = -inf: exp2f(-inf) = 0)
            const float cb = mnb == -INFINITY ? 1.0f : sycl::exp2(mb - mnb);
            const float ua = mna == -INFINITY ? 0.0f : mna, ub = mnb == -INFINITY ? 0.0f : mnb;
            ma = mna, mb = mnb;
            la *= ca, lb *= cb;
#pragma unroll
            for (int j = 0; j < kFaD / 8; ++j) o[j][0] *= ca, o[j][1] *= ca, o[j][2] *= cb, o[j][3] *= cb;
            // P (FP16) as the A fragments of P V, two 8-key tiles per 16-key step
            uint32_t pa[kFaBN / 16][4];
#pragma unroll
            for (int kk = 0; kk < kFaBN / 16; ++kk) {
                const float p00 = sycl::exp2(s[2 * kk][0] - ua),
                            p01 = sycl::exp2(s[2 * kk][1] - ua);
                const float p02 = sycl::exp2(s[2 * kk][2] - ub),
                            p03 = sycl::exp2(s[2 * kk][3] - ub);
                const float p10 = sycl::exp2(s[2 * kk + 1][0] - ua),
                            p11 = sycl::exp2(s[2 * kk + 1][1] - ua);
                const float p12 = sycl::exp2(s[2 * kk + 1][2] - ub),
                            p13 = sycl::exp2(s[2 * kk + 1][3] - ub);
                la += p00 + p01 + p10 + p11;
                lb += p02 + p03 + p12 + p13;
                pa[kk][0] = f2_to_u32(p00, p01);
                pa[kk][1] = f2_to_u32(p02, p03);
                pa[kk][2] = f2_to_u32(p10, p11);
                pa[kk][3] = f2_to_u32(p12, p13);
            }
            if constexpr (WG) {
#pragma unroll
                for (int j = 0; j < kFaD / 8; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) wg_fence_operand(o[j][e]);
                wg_fence();
#pragma unroll
                for (int kk = 0; kk < kFaBN / 16; ++kk)
                    wgmma_m64n128k16_rs<1>(&o[0][0], pa[kk], wg_desc(V + kk * 16 * 64, kFaBN * 128, 1024), 1);
                wg_commit();
                wg_wait0();
#pragma unroll
                for (int j = 0; j < kFaD / 8; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) wg_fence_operand(o[j][e]);
            } else {
#pragma unroll
                for (int kk = 0; kk < kFaBN / 16; ++kk) {
                    const uint16_t* vr = V + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * kFaLd + (lane >> 4) * 8;
#pragma unroll
                    for (int j = 0; j < kFaD / 8; j += 2) {
                        uint32_t b0, b1, b2, b3;
                        ldsm_x4_t(b0, b1, b2, b3, vr + j * 8);
                        mma16816(o[j], pa[kk], b0, b1);
                        mma16816(o[j + 1], pa[kk], b2, b3);
                    }
                }
            }
        }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        if constexpr (!F8) item_ct1.barrier(
            sycl::access::fence_space::local_space); // (the buffer is refilled
                                                     // next iteration; FP8's
                                                     // FP16 tile is written
    }                                         //  past the next iteration's barrier)
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    la += dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), la, 1);
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    la += dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), la, 2);
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    lb += dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), lb, 1);
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    lb += dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), lb, 2);
    // the split's partials: o_part [split][Tq][H][D], ml_part [split][Tq][H] = (max in log2 units, sum)
    if (ia < Tq) {
        float* op = o_part + (((size_t) split * Tq + ia) * H + h) * kFaD + tq * 2;
#pragma unroll
        for (int j = 0; j < kFaD / 8; ++j) * (sycl::float2 *)(op + j * 8) =
            sycl::float2(o[j][0], o[j][1]);
        if (tq == 0) ml_part[((size_t)split * Tq + ia) * H + h] =
            sycl::float2(ma, la);
    }
    if (ib < Tq) {
        float* op = o_part + (((size_t) split * Tq + ib) * H + h) * kFaD + tq * 2;
#pragma unroll
        for (int j = 0; j < kFaD / 8; ++j) * (sycl::float2 *)(op + j * 8) =
            sycl::float2(o[j][2], o[j][3]);
        if (tq == 0) ml_part[((size_t)split * Tq + ib) * H + h] =
            sycl::float2(mb, lb);
    }
}

// the splits merged with the sink: out [Tq] rows of o_ld, head h at h * D; a block a (query, head), D threads
__dpct_inline__ void attn_flash_combine_kernel(
    const float *__restrict__ o_part, const sycl::float2 *__restrict__ ml_part,
    int n_split, int Tq, int H, const float *__restrict__ sink, float *out,
    int o_ld, float v_inv) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2), h = item_ct1.get_group(1),
              d = item_ct1.get_local_id(2);
    float m = sink[(size_t) i * H + h] * kLog2e;
#pragma unroll
    for (int s = 0; s < n_split; ++s)
        m = sycl::fmax(m, ml_part[((size_t)s * Tq + i) * H + h].x());
    float den = sycl::exp2(sink[(size_t)i * H + h] * kLog2e - m), acc = 0.0f;
    for (int s = 0; s < n_split; ++s) {
        const sycl::float2 ml = ml_part[((size_t)s * Tq + i) * H + h];
        if (ml.x() == -INFINITY) continue;
        const float c = sycl::exp2(ml.x() - m);
        den += ml.y() * c;
        acc += o_part[(((size_t) s * Tq + i) * H + h) * kFaD + d] * c;
    }
    out[(size_t) i * o_ld + (size_t) h * kFaD + d] = acc / den * v_inv;   // (v_inv: an FP8 cache's V scale)
}

// ---- one token's attention on the tensor cores (long contexts: the scalar kernels are bound by their arithmetic)
// A warp at a time and straight from the cache into the registers: no shared memory, so a dozen and more warps an
// SM run it, and a warp's wait for its rows is another warp's arithmetic - the kernel reads at the memory's speed,
// an FP8 cache too (its bytes made FP16 in the registers).  (attn_flash's way - tiles in shared memory, one block
// of four warps an SM, one of them computing - read an FP16 cache at the memory's speed and an FP8 one at a third
// of it.)
// A block's warps share a K / V head; a warp takes a run of its keys, 16 a step, for the R = H / Hkv <= 8 heads of
// that K / V head - a key's K and V rows are read once for them all:
//   Q K^T   C [16 keys][8 heads] += A [16 keys][16 dims] B [16 dims][8 heads].  A is K as it lies in the cache: a
//           lane loads 4 consecutive values of two keys, and the 16 dims of a step are taken in the order that makes
//           those the lane's fragment; B is the heads' q in the same order.
//   P V     C [16 dims][8 heads] += A [16 dims][16 keys] B [16 keys][8 heads].  A pairs two keys' values of a dim:
//           a lane loads 16 consecutive values of four keys and interleaves them; B is P, shuffled from the lanes
//           that hold a (key, head) logit to the lanes of that head's fragment.
// The online softmax in the log2 domain: a lane has the logits of two heads (the columns 2 tq, 2 tq + 1 of both
// C's), keeps their (max, sum) and 16 dims of their output.  A warp writes its unnormalised output and (max, sum);
// attn_decode_mma_combine merges the warps and the sink (and takes the output through the layer's gate: one launch
// less).  The relative bias r . proj is taken where a step is within the extent of the token, by the lanes that hold
// the logits.

constexpr int kWaNW = 4;    // warps a block
constexpr int kWaMin = 1;   // steps a warp before the next warp has keys (1: a short context or a window is bound by
                            // a warp's latency, not by the partials to merge)
// the keys of a token at qpos: steps of 16 from k_lo; of nw warps, the ones that have keys
__dpct_inline__ int wa_active(int qpos, int window, int nw, int &k_lo,
                              int &n_steps) {
    k_lo = window < (1 << 30) ? sycl::max(0, qpos - window + 1) : 0;
    n_steps = (qpos + 1 - k_lo + 15) >> 4;
    return sycl::min(nw, (n_steps + kWaMin - 1) / kWaMin);
}

template <class KVT>
/*
DPCT1110: The total declared local variable size in device function
attn_decode_mma_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void attn_decode_mma_kernel(
    const uint16_t *__restrict__ q16, const KVT *__restrict__ kc,
    const KVT *__restrict__ vc, int kv_cap, int H, int Hkv, const int *pos_d,
    int qpos, float qk_scale, const float *__restrict__ r,
    const float *__restrict__ proj, int d_rel, bool rel4, int extent,
    int window, int n_split, float *__restrict__ o_part,
    sycl::float2 *__restrict__ ml_part) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr bool F8 = sizeof(KVT) == 1;
    if (pos_d) qpos = *pos_d;   // (a token of a replayed graph: its position read here)
    const int lane = item_ct1.get_local_id(2) & 31, gr = lane >> 2,
              tq = lane & 3;
    const int g = item_ct1.get_group(2), R = H / Hkv, KD = Hkv * kFaD;
    const int w =
        item_ct1.get_group(1) * kWaNW +
        (item_ct1.get_local_id(2) >> 5); // this warp of the K / V head's
    int k_lo, n_steps;
    const int n_act = wa_active(qpos, window, n_split * kWaNW, k_lo, n_steps);
    if (w >= n_act) return;
    const int per = (n_steps + n_act - 1) / n_act;
    const int st0 = sycl::min(n_steps, w * per),
              st1 = sycl::min(n_steps, st0 + per);
    // the lane's part: head gr's q and P (B's column gr), the logits and the output of the heads qa and qb
    const int qa = 2 * tq, qb = qa + 1;
    const bool q_ok = gr < R, a_ok = qa < R, b_ok = qb < R;
    uint32_t qf[kFaD / 16][2];
    {
        const sycl::uint2 *qr =
            (const sycl::uint2 *)(q16 +
                                  (size_t)(g * R + (q_ok ? gr : 0)) * kFaD);
#pragma unroll
        for (int kk = 0; kk < kFaD / 16; ++kk) {
            const sycl::uint2 v =
                qr[kk * 4 + tq]; // (the dims 16 kk + 4 tq .. + 3)
            qf[kk][0] = q_ok ? v.x() : 0u, qf[kk][1] = q_ok ? v.y() : 0u;
        }
    }
    float o[kFaD / 16][4];
#pragma unroll
    for (int j = 0; j < kFaD / 16; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    float ma = -INFINITY, mb = -INFINITY, la = 0.0f, lb = 0.0f;
    const float sc = qk_scale * kLog2e;
    const KVT* kg = kc + (size_t) g * kFaD + tq * 4;    // the lane's values of a K row (+ 16 kk) ...
    const KVT* vg = vc + (size_t) g * kFaD + gr * 16;   // ... and of a V row
    const float* ra = r + (size_t) (g * R + (a_ok ? qa : 0)) * d_rel;   // (the bias: the heads' r)
    const float* rb = r + (size_t) (g * R + (b_ok ? qb : 0)) * d_rel;
    const int src = tq * 8 + (gr >> 1);   // the lane with head gr's logits of the key 2 tq (+ 4: of the key 2 tq + 1)
    const uint32_t psel = (gr & 1) ? 0x7632u : 0x5410u;

    for (int st = st0; st < st1; ++st) {
        const int kt = k_lo + st * 16, kmax = qpos - kt;
        const int row0 = kv_cap > 0 ? kt % kv_cap : kt;
        // a key's row in the cache (a key past the token's: the token's row, its logits masked below)
        auto row = [&](int key) {
            int rw = row0 + sycl::min(key, kmax);
            if (kv_cap > 0 && rw >= kv_cap) rw -= kv_cap;
            return (size_t) rw * KD;
        };
        // Q K^T: the keys gr and gr + 8
        float sa[4] = {0.0f, 0.0f, 0.0f, 0.0f}, sb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        {
            const KVT* p0 = kg + row(gr);
            const KVT* p1 = kg + row(gr + 8);
            uint32_t ka[kFaD / 16][4];
            if constexpr (F8) {
                uint32_t w0[kFaD / 16], w1[kFaD / 16];
#pragma unroll
                for (int kk = 0; kk < kFaD / 16; ++kk) {
                    w0[kk] = ik2_ldg((const uint32_t*) (p0 + kk * 16));
                    w1[kk] = ik2_ldg((const uint32_t*) (p1 + kk * 16));
                }
#pragma unroll
                for (int kk = 0; kk < kFaD / 16; ++kk) {
                    e4m3x4_h4(w0[kk], ka[kk][0], ka[kk][2]);
                    e4m3x4_h4(w1[kk], ka[kk][1], ka[kk][3]);
                }
            } else {
#pragma unroll
                for (int kk = 0; kk < kFaD / 16; ++kk) {
                    const sycl::uint2 v0 = ::ik2_ldg(
                                          (const sycl::uint2 *)(p0 + kk * 16)),
                                      v1 = ::ik2_ldg(
                                          (const sycl::uint2 *)(p1 + kk * 16));
                    ka[kk][0] = v0.x(), ka[kk][2] = v0.y(), ka[kk][1] = v1.x(),
                    ka[kk][3] = v1.y();
                }
            }
#pragma unroll
            for (int kk = 0; kk < kFaD / 16; ++kk) mma16816((kk & 1) ? sb : sa, ka[kk], qf[kk][0], qf[kk][1]);
        }
        // the logits (log2 units): scale, bias, mask
        const int k0 = kt + gr, k1 = k0 + 8;
        float c0 = (sa[0] + sb[0]) * sc, c1 = (sa[1] + sb[1]) * sc, c2 = (sa[2] + sb[2]) * sc, c3 = (sa[3] + sb[3]) * sc;
        if (qpos - (kt + 15) < extent) {   // (a step within the bias's extent of the token; zero past the extent)
            const int d0 = qpos - k0, d1 = qpos - k1;
            const bool e0 = d0 >= 0 && d0 < extent, e1 = d1 >= 0 && d1 < extent;
            const float* pj0 = proj + (size_t) (e0 ? d0 : 0) * d_rel;
            const float* pj1 = proj + (size_t) (e1 ? d1 : 0) * d_rel;
            float b0a = 0.0f, b0b = 0.0f, b1a = 0.0f, b1b = 0.0f;   // (key gr | gr + 8, head qa | qb)
            if (rel4) {
                for (int e = 0; e < d_rel; e += 4) {
                    const sycl::float4 x0 = ::ik2_ldg(
                                           (const sycl::float4 *)(pj0 + e)),
                                       x1 = ::ik2_ldg(
                                           (const sycl::float4 *)(pj1 + e));
                    const sycl::float4 ya = ::ik2_ldg(
                                           (const sycl::float4 *)(ra + e)),
                                       yb = ::ik2_ldg(
                                           (const sycl::float4 *)(rb + e));
                    b0a += x0.x() * ya.x() + x0.y() * ya.y() + x0.z() * ya.z() +
                           x0.w() * ya.w();
                    b0b += x0.x() * yb.x() + x0.y() * yb.y() + x0.z() * yb.z() +
                           x0.w() * yb.w();
                    b1a += x1.x() * ya.x() + x1.y() * ya.y() + x1.z() * ya.z() +
                           x1.w() * ya.w();
                    b1b += x1.x() * yb.x() + x1.y() * yb.y() + x1.z() * yb.z() +
                           x1.w() * yb.w();
                }
            } else {
                for (int e = 0; e < d_rel; ++e) {
                    const float x0 = pj0[e], x1 = pj1[e], ya = ra[e], yb = rb[e];
                    b0a += x0 * ya, b0b += x0 * yb, b1a += x1 * ya, b1b += x1 * yb;
                }
            }
            if (e0) c0 += b0a * kLog2e, c1 += b0b * kLog2e;
            if (e1) c2 += b1a * kLog2e, c3 += b1b * kLog2e;
        }
        c0 = (k0 > qpos || !a_ok) ? -INFINITY : c0;
        c1 = (k0 > qpos || !b_ok) ? -INFINITY : c1;
        c2 = (k1 > qpos || !a_ok) ? -INFINITY : c2;
        c3 = (k1 > qpos || !b_ok) ? -INFINITY : c3;
        float mxa = sycl::fmax(c0, c2),
              mxb = sycl::fmax(
                  c1, c3); // (over the lanes of the same heads: the same tq)
#pragma unroll
        for (int sh = 4; sh < 32; sh <<= 1) {
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            mxa = sycl::fmax(
                mxa, dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(),
                         mxa, sh));
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            mxb = sycl::fmax(
                mxb, dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(),
                         mxb, sh));
        }
        if (sycl::reduce_over_group(
                sycl::ext::oneapi::this_work_item::get_sub_group(),
                (0xffffffffu &
                 (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                             .get_local_linear_id())) &&
                            mxa > ma ||
                        mxb > mb
                    ? (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                                  .get_local_linear_id())
                    : 0,
                sycl::ext::oneapi::plus<>())) { // a new maximum (few steps have
                                                // one): rescale
            const float mna = sycl::fmax(ma, mxa), mnb = sycl::fmax(mb, mxb);
            const float ca =
                mna == -INFINITY
                    ? 1.0f
                    : sycl::exp2(ma - mna); // (ma = -inf: exp2f(-inf) = 0)
            const float cb = mnb == -INFINITY ? 1.0f : sycl::exp2(mb - mnb);
            ma = mna, mb = mnb;
            la *= ca, lb *= cb;
#pragma unroll
            for (int j = 0; j < kFaD / 16; ++j) o[j][0] *= ca, o[j][1] *= cb, o[j][2] *= ca, o[j][3] *= cb;
        }
        const float ua = ma == -INFINITY ? 0.0f : ma, ub = mb == -INFINITY ? 0.0f : mb;
        const float p0 = sycl::exp2(c0 - ua), p1 = sycl::exp2(c1 - ub),
                    p2 = sycl::exp2(c2 - ua), p3 = sycl::exp2(c3 - ub);
        la += p0 + p2, lb += p1 + p3;
        // P (FP16) to the lanes of its fragment: head gr's of the keys 2 tq, 2 tq + 1 and 2 tq + 8, 2 tq + 9
        const uint32_t plo = f2_to_u32(p0, p1), phi = f2_to_u32(p2, p3);   // (key gr, key gr + 8: heads qa | qb)
        /*
        DPCT1108: '__shfl_sync' was migrated with the experimental feature
        masked sub_group function which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        const uint32_t alo = dpct::select_from_sub_group(sycl::ext::oneapi::this_work_item::get_sub_group(),
                           plo, src),
                       blo = dpct::select_from_sub_group(sycl::ext::oneapi::this_work_item::get_sub_group(),
                           plo, src + 4);
        /*
        DPCT1108: '__shfl_sync' was migrated with the experimental feature
        masked sub_group function which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        const uint32_t ahi = dpct::select_from_sub_group(sycl::ext::oneapi::this_work_item::get_sub_group(),
                           phi, src),
                       bhi = dpct::select_from_sub_group(sycl::ext::oneapi::this_work_item::get_sub_group(),
                           phi, src + 4);
        const uint32_t pb0 =
                           q_ok ? dpct::byte_level_permute(alo, blo, psel) : 0u,
                       pb1 =
                           q_ok ? dpct::byte_level_permute(ahi, bhi, psel) : 0u;
        // P V: the dims 16 gr .. + 15 of the keys 2 tq, 2 tq + 1, 2 tq + 8, 2 tq + 9; o[j]: the dims 16 gr + 2 j, + 1
        {
            const KVT* v0 = vg + row(2 * tq);
            const KVT* v1 = vg + row(2 * tq + 1);
            const KVT* v8 = vg + row(2 * tq + 8);
            const KVT* v9 = vg + row(2 * tq + 9);
            if constexpr (F8) {
                const sycl::uint4 r0 = ::ik2_ldg((const sycl::uint4 *)v0),
                                  r1 = ::ik2_ldg((const sycl::uint4 *)v1);
                const sycl::uint4 r8 = ::ik2_ldg((const sycl::uint4 *)v8),
                                  r9 = ::ik2_ldg((const sycl::uint4 *)v9);
                const uint32_t x0[4] = {r0.x(), r0.y(), r0.z(), r0.w()},
                               x1[4] = {r1.x(), r1.y(), r1.z(), r1.w()};
                const uint32_t x8[4] = {r8.x(), r8.y(), r8.z(), r8.w()},
                               x9[4] = {r9.x(), r9.y(), r9.z(), r9.w()};
#pragma unroll
                for (int i = 0; i < 4; ++i) {   // (four bytes: two dims of two keys)
                    uint32_t a[4];
                    e4m3x4_h4(dpct::byte_level_permute(x0[i], x1[i], 0x5140u),
                              a[0], a[1]);
                    e4m3x4_h4(dpct::byte_level_permute(x8[i], x9[i], 0x5140u),
                              a[2], a[3]);
                    mma16816(o[2 * i], a, pb0, pb1);
                    e4m3x4_h4(dpct::byte_level_permute(x0[i], x1[i], 0x7362u),
                              a[0], a[1]);
                    e4m3x4_h4(dpct::byte_level_permute(x8[i], x9[i], 0x7362u),
                              a[2], a[3]);
                    mma16816(o[2 * i + 1], a, pb0, pb1);
                }
            } else {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const sycl::uint4 r0 = ::ik2_ldg((const sycl::uint4 *)v0 +
                                                     i),
                                      r1 = ::ik2_ldg((const sycl::uint4 *)v1 +
                                                     i);
                    const sycl::uint4 r8 = ::ik2_ldg((const sycl::uint4 *)v8 +
                                                     i),
                                      r9 = ::ik2_ldg((const sycl::uint4 *)v9 +
                                                     i);
                    const uint32_t x0[4] = {r0.x(), r0.y(), r0.z(), r0.w()},
                                   x1[4] = {r1.x(), r1.y(), r1.z(), r1.w()};
                    const uint32_t x8[4] = {r8.x(), r8.y(), r8.z(), r8.w()},
                                   x9[4] = {r9.x(), r9.y(), r9.z(), r9.w()};
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        const uint32_t a[4] = {
                            dpct::byte_level_permute(x0[jj], x1[jj], 0x5410u),
                            dpct::byte_level_permute(x0[jj], x1[jj], 0x7632u),
                            dpct::byte_level_permute(x8[jj], x9[jj], 0x5410u),
                            dpct::byte_level_permute(x8[jj], x9[jj], 0x7632u)};
                        mma16816(o[4 * i + jj], a, pb0, pb1);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int sh = 4; sh < 32; sh <<= 1) {
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        la += dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), la,
            sh);
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        lb += dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), lb,
            sh);
    }
    // the warp's partials: o_part [warp][H][D], ml_part [warp][H] = (max in log2 units, sum)
    if (a_ok) {
        float* op = o_part + ((size_t) w * H + g * R + qa) * kFaD + gr * 16;
#pragma unroll
        for (int j = 0; j < kFaD / 16; ++j) * (sycl::float2 *)(op + 2 * j) =
            sycl::float2(o[j][0], o[j][2]);
        if (gr == 0) ml_part[(size_t)w * H + g * R + qa] = sycl::float2(ma, la);
    }
    if (b_ok) {
        float* op = o_part + ((size_t) w * H + g * R + qb) * kFaD + gr * 16;
#pragma unroll
        for (int j = 0; j < kFaD / 16; ++j) * (sycl::float2 *)(op + 2 * j) =
            sycl::float2(o[j][1], o[j][3]);
        if (gr == 0) ml_part[(size_t)w * H + g * R + qb] = sycl::float2(mb, lb);
    }
}
// the warps merged with the sink: out [H][D] (times v_inv: an FP8 cache's V scale; times the sigmoid of gate [H][D]
// if given; out16: an FP16 copy); a block a head, D threads.  A warp's weight (the exp2 of its max less the overall
// one) is found once and kept in shared memory.
__dpct_inline__ void attn_decode_mma_combine_kernel(
    const float *__restrict__ o_part, const sycl::float2 *__restrict__ ml_part,
    int n_split, int H, const float *__restrict__ sink, float v_inv,
    const int *pos_d, int qpos, int window, const float *__restrict__ gate,
    float *out, uint16_t *out16, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto cm_sm = (float *)dpct_local;
    float* red = cm_sm;         // [kFaD] the block's reductions
    float* cw = cm_sm + kFaD;   // [n_split * kWaNW] the warps' weights
    if (pos_d) qpos = *pos_d;
    int k_lo, n_steps;
    const int n_act = wa_active(qpos, window, n_split * kWaNW, k_lo, n_steps);
    const int h = item_ct1.get_group(2), d = item_ct1.get_local_id(2);
    const float sk = sink[h] * kLog2e;
    float m = d == 0 ? sk : -INFINITY;
#pragma unroll
    for (int s = d; s < n_act; s += kFaD)
        m = sycl::fmax(m, ml_part[(size_t)s * H + h].x());
    m = bmax(m, red);
    float den = d == 0 ? sycl::exp2(sk - m) : 0.0f;
    for (int s = d; s < n_act; s += kFaD) {
        const sycl::float2 ml = ml_part[(size_t)s * H + h];
        const float c = ml.x() == -INFINITY ? 0.0f : sycl::exp2(ml.x() - m);
        cw[s] = c;
        den += ml.y() * c;
    }
    den = bsum(den, red);   // (and its barriers: the weights are all there)
    float acc = 0.0f;
#pragma unroll
    for (int s = 0; s < n_act; ++s) acc +=
        o_part[((size_t)s * H + h) * kFaD + d] * cw[s];
    float res = acc / den * v_inv;
    if (gate) res =
        res * (1.0f / (1.0f + sycl::native::exp(-gate[(size_t)h * kFaD + d])));
    out[(size_t) h * kFaD + d] = res;
    if (out16) out16[(size_t) h * kFaD + d] = f2h(res);
}

template <bool WRITE_F16, bool WRITE_F32>
__dpct_inline__ void gate_mul_impl_kernel(float *o, const float *g, size_t n,
                                          uint16_t *o16) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t i =
        (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (((uintptr_t) o & 15) == 0 && ((uintptr_t) g & 15) == 0 && (!WRITE_F16 || ((uintptr_t) o16 & 7) == 0)) {
        const size_t n4 = n >> 2;
        if (i < n4) {
            const sycl::float4 ov = ((const sycl::float4 *)o)[i];
            const sycl::float4 gv = ::ik2_ldg((const sycl::float4 *)g + i);
            const float r0 =
                ov.x() * (1.0f / (1.0f + sycl::native::exp(-gv.x())));
            const float r1 =
                ov.y() * (1.0f / (1.0f + sycl::native::exp(-gv.y())));
            const float r2 =
                ov.z() * (1.0f / (1.0f + sycl::native::exp(-gv.z())));
            const float r3 =
                ov.w() * (1.0f / (1.0f + sycl::native::exp(-gv.w())));
            if (WRITE_F32)((sycl::float4 *)o)[i] = sycl::float4(r0, r1, r2, r3);
            if (WRITE_F16)((sycl::uint2 *)o16)[i] =
                sycl::uint2(f2_to_u32(r0, r1), f2_to_u32(r2, r3));
        }
        const size_t tail = (n4 << 2) + i;
        if (tail < n) {
            const float r =
                o[tail] * (1.0f / (1.0f + sycl::native::exp(-g[tail])));
            if (WRITE_F32) o[tail] = r;
            if (WRITE_F16) o16[tail] = f2h(r);
        }
        return;
    }
    if (i < n) {
        const float r = o[i] * (1.0f / (1.0f + sycl::native::exp(-g[i])));
        if (WRITE_F32) o[i] = r;
        if (WRITE_F16) o16[i] = f2h(r);
    }
}

constexpr int kRouteMax = 16;
// Single-warp zero-barrier router for E <= 512: each lane holds <= 16 experts in registers.
/*
DPCT1110: The total declared local variable size in device function
route_warp_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void route_warp_kernel(const float *logits, const float *bias,
                                       int T, int E, int K, float scale,
                                       int *top, float *w) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(2),
              lane = item_ct1.get_local_id(2); // 32 threads
    const float* l = logits + (size_t) t * E;
    float val[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        const int e = lane + m * 32;
        val[m] = (e < E) ? (1.0f / (1.0f + sycl::native::exp(-ik2_ldg(l + e))) +
                            ik2_ldg(bias + e))
                         : -INFINITY;
    }
    int ch[kRouteMax];
    for (int k = 0; k < K; ++k) {
        float best = -INFINITY;
        int idx = 0x7fffffff;
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            const int e = lane + m * 32;
            if (val[m] > best || (val[m] == best && e < idx)) {
                best = val[m];
                idx = e;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const float ob = dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(),
                best, o);
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const int oi = dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(),
                idx, o);
            if (ob > best || (ob == best && oi < idx)) {
                best = ob;
                idx = oi;
            }
        }
        ch[k] = idx;
        if (idx < E && (idx & 31) == lane) val[idx >> 5] = -INFINITY;
    }
    if (lane == 0) {
        for (int a = 1; a < K; ++a) {   // ascending expert order
            const int x = ch[a];
            int b = a - 1;
#pragma unroll
            for (; b >= 0 && ch[b] > x; --b) ch[b + 1] = ch[b];
            ch[b + 1] = x;
        }
        float sc[kRouteMax], sum = 0.0f;
#pragma unroll
        for (int k = 0; k < K; ++k) sum +=
            (sc[k] = 1.0f / (1.0f + sycl::native::exp(-ik2_ldg(l + ch[k]))));
#pragma unroll
        for (int k = 0; k < K; ++k) {
            top[(size_t) t * K + k] = ch[k];
            w[(size_t) t * K + k] = sc[k] / (sum + 1e-20f) * scale;
        }
    }
}

// a block a token: the scores in shared memory, then K rounds of the block's argmax (ties: the lower expert)
/*
DPCT1110: The total declared local variable size in device function
route_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void route_kernel(const float *logits, const float *bias, int T,
                                  int E, int K, float scale, int *top, float *w,
                                  uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto v = (float *)dpct_local; // [E]
    auto &bv = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &bi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &ch =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[kRouteMax]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_group(2), tid = item_ct1.get_local_id(2),
              lane = tid & 31, warp = tid >> 5,
              nw = item_ct1.get_local_range(2) >> 5;
    const float* l = logits + (size_t) t * E;
#pragma unroll
    for (int e = tid; e < E; e += item_ct1.get_local_range(2))
        v[e] = 1.0f / (1.0f + sycl::native::exp(-l[e])) + bias[e];
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int k = 0; k < K; ++k) {
        float best = -INFINITY;
        int idx = 0x7fffffff;
#pragma unroll
        for (int e = tid; e < E; e += item_ct1.get_local_range(2))
            if (v[e] > best || (v[e] == best && e < idx)) best = v[e], idx = e;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const float ob = dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(),
                best, o);
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const int oi = dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(),
                idx, o);
            if (ob > best || (ob == best && oi < idx)) best = ob, idx = oi;
        }
        if (lane == 0) bv[warp] = best, bi[warp] = idx;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        if (tid == 0) {
            float b = bv[0];
            int i = bi[0];
#pragma unroll
            for (int j = 1; j < nw; ++j)
                if (bv[j] > b || (bv[j] == b && bi[j] < i)) b = bv[j], i = bi[j];
            ch[k] = i;
            v[i] = -INFINITY;
        }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    if (tid == 0) {
        int c[kRouteMax];
#pragma unroll
        for (int k = 0; k < K; ++k) c[k] = ch[k];
        for (int a = 1; a < K; ++a) {   // ascending expert order
            const int x = c[a];
            int b = a - 1;
#pragma unroll
            for (; b >= 0 && c[b] > x; --b) c[b + 1] = c[b];
            c[b + 1] = x;
        }
        float sc[kRouteMax], sum = 0.0f;
#pragma unroll
        for (int k = 0; k < K; ++k) sum +=
            (sc[k] = 1.0f / (1.0f + sycl::native::exp(-l[c[k]])));
#pragma unroll
        for (int k = 0; k < K; ++k) {
            top[(size_t) t * K + k] = c[k];
            w[(size_t) t * K + k] = sc[k] / (sum + 1e-20f) * scale;
        }
    }
}

__dpct_inline__ void gather_f16_kernel(const float *x, const int *idx, int n,
                                       int C, uint16_t *y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t i =
        (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= (size_t) n * C) return;
    y[i] = f2h(x[(size_t) idx[i / C] * C + i % C]);
}

__dpct_inline__ void swiglu_cap_kernel(const uint16_t *gu, int n, int F,
                                       const float *sg, const float *su,
                                       float gcap, float ucap, uint16_t *h) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t i =
        (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= (size_t) n * F) return;
    const size_t r = i / F, c = i % F;
    const float g = h2f(gu[r * 2 * F + c]) * sg[r];
    const float u = h2f(gu[r * 2 * F + F + c]) * su[r];
    h[i] = f2h(gcap * sycl::tanh(g / gcap) *
               (1.0f / (1.0f + sycl::native::exp(-g))) * ucap *
               sycl::tanh(u / ucap));
}

__dpct_inline__ void moe_combine_kernel(const uint16_t *y, const int *inv,
                                        const float *wt, int T, int K, int C,
                                        float *out, bool add) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t i =
        (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= (size_t) T * C) return;
    const size_t t = i / C, c = i % C;
    float a = 0.0f;
#pragma unroll
    for (int k = 0; k < K; ++k) a +=
        wt[t * K + k] * h2f(y[(size_t)inv[t * K + k] * C + c]);
    out[i] = add ? out[i] + a : a;
}

__dpct_inline__ void swiglu_cap_f32_kernel(
    const float *gu, int n, int F, const float *sg, const float *su, float gcap,
    float ucap, uint16_t *h) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t i =
        (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= (size_t) n * F) return;
    const size_t r = i / F, c = i % F;
    const float g = gu[r * 2 * F + c] * sg[r];
    const float u = gu[r * 2 * F + F + c] * su[r];
    h[i] = f2h(gcap * sycl::tanh(g / gcap) *
               (1.0f / (1.0f + sycl::native::exp(-g))) * ucap *
               sycl::tanh(u / ucap));
}

// one token's moe_combine_f32 and the conv4_res after it in one launch: out = conv(sum_k wt[k] y[inv[k]]) + res
__dpct_inline__ void moe_combine_conv_kernel(
    const float *__restrict__ y, const int *__restrict__ inv,
    const float *__restrict__ wt, int K, int C, float *hist,
    const float *__restrict__ w, const float *res, float *out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= C) return;
    float a = 0.0f;
#pragma unroll
    for (int k = 0; k < K; ++k) a += wt[k] * y[(size_t)inv[k] * C + c];
    const float h0 = hist[c], h1 = hist[(size_t) C + c], h2 = hist[(size_t) 2 * C + c];
    const float* wc = w + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    float b = w0 * h0 + w1 * h1 + w2 * h2 + w3 * a;
    if (res) b += res[c];
    out[c] = b;
    hist[c] = h1;
    hist[(size_t) C + c] = h2;
    hist[(size_t) 2 * C + c] = a;
}
__dpct_inline__ void moe_combine_f32_kernel(const float *y, const int *inv,
                                            const float *wt, int T, int K,
                                            int C, float *out, bool add) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t i =
        (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= (size_t) T * C) return;
    const size_t t = i / C, c = i % C;
    float a = 0.0f;
#pragma unroll
    for (int k = 0; k < K; ++k) a +=
        wt[t * K + k] * y[(size_t)inv[t * K + k] * C + c];
    out[i] = add ? out[i] + a : a;
}

__dpct_inline__ void add_kernel(const float *a, const float *b, size_t n,
                                float *y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t i =
        (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) y[i] = a[i] + b[i];
}
__dpct_inline__ void to_f16_kernel(const float *x, size_t n, uint16_t *y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t i =
        (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) y[i] = f2h(x[i]);
}

__dpct_inline__ void to_f16_limbs_kernel(const float *x, size_t n,
                                         uint16_t *y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t i =
        (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n) return;
    const uint16_t hi = f2h(x[i]);
    y[i] = hi, y[n + i] = f2h(x[i] - h2f(hi));
}
__dpct_inline__ void softcap_kernel(float *x, size_t n, float cap) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t i =
        (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) x[i] = cap * sycl::tanh(x[i] / cap);
}

// The loop's message gates, up to three (a layer's q, k and v inputs) in two launches that spread over the SMs.
// sc [n][T][2 R + 2]: per gate and token, the two halves of each g before the norms (D's x columns . x, its m columns
// . m), then the two sums of squares.
struct LoopGateArgs {
    const float *x[3], *m[3];
    const void *D[3], *F[3], *G[3];
    float* y[3];
};
constexpr int kLoopRankMax = 128;
template <bool Compact>
__dpct_inline__ float gate_weight(const void *p, size_t i) {
    if constexpr (Compact)
        return sycl::bit_cast<float>((uint32_t)((const uint16_t *)p)[i] << 16);
    else
        return ((const float*)p)[i];
}

// (1) a block a rank (128 threads): its two dots; rank 0's block also stores the sums of squares
template <bool Compact>
__dpct_inline__ void loop_gate_dots_kernel(LoopGateArgs a, int T, int C, int R,
                                           float *sc) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sh = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[4][4]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int r = item_ct1.get_group(2), j = item_ct1.get_group(1),
              t = item_ct1.get_group(0), tid = item_ct1.get_local_id(2),
              warp = tid >> 5;
    const float* xr = a.x[j] + (size_t) t * C;
    const float* mr = a.m[j] + (size_t) t * C;
    const size_t row = (size_t)r * 2 * C;
    float ax = 0.0f, am = 0.0f, sx = 0.0f, sm = 0.0f;
    for (int c = tid; c < C; c += 128) {
        const float xv = xr[c], mv = mr[c];
        ax += gate_weight<Compact>(a.D[j], row + c) * xv, am += gate_weight<Compact>(a.D[j], row + C + c) * mv;
        sx += xv * xv, sm += mv * mv;
    }
    ax = wsum(ax), am = wsum(am), sx = wsum(sx), sm = wsum(sm);
    if ((tid & 31) == 0) sh[warp][0] = ax, sh[warp][1] = am, sh[warp][2] = sx, sh[warp][3] = sm;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (tid == 0) {
        float* o = sc + ((size_t) j * T + t) * (2 * R + 2);
        o[2 * r] = sh[0][0] + sh[1][0] + sh[2][0] + sh[3][0];
        o[2 * r + 1] = sh[0][1] + sh[1][1] + sh[2][1] + sh[3][1];
        if (r == 0) {
            o[2 * R] = sh[0][2] + sh[1][2] + sh[2][2] + sh[3][2];
            o[2 * R + 1] = sh[0][3] + sh[1][3] + sh[2][3] + sh[3][3];
        }
    }
}
// (2) y = (1 - tanh(F g)) x + tanh(G g) m: a block is kBlk channels of a token, a thread a channel; F and G by rank
// ([R][C]), so that a warp's threads read side by side.  y may be x (x's sums were taken in (1)).
template <int Rank, bool Compact>
__dpct_inline__ void loop_gate_apply_kernel(LoopGateArgs a, int T, int C, int R,
                                            float eps,
                                            const float *__restrict__ sc) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sh_g =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<float[kLoopRankMax]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int j = item_ct1.get_group(1), t = item_ct1.get_group(0),
              tid = item_ct1.get_local_id(2);
    const float* o = sc + ((size_t) j * T + t) * (2 * R + 2);
    for (int i = tid; i < R; i += item_ct1.get_local_range(2)) {
        const float rx = sycl::rsqrt(o[2 * R] / (float)C + eps),
                    rm = sycl::rsqrt(o[2 * R + 1] / (float)C + eps);
        sh_g[i] = rx * o[2 * i] + rm * o[2 * i + 1];
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int c =
        (int)item_ct1.get_group(2) * item_ct1.get_local_range(2) + tid;
    if (c >= C) return;

    float f = 0.0f, gn = 0.0f;
    for (int r = 0; r < (Rank ? Rank : R); ++r) {
        const float gv = sh_g[r];
        f += gate_weight<Compact>(a.F[j], (size_t)r * C + c) * gv,
            gn += gate_weight<Compact>(a.G[j], (size_t)r * C + c) * gv;
    }
    const size_t i = (size_t) t * C + c;
    a.y[j][i] = (1.0f - sycl::tanh(f)) * a.x[j][i] + sycl::tanh(gn) * a.m[j][i];
}
// ---- sampling: the largest logits found on the device - the host gets a thousand or two candidates and not the
// vocabulary (copying 200,000 logits out and sorting them was 0.8 ms a token).  A histogram of the values a block,
// the bin from which the bins hold K values at least, the values of those bins gathered.
constexpr int kCandBins = 1024;
__dpct_inline__ int cand_bin(float x, float lo, float inv_w) {
    return sycl::min(kCandBins - 1, sycl::max(0, (int)((x - lo) * inv_w)));
}
__dpct_inline__ void cand_hist_kernel(const float *__restrict__ x, int n,
                                      float lo, float inv_w, int per,
                                      int *__restrict__ part) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &h = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[kCandBins]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
#pragma unroll
    for (int j = item_ct1.get_local_id(2); j < kCandBins;
         j += item_ct1.get_local_range(2)) h[j] = 0;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const int i0 = item_ct1.get_group(2) * per, i1 = sycl::min(n, i0 + per);
#pragma unroll
    for (int i = i0 + item_ct1.get_local_id(2); i < i1;
         i += item_ct1.get_local_range(2))
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &h[cand_bin(x[i], lo, inv_w)], 1);
    item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int j = item_ct1.get_local_id(2); j < kCandBins;
         j += item_ct1.get_local_range(2))
        part[(size_t)item_ct1.get_group(2) * kCandBins + j] = h[j];
}
// a thread a bin (kCandBins threads): head[1] = the bin from which the bins hold K values (0 if they all hold
// fewer), head[0] = 0 (the gathered count starts there)
__dpct_inline__ void cand_threshold_kernel(const int *__restrict__ part,
                                           int n_part, int K, int *head) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &cnt =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<int[kCandBins]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &wcnt = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        int[kCandBins / 32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int j = item_ct1.get_local_id(2), lane = j & 31, warp = j >> 5;
    int c = 0;
#pragma unroll
    for (int b = 0; b < n_part; ++b) c += part[(size_t)b * kCandBins + j];
    cnt[j] = c;
    int ws = c;
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) ws +=
        dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), ws,
            o);
    if (lane == 0) wcnt[warp] = ws;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    int above = 0;   // the values of the bins over this one
#pragma unroll
    for (int w = warp + 1; w < kCandBins / 32; ++w) above += wcnt[w];
#pragma unroll
    for (int i = j + 1; i < (warp + 1) * 32; ++i) above += cnt[i];
    if (above < K && above + c >= K) head[1] = j;
    if (j == 0) {
        head[0] = 0;
        if (above + c < K) head[1] = 0;
    }
}
__dpct_inline__ void cand_gather_kernel(const float *__restrict__ x, int n,
                                        float lo, float inv_w, int cap,
                                        int *head, int *idx, float *val) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= n) return;
    const float v = x[i];
    if (cand_bin(v, lo, inv_w) < head[1]) return;
    const int at =
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(head,
                                                                           1);
    if (at < cap) idx[at] = i, val[at] = v;
}

// The gates of a chunk of tokens: their two products (g = D [rms(x); rms(m)], then F g and G g) are matrix products
// taken outside, and these are what is left.  (1) g [T][R] = a1 / rms(x) + a2 / rms(m), a1 = x D_x^T and
// a2 = m D_m^T; a block a token.
__dpct_inline__ void loop_gate_mix_kernel(
    const float *__restrict__ x, const float *__restrict__ m,
    const float *__restrict__ a1, const float *__restrict__ a2, int C, int R,
    float eps, float *__restrict__ g) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sh = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[kBlk]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_group(2), tid = item_ct1.get_local_id(2);
    const float* xr = x + (size_t) t * C;
    const float* mr = m + (size_t) t * C;
    float sx = 0.0f, sm = 0.0f;
    for (int c = tid; c < C; c += kBlk) {
        const float xv = xr[c], mv = mr[c];
        sx += xv * xv, sm += mv * mv;
    }
    sx = bsum(sx, sh), sm = bsum(sm, sh);
    if (tid < R) {
        const float rx = sycl::rsqrt(sx / (float)C + eps),
                    rm = sycl::rsqrt(sm / (float)C + eps);
        g[(size_t) t * R + tid] = rx * a1[(size_t) t * R + tid] + rm * a2[(size_t) t * R + tid];
    }
}
// (2) y = (1 - tanh f) x + tanh(gn) m, f = g F and gn = g G; y may be x
__dpct_inline__ void loop_gate_blend_kernel(
    const float *x, const float *m, const float *__restrict__ f,
    const float *__restrict__ gn, size_t n, float *y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t i =
        (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) y[i] =
        (1.0f - sycl::tanh(f[i])) * x[i] + sycl::tanh(gn[i]) * m[i];
}

// ---- the decodes: value v of a run of blocks, ggml's dequantize_row_* arithmetic

__dpct_inline__ float ue4m3(
    uint8_t x) { // ggml_ue4m3_to_fp32 (branchless bit-ops)
    const uint32_t e = (x >> 3) & 0xFu;
    const uint32_t m = x & 7u;
    const float norm_f = bits_to_float(((e + 119u) << 23) | (m << 20));
    const float sub_f = (float) m * 0x1.0p-10f;
    const float v = (e == 0u) ? sub_f : norm_f;
    return (x == 0x7Fu) ? 0.0f : v;
}
// IQ3_XXS / IQ4_XS tables: a device copy (read with __ldg: lanes index them divergently) and a host copy.
inline dpct::global_memory<const uint32_t, 1> d_iq3xxs_grid(sycl::range(256),
                                                            {IK2_IQ3XXS_GRID});
static const uint32_t h_iq3xxs_grid[256] = {IK2_IQ3XXS_GRID};
inline dpct::global_memory<const int8_t, 1> d_iq4nl(sycl::range(16),
                                                    {IK2_IQ4NL_VALUES});
static const int8_t h_iq4nl[16] = {IK2_IQ4NL_VALUES};
__dpct_inline__ uint32_t iq3xxs_grid_at(int i, const uint32_t *d_iq3xxs_grid) {

    return ik2_ldg(d_iq3xxs_grid + i);
}
__dpct_inline__ uint32_t iq3xxs_grid_at_host_ct1(int i) {

    return h_iq3xxs_grid[i];
}
__dpct_inline__ int iq4nl_at(int i, const int8_t *d_iq4nl) {

    return d_iq4nl[i];
}
__dpct_inline__ int iq4nl_at_host_ct5(int i) {

    return h_iq4nl[i];
}
// ggml's ksigns_iq2xs[s]: the 7 sign bits plus an 8th that makes the count of set bits even
__dpct_inline__ uint32_t iq_signs8(uint32_t s7) {

    return s7 | ((uint32_t)(sycl::popcount(s7) & 1) << 7);
}
inline __dpct_inline__ uint32_t iq_signs8_host_ct0(uint32_t s7) {

    uint32_t p = s7 ^ (s7 >> 4); // (the parity by folding: no GNU builtin, no
                                 // C++20 <bit> - MSVC builds this too)
    p ^= p >> 2;
    p ^= p >> 1;
    return s7 | (p & 1) << 7;
}

/*
DPCT1110: The total declared local variable size in device function
decode_value exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline float decode_value(int type, const uint8_t *blocks, size_t v,
                          const uint32_t *d_iq3xxs_grid,
                          const int8_t *d_iq4nl) {
    if (type == kNVFP4) {
        const int8_t kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
        const uint8_t* b = blocks + (v / 64) * 36;
        const int i = (int) (v % 64), sb = i / 16, j = i % 16;
        const uint8_t byte = b[4 + sb * 8 + (j & 7)];
        return (float) kv[j < 8 ? (byte & 0xF) : (byte >> 4)] * ue4m3(b[sb]);
    }
    if (type == kQ8_0) {
        const uint8_t* b = blocks + (v / 32) * 34;
        const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
        return d * (float) (int8_t) b[2 + v % 32];
    }
    if (type == kQ6_K) {   // ql [128], qh [64], scales int8 [16], d
        const uint8_t* b = blocks + (v / 256) * 210;
        const int i = (int) (v % 256), n = i / 128, lq = i % 128, qd = lq / 32, l = lq % 32;
        const uint8_t* ql = b + n * 64;
        const uint8_t* qh = b + 128 + n * 32;
        const int8_t* sc = (const int8_t*) b + 192 + n * 8;
        const float d = half_bits_to_float((uint16_t) (b[208] | (b[209] << 8)));
        const uint8_t lo = ql[l + ((qd & 1) ? 32 : 0)];
        const int nib = qd >= 2 ? (lo >> 4) : (lo & 0xF);
        const int q = (int8_t) (nib | (((qh[l] >> (2 * qd)) & 3) << 4)) - 32;
        return d * (float) sc[l / 16 + 2 * qd] * (float) q;
    }
    if (type == kIQ3_XXS) {   // d, qs [64] grid indices, [8] u32 scale (top 4 bits) + 4 x 7 sign bits
        const uint8_t* b = blocks + (v / 256) * 98;
        const int i = (int) (v % 256), ib32 = i / 32, l = (i % 32) / 8, j = i % 8;
        const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
        const uint8_t* q = b + 2 + 64 + 4 * ib32;
        const uint32_t aux = (uint32_t) q[0] | ((uint32_t) q[1] << 8) | ((uint32_t) q[2] << 16) | ((uint32_t) q[3] << 24);
        const float db = d * (0.5f + (float) (aux >> 28)) * 0.5f;
        const uint32_t signs = iq_signs8((aux >> (7 * l)) & 127u);
        const uint32_t g =
            iq3xxs_grid_at(b[2 + 8 * ib32 + 2 * l + (j >> 2)], d_iq3xxs_grid);
        const float w = (float) ((g >> (8 * (j & 3))) & 0xFFu);
        return db * w * ((signs >> j) & 1u ? -1.0f : 1.0f);
    }
    if (type == kIQ4_XS) {   // d, scales_h, scales_l [4], qs [128]
        const uint8_t* b = blocks + (v / 256) * 136;
        const int i = (int) (v % 256), ib = i / 32, j = i % 32;
        const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
        const uint32_t sh = (uint32_t) b[2] | ((uint32_t) b[3] << 8);
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (int) (((sh >> (2 * ib)) & 3u) << 4);
        const uint8_t q = b[8 + 16 * ib + (j & 15)];
        return d * (float)(ls - 32) *
               (float)iq4nl_at(j < 16 ? (q & 0xF) : (q >> 4), d_iq4nl);
    }
    if (type == kQ5_K) {   // d, dmin, scales [12], qh [32], qs [128]
        const uint8_t* b = blocks + (v / 256) * 176;
        const int i = (int) (v % 256), j64 = i / 64, half = (i % 64) / 32, l = i % 32;
        const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
        const float mn = half_bits_to_float((uint16_t) (b[2] | (b[3] << 8)));
        const uint8_t* q = b + 4;
        const int is = 2 * j64 + half;
        int sc, m;
        if (is < 4) {
            sc = q[is] & 63, m = q[is + 4] & 63;
        } else {
            sc = (q[is + 4] & 0xF) | ((q[is - 4] >> 6) << 4), m = (q[is + 4] >> 4) | ((q[is] >> 6) << 4);
        }
        const uint8_t lo = b[48 + 32 * j64 + l];
        const int hb = (b[16 + l] >> (2 * j64 + half)) & 1;
        const int val = (half ? (lo >> 4) : (lo & 0xF)) + (hb ? 16 : 0);
        return d * (float) sc * (float) val - mn * (float) m;
    }
    return 0.0f;
}

// The input vector in shared memory, padded: value i at xp(i) = (i >> 5) * 36 + (i & 31), so every 16-float
// NVFP4 sub-block and 32-float Q8_0 block is 16-byte (float4 / LDS.128) aligned with 0 bank conflicts.
__dpct_inline__ int xp(int i) { return (i >> 5) * 36 + (i & 31); }
constexpr size_t xp_floats(int K) {
    return K <= 0 ? 0 : (size_t) ((K - 1) >> 5) * 36 + ((K - 1) & 31) + 1;
}

// Fast register E2M1 pair decode via FP16 high-byte LUT in __byte_perm -> __half22float2:
// FP16 representations of {0, 1, 2, 3, 4, 6, 8, 12} all have low byte 0x00 and high bytes {0x00, 0x3c, 0x40, 0x42, 0x44, 0x46, 0x48, 0x4a}.
__dpct_inline__ float ik2_e2m1(uint32_t nib) {   // (Intel) one E2M1 nibble, doubled, as a float
    const float v = (float) ((0xC8643210u >> ((nib & 7u) * 4u)) & 15u);
    return sycl::bit_cast<float>(sycl::bit_cast<uint32_t>(v) | ((nib & 8u) << 28));
}
__dpct_inline__ sycl::float2 e2m1_pair_lo(uint32_t s) { return sycl::float2(ik2_e2m1(s), ik2_e2m1(s >> 8)); }        // nibbles 0, 2
__dpct_inline__ sycl::float2 e2m1_pair_hi(uint32_t s) { return sycl::float2(ik2_e2m1(s >> 4), ik2_e2m1(s >> 12)); }   // nibbles 1, 3

// Four signed FP16 high bytes in the order of the serialized E2M1 nibbles.
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP == 860
__device__ __forceinline__ uint32_t decode_signed_nibbles(uint32_t nib) {
    const uint32_t p = dpct::byte_level_permute(0x42403c00u, 0x4a484644u, nib);
    const uint32_t n = dpct::byte_level_permute(0xc2c0bc80u, 0xcac8c6c4u, nib);
    return dpct::byte_level_permute(p, n, 0x3210u | ((nib & 0x8888u) >> 1));
}
__device__ __forceinline__ float nvfp4_sub_dot(uint32_t q0, uint32_t q1, float4 x0, float4 x1, float4 x2, float4 x3) {
    const uint32_t a = decode_signed_nibbles(q0), b = decode_signed_nibbles(q0 >> 16);
    const uint32_t c = decode_signed_nibbles(q1), d = decode_signed_nibbles(q1 >> 16);

    const auto v01 = u32_to_f2(dpct::byte_level_permute(0, a, 0x6040)), v23 = u32_to_f2(dpct::byte_level_permute(0, b, 0x6040));
    const auto v45 = u32_to_f2(dpct::byte_level_permute(0, c, 0x6040)), v67 = u32_to_f2(dpct::byte_level_permute(0, d, 0x6040));
    const auto v89 = u32_to_f2(dpct::byte_level_permute(0, a, 0x7050)), vAB = u32_to_f2(dpct::byte_level_permute(0, b, 0x7050));
    const auto vCD = u32_to_f2(dpct::byte_level_permute(0, c, 0x7050)), vEF = u32_to_f2(dpct::byte_level_permute(0, d, 0x7050));
    return (v01.x * x0.x + v01.y * x0.y + v23.x * x0.z + v23.y * x0.w) +
           (v45.x * x1.x + v45.y * x1.y + v67.x * x1.z + v67.y * x1.w) +
           (v89.x * x2.x + v89.y * x2.y + vAB.x * x2.z + vAB.y * x2.w) +
           (vCD.x * x3.x + vCD.y * x3.y + vEF.x * x3.z + vEF.y * x3.w);
}
#else
__dpct_inline__ float nvfp4_sub_dot(uint32_t q0, uint32_t q1, sycl::float4 x0,
                                    sycl::float4 x1, sycl::float4 x2,
                                    sycl::float4 x3) {
    const sycl::float2 w01 = e2m1_pair_lo(q0), w23 = e2m1_pair_lo(q0 >> 16);
    const sycl::float2 w45 = e2m1_pair_lo(q1), w67 = e2m1_pair_lo(q1 >> 16);
    const sycl::float2 w89 = e2m1_pair_hi(q0), wAB = e2m1_pair_hi(q0 >> 16);
    const sycl::float2 wCD = e2m1_pair_hi(q1), wEF = e2m1_pair_hi(q1 >> 16);
    return (w01.x() * x0.x() + w01.y() * x0.y() + w23.x() * x0.z() +
            w23.y() * x0.w()) +
           (w45.x() * x1.x() + w45.y() * x1.y() + w67.x() * x1.z() +
            w67.y() * x1.w()) +
           (w89.x() * x2.x() + w89.y() * x2.y() + wAB.x() * x2.z() +
            wAB.y() * x2.w()) +
           (wCD.x() * x3.x() + wCD.y() * x3.y() + wEF.x() * x3.z() +
            wEF.y() * x3.w());
}
#endif

__dpct_inline__ float dot_u32_f4(uint32_t q4, sycl::float4 f) {
    return (float)(int8_t)(q4 & 0xFFu) * f.x() +
           (float)(int8_t)((q4 >> 8) & 0xFFu) * f.y() +
           (float)(int8_t)((q4 >> 16) & 0xFFu) * f.z() +
           (float)(int8_t)(q4 >> 24) * f.w();
}

__dpct_inline__ float q8_blk0_dot(
    const uint32_t *p, sycl::float4 f0, sycl::float4 f1, sycl::float4 f2,
    sycl::float4 f3, sycl::float4 f4, sycl::float4 f5, sycl::float4 f6,
    sycl::float4 f7, float &d1_out) {
    const uint32_t u0 = ik2_ldg(p + 0), u1 = ik2_ldg(p + 1), u2 = ik2_ldg(p + 2), u3 = ik2_ldg(p + 3), u4 = ik2_ldg(p + 4);
    const uint32_t u5 = ik2_ldg(p + 5), u6 = ik2_ldg(p + 6), u7 = ik2_ldg(p + 7), u8 = ik2_ldg(p + 8);
    const float d0 = half_bits_to_float((uint16_t) (u0 & 0xFFFFu));
    float a0 = (float)(int8_t)((u0 >> 16) & 0xFFu) * f0.x() +
               (float)(int8_t)(u0 >> 24) * f0.y();
    a0 += dot_u32_f4(u1, sycl::float4(f0.z(), f0.w(), f1.x(), f1.y()));
    a0 += dot_u32_f4(u2, sycl::float4(f1.z(), f1.w(), f2.x(), f2.y()));
    a0 += dot_u32_f4(u3, sycl::float4(f2.z(), f2.w(), f3.x(), f3.y()));
    a0 += dot_u32_f4(u4, sycl::float4(f3.z(), f3.w(), f4.x(), f4.y()));
    a0 += dot_u32_f4(u5, sycl::float4(f4.z(), f4.w(), f5.x(), f5.y()));
    a0 += dot_u32_f4(u6, sycl::float4(f5.z(), f5.w(), f6.x(), f6.y()));
    a0 += dot_u32_f4(u7, sycl::float4(f6.z(), f6.w(), f7.x(), f7.y()));
    a0 += (float)(int8_t)(u8 & 0xFFu) * f7.z() +
          (float)(int8_t)((u8 >> 8) & 0xFFu) * f7.w();
    d1_out = half_bits_to_float((uint16_t) (u8 >> 16));
    return d0 * a0;
}

__dpct_inline__ float q8_blk1_dot(
    const uint32_t *p, float d1, sycl::float4 g0, sycl::float4 g1,
    sycl::float4 g2, sycl::float4 g3, sycl::float4 g4, sycl::float4 g5,
    sycl::float4 g6, sycl::float4 g7) {
    return d1 * (dot_u32_f4(ik2_ldg(p + 9), g0) + dot_u32_f4(ik2_ldg(p + 10), g1) + dot_u32_f4(ik2_ldg(p + 11), g2) +
                 dot_u32_f4(ik2_ldg(p + 12), g3) + dot_u32_f4(ik2_ldg(p + 13), g4) + dot_u32_f4(ik2_ldg(p + 14), g5) +
                 dot_u32_f4(ik2_ldg(p + 15), g6) + dot_u32_f4(ik2_ldg(p + 16), g7));
}


// ---- IQ3_XXS / IQ4_XS: one 32-value sub-block against 32 inputs (xv: 8 float4, the padded xs of that sub-block)

__dpct_inline__ float sdot4_u8(uint32_t g, uint32_t s,
                               sycl::float4 x) { // bytes of g, sign bit k of s
    return sycl::bit_cast<float, int>(
               sycl::bit_cast<int>((float)(g & 0xFFu) * x.x()) ^
               ((s & 1u) << 31)) +
           sycl::bit_cast<float, int>(
               sycl::bit_cast<int>((float)((g >> 8) & 0xFFu) * x.y()) ^
               (((s >> 1) & 1u) << 31)) +
           sycl::bit_cast<float, int>(
               sycl::bit_cast<int>((float)((g >> 16) & 0xFFu) * x.z()) ^
               (((s >> 2) & 1u) << 31)) +
           sycl::bit_cast<float, int>(
               sycl::bit_cast<int>((float)(g >> 24) * x.w()) ^
               (((s >> 3) & 1u) << 31));
}
// block b (98 bytes, 2-aligned), sub-block ib32
__dpct_inline__ float iq3xxs_sub_dot(const uint8_t *b, int ib32,
                                     const sycl::float4(&xv)[8],
                                     const uint32_t *d_iq3xxs_grid) {
    const float d = half_bits_to_float(ik2_ldg((const uint16_t*) b));
    const uint16_t* qs = (const uint16_t*) (b + 2 + 8 * ib32);
    const uint16_t* sp = (const uint16_t*) (b + 66 + 4 * ib32);
    const uint32_t aux = (uint32_t) ik2_ldg(sp) | ((uint32_t) ik2_ldg(sp + 1) << 16);
    float a = 0.0f;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t qq = ik2_ldg(qs + l);
        const uint32_t g1 = ik2_ldg(d_iq3xxs_grid + (qq & 0xFFu)), g2 = ik2_ldg(d_iq3xxs_grid + (qq >> 8));
        const uint32_t sg = iq_signs8((aux >> (7 * l)) & 127u);
        a += sdot4_u8(g1, sg, xv[2 * l]) + sdot4_u8(g2, sg >> 4, xv[2 * l + 1]);
    }
    return d * (0.5f + (float) (aux >> 28)) * 0.5f * a;
}
// IQ4_NL values of the 4 low (hi = false) or high nibbles of the bytes of w, as 4 int8 in a u32
__dpct_inline__ uint32_t iq4nl_lut4(uint32_t w, bool hi) {
    const uint32_t t = (hi ? (w >> 4) : w) & 0x0F0F0F0Fu;
    const uint32_t x = t | (t >> 4);
    const uint32_t sel = (x & 0xFFu) | ((x >> 8) & 0xFF00u);
    const uint32_t T0 = 0xBFAD9881u, T1 = 0xF6EADDCFu, T2 = 0x26190D01u, T3 = 0x71594535u;
    const uint32_t lo = dpct::byte_level_permute(T0, T1, sel & 0x7777u),
                   hv = dpct::byte_level_permute(T2, T3, sel & 0x7777u);
    const uint32_t m = ((t >> 3) & 0x01010101u) * 0xFFu;
    return (lo & ~m) | (hv & m);
}
// block b (136 bytes, 4-aligned), sub-block ib
__dpct_inline__ float iq4xs_sub_dot(const uint8_t *b, int ib,
                                    const sycl::float4(&xv)[8]) {
    const uint32_t h = ik2_ldg((const uint32_t*) b), sl = ik2_ldg((const uint32_t*) (b + 4));
    const float d = half_bits_to_float((uint16_t) (h & 0xFFFFu));
    const int ls = (int) ((sl >> (4 * ib)) & 0xFu) | (int) (((h >> (16 + 2 * ib)) & 3u) << 4);
    const uint32_t* q = (const uint32_t*) (b + 8 + 16 * ib);
    float a = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t w = ik2_ldg(q + k);
        a += dot_u32_f4(iq4nl_lut4(w, false), xv[k]) + dot_u32_f4(iq4nl_lut4(w, true), xv[4 + k]);
    }
    return d * (float) (ls - 32) * a;
}
__dpct_inline__ void load_xv8(const float *x, sycl::float4(&xv)[8]) {
#pragma unroll
    for (int k = 0; k < 8; ++k) xv[k] = *(const sycl::float4 *)(x + 4 * k);
}

// a warp's dot of one row of K values (blocks of `type` from w) with xs (shared, padded: xp), every lane gets it
/*
DPCT1110: The total declared local variable size in device function row_dot
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
inline float row_dot(int type, const uint8_t *w, const float *xs, int K,
                     const uint32_t *d_iq3xxs_grid, const int8_t *d_iq4nl) {
    const int lane =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) &
        31;
    float acc = 0.0f;
    if (type == kNVFP4) {
        const int nsub = K / 16;
        for (int sb = lane; sb < nsub; sb += 32) {
            const uint8_t* b = w + (size_t) (sb >> 2) * 36;
            const int sub = sb & 3;
            const float d = ue4m3((uint8_t) (ik2_ldg((const uint32_t*) b) >> (8 * sub)));
            const uint32_t q0 = ik2_ldg((const uint32_t*) (b + 4 + sub * 8));
            const uint32_t q1 = ik2_ldg((const uint32_t*) (b + 8 + sub * 8));
            const float* x = xs + (sb >> 1) * 36 + (sb & 1) * 16;
            const sycl::float4 x0 = *(const sycl::float4 *)(x + 0),
                               x1 = *(const sycl::float4 *)(x + 4);
            const sycl::float4 x2 = *(const sycl::float4 *)(x + 8),
                               x3 = *(const sycl::float4 *)(x + 12);
            acc += d * nvfp4_sub_dot(q0, q1, x0, x1, x2, x3);
        }
    } else if (type == kQ8_0) {
        const int nb = K / 32;
        if (((uintptr_t) w & 3) == 0) {
            for (int bi = 2 * lane; bi + 1 < nb; bi += 64) {
                const uint32_t* p = (const uint32_t*) (w + (size_t) bi * 34);
                const float* x0 = xs + bi * 36;
                const sycl::float4 f0 = *(const sycl::float4 *)(x0 + 0),
                                   f1 = *(const sycl::float4 *)(x0 + 4);
                const sycl::float4 f2 = *(const sycl::float4 *)(x0 + 8),
                                   f3 = *(const sycl::float4 *)(x0 + 12);
                const sycl::float4 f4 = *(const sycl::float4 *)(x0 + 16),
                                   f5 = *(const sycl::float4 *)(x0 + 20);
                const sycl::float4 f6 = *(const sycl::float4 *)(x0 + 24),
                                   f7 = *(const sycl::float4 *)(x0 + 28);
                float d1;
                acc += q8_blk0_dot(p, f0, f1, f2, f3, f4, f5, f6, f7, d1);
                const sycl::float4 g0 = *(const sycl::float4 *)(x0 + 36),
                                   g1 = *(const sycl::float4 *)(x0 + 40);
                const sycl::float4 g2 = *(const sycl::float4 *)(x0 + 44),
                                   g3 = *(const sycl::float4 *)(x0 + 48);
                const sycl::float4 g4 = *(const sycl::float4 *)(x0 + 52),
                                   g5 = *(const sycl::float4 *)(x0 + 56);
                const sycl::float4 g6 = *(const sycl::float4 *)(x0 + 60),
                                   g7 = *(const sycl::float4 *)(x0 + 64);
                acc += q8_blk1_dot(p, d1, g0, g1, g2, g3, g4, g5, g6, g7);
            }
            if (nb & 1) {
                const int bi = nb - 1;
                const uint8_t* b = w + (size_t) bi * 34;
                const float d = half_bits_to_float((uint16_t) (ik2_ldg(b) | (ik2_ldg(b + 1) << 8)));
                acc += d * (float) (int8_t) ik2_ldg(b + 2 + lane) * xs[xp(bi * 32 + lane)];
            }
        } else {
            for (int bi = lane; bi < nb; bi += 32) {
                const uint8_t* b = w + (size_t) bi * 34;
                const float d = half_bits_to_float((uint16_t) (ik2_ldg(b) | (ik2_ldg(b + 1) << 8)));
                const float* x0 = xs + bi * 36;
                float a = 0.0f;
#pragma unroll 8
                for (int j = 0; j < 32; ++j) a += (float) (int8_t) ik2_ldg(b + 2 + j) * x0[j];
                acc += d * a;
            }
        }
    } else if (type == kQ6_K) {
        const int nb = K / 256;
        for (int bi = 0; bi < nb; ++bi) {
            const uint8_t* b = w + (size_t) bi * 210;
            const float d = half_bits_to_float((uint16_t) (ik2_ldg(b + 208) | (ik2_ldg(b + 209) << 8)));
            const int8_t* sc = (const int8_t*) (b + 192);
            const int x0 = bi * 256;
            float a = 0.0f;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                const uint8_t* ql = b + n * 64;
                const uint8_t qh = ik2_ldg(b + 128 + n * 32 + lane);
                const uint8_t l0 = ik2_ldg(ql + lane), l1 = ik2_ldg(ql + lane + 32);
                const int is = n * 8 + (lane >> 4);
                const int xn = x0 + n * 128 + lane;
                a += (float) ik2_ldg(sc + is + 0) * (float) ((int) ((l0 & 0xF) | (((qh >> 0) & 3) << 4)) - 32) * xs[xp(xn)];
                a += (float) ik2_ldg(sc + is + 2) * (float) ((int) ((l1 & 0xF) | (((qh >> 2) & 3) << 4)) - 32) * xs[xp(xn + 32)];
                a += (float) ik2_ldg(sc + is + 4) * (float) ((int) ((l0 >> 4) | (((qh >> 4) & 3) << 4)) - 32) * xs[xp(xn + 64)];
                a += (float) ik2_ldg(sc + is + 6) * (float) ((int) ((l1 >> 4) | (((qh >> 6) & 3) << 4)) - 32) * xs[xp(xn + 96)];
            }
            acc += d * a;
        }
    } else if (type == kQ5_K) {
        const int nb = K / 256;
        const bool hdr4 = (((uintptr_t) w & 3) == 0);
        for (int bi = 0; bi < nb; ++bi) {
            const uint8_t* b = w + (size_t) bi * 176;
            float d, mn;
            if (hdr4) {
                const uint32_t dm = ik2_ldg((const uint32_t*) b);
                d = half_bits_to_float((uint16_t) (dm & 0xFFFFu));
                mn = half_bits_to_float((uint16_t) (dm >> 16));
            } else {
                d = half_bits_to_float((uint16_t) (ik2_ldg(b) | (ik2_ldg(b + 1) << 8)));
                mn = half_bits_to_float((uint16_t) (ik2_ldg(b + 2) | (ik2_ldg(b + 3) << 8)));
            }
            const uint8_t* q = b + 4;
            const uint8_t qh = ik2_ldg(b + 16 + lane);
            const int x0 = bi * 256 + lane;
            float a = 0.0f;
#pragma unroll
            for (int j64 = 0; j64 < 4; ++j64) {
                const uint8_t lo = ik2_ldg(b + 48 + 32 * j64 + lane);
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int is = 2 * j64 + half;
                    int sc, m;
                    if (is < 4) {
                        sc = ik2_ldg(q + is) & 63, m = ik2_ldg(q + is + 4) & 63;
                    } else {
                        const uint8_t q_hi = ik2_ldg(q + is + 4);
                        sc = (q_hi & 0xF) | ((ik2_ldg(q + is - 4) >> 6) << 4);
                        m = (q_hi >> 4) | ((ik2_ldg(q + is) >> 6) << 4);
                    }
                    const int val = (half ? (lo >> 4) : (lo & 0xF)) + (((qh >> is) & 1) ? 16 : 0);
                    a += (d * (float) sc * (float) val - mn * (float) m) * xs[xp(x0 + 64 * j64 + 32 * half)];
                }
            }
            acc += a;
        }
    } else if (type == kIQ3_XXS || type == kIQ4_XS) {
        const bool iq3 = type == kIQ3_XXS;
        for (int sb = lane; sb < K / 32; sb += 32) {
            sycl::float4 xv[8];
            load_xv8(xs + sb * 36, xv);
            acc += iq3 ? iq3xxs_sub_dot(w + (size_t)(sb >> 3) * 98, sb & 7, xv,
                                        d_iq3xxs_grid)
                       : iq4xs_sub_dot(w + (size_t)(sb >> 3) * 136, sb & 7, xv);
        }
    } else {
#pragma unroll
        for (int i = lane; i < K; i += 32) acc +=
            decode_value(type, w, (size_t)i, d_iq3xxs_grid, d_iq4nl) *
            xs[xp(i)];
    }
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) acc +=
        dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(),
            acc, o);
    return acc;
}

// Single-pass dot of two weight rows (w0, w1) of the same quant type against xs (reading xs once for both rows).
/*
DPCT1110: The total declared local variable size in device function row_dot_2
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
inline void row_dot_2(int type, const uint8_t *wg, const uint8_t *wu,
                      const float *xs, int K, float &og, float &ou,
                      const uint32_t *d_iq3xxs_grid, const int8_t *d_iq4nl) {
    const int lane =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) &
        31;
    float ag = 0.0f, au = 0.0f;
    if (type == kNVFP4) {
        const int nsub = K / 16;
        for (int sb = lane; sb < nsub; sb += 32) {
            const float* x = xs + (sb >> 1) * 36 + (sb & 1) * 16;
            const sycl::float4 x0 = *(const sycl::float4 *)(x + 0),
                               x1 = *(const sycl::float4 *)(x + 4);
            const sycl::float4 x2 = *(const sycl::float4 *)(x + 8),
                               x3 = *(const sycl::float4 *)(x + 12);
            const size_t blk_off = (size_t) (sb >> 2) * 36;
            const int sub = sb & 3;
            const uint8_t* bg = wg + blk_off;
            const float dg = ue4m3((uint8_t) (ik2_ldg((const uint32_t*) bg) >> (8 * sub)));
            const uint32_t gq0 = ik2_ldg((const uint32_t*) (bg + 4 + sub * 8));
            const uint32_t gq1 = ik2_ldg((const uint32_t*) (bg + 8 + sub * 8));
            ag += dg * nvfp4_sub_dot(gq0, gq1, x0, x1, x2, x3);

            const uint8_t* bu = wu + blk_off;
            const float du = ue4m3((uint8_t) (ik2_ldg((const uint32_t*) bu) >> (8 * sub)));
            const uint32_t uq0 = ik2_ldg((const uint32_t*) (bu + 4 + sub * 8));
            const uint32_t uq1 = ik2_ldg((const uint32_t*) (bu + 8 + sub * 8));
            au += du * nvfp4_sub_dot(uq0, uq1, x0, x1, x2, x3);
        }
    } else if (type == kQ8_0) {
        const int nb = K / 32;
        if (((uintptr_t) wg & 3) == 0 && ((uintptr_t) wu & 3) == 0) {
            for (int bi = 2 * lane; bi + 1 < nb; bi += 64) {
                const float* x0 = xs + bi * 36;
                const sycl::float4 f0 = *(const sycl::float4 *)(x0 + 0),
                                   f1 = *(const sycl::float4 *)(x0 + 4);
                const sycl::float4 f2 = *(const sycl::float4 *)(x0 + 8),
                                   f3 = *(const sycl::float4 *)(x0 + 12);
                const sycl::float4 f4 = *(const sycl::float4 *)(x0 + 16),
                                   f5 = *(const sycl::float4 *)(x0 + 20);
                const sycl::float4 f6 = *(const sycl::float4 *)(x0 + 24),
                                   f7 = *(const sycl::float4 *)(x0 + 28);
                const uint32_t* pg = (const uint32_t*) (wg + (size_t) bi * 34);
                const uint32_t* pu = (const uint32_t*) (wu + (size_t) bi * 34);
                float dg1, du1;
                ag += q8_blk0_dot(pg, f0, f1, f2, f3, f4, f5, f6, f7, dg1);
                au += q8_blk0_dot(pu, f0, f1, f2, f3, f4, f5, f6, f7, du1);
                const sycl::float4 g0 = *(const sycl::float4 *)(x0 + 36),
                                   g1 = *(const sycl::float4 *)(x0 + 40);
                const sycl::float4 g2 = *(const sycl::float4 *)(x0 + 44),
                                   g3 = *(const sycl::float4 *)(x0 + 48);
                const sycl::float4 g4 = *(const sycl::float4 *)(x0 + 52),
                                   g5 = *(const sycl::float4 *)(x0 + 56);
                const sycl::float4 g6 = *(const sycl::float4 *)(x0 + 60),
                                   g7 = *(const sycl::float4 *)(x0 + 64);
                ag += q8_blk1_dot(pg, dg1, g0, g1, g2, g3, g4, g5, g6, g7);
                au += q8_blk1_dot(pu, du1, g0, g1, g2, g3, g4, g5, g6, g7);
            }
            if (nb & 1) {
                const int bi = nb - 1;
                const float xv = xs[xp(bi * 32 + lane)];
                const uint8_t* bg = wg + (size_t) bi * 34;
                const uint8_t* bu = wu + (size_t) bi * 34;
                ag += half_bits_to_float((uint16_t) (ik2_ldg(bg) | (ik2_ldg(bg + 1) << 8))) *
                      (float) (int8_t) ik2_ldg(bg + 2 + lane) * xv;
                au += half_bits_to_float((uint16_t) (ik2_ldg(bu) | (ik2_ldg(bu + 1) << 8))) *
                      (float) (int8_t) ik2_ldg(bu + 2 + lane) * xv;
            }
        } else {
            for (int bi = lane; bi < nb; bi += 32) {
                const float* x0 = xs + bi * 36;
                const uint8_t* bg = wg + (size_t) bi * 34;
                const uint8_t* bu = wu + (size_t) bi * 34;
                const float dg = half_bits_to_float((uint16_t) (ik2_ldg(bg) | (ik2_ldg(bg + 1) << 8)));
                const float du = half_bits_to_float((uint16_t) (ik2_ldg(bu) | (ik2_ldg(bu + 1) << 8)));
                float sg = 0.0f, su = 0.0f;
#pragma unroll 8
                for (int j = 0; j < 32; ++j) {
                    const float xv = x0[j];
                    sg += (float) (int8_t) ik2_ldg(bg + 2 + j) * xv;
                    su += (float) (int8_t) ik2_ldg(bu + 2 + j) * xv;
                }
                ag += dg * sg;
                au += du * su;
            }
        }
    } else if (type == kQ6_K) {
        const int nb = K / 256;
        for (int bi = 0; bi < nb; ++bi) {
            const uint8_t* bg = wg + (size_t) bi * 210;
            const uint8_t* bu = wu + (size_t) bi * 210;
            const float dg = half_bits_to_float((uint16_t) (ik2_ldg(bg + 208) | (ik2_ldg(bg + 209) << 8)));
            const float du = half_bits_to_float((uint16_t) (ik2_ldg(bu + 208) | (ik2_ldg(bu + 209) << 8)));
            const int8_t* scg = (const int8_t*) (bg + 192);
            const int8_t* scu = (const int8_t*) (bu + 192);
            const int x0 = bi * 256;
            float sg = 0.0f, su = 0.0f;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                const int xn = x0 + n * 128 + lane;
                const float xv0 = xs[xp(xn)], xv1 = xs[xp(xn + 32)], xv2 = xs[xp(xn + 64)], xv3 = xs[xp(xn + 96)];
                const int is = n * 8 + (lane >> 4);

                const uint8_t* qlg = bg + n * 64;
                const uint8_t qhg = ik2_ldg(bg + 128 + n * 32 + lane);
                const uint8_t lg0 = ik2_ldg(qlg + lane), lg1 = ik2_ldg(qlg + lane + 32);
                sg += (float) ik2_ldg(scg + is + 0) * (float) ((int) ((lg0 & 0xF) | (((qhg >> 0) & 3) << 4)) - 32) * xv0;
                sg += (float) ik2_ldg(scg + is + 2) * (float) ((int) ((lg1 & 0xF) | (((qhg >> 2) & 3) << 4)) - 32) * xv1;
                sg += (float) ik2_ldg(scg + is + 4) * (float) ((int) ((lg0 >> 4) | (((qhg >> 4) & 3) << 4)) - 32) * xv2;
                sg += (float) ik2_ldg(scg + is + 6) * (float) ((int) ((lg1 >> 4) | (((qhg >> 6) & 3) << 4)) - 32) * xv3;

                const uint8_t* qlu = bu + n * 64;
                const uint8_t qhu = ik2_ldg(bu + 128 + n * 32 + lane);
                const uint8_t lu0 = ik2_ldg(qlu + lane), lu1 = ik2_ldg(qlu + lane + 32);
                su += (float) ik2_ldg(scu + is + 0) * (float) ((int) ((lu0 & 0xF) | (((qhu >> 0) & 3) << 4)) - 32) * xv0;
                su += (float) ik2_ldg(scu + is + 2) * (float) ((int) ((lu1 & 0xF) | (((qhu >> 2) & 3) << 4)) - 32) * xv1;
                su += (float) ik2_ldg(scu + is + 4) * (float) ((int) ((lu0 >> 4) | (((qhu >> 4) & 3) << 4)) - 32) * xv2;
                su += (float) ik2_ldg(scu + is + 6) * (float) ((int) ((lu1 >> 4) | (((qhu >> 6) & 3) << 4)) - 32) * xv3;
            }
            ag += dg * sg;
            au += du * su;
        }
    } else if (type == kQ5_K) {
        const int nb = K / 256;
        const bool hdr4 = (((uintptr_t) wg & 3) == 0) && (((uintptr_t) wu & 3) == 0);
        for (int bi = 0; bi < nb; ++bi) {
            const uint8_t* bg = wg + (size_t) bi * 176;
            const uint8_t* bu = wu + (size_t) bi * 176;
            float dg, mng, du, mnu;
            if (hdr4) {
                const uint32_t dmg = ik2_ldg((const uint32_t*) bg);
                const uint32_t dmu = ik2_ldg((const uint32_t*) bu);
                dg = half_bits_to_float((uint16_t) (dmg & 0xFFFFu));
                mng = half_bits_to_float((uint16_t) (dmg >> 16));
                du = half_bits_to_float((uint16_t) (dmu & 0xFFFFu));
                mnu = half_bits_to_float((uint16_t) (dmu >> 16));
            } else {
                dg = half_bits_to_float((uint16_t) (ik2_ldg(bg) | (ik2_ldg(bg + 1) << 8)));
                mng = half_bits_to_float((uint16_t) (ik2_ldg(bg + 2) | (ik2_ldg(bg + 3) << 8)));
                du = half_bits_to_float((uint16_t) (ik2_ldg(bu) | (ik2_ldg(bu + 1) << 8)));
                mnu = half_bits_to_float((uint16_t) (ik2_ldg(bu + 2) | (ik2_ldg(bu + 3) << 8)));
            }
            const uint8_t* qg = bg + 4;
            const uint8_t* qu = bu + 4;
            const uint8_t qhg = ik2_ldg(bg + 16 + lane), qhu = ik2_ldg(bu + 16 + lane);
            const int x0 = bi * 256 + lane;
#pragma unroll
            for (int j64 = 0; j64 < 4; ++j64) {
                const uint8_t log_ = ik2_ldg(bg + 48 + 32 * j64 + lane);
                const uint8_t lou = ik2_ldg(bu + 48 + 32 * j64 + lane);
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int is = 2 * j64 + half;
                    const float xv = xs[xp(x0 + 64 * j64 + 32 * half)];
                    int scg, mg, scu, mu;
                    if (is < 4) {
                        scg = ik2_ldg(qg + is) & 63, mg = ik2_ldg(qg + is + 4) & 63;
                        scu = ik2_ldg(qu + is) & 63, mu = ik2_ldg(qu + is + 4) & 63;
                    } else {
                        const uint8_t qgh = ik2_ldg(qg + is + 4), quh = ik2_ldg(qu + is + 4);
                        scg = (qgh & 0xF) | ((ik2_ldg(qg + is - 4) >> 6) << 4), mg = (qgh >> 4) | ((ik2_ldg(qg + is) >> 6) << 4);
                        scu = (quh & 0xF) | ((ik2_ldg(qu + is - 4) >> 6) << 4), mu = (quh >> 4) | ((ik2_ldg(qu + is) >> 6) << 4);
                    }
                    const int vg = (half ? (log_ >> 4) : (log_ & 0xF)) + (((qhg >> is) & 1) ? 16 : 0);
                    const int vu = (half ? (lou >> 4) : (lou & 0xF)) + (((qhu >> is) & 1) ? 16 : 0);
                    ag += (dg * (float) scg * (float) vg - mng * (float) mg) * xv;
                    au += (du * (float) scu * (float) vu - mnu * (float) mu) * xv;
                }
            }
        }
    } else if (type == kIQ3_XXS || type == kIQ4_XS) {
        const bool iq3 = type == kIQ3_XXS;
        const size_t bb = iq3 ? 98 : 136;
        for (int sb = lane; sb < K / 32; sb += 32) {
            sycl::float4 xv[8];
            load_xv8(xs + sb * 36, xv);
            const size_t off = (size_t) (sb >> 3) * bb;
            ag += iq3 ? iq3xxs_sub_dot(wg + off, sb & 7, xv, d_iq3xxs_grid)
                      : iq4xs_sub_dot(wg + off, sb & 7, xv);
            au += iq3 ? iq3xxs_sub_dot(wu + off, sb & 7, xv, d_iq3xxs_grid)
                      : iq4xs_sub_dot(wu + off, sb & 7, xv);
        }
    } else {
        for (int i = lane; i < K; i += 32) {
            const float xv = xs[xp(i)];
            ag +=
                decode_value(type, wg, (size_t)i, d_iq3xxs_grid, d_iq4nl) * xv;
            au +=
                decode_value(type, wu, (size_t)i, d_iq3xxs_grid, d_iq4nl) * xv;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        ag += dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), ag,
            o);
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        au += dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), au,
            o);
    }
    og = ag;
    ou = au;
}
__dpct_inline__ void row_dot_pair(
    int type, const uint8_t *wg, const uint8_t *wu, const float *xs, int K,
    float &og, float &ou, const uint32_t *d_iq3xxs_grid,
    const int8_t *d_iq4nl) {
    row_dot_2(type, wg, wu, xs, K, og, ou, d_iq3xxs_grid, d_iq4nl);
}

__dpct_inline__ void load_xs_f32(const float *x, int K, float *xs) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (((uintptr_t)x & 15) == 0 && (K & 3) == 0) {
        const int n4 = K >> 2;
        const sycl::float4 *x4 = (const sycl::float4 *)x;
#pragma unroll
        for (int i4 = item_ct1.get_local_id(2); i4 < n4;
             i4 += item_ct1.get_local_range(2))
            *(sycl::float4 *)(xs + xp(i4 << 2)) = ::ik2_ldg(x4 + i4);
    } else {
#pragma unroll
        for (int i = item_ct1.get_local_id(2); i < K;
             i += item_ct1.get_local_range(2)) xs[xp(i)] = x[i];
    }
}

__dpct_inline__ void load_xs_f16(const uint16_t *h, int K, float *xs) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (((uintptr_t)h & 7) == 0 && (K & 3) == 0) {
        const int n4 = K >> 2;
        const sycl::uint2 *h4 = (const sycl::uint2 *)h;
        for (int i4 = item_ct1.get_local_id(2); i4 < n4;
             i4 += item_ct1.get_local_range(2)) {
            const sycl::uint2 u = ::ik2_ldg(h4 + i4);
            const sycl::float2 f0 = u32_to_f2(u.x()), f1 = u32_to_f2(u.y());
            *(sycl::float4 *)(xs + xp(i4 << 2)) =
                sycl::float4(f0.x(), f0.y(), f1.x(), f1.y());
        }
    } else {
#pragma unroll
        for (int i = item_ct1.get_local_id(2); i < K;
             i += item_ct1.get_local_range(2)) xs[xp(i)] = h2f(h[i]);
    }
}

size_t mv_row_bytes(int type, int K) {
    return (size_t) (K / block_values(type)) * block_bytes(type);
}

// Two signed-byte limbs per activation, one FP32 scale per sixteen values.
// The byte order follows the GGUF nibbles, so each four-weight lookup feeds
// two dp4a accumulators without reshuffling either set of bytes.
struct NvInput {
    int high[4], low[4];
    float scale;
};
template <bool Gate>
/*
DPCT1110: The total declared local variable size in device function
pack_limbs exceeds 128 bytes and may cause high register pressure. Consult with
your hardware vendor to find the total register size available and adjust the
code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void pack_limbs(const float *x, const uint16_t *h, int K,
                                NvInput *packed, int i, int slot, bool q8) {
    if (i >= K / 16) return;
    const int base = slot * K + 16 * i;
    float vals[16], mx = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        vals[j] = Gate ? x[base + j] : h2f(h[base + j]);
        mx = sycl::fmax(mx, sycl::fabs(vals[j]));
    }
    const float inv = mx > 0 ? 16256.f / mx : 0;
    NvInput dst;
    dst.scale = mx / 16256.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        unsigned hi = 0, lo = 0;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int ix = q8 ? 4 * j + t : 2 * j + (t >> 1) + 8 * (t & 1);
            const int q = sycl::vec<float, 1>{(vals[ix] * inv)}
                              .convert<int, sycl::rounding_mode::rte>()[0];
            const int qh = (q + 64) >> 7;
            hi |= (unsigned)(uint8_t)qh << (8 * t);
            lo |= (unsigned)(uint8_t)(q - 128 * qh) << (8 * t);
        }
        dst.high[j] = hi;
        dst.low[j] = lo;
    }
    packed[base / 16] = dst;
}

template <bool Gate>
__dpct_inline__ void nv_input(const float *x, const uint16_t *h, int K,
                              NvInput *packed, bool q8) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    pack_limbs<Gate>(x, h, K, packed,
                     item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                         item_ct1.get_local_id(2),
                     item_ct1.get_group(1), q8);
}

__dpct_inline__ float nv_limb_dot(const uint8_t *row, int sub,
                                  const NvInput &x) {
    const uint8_t* b = row + 36 * (sub / 4);
    const uint32_t* words = (const uint32_t*)(b + 4 + 8 * (sub % 4));
    int hi = 0, lo = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t nib = words[j / 2] >> (16 * (j % 2));
        const uint32_t pos =
            dpct::byte_level_permute(0x03020100, 0x0c080604, nib);
        const uint32_t neg =
            dpct::byte_level_permute(0xfdfeff00, 0xf4f8fafc, nib);
        const int val =
            dpct::byte_level_permute(pos, neg, 0x3210 | ((nib & 0x8888) >> 1));
        hi = dpct::dp4a(val, x.high[j], hi);
        lo = dpct::dp4a(val, x.low[j], lo);
    }
    return float(128 * hi + lo) * x.scale * ue4m3(b[sub % 4]);
}
__dpct_inline__ float q8_limb_dot(const uint8_t *row, int sub,
                                  const NvInput &x) {
    const uint8_t* block = row + (sub / 2) * 34;
    const uint16_t* q = (const uint16_t*)(block + 2 + (sub % 2) * 16);
    int hi = 0, lo = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int four = (uint32_t)q[2 * j] | ((uint32_t)q[2 * j + 1] << 16);
        hi = dpct::dp4a(four, x.high[j], hi);
        lo = dpct::dp4a(four, x.low[j], lo);
    }
    return float(128 * hi + lo) * x.scale * h2f(*(const uint16_t*)block);
}

template <bool Gate, bool Q8>
__dpct_inline__ void nv_limb_mv(
    const uint8_t *const *blobs, const NvInput *packed, size_t offset, int C,
    int F, const float *sg, const float *su, float gcap, float ucap,
    uint16_t *hidden, float *y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) % 32,
              warp = item_ct1.get_local_id(2) / 32,
              slot = item_ct1.get_group(1);
    const int K = Gate ? C : F, N = Gate ? F : C;
    const int r =
        (item_ct1.get_group(2) * (item_ct1.get_local_range(2) / 32) + warp) *
        (Gate ? 1 : 2);
    if (r >= N) return;
    const int stride = Q8 ? K / 32 * 34 : K / 64 * 36;
    const uint8_t* row = blobs[slot] + (Gate ? 0 : offset) + (size_t)r * stride;
    const uint8_t* second = row + (Gate ? offset : stride);
    const NvInput* xp = packed + (Gate ? 0 : slot * K / 16);
    float a = 0, b = 0;
    for (int s = lane; s < K / 16; s += 32) {
        const NvInput z = xp[s];
        a += Q8 ? q8_limb_dot(row, s, z) : nv_limb_dot(row, s, z);
        b += Q8 ? q8_limb_dot(second, s, z) : nv_limb_dot(second, s, z);
    }
    a = wsum(a);
    b = wsum(b);
    if (lane) return;
    if (Gate) {
        const float g = a * (sg ? sg[slot] : 1.f), u = b * (su ? su[slot] : 1.f);
        hidden[(size_t)slot * F + r] =
            f2h(gcap * sycl::tanh(g / gcap) *
                (1.f / (1.f + sycl::native::exp(-g))) * ucap *
                sycl::tanh(u / ucap));
    } else {
        y[(size_t)slot * C + r] = a;
        y[(size_t)slot * C + r + 1] = b;
    }
}

// Each warp multiplies 16 expert rows by the distinct token inputs in its group.
__dpct_inline__ uint32_t ik2_fp4pair(uint32_t v, uint32_t scale) {
    const uint32_t mag =
        dpct::byte_level_permute(0x42403c00u, 0x4a484644u, (v & 0x0707u) << 4);
    const uint32_t sign = ((v & 8u) << 12) | ((v & 0x800u) << 20);
    uint32_t ret;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
        asm("mul.rn.f16x2 %0, %1, %2;"
            : "=r"(ret)
            : "r"(mag | sign), "r"(scale));
#else
        ret =
            ((sycl::vec<int, 1>((mag | sign))).as<sycl::vec<sycl::half, 2>>() *
             (sycl::vec<int, 1>(scale)).as<sycl::vec<sycl::half, 2>>())
                .as<sycl::vec<int, 1>>()
                .x();
#endif
    return ret;
}
__dpct_inline__ uint32_t q8_pair(uint32_t w) {
    const uint32_t u =
        dpct::byte_level_permute(w ^ 0x80808080u, 0x64646464u, 0x4140);
    uint32_t f;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
        asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(f) : "r"(u), "r"(0x64806480u));
#else
        f = (((sycl::vec<int, 1>(u)).as<sycl::vec<sycl::half, 2>>() -
              (sycl::vec<int, 1>(0x64806480u)).as<sycl::vec<sycl::half, 2>>())
                 .as<sycl::vec<int, 1>>())
                .x();
        ;
#endif
    return f;
}
__dpct_inline__ uint32_t q8_h2(uint32_t w, uint32_t scale) {
    uint32_t f = q8_pair(w);
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(f) : "r"(f), "r"(scale));
#else
        f = ((sycl::vec<int, 1>(f)).as<sycl::vec<sycl::half, 2>>() *
             (sycl::vec<int, 1>(scale)).as<sycl::vec<sycl::half, 2>>())
                .as<sycl::vec<int, 1>>()
                .x();
#endif
    return f;
}

template <bool Gate, int Splits, bool Q8 = false>
/*
DPCT1110: The total declared local variable size in device function
grouped_expert_mma exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void grouped_expert_mma(
    const uint8_t *const *blobs, const int *tok, size_t offset, int C, int F,
    const uint16_t *h, uint16_t *hidden, float *out, const float *sg,
    const float *su, float gcap, float ucap, const int *counts,
    const int *rows_out, const uint16_t *input16 = nullptr) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    static constexpr int rows = Gate ? 8 : 16;
    const int lane = item_ct1.get_local_id(2) & 31,
              split = item_ct1.get_local_id(2) / 32, grp = lane / 4,
              sub = lane % 4;
    const int s = item_ct1.get_group(1), r = item_ct1.get_group(2) * rows + grp,
              K = Gate ? C : F;
    const int count = counts[s];
    if (count <= 0) return;
    const int gi = s * kGroupMax + (grp < count ? grp : 0);
    const int input_row = Gate ? tok[gi] : rows_out[gi];
    const int stride = Q8 ? K / 32 * 34 : K / 64 * 36;
    const uint8_t* a = blobs[s] + (Gate ? 0 : offset) + (size_t)r * stride;
    const uint8_t* b = a + (Gate ? offset : 8 * stride);
    float v[4] = {};
    if constexpr (Q8) {
        for (int k = split; k < K / 32; k += Splits) {
            const uint16_t* ap = (const uint16_t*)(a + k * 34);
            const uint16_t* bp = (const uint16_t*)(b + k * 34);
            const float sa = h2f(ap[0]), sb = h2f(bp[0]);
            const int off = k * 32 + sub * 8;
            uint32_t xx[4];
                const uint32_t* xp = (const uint32_t*)((Gate ? input16 : h) + (size_t)input_row * K + off);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    xx[j] = xp[j];
            ap += 1 + 4 * sub;
            bp += 1 + 4 * sub;
            // Q8's signed integers are exactly representable in FP16. Apply its original
            // per-row block scale after the FP32 dot, avoiding FP16 weight rounding.
            float block_sum[4]{};
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                uint32_t af[4] = {q8_pair(ap[j]), q8_pair(bp[j]), q8_pair(ap[j + 2]), q8_pair(bp[j + 2])};
                mma16816(block_sum, af, xx[j], xx[j + 2]);
            }
            v[0] += sa * block_sum[0];
            v[1] += sa * block_sum[1];
            v[2] += sb * block_sum[2];
            v[3] += sb * block_sum[3];
        }
    } else
        for (int k = split; k < K / 64; k += Splits) {
            const uint8_t* ap = a + k * 36;
            const uint8_t* bp = b + k * 36;
            uint32_t sa = f2h(ue4m3(ap[sub])), sb = f2h(ue4m3(bp[sub]));
            sa |= sa << 16;
            sb |= sb << 16;
            uint32_t xx[8];
            const int off = k * 64 + sub * 16;
                const uint32_t* xp = (const uint32_t*)((Gate ? input16 : h) + (size_t)input_row * K + off);
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    xx[i] = xp[i];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const uint32_t wa = ((const uint32_t*)(ap + 4))[sub * 2 + j];
                const uint32_t wb = ((const uint32_t*)(bp + 4))[sub * 2 + j];
#pragma unroll
                for (int m = 0; m < 2; ++m) {
                    const uint32_t ra = wa >> (16 * m), rb = wb >> (16 * m);
                    const uint32_t af[4] = {ik2_fp4pair(ra, sa), ik2_fp4pair(rb, sb), ik2_fp4pair(ra >> 4, sa),
                                            ik2_fp4pair(rb >> 4, sb)};
                    mma16816(v, af, xx[2 * j + m], xx[4 + 2 * j + m]);
                }
            }
        }
    auto &partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        sycl::float4[Splits][32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    partial[split][lane] = sycl::float4(v[0], v[1], v[2], v[3]);
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (split) return;
    sycl::float4 total = partial[0][lane];
#pragma unroll
    for (int i = 1; i < Splits; ++i) {
        const sycl::float4 z = partial[i][lane];
        total.x() += z.x();
        total.y() += z.y();
        total.z() += z.z();
        total.w() += z.w();
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int token = sub * 2 + j;
        if (token >= count) continue;
        const int dst = rows_out[s * kGroupMax + token];
        const float a0 = j ? total.y() : total.x(),
                    a1 = j ? total.w() : total.z();
        if (Gate) {
            const float g = a0 * (sg ? sg[dst] : 1.f), u = a1 * (su ? su[dst] : 1.f);
            hidden[(size_t)dst * F + r] =
                f2h(gcap * sycl::tanh(g / gcap) *
                    (1.f / (1.f + sycl::native::exp(-g))) * ucap *
                    sycl::tanh(u / ucap));
        } else {
            out[(size_t)dst * C + r] = a0;
            out[(size_t)dst * C + r + 8] = a1;
        }
    }
}

// ---- one token's NVFP4 experts through the tensor cores, straight from their blocks into the registers.
// A warp is 16 rows (gate / up: 8 gate rows and the same 8 of up, so that SwiGLU is a lane's; down: 16 rows) over
// the whole K, no shared memory.  Lane (g, t): sub-block t of rows g and g + 8 of each 64-value block - four MMA a
// block, MMA m the sub-block's bytes 2 m and 2 m + 1 (low nibbles values 2 m, 2 m + 1 in k slots 2 t, 2 t + 1; high
// nibbles values 2 m + 8, 2 m + 9 in slots 2 t + 8, 2 t + 9).  B: the input in FP16 where it is, the same in all
// eight columns.  A code times its sub-block's scale is exact in FP16 (a code is at most 12, ue4m3 at most 240 and
// at least 2^-10); the sum is the tensor cores' FP32.
__dpct_inline__ uint32_t ue4m3_h2(
    uint32_t x) { // FP16 bits of ue4m3(x) in both halves
    const uint32_t e = (x >> 3) & 15u, m = x & 7u;
    const uint32_t sub = dpct::byte_level_permute(0x1a181400u, 0x1f1e1d1cu, m)
                         << 8; // m 2^-10
    uint32_t h = e ? (((e + 7u) << 10) | (m << 7)) : sub;
    h = (x == 0x7Fu) ? 0u : h;
    return h | (h << 16);
}
__dpct_inline__ uint32_t nv_pair_h2(
    uint32_t s, uint32_t scale) { // bytes 0 and 1's low nibbles
    const uint32_t mag =
        dpct::byte_level_permute(0x42403c00u, 0x4a484644u, (s & 0x0707u) << 4);
    const uint32_t sign = ((s & 0x0808u) * 0x101000u) & 0x80008000u;
    uint32_t r;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(mag | sign), "r"(scale));
#else
        r = ((sycl::vec<int, 1>((mag | sign))).as<sycl::vec<sycl::half, 2>>() *
             (sycl::vec<int, 1>(scale)).as<sycl::vec<sycl::half, 2>>())
                .as<sycl::vec<int, 1>>()
                .x();
#endif
    return r;
}
struct NvStep {   // a lane's part of a 64-value block of its two rows: the sub-block's scales and codes
    uint32_t sa, sb, wa0, wa1, wb0, wb1;
};
__dpct_inline__ NvStep nv_step_load(const uint8_t *a, const uint8_t *b, int k,
                                    int t) {
    const uint32_t* ap = (const uint32_t*) (a + (size_t) k * 36);
    const uint32_t* bp = (const uint32_t*) (b + (size_t) k * 36);
    NvStep r;
    r.sa = ik2_ldg(ap), r.sb = ik2_ldg(bp);
    r.wa0 = ik2_ldg(ap + 1 + 2 * t), r.wa1 = ik2_ldg(ap + 2 + 2 * t);
    r.wb0 = ik2_ldg(bp + 1 + 2 * t), r.wb1 = ik2_ldg(bp + 2 + 2 * t);
    return r;
}
// A block is a group of 16 rows, its SPLIT warps each a stride of the K blocks (more rows of weights in flight than
// a warp alone keeps: a warp waits for a block's bytes before it asks for more than the next one's).
template <bool Gate, int SPLIT>
/*
DPCT1110: The total declared local variable size in device function
experts_mma_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void experts_mma_kernel(
    const uint8_t *const *__restrict__ blob, size_t off, int C, int F,
    const uint16_t *__restrict__ x16, int x_ld, uint16_t *hidden, float *y,
    const float *sg, const float *su, float gcap, float ucap) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    sycl::float2[SPLIT][8]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int lane = item_ct1.get_local_id(2) & 31,
              split = item_ct1.get_local_id(2) >> 5, g = lane >> 2,
              t = lane & 3, slot = item_ct1.get_group(1);
    const int K = Gate ? C : F;
    const int r = item_ct1.get_group(2) * (Gate ? 8 : 16) + g;
    const size_t rb = (size_t) (K / 64) * 36;
    const uint8_t* a = blob[slot] + (Gate ? 0 : off) + (size_t) r * rb;
    const uint8_t* b = Gate ? a + off : a + 8 * rb;
    // (gate / up: the input as two FP16 limbs, x = hi + lo, [2][C] - column 0 takes hi, column 1 lo, and a row's
    //  two columns add up to the product with the FP32 input; down: h is FP16 itself, every column the same)
    const sycl::uint4 *xp =
        (const sycl::uint4 *)(x16 + (Gate ? (g == 1 ? (size_t)C : 0)
                                          : (size_t)slot * x_ld)) +
        2 * t;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int nk = K / 64;
    NvStep cur = nv_step_load(a, b, split, t);
    for (int k = split; k < nk; k += SPLIT) {
        NvStep nxt = cur;
        if (k + SPLIT < nk) nxt = nv_step_load(a, b, k + SPLIT, t);
        const uint32_t sa = ue4m3_h2((cur.sa >> (8 * t)) & 0xFFu), sb = ue4m3_h2((cur.sb >> (8 * t)) & 0xFFu);
        const uint32_t wa[2] = {cur.wa0, cur.wa1}, wb[2] = {cur.wb0, cur.wb1};
        const sycl::uint4 x0 = ::ik2_ldg(xp + k * 8),
                          x1 = ::ik2_ldg(xp + k * 8 + 1);
        const uint32_t xx[8] = {x0.x(), x0.y(), x0.z(), x0.w(),
                                x1.x(), x1.y(), x1.z(), x1.w()};
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t qa = wa[m >> 1] >> (16 * (m & 1)), qb = wb[m >> 1] >> (16 * (m & 1));
            const uint32_t af[4] = {nv_pair_h2(qa, sa), nv_pair_h2(qb, sb), nv_pair_h2(qa >> 4, sa),
                                    nv_pair_h2(qb >> 4, sb)};
            mma16816(v, af, xx[m], xx[4 + m]);
        }
        cur = nxt;
    }
    if (t == 0) part[split][g] = Gate ? sycl::float2(v[0] + v[1], v[2] + v[3])
                                      : sycl::float2(v[0], v[2]);
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (split || t) return;
    sycl::float2 tot = part[0][g];
#pragma unroll
    for (int i = 1; i < SPLIT; ++i) tot.x() += part[i][g].x(),
        tot.y() += part[i][g].y();
    if (Gate) {
        const float gg = tot.x() * (sg ? sg[slot] : 1.0f),
                    u = tot.y() * (su ? su[slot] : 1.0f);
        hidden[(size_t)slot * F + r] =
            f2h(gcap * sycl::tanh(gg / gcap) *
                (1.0f / (1.0f + sycl::native::exp(-gg))) * ucap *
                sycl::tanh(u / ucap));
    } else {
        y[(size_t)slot * C + r] = tot.x();
        y[(size_t)slot * C + r + 8] = tot.y();
    }
}

template <int WARPS, int ROWS>
__dpct_inline__ void experts_gate_up_kernel(
    const uint8_t *const *blob, const int *tok, int ty_gate, int ty_up,
    size_t eb0, const float *x, int C, int F, size_t rb_gate, size_t rb_up,
    float *gu, uint8_t *dpct_local, const uint32_t *d_iq3xxs_grid,
    const int8_t *d_iq4nl) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto xs = (float *)dpct_local;
    const int s = item_ct1.get_group(1);
    load_xs_f32(x + (size_t) tok[s] * C, C, xs);
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int warp = item_ct1.get_local_id(2) >> 5;
    const uint8_t* bl = blob[s];
    for (int j = 0; j < ROWS; ++j) {
        const int row = (item_ct1.get_group(2) * WARPS + warp) * ROWS +
                        j; // of 2 F: gate's then up's
        if (row >= 2 * F) break;
        const bool up = row >= F;
        const int r = up ? row - F : row;
        const float v = row_dot(up ? ty_up : ty_gate,
                                bl + (up ? eb0 + r * rb_up : r * rb_gate), xs,
                                C, d_iq3xxs_grid, d_iq4nl);
        if ((item_ct1.get_local_id(2) & 31) == 0) gu[(size_t)s * 2 * F + row] =
            v;
    }
}

template <int WARPS, int ROWS>
__dpct_inline__ void experts_gate_up_swiglu_kernel(
    const uint8_t *const *blob, const int *tok, int ty_gate, int ty_up,
    size_t eb0, const float *x, int C, int F, size_t rb_gate, size_t rb_up,
    const float *sg, const float *su, float gcap, float ucap, uint16_t *h,
    const int *orow, int h_ld, uint8_t *dpct_local,
    const uint32_t *d_iq3xxs_grid, const int8_t *d_iq4nl) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto xs = (float *)dpct_local;
    const int s = item_ct1.get_group(1);
    load_xs_f32(x + (size_t) tok[s] * C, C, xs);
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int warp = item_ct1.get_local_id(2) >> 5;
    const uint8_t* bl = blob[s];
    const float sgs = sg ? sg[s] : 1.0f;
    const float sus = su ? su[s] : 1.0f;
    for (int j = 0; j < ROWS; ++j) {
        const int r = (item_ct1.get_group(2) * WARPS + warp) * ROWS + j;
        if (r >= F) break;
        float vg, vu;
        if (ty_gate == ty_up) {
            row_dot_pair(ty_gate, bl + (size_t)r * rb_gate,
                         bl + eb0 + (size_t)r * rb_up, xs, C, vg, vu,
                         d_iq3xxs_grid, d_iq4nl);
        } else {
            vg = row_dot(ty_gate, bl + (size_t)r * rb_gate, xs, C,
                         d_iq3xxs_grid, d_iq4nl);
            vu = row_dot(ty_up, bl + eb0 + (size_t)r * rb_up, xs, C,
                         d_iq3xxs_grid, d_iq4nl);
        }
        if ((item_ct1.get_local_id(2) & 31) == 0) {
            const float g = vg * sgs, u = vu * sus;
            h[(size_t)(orow ? orow[s] : s) * h_ld + r] =
                f2h(gcap * sycl::tanh(g / gcap) *
                    (1.0f / (1.0f + sycl::native::exp(-g))) * ucap *
                    sycl::tanh(u / ucap));
        }
    }
}

template <int WARPS, int ROWS>
__dpct_inline__ void experts_down_kernel(
    const uint8_t *const *blob, int ty_down, size_t off_down, const uint16_t *h,
    int C, int F, size_t rb_down, float *y, uint8_t *dpct_local,
    const uint32_t *d_iq3xxs_grid, const int8_t *d_iq4nl) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto xs = (float *)dpct_local;
    const int s = item_ct1.get_group(1);
    load_xs_f16(h + (size_t) s * F, F, xs);
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int warp = item_ct1.get_local_id(2) >> 5;
    const uint8_t* w = blob[s] + off_down;
    for (int j = 0; j < ROWS;) {
        const int row = (item_ct1.get_group(2) * WARPS + warp) * ROWS + j;
        if (row >= C) break;
        if (j + 1 < ROWS && row + 1 < C) {
            float v0, v1;
            row_dot_2(ty_down, w + (size_t)row * rb_down,
                      w + (size_t)(row + 1) * rb_down, xs, F, v0, v1,
                      d_iq3xxs_grid, d_iq4nl);
            if ((item_ct1.get_local_id(2) & 31) == 0) {
                y[(size_t) s * C + row] = v0;
                y[(size_t) s * C + row + 1] = v1;
            }
            j += 2;
        } else {
            const float v = row_dot(ty_down, w + (size_t)row * rb_down, xs, F,
                                    d_iq3xxs_grid, d_iq4nl);
            if ((item_ct1.get_local_id(2) & 31) == 0) y[(size_t)s * C + row] =
                v;
            j += 1;
        }
    }
}

// ---- grouped: an expert's rows read once for every token that picked it (a verify window's few tokens)

// a warp's dots of one row (blocks of `type`) with n <= N_CAP inputs (each K values, F32 or FP16 rows in global memory)
template <class X> __dpct_inline__ float xval(const X *p, int i);
template <>
__dpct_inline__ float xval<float>(const float *p, int i) {
    return ik2_ldg(p + i);
}
template <>
__dpct_inline__ float xval<uint16_t>(const uint16_t *p, int i) {
    return h2f(ik2_ldg(p + i));
}

template <class X>
__dpct_inline__ float dot16_vec(const X *p, const float *wv);

template <>
__dpct_inline__ float dot16_vec<float>(const float *p, const float *wv) {
    if (((uintptr_t) p & 15) == 0) {
        const sycl::float4 *p4 = (const sycl::float4 *)p;
        const sycl::float4 v0 = ::ik2_ldg(p4 + 0), v1 = ::ik2_ldg(p4 + 1),
                           v2 = ::ik2_ldg(p4 + 2), v3 = ::ik2_ldg(p4 + 3);
        return wv[0] * v0.x() + wv[1] * v0.y() + wv[2] * v0.z() +
               wv[3] * v0.w() + wv[4] * v1.x() + wv[5] * v1.y() +
               wv[6] * v1.z() + wv[7] * v1.w() + wv[8] * v2.x() +
               wv[9] * v2.y() + wv[10] * v2.z() + wv[11] * v2.w() +
               wv[12] * v3.x() + wv[13] * v3.y() + wv[14] * v3.z() +
               wv[15] * v3.w();
    }
    float a = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) a += wv[i] * ik2_ldg(p + i);
    return a;
}

template <>
__dpct_inline__ float dot16_vec<uint16_t>(const uint16_t *p, const float *wv) {
    if (((uintptr_t) p & 15) == 0) {
        const sycl::uint4 *p4 = (const sycl::uint4 *)p;
        const sycl::uint4 u0 = ::ik2_ldg(p4 + 0), u1 = ::ik2_ldg(p4 + 1);
        const sycl::float2 f0 = u32_to_f2(u0.x()), f1 = u32_to_f2(u0.y()),
                           f2 = u32_to_f2(u0.z()), f3 = u32_to_f2(u0.w());
        const sycl::float2 f4 = u32_to_f2(u1.x()), f5 = u32_to_f2(u1.y()),
                           f6 = u32_to_f2(u1.z()), f7 = u32_to_f2(u1.w());
        return wv[0] * f0.x() + wv[1] * f0.y() + wv[2] * f1.x() +
               wv[3] * f1.y() + wv[4] * f2.x() + wv[5] * f2.y() +
               wv[6] * f3.x() + wv[7] * f3.y() + wv[8] * f4.x() +
               wv[9] * f4.y() + wv[10] * f5.x() + wv[11] * f5.y() +
               wv[12] * f6.x() + wv[13] * f6.y() + wv[14] * f7.x() +
               wv[15] * f7.y();
    }
    float a = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) a += wv[i] * h2f(ik2_ldg(p + i));
    return a;
}

__dpct_inline__ void decode_nvfp4_wv(uint32_t q0, uint32_t q1, float *wv) {
    const sycl::float2 w01 = e2m1_pair_lo(q0), w23 = e2m1_pair_lo(q0 >> 16);
    const sycl::float2 w45 = e2m1_pair_lo(q1), w67 = e2m1_pair_lo(q1 >> 16);
    const sycl::float2 w89 = e2m1_pair_hi(q0), wAB = e2m1_pair_hi(q0 >> 16);
    const sycl::float2 wCD = e2m1_pair_hi(q1), wEF = e2m1_pair_hi(q1 >> 16);
    wv[0] = w01.x(), wv[1] = w01.y(), wv[2] = w23.x(), wv[3] = w23.y();
    wv[4] = w45.x(), wv[5] = w45.y(), wv[6] = w67.x(), wv[7] = w67.y();
    wv[8] = w89.x(), wv[9] = w89.y(), wv[10] = wAB.x(), wv[11] = wAB.y();
    wv[12] = wCD.x(), wv[13] = wCD.y(), wv[14] = wEF.x(), wv[15] = wEF.y();
}

__dpct_inline__ void unpack_u32_i8(uint32_t q4, float *wv) {
    wv[0] = (float) (int8_t) (q4 & 0xFFu);
    wv[1] = (float) (int8_t) ((q4 >> 8) & 0xFFu);
    wv[2] = (float) (int8_t) ((q4 >> 16) & 0xFFu);
    wv[3] = (float) (int8_t) (q4 >> 24);
}

template <class X, int N_CAP>
/*
DPCT1110: The total declared local variable size in device function
row_dot_multi exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__attribute__((noinline)) void row_dot_multi(int type, const uint8_t *w, const X *const *xs, int n,
                          int K, float *out, const uint32_t *d_iq3xxs_grid,
                          const int8_t *d_iq4nl) {
    const int lane =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) &
        31;
    float acc[N_CAP];
#pragma unroll
    for (int t = 0; t < N_CAP; ++t) acc[t] = 0.0f;
    if (type == kNVFP4) {
        const int nsub = K / 16;
        for (int sb = lane; sb < nsub; sb += 32) {
            const uint8_t* b = w + (size_t) (sb >> 2) * 36;
            const int sub = sb & 3;
            const float d = ue4m3((uint8_t) (*(const uint32_t*) b >> (8 * sub)));
            const uint32_t q0 = *(const uint32_t*) (b + 4 + sub * 8), q1 = *(const uint32_t*) (b + 8 + sub * 8);
            float wv[16];
            decode_nvfp4_wv(q0, q1, wv);
#pragma unroll
            for (int t = 0; t < N_CAP; ++t) {
                if (t >= n) break;
                acc[t] += d * dot16_vec(xs[t] + sb * 16, wv);
            }
        }
    } else if (type == kQ8_0) {
        const int nb = K / 32;
        if (((uintptr_t) w & 3) == 0) {
            for (int bi = 2 * lane; bi + 1 < nb; bi += 64) {
                const uint32_t* p = (const uint32_t*) (w + (size_t) bi * 34);
                const uint32_t u0 = p[0], u1 = p[1], u2 = p[2], u3 = p[3], u4 = p[4];
                const uint32_t u5 = p[5], u6 = p[6], u7 = p[7], u8 = p[8];
                const float d0 = half_bits_to_float((uint16_t) (u0 & 0xFFFFu));
                float a0[N_CAP];
#pragma unroll
                for (int t = 0; t < N_CAP; ++t) a0[t] = 0.0f;
                {
                    float wv[16];
                    wv[0] = (float) (int8_t) ((u0 >> 16) & 0xFFu);
                    wv[1] = (float) (int8_t) (u0 >> 24);
                    unpack_u32_i8(u1, wv + 2);
                    unpack_u32_i8(u2, wv + 6);
                    unpack_u32_i8(u3, wv + 10);
                    wv[14] = (float) (int8_t) (u4 & 0xFFu);
                    wv[15] = (float) (int8_t) ((u4 >> 8) & 0xFFu);
#pragma unroll
                    for (int t = 0; t < N_CAP; ++t) {
                        if (t >= n) break;
                        a0[t] += dot16_vec(xs[t] + bi * 32, wv);
                    }
                    wv[0] = (float) (int8_t) ((u4 >> 16) & 0xFFu);
                    wv[1] = (float) (int8_t) (u4 >> 24);
                    unpack_u32_i8(u5, wv + 2);
                    unpack_u32_i8(u6, wv + 6);
                    unpack_u32_i8(u7, wv + 10);
                    wv[14] = (float) (int8_t) (u8 & 0xFFu);
                    wv[15] = (float) (int8_t) ((u8 >> 8) & 0xFFu);
#pragma unroll
                    for (int t = 0; t < N_CAP; ++t) {
                        if (t >= n) break;
                        a0[t] += dot16_vec(xs[t] + bi * 32 + 16, wv);
                        acc[t] += d0 * a0[t];
                    }
                }
                const float d1 = half_bits_to_float((uint16_t) (u8 >> 16));
                const uint32_t v0 = p[9], v1 = p[10], v2 = p[11], v3 = p[12];
                const uint32_t v4 = p[13], v5 = p[14], v6 = p[15], v7 = p[16];
                float a1[N_CAP];
#pragma unroll
                for (int t = 0; t < N_CAP; ++t) a1[t] = 0.0f;
                {
                    float wv[16];
                    unpack_u32_i8(v0, wv);
                    unpack_u32_i8(v1, wv + 4);
                    unpack_u32_i8(v2, wv + 8);
                    unpack_u32_i8(v3, wv + 12);
#pragma unroll
                    for (int t = 0; t < N_CAP; ++t) {
                        if (t >= n) break;
                        a1[t] += dot16_vec(xs[t] + bi * 32 + 32, wv);
                    }
                    unpack_u32_i8(v4, wv);
                    unpack_u32_i8(v5, wv + 4);
                    unpack_u32_i8(v6, wv + 8);
                    unpack_u32_i8(v7, wv + 12);
#pragma unroll
                    for (int t = 0; t < N_CAP; ++t) {
                        if (t >= n) break;
                        a1[t] += dot16_vec(xs[t] + bi * 32 + 48, wv);
                        acc[t] += d1 * a1[t];
                    }
                }
            }
            if (nb & 1) {
                const int bi = nb - 1;
                const uint8_t* b = w + (size_t) bi * 34;
                const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
                const float wv = d * (float) (int8_t) b[2 + lane];
                const int off = bi * 32 + lane;
#pragma unroll
                for (int t = 0; t < N_CAP; ++t) {
                    if (t >= n) break;
                    acc[t] += wv * xval(xs[t], off);
                }
            }
        } else {
            for (int bi = lane; bi < nb; bi += 32) {
                const uint8_t* b = w + (size_t) bi * 34;
                const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
                for (int h = 0; h < 2; ++h) {
                    float wv[16];
#pragma unroll
                    for (int j = 0; j < 16; ++j) wv[j] = (float) (int8_t) b[2 + h * 16 + j];
#pragma unroll
                    for (int t = 0; t < N_CAP; ++t) {
                        if (t >= n) break;
                        acc[t] += d * dot16_vec(xs[t] + bi * 32 + h * 16, wv);
                    }
                }
            }
        }
    } else if (type == kQ6_K) {
        const int nb = K / 256;
        for (int bi = 0; bi < nb; ++bi) {
            const uint8_t* b = w + (size_t) bi * 210;
            const float d = half_bits_to_float((uint16_t) (b[208] | (b[209] << 8)));
            const int8_t* sc = (const int8_t*) (b + 192);
            const int x0 = bi * 256;
            float a[N_CAP];
#pragma unroll
            for (int t = 0; t < N_CAP; ++t) a[t] = 0.0f;
#pragma unroll
            for (int nn = 0; nn < 2; ++nn) {
                const uint8_t* ql = b + nn * 64;
                const uint8_t qh = b[128 + nn * 32 + lane];
                const uint8_t l0 = ql[lane], l1 = ql[lane + 32];
                const int is = nn * 8 + (lane >> 4);
                const float w0 = (float) sc[is + 0] * (float) ((int) ((l0 & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
                const float w1 = (float) sc[is + 2] * (float) ((int) ((l1 & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
                const float w2 = (float) sc[is + 4] * (float) ((int) ((l0 >> 4) | (((qh >> 4) & 3) << 4)) - 32);
                const float w3 = (float) sc[is + 6] * (float) ((int) ((l1 >> 4) | (((qh >> 6) & 3) << 4)) - 32);
                const int xn = x0 + nn * 128 + lane;
#pragma unroll
                for (int t = 0; t < N_CAP; ++t) {
                    if (t >= n) break;
                    a[t] += w0 * xval(xs[t], xn) + w1 * xval(xs[t], xn + 32) +
                            w2 * xval(xs[t], xn + 64) + w3 * xval(xs[t], xn + 96);
                }
            }
#pragma unroll
            for (int t = 0; t < N_CAP; ++t) {
                if (t >= n) break;
                acc[t] += d * a[t];
            }
        }
    } else if (type == kQ5_K) {
        const int nb = K / 256;
        for (int bi = 0; bi < nb; ++bi) {
            const uint8_t* b = w + (size_t) bi * 176;
            const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
            const float mn = half_bits_to_float((uint16_t) (b[2] | (b[3] << 8)));
            const uint8_t* q = b + 4;
            const uint8_t qh = b[16 + lane];
            const int x0 = bi * 256 + lane;
#pragma unroll
            for (int j64 = 0; j64 < 4; ++j64) {
                const uint8_t lo = b[48 + 32 * j64 + lane];
                const int is0 = 2 * j64, is1 = is0 + 1;
                int sc0, m0, sc1, m1;
                if (is0 < 4) {
                    sc0 = q[is0] & 63, m0 = q[is0 + 4] & 63;
                    sc1 = q[is1] & 63, m1 = q[is1 + 4] & 63;
                } else {
                    sc0 = (q[is0 + 4] & 0xF) | ((q[is0 - 4] >> 6) << 4), m0 = (q[is0 + 4] >> 4) | ((q[is0] >> 6) << 4);
                    sc1 = (q[is1 + 4] & 0xF) | ((q[is1 - 4] >> 6) << 4), m1 = (q[is1 + 4] >> 4) | ((q[is1] >> 6) << 4);
                }
                const float w0 = d * (float) sc0 * (float) ((lo & 0xF) + (((qh >> is0) & 1) ? 16 : 0)) - mn * (float) m0;
                const float w1 = d * (float) sc1 * (float) ((lo >> 4) + (((qh >> is1) & 1) ? 16 : 0)) - mn * (float) m1;
                const int xp0 = x0 + 64 * j64;
#pragma unroll
                for (int t = 0; t < N_CAP; ++t) {
                    if (t >= n) break;
                    acc[t] += w0 * xval(xs[t], xp0) + w1 * xval(xs[t], xp0 + 32);
                }
            }
        }
    } else {
        for (int i = lane; i < K; i += 32) {
            const float wv =
                decode_value(type, w, (size_t)i, d_iq3xxs_grid, d_iq4nl);
#pragma unroll
            for (int t = 0; t < N_CAP; ++t) {
                if (t >= n) break;
                acc[t] += wv * xval(xs[t], i);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < N_CAP; ++t) {
        float a = acc[t];
#pragma unroll
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        for (int o = 16; o > 0; o >>= 1) a +=
            dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(),
                a, o);
        out[t] = a;
    }
}

// Single-pass grouped (gate, up) dot of the same quant type against xs[0..n-1] (reading each xs[t] once)
template <int N_CAP>
/*
DPCT1110: The total declared local variable size in device function
row_dot_pair_multi exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__attribute__((noinline)) void row_dot_pair_multi(int type, const uint8_t *wg, const uint8_t *wu,
                               const float *const *xs, int n, int K, float *og,
                               float *ou, uint32_t *d_iq3xxs_grid,
                               int8_t *d_iq4nl) {
    const int lane =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) &
        31;
    float ag[N_CAP], au[N_CAP];
#pragma unroll
    for (int t = 0; t < N_CAP; ++t) ag[t] = 0.0f, au[t] = 0.0f;
    if (type == kNVFP4) {
        const int nsub = K / 16;
        for (int sb = lane; sb < nsub; sb += 32) {
            const size_t blk_off = (size_t) (sb >> 2) * 36;
            const int sub = sb & 3;
            const uint8_t* bg = wg + blk_off;
            const uint8_t* bu = wu + blk_off;
            const float dg = ue4m3((uint8_t) (*(const uint32_t*) bg >> (8 * sub)));
            const float du = ue4m3((uint8_t) (*(const uint32_t*) bu >> (8 * sub)));
            float wvg[16], wvu[16];
            decode_nvfp4_wv(*(const uint32_t*) (bg + 4 + sub * 8), *(const uint32_t*) (bg + 8 + sub * 8), wvg);
            decode_nvfp4_wv(*(const uint32_t*) (bu + 4 + sub * 8), *(const uint32_t*) (bu + 8 + sub * 8), wvu);
#pragma unroll
            for (int t = 0; t < N_CAP; ++t) {
                if (t >= n) break;
                const float* xp_t = xs[t] + sb * 16;
                if (((uintptr_t) xp_t & 15) == 0) {
                    const sycl::float4 *p4 = (const sycl::float4 *)xp_t;
                    const sycl::float4 v0 = ::ik2_ldg(p4 + 0),
                                       v1 = ::ik2_ldg(p4 + 1),
                                       v2 = ::ik2_ldg(p4 + 2),
                                       v3 = ::ik2_ldg(p4 + 3);
                    ag[t] +=
                        dg *
                        (wvg[0] * v0.x() + wvg[1] * v0.y() + wvg[2] * v0.z() +
                         wvg[3] * v0.w() + wvg[4] * v1.x() + wvg[5] * v1.y() +
                         wvg[6] * v1.z() + wvg[7] * v1.w() + wvg[8] * v2.x() +
                         wvg[9] * v2.y() + wvg[10] * v2.z() + wvg[11] * v2.w() +
                         wvg[12] * v3.x() + wvg[13] * v3.y() +
                         wvg[14] * v3.z() + wvg[15] * v3.w());
                    au[t] +=
                        du *
                        (wvu[0] * v0.x() + wvu[1] * v0.y() + wvu[2] * v0.z() +
                         wvu[3] * v0.w() + wvu[4] * v1.x() + wvu[5] * v1.y() +
                         wvu[6] * v1.z() + wvu[7] * v1.w() + wvu[8] * v2.x() +
                         wvu[9] * v2.y() + wvu[10] * v2.z() + wvu[11] * v2.w() +
                         wvu[12] * v3.x() + wvu[13] * v3.y() +
                         wvu[14] * v3.z() + wvu[15] * v3.w());
                } else {
                    float sg = 0.0f, su = 0.0f;
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const float xv = ik2_ldg(xp_t + i);
                        sg += wvg[i] * xv;
                        su += wvu[i] * xv;
                    }
                    ag[t] += dg * sg;
                    au[t] += du * su;
                }
            }
        }
    } else if (type == kQ6_K) {
        const int nb = K / 256;
        for (int bi = 0; bi < nb; ++bi) {
            const uint8_t* bg = wg + (size_t) bi * 210;
            const uint8_t* bu = wu + (size_t) bi * 210;
            const float dg = half_bits_to_float((uint16_t) (bg[208] | (bg[209] << 8)));
            const float du = half_bits_to_float((uint16_t) (bu[208] | (bu[209] << 8)));
            const int8_t* scg = (const int8_t*) (bg + 192);
            const int8_t* scu = (const int8_t*) (bu + 192);
            const int x0 = bi * 256;
            float sg[N_CAP], su[N_CAP];
#pragma unroll
            for (int t = 0; t < N_CAP; ++t) sg[t] = 0.0f, su[t] = 0.0f;
#pragma unroll
            for (int nn = 0; nn < 2; ++nn) {
                const int is = nn * 8 + (lane >> 4);
                const uint8_t* qlg = bg + nn * 64;
                const uint8_t qhg = bg[128 + nn * 32 + lane];
                const uint8_t lg0 = qlg[lane], lg1 = qlg[lane + 32];
                const float wg0 = (float) scg[is + 0] * (float) ((int) ((lg0 & 0xF) | (((qhg >> 0) & 3) << 4)) - 32);
                const float wg1 = (float) scg[is + 2] * (float) ((int) ((lg1 & 0xF) | (((qhg >> 2) & 3) << 4)) - 32);
                const float wg2 = (float) scg[is + 4] * (float) ((int) ((lg0 >> 4) | (((qhg >> 4) & 3) << 4)) - 32);
                const float wg3 = (float) scg[is + 6] * (float) ((int) ((lg1 >> 4) | (((qhg >> 6) & 3) << 4)) - 32);

                const uint8_t* qlu = bu + nn * 64;
                const uint8_t qhu = bu[128 + nn * 32 + lane];
                const uint8_t lu0 = qlu[lane], lu1 = qlu[lane + 32];
                const float wu0 = (float) scu[is + 0] * (float) ((int) ((lu0 & 0xF) | (((qhu >> 0) & 3) << 4)) - 32);
                const float wu1 = (float) scu[is + 2] * (float) ((int) ((lu1 & 0xF) | (((qhu >> 2) & 3) << 4)) - 32);
                const float wu2 = (float) scu[is + 4] * (float) ((int) ((lu0 >> 4) | (((qhu >> 4) & 3) << 4)) - 32);
                const float wu3 = (float) scu[is + 6] * (float) ((int) ((lu1 >> 4) | (((qhu >> 6) & 3) << 4)) - 32);

                const int xn = x0 + nn * 128 + lane;
#pragma unroll
                for (int t = 0; t < N_CAP; ++t) {
                    if (t >= n) break;
                    const float xv0 = ik2_ldg(xs[t] + xn), xv1 = ik2_ldg(xs[t] + xn + 32);
                    const float xv2 = ik2_ldg(xs[t] + xn + 64), xv3 = ik2_ldg(xs[t] + xn + 96);
                    sg[t] += wg0 * xv0 + wg1 * xv1 + wg2 * xv2 + wg3 * xv3;
                    su[t] += wu0 * xv0 + wu1 * xv1 + wu2 * xv2 + wu3 * xv3;
                }
            }
#pragma unroll
            for (int t = 0; t < N_CAP; ++t) {
                if (t >= n) break;
                ag[t] += dg * sg[t];
                au[t] += du * su[t];
            }
        }
    } else if (type == kQ5_K) {
        const int nb = K / 256;
        for (int bi = 0; bi < nb; ++bi) {
            const uint8_t* bg = wg + (size_t) bi * 176;
            const uint8_t* bu = wu + (size_t) bi * 176;
            const float dg = half_bits_to_float((uint16_t) (bg[0] | (bg[1] << 8)));
            const float mng = half_bits_to_float((uint16_t) (bg[2] | (bg[3] << 8)));
            const float du = half_bits_to_float((uint16_t) (bu[0] | (bu[1] << 8)));
            const float mnu = half_bits_to_float((uint16_t) (bu[2] | (bu[3] << 8)));
            const uint8_t* qg = bg + 4;
            const uint8_t* qu = bu + 4;
            const uint8_t qhg = bg[16 + lane], qhu = bu[16 + lane];
            const int x0 = bi * 256 + lane;
#pragma unroll
            for (int j64 = 0; j64 < 4; ++j64) {
                const uint8_t log_ = bg[48 + 32 * j64 + lane], lou = bu[48 + 32 * j64 + lane];
                const int is0 = 2 * j64, is1 = is0 + 1;
                int scg0, mg0, scg1, mg1, scu0, mu0, scu1, mu1;
                if (is0 < 4) {
                    scg0 = qg[is0] & 63, mg0 = qg[is0 + 4] & 63, scg1 = qg[is1] & 63, mg1 = qg[is1 + 4] & 63;
                    scu0 = qu[is0] & 63, mu0 = qu[is0 + 4] & 63, scu1 = qu[is1] & 63, mu1 = qu[is1 + 4] & 63;
                } else {
                    scg0 = (qg[is0 + 4] & 0xF) | ((qg[is0 - 4] >> 6) << 4), mg0 = (qg[is0 + 4] >> 4) | ((qg[is0] >> 6) << 4);
                    scg1 = (qg[is1 + 4] & 0xF) | ((qg[is1 - 4] >> 6) << 4), mg1 = (qg[is1 + 4] >> 4) | ((qg[is1] >> 6) << 4);
                    scu0 = (qu[is0 + 4] & 0xF) | ((qu[is0 - 4] >> 6) << 4), mu0 = (qu[is0 + 4] >> 4) | ((qu[is0] >> 6) << 4);
                    scu1 = (qu[is1 + 4] & 0xF) | ((qu[is1 - 4] >> 6) << 4), mu1 = (qu[is1 + 4] >> 4) | ((qu[is1] >> 6) << 4);
                }
                const float wg0 = dg * (float) scg0 * (float) ((log_ & 0xF) + (((qhg >> is0) & 1) ? 16 : 0)) - mng * (float) mg0;
                const float wg1 = dg * (float) scg1 * (float) ((log_ >> 4) + (((qhg >> is1) & 1) ? 16 : 0)) - mng * (float) mg1;
                const float wu0 = du * (float) scu0 * (float) ((lou & 0xF) + (((qhu >> is0) & 1) ? 16 : 0)) - mnu * (float) mu0;
                const float wu1 = du * (float) scu1 * (float) ((lou >> 4) + (((qhu >> is1) & 1) ? 16 : 0)) - mnu * (float) mu1;
                const int xp0 = x0 + 64 * j64;
#pragma unroll
                for (int t = 0; t < N_CAP; ++t) {
                    if (t >= n) break;
                    const float xv0 = ik2_ldg(xs[t] + xp0), xv1 = ik2_ldg(xs[t] + xp0 + 32);
                    ag[t] += wg0 * xv0 + wg1 * xv1;
                    au[t] += wu0 * xv0 + wu1 * xv1;
                }
            }
        }
    } else {
        row_dot_multi<float, N_CAP>(type, wg, xs, n, K, og, d_iq3xxs_grid,
                                    d_iq4nl);
        row_dot_multi<float, N_CAP>(type, wu, xs, n, K, ou, d_iq3xxs_grid,
                                    d_iq4nl);
        return;
    }
#pragma unroll
    for (int t = 0; t < N_CAP; ++t) {
        float g_a = ag[t], u_a = au[t];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            g_a += dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(),
                g_a, o);
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            u_a += dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(),
                u_a, o);
        }
        og[t] = g_a;
        ou[t] = u_a;
    }
}

constexpr int kGroupWarps = 4, kGroupRows = 4, kGroupSwigluRows = 2;

template <int N_CAP>
__dpct_inline__ void experts_gate_up_g_run(
    const uint8_t *bl, const int *tok_g, const int *orow_g, int ng, int ty_gate,
    int ty_up, size_t eb0, const float *x, int C, int F, size_t rb_gate,
    size_t rb_up, float *gu, uint32_t *d_iq3xxs_grid, int8_t *d_iq4nl) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const float *xs[N_CAP];
#pragma unroll
    for (int t = 0; t < N_CAP; ++t) xs[t] = x + (size_t) tok_g[t < ng ? t : 0] * C;
    const int warp = item_ct1.get_local_id(2) >> 5;
    float out[N_CAP];
    for (int j = 0; j < kGroupRows; ++j) {
        const int row =
            (item_ct1.get_group(2) * kGroupWarps + warp) * kGroupRows + j;
        if (row >= 2 * F) break;
        const bool up = row >= F;
        const int r = up ? row - F : row;
        row_dot_multi<float, N_CAP>(up ? ty_up : ty_gate,
                                    bl + (up ? eb0 + r * rb_up : r * rb_gate),
                                    xs, ng, C, out, d_iq3xxs_grid, d_iq4nl);
        if ((item_ct1.get_local_id(2) & 31) == 0)
#pragma unroll
            for (int t = 0; t < ng; ++t)
                gu[(size_t)orow_g[t] * 2 * F + row] = out[t];
    }
}

template <int N_CAP>
/*
DPCT1110: The total declared local variable size in device function
experts_gate_up_swiglu_g_run exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
__dpct_inline__ void experts_gate_up_swiglu_g_run(
    const uint8_t *bl, const int *tok_g, const int *orow_g, int ng, int ty_gate,
    int ty_up, size_t eb0, const float *x, int C, int F, size_t rb_gate,
    size_t rb_up, const float *sg, const float *su, float gcap, float ucap,
    uint16_t *h, uint32_t *d_iq3xxs_grid, int8_t *d_iq4nl) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const float *xs[N_CAP];
#pragma unroll
    for (int t = 0; t < N_CAP; ++t) xs[t] = x + (size_t) tok_g[t < ng ? t : 0] * C;
    const int warp = item_ct1.get_local_id(2) >> 5;
    float og[N_CAP], ou[N_CAP];
    for (int j = 0; j < kGroupSwigluRows; ++j) {
        const int r =
            (item_ct1.get_group(2) * kGroupWarps + warp) * kGroupSwigluRows + j;
        if (r >= F) break;
        if (ty_gate == ty_up) {
            row_dot_pair_multi<N_CAP>(ty_gate, bl + (size_t)r * rb_gate,
                                      bl + eb0 + (size_t)r * rb_up, xs, ng, C,
                                      og, ou, d_iq3xxs_grid, d_iq4nl);
        } else {
            row_dot_multi<float, N_CAP>(ty_gate, bl + (size_t)r * rb_gate, xs,
                                        ng, C, og, d_iq3xxs_grid, d_iq4nl);
            row_dot_multi<float, N_CAP>(ty_up, bl + eb0 + (size_t)r * rb_up, xs,
                                        ng, C, ou, d_iq3xxs_grid, d_iq4nl);
        }
        if ((item_ct1.get_local_id(2) & 31) == 0) {
            for (int t = 0; t < ng; ++t) {
                const int ro = orow_g[t];
                const float g = og[t] * (sg ? sg[ro] : 1.0f);
                const float u = ou[t] * (su ? su[ro] : 1.0f);
                h[(size_t)ro * F + r] =
                    f2h(gcap * sycl::tanh(g / gcap) *
                        (1.0f / (1.0f + sycl::native::exp(-g))) * ucap *
                        sycl::tanh(u / ucap));
            }
        }
    }
}

template <int N_CAP>
__dpct_inline__ void experts_down_g_run(
    const uint8_t *w, const int *orow_g, int ng, int ty_down, const uint16_t *h,
    int C, int F, size_t rb_down, float *y, uint32_t *d_iq3xxs_grid,
    int8_t *d_iq4nl) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint16_t *xs[N_CAP];
#pragma unroll
    for (int t = 0; t < N_CAP; ++t) xs[t] = h + (size_t) orow_g[t < ng ? t : 0] * F;
    const int warp = item_ct1.get_local_id(2) >> 5;
    float out[N_CAP];
    for (int j = 0; j < kGroupRows; ++j) {
        const int row =
            (item_ct1.get_group(2) * kGroupWarps + warp) * kGroupRows + j;
        if (row >= C) break;
        row_dot_multi<uint16_t, N_CAP>(ty_down, w + row * rb_down, xs, ng, F,
                                       out, d_iq3xxs_grid, d_iq4nl);
        if ((item_ct1.get_local_id(2) & 31) == 0)
#pragma unroll
            for (int t = 0; t < ng; ++t)
                y[(size_t)orow_g[t] * C + row] = out[t];
    }
}

// group g: blob[g], n[g] tokens, their rows of x tok[g][j] and their output rows row[g][j]
__dpct_inline__ void experts_gate_up_g_kernel(
    const uint8_t *const *blob, const int *n, const int *tok, const int *orow,
    int ty_gate, int ty_up, size_t eb0, const float *x, int C, int F,
    size_t rb_gate, size_t rb_up, float *gu, uint32_t *d_iq3xxs_grid,
    int8_t *d_iq4nl) {
    const int g = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(
                  1),
              ng = n[g];
    if (ng <= 0) return;
    const uint8_t* bl = blob[g];
    const int* tok_g = tok + g * kGroupMax;
    const int* orow_g = orow + g * kGroupMax;
    if (ng == 1)
        experts_gate_up_g_run<1>(bl, tok_g, orow_g, ng, ty_gate, ty_up, eb0, x,
                                 C, F, rb_gate, rb_up, gu, d_iq3xxs_grid,
                                 d_iq4nl);
    else if (ng == 2)
        experts_gate_up_g_run<2>(bl, tok_g, orow_g, ng, ty_gate, ty_up, eb0, x,
                                 C, F, rb_gate, rb_up, gu, d_iq3xxs_grid,
                                 d_iq4nl);
    else if (ng <= 4)
        experts_gate_up_g_run<4>(bl, tok_g, orow_g, ng, ty_gate, ty_up, eb0, x,
                                 C, F, rb_gate, rb_up, gu, d_iq3xxs_grid,
                                 d_iq4nl);
    else
        experts_gate_up_g_run<kGroupMax>(bl, tok_g, orow_g, ng, ty_gate, ty_up,
                                         eb0, x, C, F, rb_gate, rb_up, gu,
                                         d_iq3xxs_grid, d_iq4nl);
}

__dpct_inline__ void experts_gate_up_swiglu_g_kernel(
    const uint8_t *const *blob, const int *n, const int *tok, const int *orow,
    int ty_gate, int ty_up, size_t eb0, const float *x, int C, int F,
    size_t rb_gate, size_t rb_up, const float *sg, const float *su, float gcap,
    float ucap, uint16_t *h, uint32_t *d_iq3xxs_grid, int8_t *d_iq4nl) {
    const int g = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(
                  1),
              ng = n[g];
    if (ng <= 0) return;
    const uint8_t* bl = blob[g];
    const int* tok_g = tok + g * kGroupMax;
    const int* orow_g = orow + g * kGroupMax;
    if (ng == 1)
        experts_gate_up_swiglu_g_run<1>(bl, tok_g, orow_g, ng, ty_gate, ty_up,
                                        eb0, x, C, F, rb_gate, rb_up, sg, su,
                                        gcap, ucap, h, d_iq3xxs_grid, d_iq4nl);
    else if (ng == 2)
        experts_gate_up_swiglu_g_run<2>(bl, tok_g, orow_g, ng, ty_gate, ty_up,
                                        eb0, x, C, F, rb_gate, rb_up, sg, su,
                                        gcap, ucap, h, d_iq3xxs_grid, d_iq4nl);
    else if (ng <= 4)
        experts_gate_up_swiglu_g_run<4>(bl, tok_g, orow_g, ng, ty_gate, ty_up,
                                        eb0, x, C, F, rb_gate, rb_up, sg, su,
                                        gcap, ucap, h, d_iq3xxs_grid, d_iq4nl);
    else
        experts_gate_up_swiglu_g_run<kGroupMax>(
            bl, tok_g, orow_g, ng, ty_gate, ty_up, eb0, x, C, F, rb_gate, rb_up,
            sg, su, gcap, ucap, h, d_iq3xxs_grid, d_iq4nl);
}

__dpct_inline__ void experts_down_g_kernel(
    const uint8_t *const *blob, const int *n, const int *orow, int ty_down,
    size_t off_down, const uint16_t *h, int C, int F, size_t rb_down, float *y,
    uint32_t *d_iq3xxs_grid, int8_t *d_iq4nl) {
    const int g = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(
                  1),
              ng = n[g];
    if (ng <= 0) return;
    const uint8_t* w = blob[g] + off_down;
    const int* orow_g = orow + g * kGroupMax;
    if (ng == 1)
        experts_down_g_run<1>(w, orow_g, ng, ty_down, h, C, F, rb_down, y,
                              d_iq3xxs_grid, d_iq4nl);
    else if (ng == 2)
        experts_down_g_run<2>(w, orow_g, ng, ty_down, h, C, F, rb_down, y,
                              d_iq3xxs_grid, d_iq4nl);
    else if (ng <= 4)
        experts_down_g_run<4>(w, orow_g, ng, ty_down, h, C, F, rb_down, y,
                              d_iq3xxs_grid, d_iq4nl);
    else
        experts_down_g_run<kGroupMax>(w, orow_g, ng, ty_down, h, C, F, rb_down,
                                      y, d_iq3xxs_grid, d_iq4nl);
}

// ---- attention decode & multi-query decode (F32, FP16, or FP8 KV cache, optional kv_cap ring buffer)

template <class KV>
__dpct_inline__ sycl::float4 v_load4(const KV *vr, int lane);
template <>
__dpct_inline__ sycl::float4 v_load4<float>(const float *vr, int lane) {
    const int o = lane * 4;
    if (((uintptr_t)vr & 15) ==
        0) return ::ik2_ldg((const sycl::float4 *)(vr + o));
    return sycl::float4(vr[o], vr[o + 1], vr[o + 2], vr[o + 3]);
}
template <>
__dpct_inline__ sycl::float4 v_load4<uint16_t>(const uint16_t *vr, int lane) {
    const int o = lane * 4;
    if (((uintptr_t) vr & 7) == 0) {
        const sycl::uint2 v4 = ::ik2_ldg((const sycl::uint2 *)(vr + o));
        const sycl::float2 f0 = u32_to_f2(v4.x()), f1 = u32_to_f2(v4.y());
        return sycl::float4(f0.x(), f0.y(), f1.x(), f1.y());
    }
    return sycl::float4(h2f(vr[o]), h2f(vr[o + 1]), h2f(vr[o + 2]),
                        h2f(vr[o + 3]));
}
template <>
__dpct_inline__ sycl::float4 v_load4<uint8_t>(const uint8_t *vr, int lane) {
    const int o = lane * 4;
    if (((uintptr_t) vr & 3) == 0) {
        const uint32_t v4 = ik2_ldg((const uint32_t*) (vr + o));
        const sycl::float2 f0 = e4m3fnx2_to_f2((uint16_t)(v4 & 0xFFFFu));
        const sycl::float2 f1 = e4m3fnx2_to_f2((uint16_t)(v4 >> 16));
        return sycl::float4(f0.x(), f0.y(), f1.x(), f1.y());
    }
    return sycl::float4(e4m3fn_to_f32(vr[o]), e4m3fn_to_f32(vr[o + 1]),
                        e4m3fn_to_f32(vr[o + 2]), e4m3fn_to_f32(vr[o + 3]));
}

template <class KV>
/*
DPCT1110: The total declared local variable size in device function
attn_flash_range exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void attn_flash_range(
    const float *qh, const float *rh, const KV *kc, const KV *vc, int KD, int g,
    int kv_cap, int k0, int j0, int j1, int qpos, float qk_scale,
    const float *proj, int d_rel, int extent, int window, float *sm_m,
    float *sm_s, float sm_o[][128]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5;
    const int o = lane * 4;
    const sycl::float4 q4 =
        (((uintptr_t)qh & 15) == 0)
            ? ::ik2_ldg((const sycl::float4 *)(qh + o))
            : sycl::float4(qh[o], qh[o + 1], qh[o + 2], qh[o + 3]);
    const float r_lane = (lane < d_rel) ? ik2_ldg(rh + lane) : 0.0f;
    float m_w = -INFINITY, s_w = 0.0f;
    sycl::float4 o_w = sycl::float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int j = j0 + warp; j < j1; j += 4) {
        const int pos = k0 + j;
        const int dist = qpos - pos;
        if (dist < 0 || dist >= window) continue;
        const int slot = kv_cap > 0 ? pos % kv_cap : pos;
        const size_t head_off = (size_t) slot * KD + (size_t) g * 128;
        const sycl::float4 k4 = v_load4(kc + head_off, lane);
        float a = q4.x() * k4.x() + q4.y() * k4.y() + q4.z() * k4.z() +
                  q4.w() * k4.w();
        float l = wsum(a) * qk_scale;
        if (dist < extent) {
            const float b = (lane < d_rel) ? r_lane * ik2_ldg(proj + (size_t) dist * d_rel + lane) : 0.0f;
            l += wsum(b);
        }
        if (l > -INFINITY) {
            const sycl::float4 v4 = v_load4(vc + head_off, lane);
            const float m_new = sycl::fmax(m_w, l);
            const float alpha =
                (m_w == -INFINITY) ? 0.0f : sycl::native::exp(m_w - m_new);
            const float beta = sycl::native::exp(l - m_new);
            m_w = m_new;
            s_w = s_w * alpha + beta;
            o_w.x() = o_w.x() * alpha + v4.x() * beta;
            o_w.y() = o_w.y() * alpha + v4.y() * beta;
            o_w.z() = o_w.z() * alpha + v4.z() * beta;
            o_w.w() = o_w.w() * alpha + v4.w() * beta;
        }
    }
    if (lane == 0) {
        sm_m[warp] = m_w;
        sm_s[warp] = s_w;
    }
    *(sycl::float4 *)(&sm_o[warp][o]) = o_w;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
}

// Unified split kernel for both attn_decode (gridDim.z = 1) and attn_multi (gridDim.z = nq).
template <class KV>
__dpct_inline__ void attn_split_kernel(
    const float *q, const float *r, const KV *kc, const KV *vc, int kv_cap,
    int H, int Hkv, int k0, int nk, int qpos0, float qk_scale,
    const float *proj, int d_rel, int extent, int window, float *part,
    const int *pos_d) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (pos_d) { // (tokens of a replayed graph: the first's position read here,
                 // the keys they may see from it)
        qpos0 = *pos_d;
        k0 = window < (1 << 30) ? sycl::max(0, qpos0 - window + 1) : 0;
        nk = qpos0 + (int)item_ct1.get_group_range(0) - k0;
    }
    auto &sm_m = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[4]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sm_s = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[4]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sm_o =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[4][128]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int h = item_ct1.get_group(2), p = item_ct1.get_group(1),
              P = item_ct1.get_group_range(1), qi = item_ct1.get_group(0),
              g = h / (H / Hkv), tid = item_ct1.get_local_id(2);
    const int KD = Hkv * 128, qpos = qpos0 + qi;
    const int len = (nk + P - 1) / P, j0 = p * len,
              j1 = sycl::min(nk, j0 + len);
    const float* qh = q + ((size_t) qi * H + h) * 128;
    const float* rh = r + ((size_t) qi * H + h) * d_rel;
    attn_flash_range(qh, rh, kc, vc, KD, g, kv_cap, k0, j0, j1, qpos, qk_scale, proj, d_rel, extent, window, sm_m,
                     sm_s, sm_o);
    const float M =
        sycl::fmax(sycl::fmax(sm_m[0], sm_m[1]), sycl::fmax(sm_m[2], sm_m[3]));
    float sum = 0.0f, o = 0.0f;
    if (M > -INFINITY) {
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (sm_m[w] == -INFINITY) continue;
            const float f = sycl::native::exp(sm_m[w] - M);
            sum += sm_s[w] * f;
            o += sm_o[w][tid] * f;
        }
    }
    float* pp = part + (((size_t) qi * H + h) * P + p) * 130;
    if (tid == 0) pp[0] = M, pp[1] = sum;
    pp[2 + tid] = o;
}

__dpct_inline__ void attn_combine_kernel(const float *sink, const float *part,
                                         int H, int P, float v_inv_scale,
                                         float *out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int h = item_ct1.get_group(2), qi = item_ct1.get_group(1),
              tid = item_ct1.get_local_id(2);
    const float* pp = part + ((size_t) qi * H + h) * P * 130;
    const float sk = sink[(size_t) qi * H + h];
    float M = sk;
#pragma unroll
    for (int p = 0; p < P; ++p) M = sycl::fmax(M, pp[(size_t)p * 130]);
    float den = sycl::native::exp(sk - M), o = 0.0f;
    for (int p = 0; p < P; ++p) {
        const float mp = pp[(size_t) p * 130];
        if (mp == -INFINITY) continue;
        const float f = sycl::native::exp(mp - M);
        den += pp[(size_t) p * 130 + 1] * f;
        o += pp[(size_t) p * 130 + 2 + tid] * f;
    }
    out[((size_t) qi * H + h) * 128 + tid] = (o / den) * v_inv_scale;
}

template <int WARPS, int ROWS>
__dpct_inline__ void mv_blocks_kernel(
    int type, const uint8_t *W, int N, int K, size_t rb, const float *x,
    float *y, uint8_t *dpct_local, const uint32_t *d_iq3xxs_grid,
    const int8_t *d_iq4nl) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto xs = (float *)dpct_local;
    load_xs_f32(x, K, xs);
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int warp = item_ct1.get_local_id(2) >> 5;
    for (int j = 0; j < ROWS; ++j) {
        const int row = (item_ct1.get_group(2) * WARPS + warp) * ROWS + j;
        if (row >= N) break;
        const float v =
            row_dot(type, W + (size_t)row * rb, xs, K, d_iq3xxs_grid, d_iq4nl);
        if ((item_ct1.get_local_id(2) & 31) == 0) y[row] = v;
    }
}

// one block a head, 4 warps: register Flash-Decode online softmax per warp, 1 SMEM combine
template <class KV>
__dpct_inline__ void attn_decode_kernel(
    const float *q, const float *r, const float *sink, const KV *kc,
    const KV *vc, int kv_cap, int H, int Hkv, int k0, int nk, int qpos,
    float qk_scale, float v_inv_scale, const float *proj, int d_rel, int extent,
    int window, float *out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sm_m = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[4]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sm_s = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[4]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sm_o =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[4][128]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int h = item_ct1.get_group(2), g = h / (H / Hkv),
              tid = item_ct1.get_local_id(2);
    const int KD = Hkv * 128;
    const float* qh = q + (size_t) h * 128;
    const float* rh = r + (size_t) h * d_rel;
    attn_flash_range(qh, rh, kc, vc, KD, g, kv_cap, k0, 0, nk, qpos, qk_scale, proj, d_rel, extent, window, sm_m, sm_s,
                     sm_o);
    const float sk = sink[h];
    const float M = sycl::fmax(sk, sycl::fmax(sycl::fmax(sm_m[0], sm_m[1]),
                                              sycl::fmax(sm_m[2], sm_m[3])));
    float den = sycl::native::exp(sk - M), o = 0.0f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        if (sm_m[w] == -INFINITY) continue;
        const float f = sycl::native::exp(sm_m[w] - M);
        den += sm_s[w] * f;
        o += sm_o[w][tid] * f;
    }
    out[(size_t) h * 128 + tid] = (o / den) * v_inv_scale;
}

__dpct_inline__ void unpack_kv_fp8_to_f16_vec4_kernel(
    const uint8_t *kc, const uint8_t *vc, int kv_cap, int k0, int nk, int KD,
    float k_inv_scale, float v_inv_scale, uint16_t *k_out16,
    uint16_t *v_out16) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int kd4 = KD >> 2;
    const int d4 = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                   item_ct1.get_local_id(2);
    if (d4 >= kd4) return;
    for (int j = item_ct1.get_group(1); j < nk;
         j += item_ct1.get_group_range(1)) {
        const int slot = kv_cap > 0 ? (k0 + j) % kv_cap : (k0 + j);
        const size_t src_off = (size_t) slot * KD;
        const size_t dst_off = (size_t) j * KD;
        if (k_out16) {
            const uint32_t k4 = ik2_ldg((const uint32_t*) (kc + src_off) + d4);
            const sycl::float2 f0 = e4m3fnx2_to_f2((uint16_t)(k4 & 0xFFFFu));
            const sycl::float2 f1 = e4m3fnx2_to_f2((uint16_t)(k4 >> 16));
            ((sycl::uint2 *)(k_out16 + dst_off))[d4] = sycl::uint2(
                f2_to_u32(f0.x() * k_inv_scale, f0.y() * k_inv_scale),
                f2_to_u32(f1.x() * k_inv_scale, f1.y() * k_inv_scale));
        }
        if (v_out16) {
            const uint32_t v4 = ik2_ldg((const uint32_t*) (vc + src_off) + d4);
            const sycl::float2 f0 = e4m3fnx2_to_f2((uint16_t)(v4 & 0xFFFFu));
            const sycl::float2 f1 = e4m3fnx2_to_f2((uint16_t)(v4 >> 16));
            ((sycl::uint2 *)(v_out16 + dst_off))[d4] = sycl::uint2(
                f2_to_u32(f0.x() * v_inv_scale, f0.y() * v_inv_scale),
                f2_to_u32(f1.x() * v_inv_scale, f1.y() * v_inv_scale));
        }
    }
}

__dpct_inline__ void unpack_kv_fp8_to_f16_kernel(
    const uint8_t *kc, const uint8_t *vc, int kv_cap, int k0, int nk, int KD,
    float k_inv_scale, float v_inv_scale, uint16_t *k_out16,
    uint16_t *v_out16) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t total = (size_t)nk * KD;
    for (size_t idx =
             (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         idx < total; idx += (size_t)item_ct1.get_group_range(2) *
                             item_ct1.get_local_range(2)) {
        const int j = (int) (idx / (size_t) KD);
        const int d = (int) (idx % (size_t) KD);
        const int slot = kv_cap > 0 ? (k0 + j) % kv_cap : (k0 + j);
        const size_t src = (size_t) slot * KD + d;
        if (k_out16) k_out16[idx] = f2h(e4m3fn_to_f32(kc[src]) * k_inv_scale);
        if (v_out16) v_out16[idx] = f2h(e4m3fn_to_f32(vc[src]) * v_inv_scale);
    }
}

// ---- the resident token path: every expert in VRAM and nothing of a token on the host

// y [N] = W [N][K] FP16 . x [K]: a warp a row (the small dense matrices kept in FP16: sinks, dynamic taps, the router)
template <int WARPS>
__dpct_inline__ void mv_f16_kernel(const uint16_t *__restrict__ W, int N, int K,
                                   const float *__restrict__ x,
                                   float *__restrict__ y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2) * WARPS +
                    (item_ct1.get_local_id(2) >> 5),
              lane = item_ct1.get_local_id(2) & 31;
    if (row >= N) return;
    const uint16_t* w = W + (size_t) row * K;
    float a = 0.0f;
#pragma unroll
    for (int i = lane; i < K; i += 32) a += h2f(w[i]) * x[i];
    a = wsum(a);
    if (lane == 0) y[row] = a;
}

// a product's row through its short conv (conv4_fused_kernel's arithmetic for one token): false if the row has none
__dpct_inline__ bool mv_conv_row(const MvConv &cv, int row, float x_new) {
    const int c = row - cv.r0;
    if (c < 0 || c >= cv.n) return false;
    const float h0 = cv.hist[c], h1 = cv.hist[(size_t) cv.n + c], h2 = cv.hist[(size_t) 2 * cv.n + c];
    const float* wc = cv.w + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    float a = w0 * h0 + w1 * h1 + w2 * h2 + w3 * x_new;
    if (cv.dyn) {
        const float* dc = cv.dyn + (size_t) (c / (cv.n / cv.heads)) * 4;
        a += dc[0] * x_new + dc[1] * h2 + dc[2] * h1 + dc[3] * h0;
    }
    if (cv.res) a += cv.res[c];
    cv.y[c] = a;
    cv.hist[c] = h1;
    cv.hist[(size_t) cv.n + c] = h2;
    cv.hist[(size_t) 2 * cv.n + c] = x_new;
    return true;
}

// y [N] = W . x for a Q8_0 matrix kept planar - its values q [N][K] int8 and its blocks' scales d [N][K / 32] FP16 -
// so that a warp reads 128 values an instruction; x where it is.  A warp a row; a row of c0's or c1's goes through
// its short conv instead of to y (a launch less a conv: a token's launches each cost 2 us and more).
// Quantized planar Q8 GEMM for the small verify/prefill tiles. One scale covers
// both 16-wide MMA instructions. Packing the signed bytes via 1024+u is exact
// for all 256 inputs; the product rounds to half as in decode_q8p_f16_kernel.
template <int Splits>
/*
DPCT1110: The total declared local variable size in device function
q8_mma_tile exceeds 128 bytes and may cause high register pressure. Consult with
your hardware vendor to find the total register size available and adjust the
code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void q8_mma_tile(const int8_t *q, const uint16_t *scales, int N,
                                 int K, const uint16_t *x, int T, float *out,
                                 int ldy) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) % 32,
              split = item_ct1.get_local_id(2) / 32, g = lane / 4, t = lane % 4;
    const int row = 16 * item_ct1.get_group(2) + g,
              token = 8 * item_ct1.get_group(1) + g;
    const int row0 = sycl::min(row, N - 1), row1 = sycl::min(row + 8, N - 1);
    const int blocks = K / 32;
    float acc[4] = {};
    for (int i = split; i < blocks; i += Splits) {
        const int col = i * 32 + t * 8;
        const sycl::uint2 a0 =
            *(const sycl::uint2 *)(q + (size_t)row0 * K + col);
        const sycl::uint2 a1 =
            *(const sycl::uint2 *)(q + (size_t)row1 * K + col);
        uint32_t s0 = scales[(size_t)row0 * blocks + i], s1 = scales[(size_t)row1 * blocks + i];
        s0 |= s0 << 16;
        s1 |= s1 << 16;
        const sycl::uint4 v = *(
            const sycl::uint4 *)(x + (size_t)sycl::min(token, T - 1) * K + col);
        uint32_t a[4] = {q8_h2(a0.x(), s0), q8_h2(a1.x(), s1),
                         q8_h2(a0.y(), s0), q8_h2(a1.y(), s1)};
        mma16816(acc, a, v.x(), v.z());
        a[0] = q8_h2(a0.x() >> 16, s0);
        a[1] = q8_h2(a1.x() >> 16, s1);
        a[2] = q8_h2(a0.y() >> 16, s0);
        a[3] = q8_h2(a1.y() >> 16, s1);
        mma16816(acc, a, v.y(), v.w());
    }
    auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        sycl::float4[Splits][32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    part[split][lane] = sycl::float4(acc[0], acc[1], acc[2], acc[3]);
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (split) return;
    sycl::float4 v = part[0][lane];
#pragma unroll
    for (int i = 1; i < Splits; ++i) {
        const sycl::float4 z = part[i][lane];
        v.x() += z.x();
        v.y() += z.y();
        v.z() += z.z();
        v.w() += z.w();
    }
    const int token0 = item_ct1.get_group(1) * 8 + 2 * t;
    if (token0 < T) {
        if (row < N) out[(size_t)token0 * ldy + row] = v.x();
        if (row + 8 < N) out[(size_t)token0 * ldy + row + 8] = v.z();
    }
    if (token0 + 1 < T) {
        if (row < N) out[(size_t)(token0 + 1) * ldy + row] = v.y();
        if (row + 8 < N) out[(size_t)(token0 + 1) * ldy + row + 8] = v.w();
    }
}

template <int WARPS, int StaticK = 0>
__dpct_inline__ void mv_q8p_kernel(
    const int8_t *__restrict__ q, const uint16_t *__restrict__ d, int N,
    int runtime_k, const float *__restrict__ x, float *y, MvConv c0, MvConv c1,
    MvAlt a0, MvAlt a1) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int K = StaticK ? StaticK : runtime_k;
    const int row = item_ct1.get_group(2) * WARPS +
                    (item_ct1.get_local_id(2) >> 5),
              lane = item_ct1.get_local_id(2) & 31;
    if (row >= N) return;
    const uint32_t* qr = (const uint32_t*) (q + (size_t) row * K);
    const uint16_t* dr = d + (size_t) row * (K >> 5);
    // (a0's and a1's rows have an input of their own: several products of a layer's fused matrix in one launch)
    const sycl::float4 *x4 =
        (const sycl::float4 *)((unsigned)(row - a0.r0) < (unsigned)a0.n   ? a0.x
                               : (unsigned)(row - a1.r0) < (unsigned)a1.n ? a1.x
                                                                          : x);
    float acc = 0.0f;
#pragma unroll 4
    for (int it = 0, n_it = K >> 7; it < n_it; ++it) {
        const uint32_t w = ik2_ldg(qr + it * 32 + lane);
        const sycl::float4 xv = ::ik2_ldg(x4 + it * 32 + lane);
        const float dv = h2f(ik2_ldg(dr + it * 4 + (lane >> 3)));
        acc += dv * ((float)(int8_t)(w & 0xFFu) * xv.x() +
                     (float)(int8_t)((w >> 8) & 0xFFu) * xv.y() +
                     (float)(int8_t)((w >> 16) & 0xFFu) * xv.z() +
                     (float)(int8_t)(w >> 24) * xv.w());
    }
    acc = wsum(acc);
    if (lane == 0 && !mv_conv_row(c0, row, acc) && !mv_conv_row(c1, row, acc)) y[row] = acc;
}
__dpct_inline__ void decode_q8p_f16_kernel(
    const int8_t *__restrict__ q, const uint16_t *__restrict__ d, size_t n,
    uint16_t *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t i =
        (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) out[i] = f2h(h2f(d[i >> 5]) * (float) q[i]);
}

// the routed slots' tables from the picks, on the device: slot i's expert e = top[i] - its blocks, its token, its
// scales; wt = the route weight times down's scale
__dpct_inline__ void route_gather_kernel(
    const int *__restrict__ top, const float *__restrict__ w,
    const uint8_t *const *__restrict__ tab, const float *__restrict__ sg_t,
    const float *__restrict__ su_t, const float *__restrict__ sd_t, int S,
    int K, const uint8_t **blob, int *tok, int *inv, float *sg, float *su,
    float *wt) {
    const int i =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (i >= S) return;
    const int e = top[i];
    blob[i] = tab[e];
    tok[i] = i / K;
    inv[i] = i;
    sg[i] = sg_t ? sg_t[e] : 1.0f;
    su[i] = su_t ? su_t[e] : 1.0f;
    wt[i] = w[i] * (sd_t ? sd_t[e] : 1.0f);
}
// One token's routing in one launch: route's picks and weights and route_gather's tables, the same values.  A
// thread an expert: its rank is the count of the experts ahead of it (larger sigmoid + bias; a tie: the lower
// expert), the K of rank under K are the picks, and a pick's place among them - ascending experts - is the count of
// picks under it.  (route_warp_kernel finds them by K rounds of one warp's argmax: 10 us of a lone warp's latency.)
__dpct_inline__ void route_token_kernel(
    const float *__restrict__ logits, const float *__restrict__ bias, int E,
    int K, float scale, const uint8_t *const *__restrict__ tab,
    const float *__restrict__ sg_t, const float *__restrict__ su_t,
    const float *__restrict__ sd_t, int *top, float *w, const uint8_t **blob,
    int *tok, int *inv, float *sg, float *su, float *wt, const float *x,
    int width, NvInput *packed, bool q8, uint16_t *x_limbs,
    uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (item_ct1.get_group(2)) {
        const int i =
            (item_ct1.get_group(2) - 1) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
        if (x_limbs) {   // x = hi + lo in FP16, [2][width] (experts_gate_up_swiglu_mma's input)
            if (i < width) {
                const uint16_t hi = f2h(x[i]);
                x_limbs[i] = hi, x_limbs[width + i] = f2h(x[i] - h2f(hi));
            }
        } else
            pack_limbs<true>(x, nullptr, width, packed, i, 0, q8);
        return;
    }
    auto rt_sm = (float *)dpct_local;
    float* val = rt_sm;                                       // [E] sigmoid + bias
    uint32_t* picked = (uint32_t*) (rt_sm + E);               // [warps] a warp's picks, a bit a lane
    float *sig =
        (float *)(picked +
                  (item_ct1.get_local_range(2) >>
                   5)); // [K] the picks' sigmoids in order, [K]: their sum
    const int e = item_ct1.get_local_id(2), lane = e & 31, warp = e >> 5;
    const float sl =
        e < E ? 1.0f / (1.0f + sycl::native::exp(-ik2_ldg(logits + e))) : 0.0f;
    const float v = e < E ? sl + ik2_ldg(bias + e) : -INFINITY;
    if (e < E) val[e] = v;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    int rank = 0;
    for (int j = 0; j < E; ++j) {
        const float o = val[j];
        rank += (o > v || (o == v && j < e)) ? 1 : 0;
    }
    const bool pick = e < E && rank < K;
    const uint32_t mine = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(),
        (0xffffffffu &
         (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                     .get_local_linear_id())) &&
                pick
            ? (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                          .get_local_linear_id())
            : 0,
        sycl::ext::oneapi::plus<>());
    if (lane == 0) picked[warp] = mine;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    int pos = sycl::popcount(mine & ((1u << lane) - 1u));
#pragma unroll
    for (int ww = 0; ww < warp; ++ww) pos += sycl::popcount(picked[ww]);
    if (pick) sig[pos] = sl;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (e == 0) {   // (the sum in route's order: the same weights to the bit)
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < K; ++k) sum += sig[k];
        sig[K] = sum;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (pick) {
        const float wk = sl / (sig[K] + 1e-20f) * scale;
        top[pos] = e;
        w[pos] = wk;
        blob[pos] = tab[e];
        tok[pos] = 0;
        inv[pos] = pos;
        sg[pos] = sg_t ? sg_t[e] : 1.0f;
        su[pos] = su_t ? su_t[e] : 1.0f;
        wt[pos] = wk * (sd_t ? sd_t[e] : 1.0f);
    }
}

// ---- speculative decoding

__dpct_inline__ void conv_hist_after_kernel(const float *old, const float *x,
                                            int T, int C, float *hist) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= C) return;
    float v[3];
    for (int j = 0; j < 3; ++j) {
        const int r = T + j;   // row r of [old; x]
        v[j] = r < 3 ? old[(size_t) r * C + c] : x[(size_t) (r - 3) * C + c];
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) hist[(size_t)j * C + c] = v[j];
}

__dpct_inline__ uint16_t f2bf_rn(float f) {
    uint32_t u = sycl::bit_cast<unsigned int>(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;   // NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
__dpct_inline__ float bf2f(uint16_t b) {
    return sycl::bit_cast<float>((uint32_t)b << 16);
}

__dpct_inline__ void to_bf16_kernel(const float *x, size_t n, uint16_t *y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t i =
        (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) y[i] = f2bf_rn(x[i]);
}

__dpct_inline__ void mix_rms_kernel(
    const float *x, int n_tap, size_t tap_stride, int C, const float *alpha,
    const float *w, float eps, float *out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sh = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[kBlk]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int row = item_ct1.get_group(2);
    float a = 0.0f;
    for (int c = item_ct1.get_local_id(2); c < C; c += kBlk) {
        float m = 0.0f;
#pragma unroll
        for (int t = 0; t < n_tap; ++t) m +=
            alpha[t] * x[(size_t)t * tap_stride + (size_t)row * C + c];
        out[(size_t) row * C + c] = m;
        a += m * m;
    }
    const float r = sycl::rsqrt(bsum(a, sh) / (float)C + eps);
#pragma unroll
    for (int c = item_ct1.get_local_id(2); c < C; c += kBlk)
        out[(size_t)row * C + c] *= r * w[c];
}

__dpct_inline__ void rope_neox_kernel(float *x, int heads, const int *pos,
                                      float theta) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2), h = item_ct1.get_group(1),
              i = item_ct1.get_local_id(2); // i < 64
    const float inv = 1.0f / dpct::pow(theta, (float)(2 * i) / 128.0f);
    const float ang = (float) pos[row] * inv;
    float sn, cs;
    /*
    DPCT1017: The sycl::sin/cos call is used instead of the sincosf call.
    These two calls do not provide exactly the same functionality. Check the
    potential precision and/or performance issues for the generated code.
    */
    sn = sycl::sin(ang);
    cs = sycl::cos(ang);
    float* p = x + ((size_t) row * heads + h) * 128;
    const float a = p[i], b = p[i + 64];
    p[i] = a * cs - b * sn;
    p[i + 64] = b * cs + a * sn;
}

// a block a (head, query): the logits a key a warp, softmax, the values
// the drafter's attention, flash-decoding style: a block a (head, key chunk) for all the block's queries (nq <= 8) -
// a key's K and V rows read once for them - its unnormalised output and (max, sum) a query, then merged.  Keys: the
// context [k_lo, k_hi) (rows of a ring of cap: (k_lo + j) % cap) and the block's own nq (kb, vb).
constexpr int kDraftQ = 8;
__dpct_inline__ float ld_f(const float *p) { return *p; }
__dpct_inline__ float ld_f(const uint16_t *p) { return h2f(*p); }
template <class KVT>
/*
DPCT1110: The total declared local variable size in device function
draft_attn_part_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void draft_attn_part_kernel(
    const float *q, const KVT *kc, const KVT *vc, int k_lo, int k_hi,
    const float *kb, const float *vb, int nq, int H, int Hkv, float scale,
    int per, float *lgs, float *part_o, sycl::float2 *part_ml, int cap) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sh = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &qs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[kDraftQ][128]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &mrow =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[kDraftQ]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int h = item_ct1.get_group(2), sp = item_ct1.get_group(1),
              S = item_ct1.get_group_range(1), g = h / (H / Hkv),
              tid = item_ct1.get_local_id(2), lane = tid & 31, warp = tid >> 5;
    const int KD = Hkv * 128, nc = k_hi - k_lo, nk = nc + nq;
    const int j0 = sycl::min(nk, sp * per), j1 = sycl::min(nk, j0 + per),
              n = j1 - j0;
#pragma unroll
    for (int qi = 0; qi < nq; ++qi)
        qs[qi][tid] = q[((size_t)qi * H + h) * 128 + tid];
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    float* lg = lgs + ((size_t) h * S + sp) * kDraftQ * per;   // [nq][per]
    float mx[kDraftQ];
#pragma unroll
    for (int qi = 0; qi < kDraftQ; ++qi) mx[qi] = -INFINITY;
    for (int jj = warp; jj < n; jj += 4) {
        const int j = j0 + jj;
        float k0, k1, k2, k3;
        if (j < nc) {
            const KVT* kr = kc + (size_t) ((k_lo + j) % cap) * KD + (size_t) g * 128;
            k0 = ld_f(kr + lane), k1 = ld_f(kr + lane + 32), k2 = ld_f(kr + lane + 64), k3 = ld_f(kr + lane + 96);
        } else {
            const float* kr = kb + (size_t) (j - nc) * KD + (size_t) g * 128;
            k0 = kr[lane], k1 = kr[lane + 32], k2 = kr[lane + 64], k3 = kr[lane + 96];
        }
#pragma unroll
        for (int qi = 0; qi < kDraftQ; ++qi) {
            if (qi >= nq) break;
            float a = qs[qi][lane] * k0 + qs[qi][lane + 32] * k1 + qs[qi][lane + 64] * k2 + qs[qi][lane + 96] * k3;
#pragma unroll
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            for (int o = 16; o > 0; o >>= 1) a +=
                dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), a, o);
            a *= scale;
            if (lane == 0) lg[(size_t) qi * per + jj] = a;
            mx[qi] = sycl::fmax(mx[qi], a);
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int qi = 0; qi < nq; ++qi) {
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        const float m = bmax(mx[qi], sh);
        float a = 0.0f;
        for (int jj = tid; jj < n; jj += 128) {
            const float e =
                m == -INFINITY
                    ? 0.0f
                    : sycl::native::exp(lg[(size_t)qi * per + jj] - m);
            lg[(size_t) qi * per + jj] = e;
            a += e;
        }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        const float l = bsum(a, sh);
        if (tid == 0) mrow[qi] = m,
                      part_ml[((size_t)sp * H + h) * kDraftQ + qi] =
                          sycl::float2(m, l);
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    float o[kDraftQ];
#pragma unroll
    for (int qi = 0; qi < kDraftQ; ++qi) o[qi] = 0.0f;
    for (int jj = 0; jj < n; ++jj) {
        const int j = j0 + jj;
        const float v = j < nc ? ld_f(vc + (size_t) ((k_lo + j) % cap) * KD + (size_t) g * 128 + tid)
                               : vb[(size_t) (j - nc) * KD + (size_t) g * 128 + tid];
#pragma unroll
        for (int qi = 0; qi < kDraftQ; ++qi)
            if (qi < nq) o[qi] += lg[(size_t) qi * per + jj] * v;
    }
#pragma unroll
    for (int qi = 0; qi < nq; ++qi)
        part_o[(((size_t)sp * H + h) * kDraftQ + qi) * 128 + tid] = o[qi];
}
__dpct_inline__ void draft_attn_merge_kernel(const float *part_o,
                                             const sycl::float2 *part_ml, int S,
                                             int H, float *out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int h = item_ct1.get_group(2), qi = item_ct1.get_group(1),
              tid = item_ct1.get_local_id(2);
    float m = -INFINITY;
#pragma unroll
    for (int sp = 0; sp < S; ++sp)
        m = sycl::fmax(m, part_ml[((size_t)sp * H + h) * kDraftQ + qi].x());
    float den = 0.0f, acc = 0.0f;
    for (int sp = 0; sp < S; ++sp) {
        const sycl::float2 ml = part_ml[((size_t)sp * H + h) * kDraftQ + qi];
        const float c =
            ml.x() == -INFINITY ? 0.0f : sycl::native::exp(ml.x() - m);
        den += ml.y() * c;
        acc += part_o[(((size_t) sp * H + h) * kDraftQ + qi) * 128 + tid] * c;
    }
    out[((size_t) qi * H + h) * 128 + tid] = acc / den;
}

__dpct_inline__ void silu_mul_bf16_kernel(const float *gu, int T, int F,
                                          uint16_t *out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t i =
        (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= (size_t) T * F) return;
    const size_t t = i / F, f = i % F;
    const float g = gu[t * 2 * F + f], u = gu[t * 2 * F + F + f];
    out[i] = f2bf_rn(g / (1.0f + sycl::native::exp(-g)) * u);
}

__dpct_inline__ void rnn_inputs_kernel(
    const float *state, const uint16_t *emb, const int *prev,
    const float *hidden, const float *f, int R, int C, uint16_t *out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    const int W = 2 * R + 2 * C;
    if (i >= W) return;
    float v;
    if (i < R) v = state[i];
    else if (i < 2 * R) v = bf2f(emb[(size_t) (*prev) * R + (i - R)]);
    else if (i < 2 * R + C) v = hidden[i - 2 * R];
    else v = f[i - 2 * R - C];
    out[i] = f2bf_rn(v);
}

__dpct_inline__ void rnn_gates_kernel(const float *proj, const float *bias,
                                      int R, float *state, float *latent) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= R) return;
    const float gr = proj[i] + bias[i], cr = proj[R + i] + bias[R + i], orw = proj[2 * R + i] + bias[2 * R + i];
    const float gate = 1.0f / (1.0f + sycl::native::exp(-gr));
    state[i] = gate * state[i] + (1.0f - gate) * sycl::tanh(cr);
    latent[i] = sycl::tanh(orw);
}

constexpr int kArgmaxMaxPartials = 1024;
inline dpct::global_memory<float, 1> g_argmax_v(kArgmaxMaxPartials);
inline dpct::global_memory<int, 1> g_argmax_i(kArgmaxMaxPartials);

__dpct_inline__ void argmax_better(float v, int i, float &best, int &idx) {
    if (v > best || (v == best && i < idx)) best = v, idx = i;
}

__dpct_inline__ void argmax_block_reduce(float &best, int &idx) {
auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        const float ob = dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(),
            best, o);
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        const int oi = dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(),
            idx, o);
        argmax_better(ob, oi, best, idx);
    }
    /*
    DPCT1115: The sycl::ext::oneapi::group_local_memory_for_overwrite is
    used to allocate group-local memory at the none kernel functor scope of a
    work-group data parallel kernel. You may need to adjust the code.
    */
    auto &bv = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    /*
    DPCT1115: The sycl::ext::oneapi::group_local_memory_for_overwrite is
    used to allocate group-local memory at the none kernel functor scope of a
    work-group data parallel kernel. You may need to adjust the code.
    */
    auto &bi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5,
              nw = item_ct1.get_local_range(2) >> 5;
    if (lane == 0) bv[warp] = best, bi[warp] = idx;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (warp == 0) {
        best = (lane < nw) ? bv[lane] : -INFINITY;
        idx = (lane < nw) ? bi[lane] : 0x7fffffff;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const float ob = dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(),
                best, o);
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const int oi = dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(),
                idx, o);
            argmax_better(ob, oi, best, idx);
        }
    }
}

template <bool SINGLE_BLOCK>
__dpct_inline__ void argmax_sum_part_kernel(
    const float *a, const float *b, const float *c, int n, int *out,
    float *g_argmax_v, int *g_argmax_i) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int gtid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);
    const int gstride =
        item_ct1.get_group_range(2) * item_ct1.get_local_range(2);
    float best = -INFINITY;
    int idx = 0x7fffffff;
    const bool vec4 = (((uintptr_t) a & 15) == 0) && (!b || ((uintptr_t) b & 15) == 0) && (!c || ((uintptr_t) c & 15) == 0);
    if (vec4) {
        const int n4 = n >> 2;
        const sycl::float4 *a4 = (const sycl::float4 *)a;
        const sycl::float4 *b4 = (const sycl::float4 *)b;
        const sycl::float4 *c4 = (const sycl::float4 *)c;
        for (int v4 = gtid; v4 < n4; v4 += gstride) {
            const sycl::float4 va = ::ik2_ldg(a4 + v4);
            const sycl::float4 vb =
                b4 ? ::ik2_ldg(b4 + v4) : sycl::float4(0.f, 0.f, 0.f, 0.f);
            const sycl::float4 vc =
                c4 ? ::ik2_ldg(c4 + v4) : sycl::float4(0.f, 0.f, 0.f, 0.f);
            const int base = v4 << 2;
            argmax_better(va.x() + vb.x() + vc.x(), base + 0, best, idx);
            argmax_better(va.y() + vb.y() + vc.y(), base + 1, best, idx);
            argmax_better(va.z() + vb.z() + vc.z(), base + 2, best, idx);
            argmax_better(va.w() + vb.w() + vc.w(), base + 3, best, idx);
        }
#pragma unroll
        for (int v = (n4 << 2) + gtid; v < n; v += gstride)
            argmax_better(ik2_ldg(a + v) + (b ? ik2_ldg(b + v) : 0.0f) + (c ? ik2_ldg(c + v) : 0.0f), v, best, idx);
    } else {
#pragma unroll
        for (int v = gtid; v < n; v += gstride)
            argmax_better(ik2_ldg(a + v) + (b ? ik2_ldg(b + v) : 0.0f) + (c ? ik2_ldg(c + v) : 0.0f), v, best, idx);
    }
    argmax_block_reduce(best, idx);
    if (item_ct1.get_local_id(2) == 0) {
        if (SINGLE_BLOCK) *out = idx;
        else g_argmax_v[item_ct1.get_group(2)] = best,
             g_argmax_i[item_ct1.get_group(2)] = idx;
    }
}

template <bool SINGLE_BLOCK>
__dpct_inline__ void argmax_rows_part_kernel(const float *x, int ld, int n_cols,
                                             int *out, float *g_argmax_v,
                                             int *g_argmax_i) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int r = item_ct1.get_group(1);
    const int gtid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);
    const int gstride =
        item_ct1.get_group_range(2) * item_ct1.get_local_range(2);
    const float* row = x + (size_t) r * ld;
    float best = -INFINITY;
    int idx = 0x7fffffff;
    if (((uintptr_t) row & 15) == 0) {
        const int n4 = n_cols >> 2;
        const sycl::float4 *r4 = (const sycl::float4 *)row;
        for (int v4 = gtid; v4 < n4; v4 += gstride) {
            const sycl::float4 f = ::ik2_ldg(r4 + v4);
            const int base = v4 << 2;
            argmax_better(f.x(), base + 0, best, idx);
            argmax_better(f.y(), base + 1, best, idx);
            argmax_better(f.z(), base + 2, best, idx);
            argmax_better(f.w(), base + 3, best, idx);
        }
#pragma unroll
        for (int v = (n4 << 2) + gtid; v < n_cols; v += gstride)
            argmax_better(ik2_ldg(row + v), v, best, idx);
    } else {
#pragma unroll
        for (int v = gtid; v < n_cols; v += gstride)
            argmax_better(ik2_ldg(row + v), v, best, idx);
    }
    argmax_block_reduce(best, idx);
    if (item_ct1.get_local_id(2) == 0) {
        if (SINGLE_BLOCK) out[r] = idx;
        else g_argmax_v[r * item_ct1.get_group_range(2) +
                        item_ct1.get_group(2)] = best,
                        g_argmax_i[r * item_ct1.get_group_range(2) +
                                   item_ct1.get_group(2)] = idx;
    }
}

template <bool SINGLE_BLOCK>
/*
DPCT1110: The total declared local variable size in device function
argmax_gemv_f16_part_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
__dpct_inline__ void argmax_gemv_f16_part_kernel(
    const uint16_t *W, int ld, const float *x, const float *base,
    const uint16_t *bias_f16, int R, int n, bool use_smem, int *out,
    uint8_t *dpct_local, float *g_argmax_v, int *g_argmax_i) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sx = (float *)dpct_local;
    if (use_smem) {
#pragma unroll
        for (int r = item_ct1.get_local_id(2); r < R;
             r += item_ct1.get_local_range(2)) sx[r] = x[r];
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    const int gtid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);
    const int gstride =
        item_ct1.get_group_range(2) * item_ct1.get_local_range(2);
    float best = -INFINITY;
    int idx = 0x7fffffff;
    const bool vec4 = (((uintptr_t) W & 7) == 0) && ((ld & 3) == 0) &&
                      (!base || ((uintptr_t) base & 15) == 0) && (!bias_f16 || ((uintptr_t) bias_f16 & 7) == 0);
    if (vec4) {
        const int n4 = n >> 2;
        for (int v4 = gtid; v4 < n4; v4 += gstride) {
            const int v = v4 << 2;
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
            const uint16_t* wr = W + v;
            for (int r = 0; r < R; ++r, wr += ld) {
                const float xr = use_smem ? sx[r] : ik2_ldg(x + r);
                const sycl::uint2 w4 = ::ik2_ldg((const sycl::uint2 *)wr);
                const sycl::float2 f0 = u32_to_f2(w4.x()),
                                   f1 = u32_to_f2(w4.y());
                d0 = sycl::fma((float)(f0.x()), (float)xr, d0);
                d1 = sycl::fma((float)(f0.y()), (float)xr, d1);
                d2 = sycl::fma((float)(f1.x()), (float)xr, d2);
                d3 = sycl::fma((float)(f1.y()), (float)xr, d3);
            }
            if (base) {
                const sycl::float4 bs =
                    ::ik2_ldg((const sycl::float4 *)(base + v));
                d0 += bs.x(), d1 += bs.y(), d2 += bs.z(), d3 += bs.w();
            }
            if (bias_f16) {
                const sycl::uint2 b4 =
                    ::ik2_ldg((const sycl::uint2 *)(bias_f16 + v));
                const sycl::float2 f0 = u32_to_f2(b4.x()),
                                   f1 = u32_to_f2(b4.y());
                d0 += f0.x(), d1 += f0.y(), d2 += f1.x(), d3 += f1.y();
            }
            argmax_better(d0, v + 0, best, idx);
            argmax_better(d1, v + 1, best, idx);
            argmax_better(d2, v + 2, best, idx);
            argmax_better(d3, v + 3, best, idx);
        }
        for (int v = (n4 << 2) + gtid; v < n; v += gstride) {
            float d = 0.0f;
            const uint16_t* wr = W + v;
#pragma unroll
            for (int r = 0; r < R; ++r, wr += ld)
                d = sycl::fma(h2f(ik2_ldg(wr)),
                              use_smem ? sx[r] : ik2_ldg(x + r), d);
            if (base) d += ik2_ldg(base + v);
            if (bias_f16) d += h2f(ik2_ldg(bias_f16 + v));
            argmax_better(d, v, best, idx);
        }
    } else {
        for (int v = gtid; v < n; v += gstride) {
            float d = 0.0f;
            const uint16_t* wr = W + v;
#pragma unroll
            for (int r = 0; r < R; ++r, wr += ld)
                d = sycl::fma(h2f(ik2_ldg(wr)),
                              use_smem ? sx[r] : ik2_ldg(x + r), d);
            if (base) d += ik2_ldg(base + v);
            if (bias_f16) d += h2f(ik2_ldg(bias_f16 + v));
            argmax_better(d, v, best, idx);
        }
    }
    argmax_block_reduce(best, idx);
    if (item_ct1.get_local_id(2) == 0) {
        if (SINGLE_BLOCK) *out = idx;
        else g_argmax_v[item_ct1.get_group(2)] = best,
             g_argmax_i[item_ct1.get_group(2)] = idx;
    }
}

__dpct_inline__ void argmax_final_kernel(int n_partials, int stride_partials,
                                         int *out, float *g_argmax_v,
                                         int *g_argmax_i) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int r = item_ct1.get_group(2);
    float best = -INFINITY;
    int idx = 0x7fffffff;
#pragma unroll
    for (int j = item_ct1.get_local_id(2); j < n_partials;
         j += item_ct1.get_local_range(2))
        argmax_better(g_argmax_v[r * stride_partials + j], g_argmax_i[r * stride_partials + j], best, idx);
    argmax_block_reduce(best, idx);
    if (item_ct1.get_local_id(2) == 0) out[r] = idx;
}

// fast FP16 decodes: NVFP4 a thread a 16-value sub-block (__byte_perm + uint4 stores); Q8_0 8 threads a 32-value block
// (uint2 stores); Q5_K / Q6_K a warp a 256-value block (4 values/lane with coalesced uint2 stores)
/*
DPCT1110: The total declared local variable size in device function
decode_f16_nvfp4_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void decode_f16_nvfp4_kernel(const uint8_t *blocks, size_t nsub,
                                             uint16_t *out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t sb =
        (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (sb >= nsub) return;
    const uint8_t* b = blocks + (sb >> 2) * 36;
    const int sub = (int) (sb & 3);
    const float d = ue4m3((uint8_t) (*(const uint32_t*) b >> (8 * sub)));
    const uint32_t q0 = *(const uint32_t*) (b + 4 + sub * 8), q1 = *(const uint32_t*) (b + 8 + sub * 8);
    const sycl::float2 w01 = e2m1_pair_lo(q0), w23 = e2m1_pair_lo(q0 >> 16);
    const sycl::float2 w45 = e2m1_pair_lo(q1), w67 = e2m1_pair_lo(q1 >> 16);
    const sycl::float2 w89 = e2m1_pair_hi(q0), wAB = e2m1_pair_hi(q0 >> 16);
    const sycl::float2 wCD = e2m1_pair_hi(q1), wEF = e2m1_pair_hi(q1 >> 16);
    const sycl::uint4 o0 = sycl::uint4(f2_to_u32(w01.x() * d, w01.y() * d),
                                       f2_to_u32(w23.x() * d, w23.y() * d),
                                       f2_to_u32(w45.x() * d, w45.y() * d),
                                       f2_to_u32(w67.x() * d, w67.y() * d));
    const sycl::uint4 o1 = sycl::uint4(f2_to_u32(w89.x() * d, w89.y() * d),
                                       f2_to_u32(wAB.x() * d, wAB.y() * d),
                                       f2_to_u32(wCD.x() * d, wCD.y() * d),
                                       f2_to_u32(wEF.x() * d, wEF.y() * d));
    sycl::uint4 *o = (sycl::uint4 *)(out + sb * 16);
    o[0] = o0;
    o[1] = o1;
}

__dpct_inline__ void decode_f16_q8_kernel(const uint8_t *blocks, size_t nblk,
                                          uint16_t *out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t tid_g =
        (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    const size_t bi = tid_g >> 3;
    if (bi >= nblk) return;
    const int sub = (int) (tid_g & 7);
    const uint8_t* b = blocks + bi * 34;
    const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
    const uint8_t* qp = b + 2 + sub * 4;
    int8_t q0, q1, q2, q3;
    if (((uintptr_t) qp & 3) == 0) {
        const uint32_t q4 = *(const uint32_t*) qp;
        q0 = (int8_t) (q4 & 0xFFu);
        q1 = (int8_t) ((q4 >> 8) & 0xFFu);
        q2 = (int8_t) ((q4 >> 16) & 0xFFu);
        q3 = (int8_t) (q4 >> 24);
    } else {
        q0 = (int8_t) qp[0], q1 = (int8_t) qp[1], q2 = (int8_t) qp[2], q3 = (int8_t) qp[3];
    }
    uint16_t* dst = out + bi * 32 + sub * 4;
    if (((uintptr_t) dst & 7) == 0) {
        *(sycl::uint2 *)dst =
            sycl::uint2(f2_to_u32(d * (float)q0, d * (float)q1),
                        f2_to_u32(d * (float)q2, d * (float)q3));
    } else {
        dst[0] = f2h(d * (float) q0), dst[1] = f2h(d * (float) q1);
        dst[2] = f2h(d * (float) q2), dst[3] = f2h(d * (float) q3);
    }
}

__dpct_inline__ void store_f16x4(uint16_t *dst, float v0, float v1, float v2,
                                 float v3) {
    if (((uintptr_t) dst & 7) == 0) {
        *(sycl::uint2 *)dst = sycl::uint2(f2_to_u32(v0, v1), f2_to_u32(v2, v3));
    } else {
        dst[0] = f2h(v0), dst[1] = f2h(v1), dst[2] = f2h(v2), dst[3] = f2h(v3);
    }
}

// one thread per 32-value sub-block of IQ3_XXS / IQ4_XS -> 32 FP16 (ggml's dequantize_row_iq3_xxs / iq4_xs values)
/*
DPCT1110: The total declared local variable size in device function
decode_f16_iq_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void decode_f16_iq_kernel(
    int type, const uint8_t *blocks, size_t nsub, uint16_t *out,
    const uint32_t *d_iq3xxs_grid, const int8_t *d_iq4nl) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t sb =
        (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (sb >= nsub) return;
    const int ib = (int) (sb & 7);
    float y[32];
    if (type == kIQ3_XXS) {
        const uint8_t* b = blocks + (sb >> 3) * 98;
        const float d = half_bits_to_float(ik2_ldg((const uint16_t*) b));
        const uint16_t* qs = (const uint16_t*) (b + 2 + 8 * ib);
        const uint16_t* sp = (const uint16_t*) (b + 66 + 4 * ib);
        const uint32_t aux = (uint32_t) ik2_ldg(sp) | ((uint32_t) ik2_ldg(sp + 1) << 16);
        const float db = d * (0.5f + (float) (aux >> 28)) * 0.5f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t qq = ik2_ldg(qs + l);
            const uint32_t g1 = ik2_ldg(d_iq3xxs_grid + (qq & 0xFFu)), g2 = ik2_ldg(d_iq3xxs_grid + (qq >> 8));
            const uint32_t sg = iq_signs8((aux >> (7 * l)) & 127u);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                y[8 * l + j] = db * (float) ((g1 >> (8 * j)) & 0xFFu) * ((sg >> j) & 1u ? -1.0f : 1.0f);
                y[8 * l + 4 + j] = db * (float) ((g2 >> (8 * j)) & 0xFFu) * ((sg >> (4 + j)) & 1u ? -1.0f : 1.0f);
            }
        }
    } else {
        const uint8_t* b = blocks + (sb >> 3) * 136;
        const uint32_t h = ik2_ldg((const uint32_t*) b), sl = ik2_ldg((const uint32_t*) (b + 4));
        const float d = half_bits_to_float((uint16_t) (h & 0xFFFFu));
        const int ls = (int) ((sl >> (4 * ib)) & 0xFu) | (int) (((h >> (16 + 2 * ib)) & 3u) << 4);
        const float dl = d * (float) (ls - 32);
        const uint8_t* q = b + 8 + 16 * ib;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const uint8_t v = ik2_ldg(q + j);
            y[j] = dl * (float) d_iq4nl[v & 0xF];
            y[16 + j] = dl * (float) d_iq4nl[v >> 4];
        }
    }
    uint32_t* o = (uint32_t*) (out + sb * 32);
#pragma unroll
    for (int j = 0; j < 16; ++j) o[j] = (uint32_t) f2h(y[2 * j]) | ((uint32_t) f2h(y[2 * j + 1]) << 16);
}

/*
DPCT1110: The total declared local variable size in device function
decode_f16_k_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void decode_f16_k_kernel(int type, const uint8_t *blocks,
                                         size_t nblk, uint16_t *out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t bi =
        (size_t)item_ct1.get_group(2) * (item_ct1.get_local_range(2) >> 5) +
        (item_ct1.get_local_id(2) >> 5);
    if (bi >= nblk) return;
    const int lane = item_ct1.get_local_id(2) & 31;
    const int gr = lane >> 3, l0 = (lane & 7) * 4;
    uint16_t* y = out + bi * 256;
    if (type == kQ6_K) {
        const uint8_t* b = blocks + bi * 210;
        const float d = half_bits_to_float((uint16_t) (b[208] | (b[209] << 8)));
        const int8_t* sc = (const int8_t*) (b + 192);
        const int qd = gr;
        const int sh = 2 * qd;
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            const float d_sc = d * (float) sc[n * 8 + (l0 >> 4) + 2 * qd];
            const uint8_t* qlp = b + n * 64 + ((qd & 1) ? 32 : 0) + l0;
            const uint8_t* qhp = b + 128 + n * 32 + l0;
            uint8_t lo[4], qh[4];
            if (((uintptr_t) qlp & 1) == 0) {
                const uint16_t lo0 = *(const uint16_t*) qlp, lo1 = *(const uint16_t*) (qlp + 2);
                const uint16_t qh0 = *(const uint16_t*) qhp, qh1 = *(const uint16_t*) (qhp + 2);
                lo[0] = (uint8_t) lo0, lo[1] = (uint8_t) (lo0 >> 8);
                lo[2] = (uint8_t) lo1, lo[3] = (uint8_t) (lo1 >> 8);
                qh[0] = (uint8_t) qh0, qh[1] = (uint8_t) (qh0 >> 8);
                qh[2] = (uint8_t) qh1, qh[3] = (uint8_t) (qh1 >> 8);
            } else {
#pragma unroll
                for (int k = 0; k < 4; ++k) lo[k] = qlp[k], qh[k] = qhp[k];
            }
            float f[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int nib = qd >= 2 ? (lo[k] >> 4) : (lo[k] & 0xF);
                const int q = (int) (nib | (((qh[k] >> sh) & 3) << 4)) - 32;
                f[k] = d_sc * (float) q;
            }
            store_f16x4(y + n * 128 + qd * 32 + l0, f[0], f[1], f[2], f[3]);
        }
    } else {   // Q5_K
        const uint8_t* b = blocks + bi * 176;
        const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
        const float mn = half_bits_to_float((uint16_t) (b[2] | (b[3] << 8)));
        const uint8_t* q = b + 4;
        const uint8_t* qhp = b + 16 + l0;
        const uint32_t qh4 = (((uintptr_t) qhp & 3) == 0)
                                 ? *(const uint32_t*) qhp
                                 : ((uint32_t) qhp[0] | ((uint32_t) qhp[1] << 8) | ((uint32_t) qhp[2] << 16) |
                                    ((uint32_t) qhp[3] << 24));
#pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const int is = pass * 4 + gr, j64 = is >> 1, half = is & 1;
            int sc, m;
            if (is < 4) {
                sc = q[is] & 63, m = q[is + 4] & 63;
            } else {
                sc = (q[is + 4] & 0xF) | ((q[is - 4] >> 6) << 4), m = (q[is + 4] >> 4) | ((q[is] >> 6) << 4);
            }
            const float dsc = d * (float) sc, mnm = mn * (float) m;
            const uint8_t* lop = b + 48 + 32 * j64 + l0;
            const uint32_t lo4 = (((uintptr_t) lop & 3) == 0)
                                     ? *(const uint32_t*) lop
                                     : ((uint32_t) lop[0] | ((uint32_t) lop[1] << 8) | ((uint32_t) lop[2] << 16) |
                                        ((uint32_t) lop[3] << 24));
            float f[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t lo_k = (lo4 >> (8 * k)) & 0xFFu;
                const uint32_t qh_k = (qh4 >> (8 * k)) & 0xFFu;
                const int val = (half ? (lo_k >> 4) : (lo_k & 0xF)) + (((qh_k >> is) & 1) ? 16 : 0);
                f[k] = dsc * (float) val - mnm;
            }
            store_f16x4(y + is * 32 + l0, f[0], f[1], f[2], f[3]);
        }
    }
}

__dpct_inline__ void decode_f16_kernel(
    int type, const uint8_t *blocks, size_t n, uint16_t *out,
    const uint32_t *d_iq3xxs_grid, const int8_t *d_iq4nl) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t i =
        (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) out[i] =
        f2h(decode_value(type, blocks, i, d_iq3xxs_grid, d_iq4nl));
}
__dpct_inline__ void decode_f32_kernel(
    int type, const uint8_t *blocks, size_t n, float *out,
    const uint32_t *d_iq3xxs_grid, const int8_t *d_iq4nl) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t i =
        (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) out[i] = decode_value(type, blocks, i, d_iq3xxs_grid, d_iq4nl);
}

unsigned grid(size_t n, int b = kBlk) { return (unsigned) ((n + b - 1) / b); }


// ---- the block path: a few tokens at once (2 .. 8, a speculative step's verify window) on the resident path

// the keys of a window of T tokens at pos0 .. pos0 + T - 1: steps of 16 from the first token's k_lo to the last token
__dpct_inline__ int wa_active_block(int pos0, int T, int window, int nw,
                                    int &k_lo, int &n_steps) {
    k_lo = window < (1 << 30) ? sycl::max(0, pos0 - window + 1) : 0;
    n_steps = (pos0 + T - k_lo + 15) >> 4;
    return sycl::min(nw, (n_steps + kWaMin - 1) / kWaMin);
}

// attn_decode_mma_kernel for the T tokens of a window: the B columns are (token, head) pairs of the K / V head -
// column c is token c / R, head g R + c % R - so that the tokens' queries read a key's K and V rows once; NT tiles of
// 8 columns (T R <= 8 NT).  A column's keys: its token's (causal, its own window); the bias by its own distance.
template <class KVT, int NT>
/*
DPCT1110: The total declared local variable size in device function
attn_block_mma_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void attn_block_mma_kernel(
    const uint16_t *__restrict__ q16, const KVT *__restrict__ kc,
    const KVT *__restrict__ vc, int kv_cap, int H, int Hkv,
    const int *__restrict__ pos_d, int T, float qk_scale,
    const float *__restrict__ r, int r_ld, const float *__restrict__ proj,
    int d_rel, bool rel4, int extent, int window, int n_split,
    float *__restrict__ o_part, sycl::float2 *__restrict__ ml_part) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr bool F8 = sizeof(KVT) == 1;
    const int pos0 = *pos_d;
    const int lane = item_ct1.get_local_id(2) & 31, gr = lane >> 2,
              tq = lane & 3;
    const int g = item_ct1.get_group(2), R = H / Hkv, KD = Hkv * kFaD,
              NC = T * R;
    const int w =
        item_ct1.get_group(1) * kWaNW + (item_ct1.get_local_id(2) >> 5);
    int k_lo, n_steps;
    const int n_act = wa_active_block(pos0, T, window, n_split * kWaNW, k_lo, n_steps);
    if (w >= n_act) return;
    const int per = (n_steps + n_act - 1) / n_act;
    const int st0 = sycl::min(n_steps, w * per),
              st1 = sycl::min(n_steps, st0 + per);
    const int qlast = pos0 + T - 1;
    uint32_t qf[NT][kFaD / 16][2];
    int qpa[NT], qpb[NT];             // the positions of the lane's columns a = 2 tq, b = 2 tq + 1 (-1: none)
    const float* ra[NT];
    const float* rb[NT];
    bool q_ok[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int cq = j * 8 + gr, ca = j * 8 + 2 * tq, cb = ca + 1;
        q_ok[j] = cq < NC;
        const int tqq = q_ok[j] ? cq / R : 0, hq = q_ok[j] ? cq % R : 0;
        const sycl::uint2 *qr =
            (const sycl::uint2 *)(q16 + ((size_t)tqq * H + g * R + hq) * kFaD);
#pragma unroll
        for (int kk = 0; kk < kFaD / 16; ++kk) {
            const sycl::uint2 v = qr[kk * 4 + tq];
            qf[j][kk][0] = q_ok[j] ? v.x() : 0u,
            qf[j][kk][1] = q_ok[j] ? v.y() : 0u;
        }
        const bool a_ok = ca < NC, b_ok = cb < NC;
        qpa[j] = a_ok ? pos0 + ca / R : -1;
        qpb[j] = b_ok ? pos0 + cb / R : -1;
        ra[j] = r + (size_t) (a_ok ? ca / R : 0) * r_ld + (size_t) (g * R + (a_ok ? ca % R : 0)) * d_rel;
        rb[j] = r + (size_t) (b_ok ? cb / R : 0) * r_ld + (size_t) (g * R + (b_ok ? cb % R : 0)) * d_rel;
    }
    float o[NT][kFaD / 16][4];
    float ma[NT], mb[NT], la[NT], lb[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        ma[j] = mb[j] = -INFINITY, la[j] = lb[j] = 0.0f;
#pragma unroll
        for (int jj = 0; jj < kFaD / 16; ++jj) o[j][jj][0] = o[j][jj][1] = o[j][jj][2] = o[j][jj][3] = 0.0f;
    }
    const float sc = qk_scale * kLog2e;
    const KVT* kg = kc + (size_t) g * kFaD + tq * 4;
    const KVT* vg = vc + (size_t) g * kFaD + gr * 16;
    const int src = tq * 8 + (gr >> 1);
    const uint32_t psel = (gr & 1) ? 0x7632u : 0x5410u;
    const bool windowed = window < (1 << 30);

    for (int st = st0; st < st1; ++st) {
        const int kt = k_lo + st * 16, kmax = qlast - kt;
        const int row0 = kv_cap > 0 ? kt % kv_cap : kt;
        auto row = [&](int key) {
            int rw = row0 + sycl::min(key, kmax);
            if (kv_cap > 0 && rw >= kv_cap) rw -= kv_cap;
            return (size_t) rw * KD;
        };
        uint32_t ka[kFaD / 16][4];
        {
            const KVT* p0 = kg + row(gr);
            const KVT* p1 = kg + row(gr + 8);
            if constexpr (F8) {
                uint32_t w0[kFaD / 16], w1[kFaD / 16];
#pragma unroll
                for (int kk = 0; kk < kFaD / 16; ++kk) {
                    w0[kk] = ik2_ldg((const uint32_t*) (p0 + kk * 16));
                    w1[kk] = ik2_ldg((const uint32_t*) (p1 + kk * 16));
                }
#pragma unroll
                for (int kk = 0; kk < kFaD / 16; ++kk) {
                    e4m3x4_h4(w0[kk], ka[kk][0], ka[kk][2]);
                    e4m3x4_h4(w1[kk], ka[kk][1], ka[kk][3]);
                }
            } else {
#pragma unroll
                for (int kk = 0; kk < kFaD / 16; ++kk) {
                    const sycl::uint2 v0 = ::ik2_ldg(
                                          (const sycl::uint2 *)(p0 + kk * 16)),
                                      v1 = ::ik2_ldg(
                                          (const sycl::uint2 *)(p1 + kk * 16));
                    ka[kk][0] = v0.x(), ka[kk][2] = v0.y(), ka[kk][1] = v1.x(),
                    ka[kk][3] = v1.y();
                }
            }
        }
        const int k0 = kt + gr, k1 = k0 + 8;
        const bool near = qlast - (kt + 15) < extent;
        uint32_t pb0[NT], pb1[NT];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            float sa[4] = {0.0f, 0.0f, 0.0f, 0.0f}, sb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int kk = 0; kk < kFaD / 16; ++kk) mma16816((kk & 1) ? sb : sa, ka[kk], qf[j][kk][0], qf[j][kk][1]);
            float c0 = (sa[0] + sb[0]) * sc, c1 = (sa[1] + sb[1]) * sc, c2 = (sa[2] + sb[2]) * sc,
                  c3 = (sa[3] + sb[3]) * sc;
            const int pa = qpa[j], pb = qpb[j];
            if (near) {
                const int da0 = pa - k0, da1 = pa - k1, db0 = pb - k0, db1 = pb - k1;
                const bool ea0 = pa >= 0 && da0 >= 0 && da0 < extent, ea1 = pa >= 0 && da1 >= 0 && da1 < extent;
                const bool eb0 = pb >= 0 && db0 >= 0 && db0 < extent, eb1 = pb >= 0 && db1 >= 0 && db1 < extent;
                const float* pa0 = proj + (size_t) (ea0 ? da0 : 0) * d_rel;
                const float* pa1 = proj + (size_t) (ea1 ? da1 : 0) * d_rel;
                const float* pb0p = proj + (size_t) (eb0 ? db0 : 0) * d_rel;
                const float* pb1p = proj + (size_t) (eb1 ? db1 : 0) * d_rel;
                float b0a = 0.0f, b1a = 0.0f, b0b = 0.0f, b1b = 0.0f;
                if (rel4) {
                    for (int e = 0; e < d_rel; e += 4) {
                        const sycl::float4
                            ya = ::ik2_ldg((const sycl::float4 *)(ra[j] + e)),
                            yb = ::ik2_ldg((const sycl::float4 *)(rb[j] + e));
                        const sycl::float4 x0 = ::ik2_ldg(
                                               (const sycl::float4 *)(pa0 + e)),
                                           x1 = ::ik2_ldg(
                                               (const sycl::float4 *)(pa1 + e));
                        const sycl::float4 z0 = ::ik2_ldg((
                                               const sycl::float4 *)(pb0p + e)),
                                           z1 = ::ik2_ldg((
                                               const sycl::float4 *)(pb1p + e));
                        b0a += x0.x() * ya.x() + x0.y() * ya.y() +
                               x0.z() * ya.z() + x0.w() * ya.w();
                        b1a += x1.x() * ya.x() + x1.y() * ya.y() +
                               x1.z() * ya.z() + x1.w() * ya.w();
                        b0b += z0.x() * yb.x() + z0.y() * yb.y() +
                               z0.z() * yb.z() + z0.w() * yb.w();
                        b1b += z1.x() * yb.x() + z1.y() * yb.y() +
                               z1.z() * yb.z() + z1.w() * yb.w();
                    }
                } else {
#pragma unroll
                    for (int e = 0; e < d_rel; ++e) {
                        b0a += pa0[e] * ra[j][e], b1a += pa1[e] * ra[j][e];
                        b0b += pb0p[e] * rb[j][e], b1b += pb1p[e] * rb[j][e];
                    }
                }
                if (ea0) c0 += b0a * kLog2e;
                if (eb0) c1 += b0b * kLog2e;
                if (ea1) c2 += b1a * kLog2e;
                if (eb1) c3 += b1b * kLog2e;
            }
            // (causal and windowed by the column's own token)
            const bool va0 = pa >= 0 && k0 <= pa && (!windowed || k0 > pa - window);
            const bool vb0 = pb >= 0 && k0 <= pb && (!windowed || k0 > pb - window);
            const bool va1 = pa >= 0 && k1 <= pa && (!windowed || k1 > pa - window);
            const bool vb1 = pb >= 0 && k1 <= pb && (!windowed || k1 > pb - window);
            c0 = va0 ? c0 : -INFINITY, c1 = vb0 ? c1 : -INFINITY, c2 = va1 ? c2 : -INFINITY, c3 = vb1 ? c3 : -INFINITY;
            float mxa = sycl::fmax(c0, c2), mxb = sycl::fmax(c1, c3);
#pragma unroll
            for (int sh = 4; sh < 32; sh <<= 1) {
                /*
                DPCT1108: '__shfl_xor_sync' was migrated with the
                experimental feature masked sub_group function which may not be
                supported by all compilers or runtimes. You may need to adjust
                the code.
                */
                mxa = sycl::fmax(
                    mxa, dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(),
                             mxa, sh));
                /*
                DPCT1108: '__shfl_xor_sync' was migrated with the
                experimental feature masked sub_group function which may not be
                supported by all compilers or runtimes. You may need to adjust
                the code.
                */
                mxb = sycl::fmax(
                    mxb, dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(),
                             mxb, sh));
            }
            if (sycl::reduce_over_group(
                    sycl::ext::oneapi::this_work_item::get_sub_group(),
                    (0xffffffffu &
                     (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                                 .get_local_linear_id())) &&
                                mxa > ma[j] ||
                            mxb > mb[j]
                        ? (0x1
                           << sycl::ext::oneapi::this_work_item::get_sub_group()
                                  .get_local_linear_id())
                        : 0,
                    sycl::ext::oneapi::plus<>())) {
                const float mna = sycl::fmax(ma[j], mxa),
                            mnb = sycl::fmax(mb[j], mxb);
                const float ca =
                    mna == -INFINITY ? 1.0f : sycl::exp2(ma[j] - mna);
                const float cb =
                    mnb == -INFINITY ? 1.0f : sycl::exp2(mb[j] - mnb);
                ma[j] = mna, mb[j] = mnb;
                la[j] *= ca, lb[j] *= cb;
#pragma unroll
                for (int jj = 0; jj < kFaD / 16; ++jj) o[j][jj][0] *= ca, o[j][jj][1] *= cb, o[j][jj][2] *= ca, o[j][jj][3] *= cb;
            }
            const float ua = ma[j] == -INFINITY ? 0.0f : ma[j], ub = mb[j] == -INFINITY ? 0.0f : mb[j];
            const float p0 = sycl::exp2(c0 - ua), p1 = sycl::exp2(c1 - ub),
                        p2 = sycl::exp2(c2 - ua), p3 = sycl::exp2(c3 - ub);
            la[j] += p0 + p2, lb[j] += p1 + p3;
            const uint32_t plo = f2_to_u32(p0, p1), phi = f2_to_u32(p2, p3);
            /*
            DPCT1108: '__shfl_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const uint32_t
                alo = dpct::select_from_sub_group(sycl::ext::oneapi::this_work_item::get_sub_group(), plo,
                    src),
                blo = dpct::select_from_sub_group(sycl::ext::oneapi::this_work_item::get_sub_group(), plo,
                    src + 4);
            /*
            DPCT1108: '__shfl_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const uint32_t
                ahi = dpct::select_from_sub_group(sycl::ext::oneapi::this_work_item::get_sub_group(), phi,
                    src),
                bhi = dpct::select_from_sub_group(sycl::ext::oneapi::this_work_item::get_sub_group(), phi,
                    src + 4);
            pb0[j] = q_ok[j] ? dpct::byte_level_permute(alo, blo, psel) : 0u,
            pb1[j] = q_ok[j] ? dpct::byte_level_permute(ahi, bhi, psel) : 0u;
        }
        {
            const KVT* v0 = vg + row(2 * tq);
            const KVT* v1 = vg + row(2 * tq + 1);
            const KVT* v8 = vg + row(2 * tq + 8);
            const KVT* v9 = vg + row(2 * tq + 9);
            if constexpr (F8) {
                const sycl::uint4 r0 = ::ik2_ldg((const sycl::uint4 *)v0),
                                  r1 = ::ik2_ldg((const sycl::uint4 *)v1);
                const sycl::uint4 r8 = ::ik2_ldg((const sycl::uint4 *)v8),
                                  r9 = ::ik2_ldg((const sycl::uint4 *)v9);
                const uint32_t x0[4] = {r0.x(), r0.y(), r0.z(), r0.w()},
                               x1[4] = {r1.x(), r1.y(), r1.z(), r1.w()};
                const uint32_t x8[4] = {r8.x(), r8.y(), r8.z(), r8.w()},
                               x9[4] = {r9.x(), r9.y(), r9.z(), r9.w()};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint32_t a[4];
                    e4m3x4_h4(dpct::byte_level_permute(x0[i], x1[i], 0x5140u),
                              a[0], a[1]);
                    e4m3x4_h4(dpct::byte_level_permute(x8[i], x9[i], 0x5140u),
                              a[2], a[3]);
#pragma unroll
                    for (int j = 0; j < NT; ++j) mma16816(o[j][2 * i], a, pb0[j], pb1[j]);
                    e4m3x4_h4(dpct::byte_level_permute(x0[i], x1[i], 0x7362u),
                              a[0], a[1]);
                    e4m3x4_h4(dpct::byte_level_permute(x8[i], x9[i], 0x7362u),
                              a[2], a[3]);
#pragma unroll
                    for (int j = 0; j < NT; ++j) mma16816(o[j][2 * i + 1], a, pb0[j], pb1[j]);
                }
            } else {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const sycl::uint4 r0 = ::ik2_ldg((const sycl::uint4 *)v0 +
                                                     i),
                                      r1 = ::ik2_ldg((const sycl::uint4 *)v1 +
                                                     i);
                    const sycl::uint4 r8 = ::ik2_ldg((const sycl::uint4 *)v8 +
                                                     i),
                                      r9 = ::ik2_ldg((const sycl::uint4 *)v9 +
                                                     i);
                    const uint32_t x0[4] = {r0.x(), r0.y(), r0.z(), r0.w()},
                                   x1[4] = {r1.x(), r1.y(), r1.z(), r1.w()};
                    const uint32_t x8[4] = {r8.x(), r8.y(), r8.z(), r8.w()},
                                   x9[4] = {r9.x(), r9.y(), r9.z(), r9.w()};
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        const uint32_t a[4] = {
                            dpct::byte_level_permute(x0[jj], x1[jj], 0x5410u),
                            dpct::byte_level_permute(x0[jj], x1[jj], 0x7632u),
                            dpct::byte_level_permute(x8[jj], x9[jj], 0x5410u),
                            dpct::byte_level_permute(x8[jj], x9[jj], 0x7632u)};
#pragma unroll
                        for (int j = 0; j < NT; ++j) mma16816(o[j][4 * i + jj], a, pb0[j], pb1[j]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
        for (int sh = 4; sh < 32; sh <<= 1) {
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            la[j] += dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(),
                la[j], sh);
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            lb[j] += dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(),
                lb[j], sh);
        }
        const int ca = j * 8 + 2 * tq, cb = ca + 1;
        // the warp's partials: o_part [warp][T][H][D], ml_part [warp][T][H]
        if (ca < NC) {
            const size_t ix = (size_t) w * T * H + (size_t) (ca / R) * H + g * R + ca % R;
            float* op = o_part + ix * kFaD + gr * 16;
#pragma unroll
            for (int jj = 0; jj < kFaD / 16; ++jj) *
                (sycl::float2 *)(op + 2 * jj) =
                sycl::float2(o[j][jj][0], o[j][jj][2]);
            if (gr == 0) ml_part[ix] = sycl::float2(ma[j], la[j]);
        }
        if (cb < NC) {
            const size_t ix = (size_t) w * T * H + (size_t) (cb / R) * H + g * R + cb % R;
            float* op = o_part + ix * kFaD + gr * 16;
#pragma unroll
            for (int jj = 0; jj < kFaD / 16; ++jj) *
                (sycl::float2 *)(op + 2 * jj) =
                sycl::float2(o[j][jj][1], o[j][jj][3]);
            if (gr == 0) ml_part[ix] = sycl::float2(mb[j], lb[j]);
        }
    }
}
// attn_decode_mma_combine_kernel for a window: a block a (head, token); sink [T][H]; gate rows gate_ld apart;
// out, out16 [T][H][D]
__dpct_inline__ void attn_block_combine_kernel(
    const float *__restrict__ o_part, const sycl::float2 *__restrict__ ml_part,
    int n_split, int H, int T, const float *__restrict__ sink, float v_inv,
    const int *pos_d, int window, const float *__restrict__ gate, int gate_ld,
    float *out, uint16_t *out16, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto cm_sm = (float *)dpct_local;
    float* red = cm_sm;
    float* cw = cm_sm + kFaD;
    int k_lo, n_steps;
    const int n_act = wa_active_block(*pos_d, T, window, n_split * kWaNW, k_lo, n_steps);
    const int h = item_ct1.get_group(2), t = item_ct1.get_group(1),
              d = item_ct1.get_local_id(2), TH = T * H, ix = t * H + h;
    const float sk = sink[ix] * kLog2e;
    float m = d == 0 ? sk : -INFINITY;
#pragma unroll
    for (int s = d; s < n_act; s += kFaD)
        m = sycl::fmax(m, ml_part[(size_t)s * TH + ix].x());
    m = bmax(m, red);
    float den = d == 0 ? sycl::exp2(sk - m) : 0.0f;
    for (int s = d; s < n_act; s += kFaD) {
        const sycl::float2 ml = ml_part[(size_t)s * TH + ix];
        const float c = ml.x() == -INFINITY ? 0.0f : sycl::exp2(ml.x() - m);
        cw[s] = c;
        den += ml.y() * c;
    }
    den = bsum(den, red);
    float acc = 0.0f;
#pragma unroll
    for (int s = 0; s < n_act; ++s) acc +=
        o_part[((size_t)s * TH + ix) * kFaD + d] * cw[s];
    float res = acc / den * v_inv;
    if (gate) res =
        res * (1.0f /
               (1.0f + sycl::native::exp(
                           -gate[(size_t)t * gate_ld + (size_t)h * kFaD + d])));
    out[(size_t) ix * kFaD + d] = res;
    if (out16) out16[(size_t) ix * kFaD + d] = f2h(res);
}

// a few tokens' short conv (conv4_fused_kernel's arithmetic) with row strides; the history before and the inputs
// kept for a rollback (rec_old [3][C], rec_x [T][C]: conv_hist_after's) when given
__dpct_inline__ void conv_block_kernel(
    const float *__restrict__ x, int x_ld, float *hist,
    const float *__restrict__ w, const float *__restrict__ dyn, int d_ld,
    int heads, int T, int C, const float *__restrict__ res, int res_ld,
    float *y, int y_ld, float *rec_old, float *rec_x) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= C) return;
    float h0 = hist[c], h1 = hist[(size_t) C + c], h2 = hist[(size_t) 2 * C + c];
    if (rec_old) rec_old[c] = h0, rec_old[(size_t) C + c] = h1, rec_old[(size_t) 2 * C + c] = h2;
    const float* wc = w + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    const float* dc_base = dyn ? dyn + (size_t) (c / (C / heads)) * 4 : nullptr;
    for (int t = 0; t < T; ++t) {
        const float xn = x[(size_t) t * x_ld + c];
        if (rec_x) rec_x[(size_t) t * C + c] = xn;
        float a = w0 * h0 + w1 * h1 + w2 * h2 + w3 * xn;
        if (dc_base) {
            const float* dc = dc_base + (size_t) t * d_ld;
            a += dc[0] * xn + dc[1] * h2 + dc[2] * h1 + dc[3] * h0;
        }
        if (res) a += res[(size_t) t * res_ld + c];
        y[(size_t) t * y_ld + c] = a;
        h0 = h1, h1 = h2, h2 = xn;
    }
    hist[c] = h0, hist[(size_t) C + c] = h1, hist[(size_t) 2 * C + c] = h2;
}

// every short conv's history as after the first n of a window's tokens, all the layers' convs in one launch
__dpct_inline__ void rollback_block_kernel(const ConvRec *__restrict__ recs,
                                           int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const ConvRec rc = recs[item_ct1.get_group(1)];
    for (int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
         c < rc.C;
         c += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
        float v[3];
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int r = n + j;
            v[j] = r < 3 ? rc.old[(size_t) r * rc.C + c] : rc.x[(size_t) (r - 3) * rc.C + c];
        }
#pragma unroll
        for (int j = 0; j < 3; ++j) rc.hist[(size_t) j * rc.C + c] = v[j];
    }
}

// y [T][N] = W [N][K] (FP16) . x [T][K]: mv_f16_kernel a token a grid row
template <int WARPS>
__dpct_inline__ void mv_f16_rows_kernel(
    const uint16_t *__restrict__ W, int N, int K, const float *__restrict__ x,
    int x_ld, float *__restrict__ y, int y_ld) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2) * WARPS +
                    (item_ct1.get_local_id(2) >> 5),
              lane = item_ct1.get_local_id(2) & 31, t = item_ct1.get_group(1);
    if (row >= N) return;
    const uint16_t* w = W + (size_t) row * K;
    const float* xt = x + (size_t) t * x_ld;
    float a = 0.0f;
#pragma unroll
    for (int i = lane; i < K; i += 32) a += h2f(w[i]) * xt[i];
    a = wsum(a);
    if (lane == 0) y[(size_t) t * y_ld + row] = a;
}

// x [T][C] as two FP16 limbs a token, out [T][2][C] (hi, lo; x = hi + lo)
__dpct_inline__ void f16_limbs_rows_kernel(const float *__restrict__ x, int T,
                                           int C, uint16_t *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= T * C) return;
    const int t = i / C, c = i % C;
    const uint16_t hi = f2h(x[i]);
    out[(size_t) t * 2 * C + c] = hi, out[(size_t) t * 2 * C + C + c] = f2h(x[i] - h2f(hi));
}

// a window's picks grouped by expert on the device (one block of S = T K <= 64 threads): top / w [S] (route()'s)
// -> for each expert picked, group g (in the order of its first pick): blob[g] = tab[e], n[g] its tokens (<= T),
// tok / row [g][kGroupMax] (the token, and the slot t K + k: the row of h16 / y32); for each slot sg, su, wt (w with
// down's scale) and inv (the slot itself) - the arrays experts_*_g, experts_block_* and moe_combine_f32 take.
// Groups past the last have n = 0.
__dpct_inline__ void group_route_kernel(
    const int *__restrict__ top, const float *__restrict__ w, int S, int K,
    const uint8_t *const *__restrict__ tab, const float *sg_t,
    const float *su_t, const float *sd_t, const uint8_t **blob, int *n,
    int *tok, int *row, float *sg, float *su, float *wt, int *inv) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &ex = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[64]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &lead = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[64]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &gid = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[64]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int i = item_ct1.get_local_id(2);
    const int e = i < S ? top[i] : -1;
    ex[i] = e;
    if (i < S) n[i] = 0;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    int first = i, m = 0, cnt = 0;
    if (i < S) {
#pragma unroll
        for (int j = 0; j < S; ++j)
            if (ex[j] == e) {
                if (j < first) first = j;
                if (j < i) ++m;
                ++cnt;
            }
    }
    lead[i] = i < S && first == i;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (lead[i]) {
        int gx = 0;
#pragma unroll
        for (int j = 0; j < i; ++j) gx += lead[j];
        gid[i] = gx;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (i >= S) return;
    const int g = gid[first];
    tok[g * kGroupMax + m] = i / K, row[g * kGroupMax + m] = i;
    if (first == i) blob[g] = tab[e], n[g] = cnt;
    sg[i] = sg_t ? sg_t[e] : 1.0f, su[i] = su_t ? su_t[e] : 1.0f;
    wt[i] = w[i] * (sd_t ? sd_t[e] : 1.0f);
    inv[i] = i;
}

// A group's NVFP4 experts on the tensor cores (experts_mma_kernel for the tokens of a group): gate | up, a block 8
// gate rows and the same 8 of up, SPLIT warps each a stride of the 64-value blocks; the B columns are (member,
// limb) pairs - column c the member c / 2's limb c % 2 of its input (limbs [T][2][C]) - NT tiles of 8 (members <=
// 4 NT), so that the members' products read the expert's rows once.  h [slot][F] = SwiGLU of the member's rows.
template <int SPLIT, int NT>
/*
DPCT1110: The total declared local variable size in device function
experts_block_gu_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void experts_block_gu_kernel(
    const uint8_t *const *__restrict__ blob, const int *__restrict__ n_g,
    const int *__restrict__ tok_g, const int *__restrict__ row_g, size_t off_up,
    int C, int F, const uint16_t *__restrict__ limbs,
    const float *__restrict__ sg, const float *__restrict__ su, float gcap,
    float ucap, uint16_t *h) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    sycl::float4[SPLIT][NT][32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int grp = item_ct1.get_group(1), n = n_g[grp];
    if (n <= 0) return;
    const int lane = item_ct1.get_local_id(2) & 31,
              split = item_ct1.get_local_id(2) >> 5, gr = lane >> 2,
              t = lane & 3;
    const int r = item_ct1.get_group(2) * 8 + gr;
    const size_t rb = (size_t) (C / 64) * 36;
    const uint8_t* a = blob[grp] + (size_t) r * rb;
    const uint8_t* b = a + off_up;
    const sycl::uint4 *xp[NT];
    bool x_ok[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int mem = (j * 8 + gr) >> 1, limb = gr & 1;
        x_ok[j] = mem < n;
        const int tk = x_ok[j] ? tok_g[grp * kGroupMax + mem] : 0;
        xp[j] =
            (const sycl::uint4 *)(limbs + ((size_t)tk * 2 + limb) * C) + 2 * t;
    }
    float v[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j) v[j][0] = v[j][1] = v[j][2] = v[j][3] = 0.0f;
    const int nk = C / 64;
    NvStep cur = nv_step_load(a, b, split, t);
    for (int k = split; k < nk; k += SPLIT) {
        NvStep nxt = cur;
        if (k + SPLIT < nk) nxt = nv_step_load(a, b, k + SPLIT, t);
        const uint32_t sa = ue4m3_h2((cur.sa >> (8 * t)) & 0xFFu), sb = ue4m3_h2((cur.sb >> (8 * t)) & 0xFFu);
        const uint32_t wa[2] = {cur.wa0, cur.wa1}, wb[2] = {cur.wb0, cur.wb1};
        uint32_t af[4][4];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t qa = wa[m >> 1] >> (16 * (m & 1)), qb = wb[m >> 1] >> (16 * (m & 1));
            af[m][0] = nv_pair_h2(qa, sa), af[m][1] = nv_pair_h2(qb, sb);
            af[m][2] = nv_pair_h2(qa >> 4, sa), af[m][3] = nv_pair_h2(qb >> 4, sb);
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            sycl::uint4 x0 = sycl::uint4(0, 0, 0, 0), x1 = x0;
            if (x_ok[j]) x0 = ::ik2_ldg(xp[j] + k * 8),
                         x1 = ::ik2_ldg(xp[j] + k * 8 + 1);
            const uint32_t xx[8] = {x0.x(), x0.y(), x0.z(), x0.w(),
                                    x1.x(), x1.y(), x1.z(), x1.w()};
#pragma unroll
            for (int m = 0; m < 4; ++m) mma16816(v[j], af[m], xx[m], xx[4 + m]);
        }
        cur = nxt;
    }
#pragma unroll
    for (int j = 0; j < NT; ++j)
        part[split][j][lane] = sycl::float4(v[j][0], v[j][1], v[j][2], v[j][3]);
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (split) return;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        sycl::float4 tot = part[0][j][lane];
#pragma unroll
        for (int s = 1; s < SPLIT; ++s) {
            const sycl::float4 z = part[s][j][lane];
            tot.x() += z.x(), tot.y() += z.y(), tot.z() += z.z(),
                tot.w() += z.w();
        }
        const int mem = j * 4 + t;   // (the lane's columns 2 t, 2 t + 1 of tile j: member 4 j + t's two limbs)
        if (mem < n) {
            const int slot = row_g[grp * kGroupMax + mem];
            const float gg = (tot.x() + tot.y()) * (sg ? sg[slot] : 1.0f),
                        u = (tot.z() + tot.w()) * (su ? su[slot] : 1.0f);
            h[(size_t)slot * F + r] =
                f2h(gcap * sycl::tanh(gg / gcap) *
                    (1.0f / (1.0f + sycl::native::exp(-gg))) * ucap *
                    sycl::tanh(u / ucap));
        }
    }
}
// ... down: a block 16 rows, the B columns a group's members (<= 8, their FP16 h rows); y [slot][C]
template <int SPLIT>
/*
DPCT1110: The total declared local variable size in device function
experts_block_down_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
__dpct_inline__ void experts_block_down_kernel(
    const uint8_t *const *__restrict__ blob, const int *__restrict__ n_g,
    const int *__restrict__ row_g, size_t off_down, int C, int F,
    const uint16_t *__restrict__ h, float *y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    sycl::float4[SPLIT][32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int grp = item_ct1.get_group(1), n = n_g[grp];
    if (n <= 0) return;
    const int lane = item_ct1.get_local_id(2) & 31,
              split = item_ct1.get_local_id(2) >> 5, gr = lane >> 2,
              t = lane & 3;
    const int r = item_ct1.get_group(2) * 16 + gr;
    const size_t rb = (size_t) (F / 64) * 36;
    const uint8_t* a = blob[grp] + off_down + (size_t) r * rb;
    const uint8_t* b = a + 8 * rb;
    const bool x_ok = gr < n;
    const sycl::uint4 *xp =
        (const sycl::uint4 *)(h +
                              (size_t)(x_ok ? row_g[grp * kGroupMax + gr] : 0) *
                                  F) +
        2 * t;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int nk = F / 64;
    NvStep cur = nv_step_load(a, b, split, t);
    for (int k = split; k < nk; k += SPLIT) {
        NvStep nxt = cur;
        if (k + SPLIT < nk) nxt = nv_step_load(a, b, k + SPLIT, t);
        const uint32_t sa = ue4m3_h2((cur.sa >> (8 * t)) & 0xFFu), sb = ue4m3_h2((cur.sb >> (8 * t)) & 0xFFu);
        const uint32_t wa[2] = {cur.wa0, cur.wa1}, wb[2] = {cur.wb0, cur.wb1};
        sycl::uint4 x0 = sycl::uint4(0, 0, 0, 0), x1 = x0;
        if (x_ok) x0 = ::ik2_ldg(xp + k * 8), x1 = ::ik2_ldg(xp + k * 8 + 1);
        const uint32_t xx[8] = {x0.x(), x0.y(), x0.z(), x0.w(),
                                x1.x(), x1.y(), x1.z(), x1.w()};
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t qa = wa[m >> 1] >> (16 * (m & 1)), qb = wb[m >> 1] >> (16 * (m & 1));
            const uint32_t af[4] = {nv_pair_h2(qa, sa), nv_pair_h2(qb, sb), nv_pair_h2(qa >> 4, sa),
                                    nv_pair_h2(qb >> 4, sb)};
            mma16816(v, af, xx[m], xx[4 + m]);
        }
        cur = nxt;
    }
    part[split][lane] = sycl::float4(v[0], v[1], v[2], v[3]);
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (split) return;
    sycl::float4 tot = part[0][lane];
#pragma unroll
    for (int s = 1; s < SPLIT; ++s) {
        const sycl::float4 z = part[s][lane];
        tot.x() += z.x(), tot.y() += z.y(), tot.z() += z.z(), tot.w() += z.w();
    }
    // the lane's columns 2 t, 2 t + 1: members 2 t, 2 t + 1; rows r and r + 8
    if (2 * t < n) {
        const int slot = row_g[grp * kGroupMax + 2 * t];
        y[(size_t)slot * C + r] = tot.x(),
                             y[(size_t)slot * C + r + 8] = tot.z();
    }
    if (2 * t + 1 < n) {
        const int slot = row_g[grp * kGroupMax + 2 * t + 1];
        y[(size_t)slot * C + r] = tot.y(),
                             y[(size_t)slot * C + r + 8] = tot.w();
    }
}


// route() for a window, a block of E threads a token (route_token_kernel's picks: the same weights to the bit)
__dpct_inline__ void route_rows_kernel(
    const float *__restrict__ logits, const float *__restrict__ bias, int E,
    int K, float scale, int *top, float *w, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto rr_sm = (float *)dpct_local;
    float* val = rr_sm;
    uint32_t* picked = (uint32_t*) (rr_sm + E);
    float *sig = (float *)(picked + (item_ct1.get_local_range(2) >> 5));
    const int t = item_ct1.get_group(2), e = item_ct1.get_local_id(2),
              lane = e & 31, warp = e >> 5;
    const float* lg = logits + (size_t) t * E;
    const float sl =
        e < E ? 1.0f / (1.0f + sycl::native::exp(-ik2_ldg(lg + e))) : 0.0f;
    const float v = e < E ? sl + ik2_ldg(bias + e) : -INFINITY;
    if (e < E) val[e] = v;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    int rank = 0;
    for (int j = 0; j < E; ++j) {
        const float o = val[j];
        rank += (o > v || (o == v && j < e)) ? 1 : 0;
    }
    const bool pick = e < E && rank < K;
    const uint32_t mine = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(),
        (0xffffffffu &
         (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                     .get_local_linear_id())) &&
                pick
            ? (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                          .get_local_linear_id())
            : 0,
        sycl::ext::oneapi::plus<>());
    if (lane == 0) picked[warp] = mine;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    int pos = sycl::popcount(mine & ((1u << lane) - 1u));
#pragma unroll
    for (int ww = 0; ww < warp; ++ww) pos += sycl::popcount(picked[ww]);
    if (pick) sig[pos] = sl;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (e == 0) {
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < K; ++k) sum += sig[k];
        sig[K] = sum;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (pick) top[(size_t) t * K + pos] = e, w[(size_t) t * K + pos] = sl / (sig[K] + 1e-20f) * scale;
}

// a lane's part of a Q8_0 block of its two rows (34 bytes: the FP16 scale, 32 int8): the scales, and the values
// 8 t .. 8 t + 7 as four pairs (2-byte loads: a block is 2-byte aligned)
struct Q8Step {
    uint32_t sa, sb;
    uint32_t a[4], b[4];
};
__dpct_inline__ Q8Step q8_step_load(const uint8_t *a, const uint8_t *b, int k,
                                    int t) {
    const uint16_t* ap = (const uint16_t*) (a + (size_t) k * 34);
    const uint16_t* bp = (const uint16_t*) (b + (size_t) k * 34);
    Q8Step r;
    r.sa = ik2_ldg(ap), r.sb = ik2_ldg(bp);
#pragma unroll
    for (int i = 0; i < 4; ++i) r.a[i] = ik2_ldg(ap + 1 + 4 * t + i), r.b[i] = ik2_ldg(bp + 1 + 4 * t + i);
    return r;
}
// experts_block_gu_kernel over Q8_0 blocks: a block's two MMA (values 8 t .. + 3, 8 t + 4 .. + 7 of the lane, B
// taken in the same order) into a sum of its own, scaled by the row's block scale in FP32 - the int8 values exact
// in FP16, nothing rounded but the input's limbs
template <int SPLIT, int NT>
/*
DPCT1110: The total declared local variable size in device function
experts_block_gu_q8_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
__dpct_inline__ void experts_block_gu_q8_kernel(
    const uint8_t *const *__restrict__ blob, const int *__restrict__ n_g,
    const int *__restrict__ tok_g, const int *__restrict__ row_g, size_t off_up,
    int C, int F, const uint16_t *__restrict__ limbs,
    const float *__restrict__ sg, const float *__restrict__ su, float gcap,
    float ucap, uint16_t *h) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    sycl::float4[SPLIT][NT][32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int grp = item_ct1.get_group(1), n = n_g[grp];
    if (n <= 0) return;
    const int lane = item_ct1.get_local_id(2) & 31,
              split = item_ct1.get_local_id(2) >> 5, gr = lane >> 2,
              t = lane & 3;
    const int r = item_ct1.get_group(2) * 8 + gr;
    const size_t rb = (size_t) (C / 32) * 34;
    const uint8_t* a = blob[grp] + (size_t) r * rb;
    const uint8_t* b = a + off_up;
    const sycl::uint4 *xp[NT];
    bool x_ok[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int mem = (j * 8 + gr) >> 1, limb = gr & 1;
        x_ok[j] = mem < n;
        const int tk = x_ok[j] ? tok_g[grp * kGroupMax + mem] : 0;
        xp[j] = (const sycl::uint4 *)(limbs + ((size_t)tk * 2 + limb) * C) + t;
    }
    float v[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j) v[j][0] = v[j][1] = v[j][2] = v[j][3] = 0.0f;
    const int nk = C / 32;
    Q8Step cur = q8_step_load(a, b, split, t);
    for (int k = split; k < nk; k += SPLIT) {
        Q8Step nxt = cur;
        if (k + SPLIT < nk) nxt = q8_step_load(a, b, k + SPLIT, t);
        const float sa = h2f((uint16_t) cur.sa), sb = h2f((uint16_t) cur.sb);
        const uint32_t af0[4] = {q8_pair(cur.a[0]), q8_pair(cur.b[0]), q8_pair(cur.a[1]), q8_pair(cur.b[1])};
        const uint32_t af1[4] = {q8_pair(cur.a[2]), q8_pair(cur.b[2]), q8_pair(cur.a[3]), q8_pair(cur.b[3])};
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            sycl::uint4 x = sycl::uint4(0, 0, 0, 0);
            if (x_ok[j]) x = ::ik2_ldg(xp[j] + k * 4);
            float bs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            mma16816(bs, af0, x.x(), x.y());
            mma16816(bs, af1, x.z(), x.w());
            v[j][0] += sa * bs[0], v[j][1] += sa * bs[1], v[j][2] += sb * bs[2], v[j][3] += sb * bs[3];
        }
        cur = nxt;
    }
#pragma unroll
    for (int j = 0; j < NT; ++j)
        part[split][j][lane] = sycl::float4(v[j][0], v[j][1], v[j][2], v[j][3]);
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (split) return;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        sycl::float4 tot = part[0][j][lane];
#pragma unroll
        for (int s = 1; s < SPLIT; ++s) {
            const sycl::float4 z = part[s][j][lane];
            tot.x() += z.x(), tot.y() += z.y(), tot.z() += z.z(),
                tot.w() += z.w();
        }
        const int mem = j * 4 + t;
        if (mem < n) {
            const int slot = row_g[grp * kGroupMax + mem];
            const float gg = (tot.x() + tot.y()) * (sg ? sg[slot] : 1.0f),
                        u = (tot.z() + tot.w()) * (su ? su[slot] : 1.0f);
            h[(size_t)slot * F + r] =
                f2h(gcap * sycl::tanh(gg / gcap) *
                    (1.0f / (1.0f + sycl::native::exp(-gg))) * ucap *
                    sycl::tanh(u / ucap));
        }
    }
}
template <int SPLIT>
/*
DPCT1110: The total declared local variable size in device function
experts_block_down_q8_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
__dpct_inline__ void experts_block_down_q8_kernel(
    const uint8_t *const *__restrict__ blob, const int *__restrict__ n_g,
    const int *__restrict__ row_g, size_t off_down, int C, int F,
    const uint16_t *__restrict__ h, float *y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    sycl::float4[SPLIT][32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int grp = item_ct1.get_group(1), n = n_g[grp];
    if (n <= 0) return;
    const int lane = item_ct1.get_local_id(2) & 31,
              split = item_ct1.get_local_id(2) >> 5, gr = lane >> 2,
              t = lane & 3;
    const int r = item_ct1.get_group(2) * 16 + gr;
    const size_t rb = (size_t) (F / 32) * 34;
    const uint8_t* a = blob[grp] + off_down + (size_t) r * rb;
    const uint8_t* b = a + 8 * rb;
    const bool x_ok = gr < n;
    const sycl::uint4 *xp =
        (const sycl::uint4 *)(h +
                              (size_t)(x_ok ? row_g[grp * kGroupMax + gr] : 0) *
                                  F) +
        t;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int nk = F / 32;
    Q8Step cur = q8_step_load(a, b, split, t);
    for (int k = split; k < nk; k += SPLIT) {
        Q8Step nxt = cur;
        if (k + SPLIT < nk) nxt = q8_step_load(a, b, k + SPLIT, t);
        const float sa = h2f((uint16_t) cur.sa), sb = h2f((uint16_t) cur.sb);
        const uint32_t af0[4] = {q8_pair(cur.a[0]), q8_pair(cur.b[0]), q8_pair(cur.a[1]), q8_pair(cur.b[1])};
        const uint32_t af1[4] = {q8_pair(cur.a[2]), q8_pair(cur.b[2]), q8_pair(cur.a[3]), q8_pair(cur.b[3])};
        sycl::uint4 x = sycl::uint4(0, 0, 0, 0);
        if (x_ok) x = ::ik2_ldg(xp + k * 4);
        float bs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        mma16816(bs, af0, x.x(), x.y());
        mma16816(bs, af1, x.z(), x.w());
        v[0] += sa * bs[0], v[1] += sa * bs[1], v[2] += sb * bs[2], v[3] += sb * bs[3];
        cur = nxt;
    }
    part[split][lane] = sycl::float4(v[0], v[1], v[2], v[3]);
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (split) return;
    sycl::float4 tot = part[0][lane];
#pragma unroll
    for (int s = 1; s < SPLIT; ++s) {
        const sycl::float4 z = part[s][lane];
        tot.x() += z.x(), tot.y() += z.y(), tot.z() += z.z(), tot.w() += z.w();
    }
    if (2 * t < n) {
        const int slot = row_g[grp * kGroupMax + 2 * t];
        y[(size_t)slot * C + r] = tot.x(),
                             y[(size_t)slot * C + r + 8] = tot.z();
    }
    if (2 * t + 1 < n) {
        const int slot = row_g[grp * kGroupMax + 2 * t + 1];
        y[(size_t)slot * C + r] = tot.y(),
                             y[(size_t)slot * C + r + 8] = tot.w();
    }
}

}  // namespace

void rms(const float *x, const float *w, int rows, int D, float eps, float *y,
         dpct::queue_ptr s) {
    if (rows <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<dpct_kernel_name<class rms_impl_kernel_7b9e9b,
                                         dpct_kernel_scalar<false>,
                                         dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(sycl::range(1, 1, rows) * sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                rms_impl_kernel<false, true>(x, w, D, eps, nullptr, y);
            });
    }
    check("rms");
}
void rms_multi(const RmsJobs &jobs, int n, int D, float eps,
               dpct::queue_ptr s) {
    if (n <= 0) return;
    if (n > 3) {
        std::fprintf(stderr, "ik2::k::rms_multi: %d vectors\n", n);
        std::abort();
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class rms_multi_kernel_e17a7a>>(
            sycl::nd_range<3>(sycl::range(1, 1, (unsigned)n) *
                                  sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                rms_multi_kernel(jobs, D, eps);
            });
    }
    check("rms_multi");
}
void rms_f16(const float *x, const float *w, int rows, int D, float eps,
             uint16_t *y16, float *y32, dpct::queue_ptr s) {
    if (rows <= 0) return;
    if (y16 && y32) {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<dpct_kernel_name<class rms_impl_kernel_992ad9,
                                         dpct_kernel_scalar<true>,
                                         dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(sycl::range(1, 1, rows) * sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                rms_impl_kernel<true, true>(x, w, D, eps, y16, y32);
            });
    } else if (y16) {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<dpct_kernel_name<class rms_impl_kernel_7caf21,
                                         dpct_kernel_scalar<true>,
                                         dpct_kernel_scalar<false>>>(
            sycl::nd_range<3>(sycl::range(1, 1, rows) * sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                rms_impl_kernel<true, false>(x, w, D, eps, y16, nullptr);
            });
    } else if (y32) {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<dpct_kernel_name<class rms_impl_kernel_5f7f37,
                                         dpct_kernel_scalar<false>,
                                         dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(sycl::range(1, 1, rows) * sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                rms_impl_kernel<false, true>(x, w, D, eps, nullptr, y32);
            });
    }
    check("rms_f16");
}

void l2clip(const float *x, const float *scale, int rows, int G, int D,
            float th, const int *pos, float alpha, float n_floor, float *y,
            dpct::queue_ptr s) {
    if (rows <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class l2clip_kernel_60f833>>(
            sycl::nd_range<3>(sycl::range(1, 1, rows * G) *
                                  sycl::range(1, 1, 128),
                              sycl::range(1, 1, 128)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                l2clip_kernel(x, scale, G, D, th, pos, alpha, n_floor, y);
            });
    }
    check("l2clip");
}

void conv4_res(const float *x, float *hist, const float *w, const float *dyn,
               int heads, int T, int C, const float *residual, float *y,
               dpct::queue_ptr s) {
    if (T <= 0) return;
    if (T <= kConvFusedMaxT) {
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->parallel_for<dpct_kernel_name<class conv4_fused_kernel_9d8829>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((C + 127) / 128)) *
                        sycl::range(1, 1, 128),
                    sycl::range(1, 1, 128)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    conv4_fused_kernel(x, hist, w, dyn, heads, T, C, residual,
                                       y);
                });
        }
        check("conv4_res");
        return;
    }
    const dpct::dim3 g((unsigned)((C + 127) / 128),
                       (unsigned)((T + kConvTileT - 1) / kConvTileT));
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class conv4_tiled_kernel_aa2d23>>(
            sycl::nd_range<3>(g * sycl::range(1, 1, 128),
                              sycl::range(1, 1, 128)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                conv4_tiled_kernel(x, hist, w, dyn, heads, T, C, residual, y);
            });
    }
    check("conv4_res");
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class conv4_hist_kernel_1b75dd>>(
            sycl::nd_range<3>(sycl::range(1, 1, grid(C)) *
                                  sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                conv4_hist_kernel(x, hist, T, C);
            });
    }
    check("conv4_hist");
}

void conv4(const float *x, float *hist, const float *w, const float *dyn,
           int heads, int T, int C, float *y, dpct::queue_ptr s) {
    conv4_res(x, hist, w, dyn, heads, T, C, nullptr, y, s);
}

template <bool FUSED_RMS>
void launch_attn_res(const ResStates &states, int n, const float *q, int nq,
                     int T, int C, float res_eps, float rms_eps, float *out,
                     const RmsOuts &outs, dpct::queue_ptr s, int jobs = 1) {
    if (T <= 0) return;
    const size_t q_step = (size_t) nq * C, o_step = (size_t) nq * T * C;   // (a job's queries, its outputs)
    const dpct::dim3 g((unsigned)T, (unsigned)jobs);
    if (n < 1 || n > kResMax || nq < 1 || nq > 3) {
        std::fprintf(stderr, "ik2::k::attn_res: n %d, nq %d\n", n, nq);
        std::abort();
    }
    bool vec4 = ((C & 3) == 0) && (((uintptr_t) q & 15) == 0);
    for (int i = 0; i < n && vec4; ++i) vec4 &= (((uintptr_t) states.st[i] & 15) == 0);
    if constexpr (!FUSED_RMS) {
        vec4 &= (((uintptr_t) out & 15) == 0);
    } else {
        for (int qi = 0; qi < nq && vec4; ++qi) {
            vec4 &= (!outs.w[qi] || (((uintptr_t) outs.w[qi] & 15) == 0)) &&
                    (!outs.y16[qi] || (((uintptr_t) outs.y16[qi] & 7) == 0)) &&
                    (!outs.y32[qi] || (((uintptr_t) outs.y32[qi] & 15) == 0));
        }
    }
    if (vec4 && C == 4608) {
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<dpct_kernel_name<class attn_res_vec4_kernel_67969c,
                                         dpct_kernel_scalar<FUSED_RMS>,
                                         dpct_kernel_scalar<576>>>(
            sycl::nd_range<3>(g * sycl::range(1, 1, 576),
                              sycl::range(1, 1, 576)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_res_vec4_kernel<FUSED_RMS, 576>(states, n, q, nq, T, C,
                                                     res_eps, rms_eps, out,
                                                     outs, q_step, o_step);
            });
    } else if (vec4 && C <= 4096) {
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<dpct_kernel_name<class attn_res_vec4_kernel_e0c2d5,
                                         dpct_kernel_scalar<FUSED_RMS>,
                                         dpct_kernel_scalar<512>>>(
            sycl::nd_range<3>(g * sycl::range(1, 1, 512),
                              sycl::range(1, 1, 512)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_res_vec4_kernel<FUSED_RMS, 512>(states, n, q, nq, T, C,
                                                     res_eps, rms_eps, out,
                                                     outs, q_step, o_step);
            });
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class attn_res_scalar_kernel_352dbe,
                                         dpct_kernel_scalar<FUSED_RMS>>>(
            sycl::nd_range<3>(g * sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_res_scalar_kernel<FUSED_RMS>(states, n, q, nq, T, C,
                                                  res_eps, rms_eps, out, outs,
                                                  q_step, o_step);
            });
    }
    check(FUSED_RMS ? "attn_res_rms_f16" : "attn_res");
}

void attn_res(const ResStates &states, int n, const float *q, int nq, int T,
              int C, float eps, float *out, dpct::queue_ptr s) {
    launch_attn_res<false>(states, n, q, nq, T, C, eps, 0.0f, out, RmsOuts{}, s);
}
void attn_res_jobs(const ResStates &states, int n, const float *q, int nq,
                   int jobs, int C, float eps, float *out, dpct::queue_ptr s) {
    launch_attn_res<false>(states, n, q, nq, 1, C, eps, 0.0f, out, RmsOuts{}, s, jobs);
}

void attn_res_rms_f16(const ResStates &states, int n, const float *q, int nq,
                      int T, int C, float res_eps, float rms_eps,
                      const RmsOuts &outs, dpct::queue_ptr s) {
    launch_attn_res<true>(states, n, q, nq, T, C, res_eps, rms_eps, nullptr, outs, s);
}

void attn_probs(float *lg, int Tq, int nk, int qpos0, int kpos0, float qk_scale,
                const float *r, int r_ld, int d_rel, const float *proj,
                int extent, int window, const float *sink, int sink_ld,
                dpct::queue_ptr s) {
    if (Tq <= 0 || nk <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class attn_probs_kernel_416df8>>(
            sycl::nd_range<3>(sycl::range(1, 1, Tq) * sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                attn_probs_kernel(lg, nk, qpos0, kpos0, qk_scale, r, r_ld,
                                  d_rel, proj, extent, window, sink, sink_ld);
            });
    }
    check("attn_probs");
}
void attn_probs_bias_f16(const float *lg, uint16_t *out16, int H, int Tq,
                         int nk, int qpos0, int kpos0, float qk_scale,
                         const float *bias, int extent, int window,
                         const float *sink, dpct::queue_ptr s) {
    if (H <= 0 || Tq <= 0 || nk <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<
            dpct_kernel_name<class attn_probs_bias_f16_kernel_574358>>(
            sycl::nd_range<3>(sycl::range(1, (unsigned)H, (unsigned)Tq) *
                                  sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                attn_probs_bias_f16_kernel(lg, out16, H, Tq, nk, qpos0, kpos0,
                                           qk_scale, bias, extent, window,
                                           sink);
            });
    }
    check("attn_probs_bias_f16");
}
// the Hopper kernel (wgmma) on a 9.x GPU of a build with sm_90a; IK2_WGMMA=0: the mma.sync one everywhere
bool fa_wgmma() try {
    static const int on = [] {
        if (getenv("IK2_WGMMA") && !std::atoi(getenv("IK2_WGMMA"))) return 0;
        int dev = 0, major = 0;
        dev = dpct::get_current_device_id();
        major = dpct::get_device(dev).get_major_version();
        return major == 9 ? 1 : 0;
    }();
    return on != 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
template <int NW, bool WG, class KVT>
void attn_flash_launch(const uint16_t *q16, int q_ld, const KVT *kc,
                       const KVT *vc, int kv_cap, int H, int KV, int Tq,
                       int qpos0, float qk_scale, const float *bias, int extent,
                       int window, int n_split, float *o_part,
                       sycl::float2 *ml_part, dpct::queue_ptr s) {
    static bool attr[16] = {};   // (per device: the attribute is a device's, and the layer split runs on several)
    int dev = 0;
    dev = dpct::get_current_device_id();
    if (dev < 16 && !attr[dev]) {
        /*
        DPCT1026: The call to cudaFuncSetAttribute was removed because SYCL
        currently does not support corresponding setting.
        */
        check("attn_flash (shared memory)");
        attr[dev] = true;
    }
    const dpct::dim3 grid((unsigned)((Tq + 16 * NW - 1) / (16 * NW)),
                          (unsigned)H, (unsigned)n_split);
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(kFaSmem), cgh);

            cgh.parallel_for<dpct_kernel_name<
                class attn_flash_f16_kernel_9649ed, dpct_kernel_scalar<NW>,
                dpct_kernel_scalar<WG>, KVT>>(
                sycl::nd_range<3>(grid * sycl::range(1, 1, NW * 32),
                                  sycl::range(1, 1, NW * 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        attn_flash_f16_kernel<NW, WG, KVT>(
                            q16, q_ld, kc, vc, kv_cap, H, KV, Tq, qpos0,
                            qk_scale, bias, extent, window, n_split, o_part,
                            ml_part,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    }
    check("attn_flash");
}
#include "ik2_flash_xmx.dp.inc"
#include "ik2_moe_xmx.dp.inc"
void attn_flash_f16(const uint16_t *q16, int q_ld, const uint16_t *kc,
                    const uint16_t *vc, int kv_cap, int H, int KV, int D,
                    int Tq, int qpos0, float qk_scale, const float *bias,
                    int extent, int window, const float *sink, float *out,
                    int o_ld, float *scratch, size_t scratch_floats,
                    int n_split, dpct::queue_ptr s) {
    if (Tq <= 0) return;
    if (D != kFaD) {
        std::fprintf(stderr, "ik2::k::attn_flash_f16: D = %d, not 128\n", D);
        std::abort();
    }
    const size_t per_split = (size_t) Tq * H * (kFaD + 2);
    n_split = std::max(1, std::min<int>(n_split, (int) (scratch_floats / per_split)));
    float* o_part = scratch;
    sycl::float2 *ml_part =
        (sycl::float2 *)(scratch + (size_t)n_split * Tq * H * kFaD);
    if (g_flash_xmx) attn_flash_xmx_launch(q16, q_ld, kc, vc, kv_cap, H, KV, Tq, qpos0, qk_scale, bias, extent, window, n_split, o_part, ml_part, s);
    else if (fa_wgmma())
        attn_flash_launch<kFaNW, true, uint16_t>(q16, q_ld, kc, vc, kv_cap, H, KV, Tq, qpos0, qk_scale, bias, extent,
                                                 window, n_split, o_part, ml_part, s);
    else
        attn_flash_launch<kFaNW, false, uint16_t>(q16, q_ld, kc, vc, kv_cap, H, KV, Tq, qpos0, qk_scale, bias, extent,
                                                  window, n_split, o_part, ml_part, s);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<
            dpct_kernel_name<class attn_flash_combine_kernel_40cc38>>(
            sycl::nd_range<3>(sycl::range(1, (unsigned)H, (unsigned)Tq) *
                                  sycl::range(1, 1, kFaD),
                              sycl::range(1, 1, kFaD)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                attn_flash_combine_kernel(o_part, ml_part, n_split, Tq, H, sink,
                                          out, o_ld, 1.0f);
            });
    }
    check("attn_flash_combine");
}
void attn_flash_fp8(const uint16_t *q16, int q_ld, const uint8_t *kc,
                    const uint8_t *vc, int kv_cap, int H, int KV, int D, int Tq,
                    int qpos0, float qk_scale, float v_inv_scale,
                    const float *bias, int extent, int window,
                    const float *sink, float *out, int o_ld, float *scratch,
                    size_t scratch_floats, int n_split, dpct::queue_ptr s) {
    if (Tq <= 0) return;
    if (D != kFaD) {
        std::fprintf(stderr, "ik2::k::attn_flash_fp8: D = %d, not 128\n", D);
        std::abort();
    }
    const size_t per_split = (size_t) Tq * H * (kFaD + 2);
    n_split = std::max(1, std::min<int>(n_split, (int) (scratch_floats / per_split)));
    float* o_part = scratch;
    sycl::float2 *ml_part =
        (sycl::float2 *)(scratch + (size_t)n_split * Tq * H * kFaD);
    attn_flash_launch<kFaNW, false, uint8_t>(q16, q_ld, kc, vc, kv_cap, H, KV, Tq, qpos0, qk_scale, bias, extent, window,
                                             n_split, o_part, ml_part, s);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<
            dpct_kernel_name<class attn_flash_combine_kernel_337d44>>(
            sycl::nd_range<3>(sycl::range(1, (unsigned)H, (unsigned)Tq) *
                                  sycl::range(1, 1, kFaD),
                              sycl::range(1, 1, kFaD)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                attn_flash_combine_kernel(o_part, ml_part, n_split, Tq, H, sink,
                                          out, o_ld, v_inv_scale);
            });
    }
    check("attn_flash_combine");
}
int attn_flash_splits(int Tq, int H, int nk, int n_sm) {
    const int blocks = (Tq + 16 * kFaNW - 1) / (16 * kFaNW) * H;
    const int want = (6 * n_sm + blocks - 1) / blocks;   // (about three waves of two blocks an SM)
    const int most = std::max(1, (nk + 16 * kFaBN - 1) / (16 * kFaBN));   // (>= 1024 keys a split)
    return std::max(1, std::min({want, most, 16}));
}
void attn_probs_f16(const float *lg, uint16_t *out16, int H, int Tq, int nk,
                    int qpos0, int kpos0, float qk_scale, const float *r,
                    int d_rel, const float *proj, int extent, int window,
                    const float *sink, dpct::queue_ptr s) {
    if (H <= 0 || Tq <= 0 || nk <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class attn_probs_f16_kernel_3cfced>>(
            sycl::nd_range<3>(sycl::range(1, (unsigned)H, (unsigned)Tq) *
                                  sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                attn_probs_f16_kernel(lg, out16, H, Tq, nk, qpos0, kpos0,
                                      qk_scale, r, d_rel, proj, extent, window,
                                      sink);
            });
    }
    check("attn_probs_f16");
}

void gate_mul(float *o, const float *g, size_t n, dpct::queue_ptr s) {
    if (!n) return;
    const bool vec4 = (((uintptr_t) o & 15) == 0) && (((uintptr_t) g & 15) == 0);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<dpct_kernel_name<class gate_mul_impl_kernel_588306,
                                         dpct_kernel_scalar<false>,
                                         dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(
                sycl::range(1, 1, vec4 ? grid((n + 3) >> 2) : grid(n)) *
                    sycl::range(1, 1, kBlk),
                sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                gate_mul_impl_kernel<false, true>(o, g, n, nullptr);
            });
    }
    check("gate_mul");
}
void gate_mul_f16(float *o, const float *g, size_t n, uint16_t *o16,
                  bool write_f32, dpct::queue_ptr s) {
    if (!n) return;
    const bool vec4 = (((uintptr_t) o & 15) == 0) && (((uintptr_t) g & 15) == 0) &&
                      (!o16 || ((uintptr_t) o16 & 7) == 0);
    const unsigned nb = vec4 ? grid((n + 3) >> 2) : grid(n);
    if (o16 && write_f32) {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<dpct_kernel_name<class gate_mul_impl_kernel_8bbd6d,
                                         dpct_kernel_scalar<true>,
                                         dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(sycl::range(1, 1, nb) * sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                gate_mul_impl_kernel<true, true>(o, g, n, o16);
            });
    } else if (o16) {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<dpct_kernel_name<class gate_mul_impl_kernel_de5789,
                                         dpct_kernel_scalar<true>,
                                         dpct_kernel_scalar<false>>>(
            sycl::nd_range<3>(sycl::range(1, 1, nb) * sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                gate_mul_impl_kernel<true, false>(o, g, n, o16);
            });
    } else if (write_f32) {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<dpct_kernel_name<class gate_mul_impl_kernel_cfc843,
                                         dpct_kernel_scalar<false>,
                                         dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(sycl::range(1, 1, nb) * sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                gate_mul_impl_kernel<false, true>(o, g, n, nullptr);
            });
    }
    check("gate_mul_f16");
}

void route(const float *logits, const float *bias, int T, int E, int K,
           float scale, int *top, float *w, dpct::queue_ptr s) {
    if (T <= 0) return;
    if (K < 1 || K > kRouteMax) {
        std::fprintf(stderr, "ik2::k::route: K %d\n", K);
        std::abort();
    }
    if (E <= 512) {
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->parallel_for<dpct_kernel_name<class route_warp_kernel_6f2ea8>>(
                sycl::nd_range<3>(sycl::range(1, 1, T) * sycl::range(1, 1, 32),
                                  sycl::range(1, 1, 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        route_warp_kernel(logits, bias, T, E, K, scale, top, w);
                    });
        }
        check("route_warp");
        return;
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range((size_t)E * 4), cgh);

            cgh.parallel_for<dpct_kernel_name<class route_kernel_ad3e38>>(
                sycl::nd_range<3>(sycl::range(1, 1, T) * sycl::range(1, 1, 128),
                                  sycl::range(1, 1, 128)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        route_kernel(
                            logits, bias, T, E, K, scale, top, w,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    }
    check("route");
}

void gather_f16(const float *x, const int *idx, int n, int C, uint16_t *y,
                dpct::queue_ptr s) {
    if (n <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class gather_f16_kernel_887b41>>(
            sycl::nd_range<3>(sycl::range(1, 1, grid((size_t)n * C)) *
                                  sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                gather_f16_kernel(x, idx, n, C, y);
            });
    }
    check("gather_f16");
}

void swiglu_cap(const uint16_t *gu, int n, int F, const float *sg,
                const float *su, float gcap, float ucap, uint16_t *h,
                dpct::queue_ptr s) {
    if (n <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class swiglu_cap_kernel_8e4214>>(
            sycl::nd_range<3>(sycl::range(1, 1, grid((size_t)n * F)) *
                                  sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                swiglu_cap_kernel(gu, n, F, sg, su, gcap, ucap, h);
            });
    }
    check("swiglu_cap");
}

void moe_combine(const uint16_t *y, const int *inv, const float *wt, int T,
                 int K, int C, float *out, bool add, dpct::queue_ptr s) {
    if (T <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class moe_combine_kernel_caff08>>(
            sycl::nd_range<3>(sycl::range(1, 1, grid((size_t)T * C)) *
                                  sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                moe_combine_kernel(y, inv, wt, T, K, C, out, add);
            });
    }
    check("moe_combine");
}

void swiglu_cap_f32(const float *gu, int n, int F, const float *sg,
                    const float *su, float gcap, float ucap, uint16_t *h,
                    dpct::queue_ptr s) {
    if (n <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class swiglu_cap_f32_kernel_7a1635>>(
            sycl::nd_range<3>(sycl::range(1, 1, grid((size_t)n * F)) *
                                  sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                swiglu_cap_f32_kernel(gu, n, F, sg, su, gcap, ucap, h);
            });
    }
    check("swiglu_cap_f32");
}
void moe_combine_f32(const float *y, const int *inv, const float *wt, int T,
                     int K, int C, float *out, bool add, dpct::queue_ptr s) {
    if (T <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class moe_combine_f32_kernel_141be3>>(
            sycl::nd_range<3>(sycl::range(1, 1, grid((size_t)T * C)) *
                                  sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                moe_combine_f32_kernel(y, inv, wt, T, K, C, out, add);
            });
    }
    check("moe_combine_f32");
}
void moe_combine_conv(const float *y, const int *inv, const float *wt, int K,
                      int C, float *hist, const float *w, const float *res,
                      float *out, dpct::queue_ptr s) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class moe_combine_conv_kernel_4bdf0b>>(
            sycl::nd_range<3>(sycl::range(1, 1, (unsigned)((C + 127) / 128)) *
                                  sycl::range(1, 1, 128),
                              sycl::range(1, 1, 128)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                moe_combine_conv_kernel(y, inv, wt, K, C, hist, w, res, out);
            });
    }
    check("moe_combine_conv");
}
void add(const float *a, const float *b, size_t n, float *y,
         dpct::queue_ptr s) {
    if (!n) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class add_kernel_f5e85c>>(
            sycl::nd_range<3>(sycl::range(1, 1, grid(n)) *
                                  sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                add_kernel(a, b, n, y);
            });
    }
    check("add");
}
void to_f16(const float *x, size_t n, uint16_t *y, dpct::queue_ptr s) {
    if (!n) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class to_f16_kernel_1dcff1>>(
            sycl::nd_range<3>(sycl::range(1, 1, grid(n)) *
                                  sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                to_f16_kernel(x, n, y);
            });
    }
    check("to_f16");
}

void to_f16_limbs(const float *x, size_t n, uint16_t *y, dpct::queue_ptr s) {
    if (!n) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class to_f16_limbs_kernel_b59791>>(
            sycl::nd_range<3>(sycl::range(1, 1, grid(n)) *
                                  sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                to_f16_limbs_kernel(x, n, y);
            });
    }
    check("to_f16_limbs");
}
void softcap(float *x, size_t n, float cap, dpct::queue_ptr s) {
    if (!n) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class softcap_kernel_6f6dfe>>(
            sycl::nd_range<3>(sycl::range(1, 1, grid(n)) *
                                  sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                softcap_kernel(x, n, cap);
            });
    }
    check("softcap");
}

void loop_gates(const LoopGate *gt, int n, int T, int C, int R, float eps,
                float *sc, dpct::queue_ptr s) {
    if (T <= 0 || n <= 0) return;
    if (n > 3 || R > kLoopRankMax || T > 65535) {
        std::fprintf(stderr, "ik2::k::loop_gates: %d gates, rank %d, %d tokens\n", n, R, T);
        std::abort();
    }
    bool compact = true;
    for (int j = 0; j < n; ++j)
        compact = compact && gt[j].compact[0] && gt[j].compact[1] && gt[j].compact[2];
    LoopGateArgs a{};
    for (int j = 0; j < n; ++j) {
        a.x[j] = gt[j].x;
        a.m[j] = gt[j].m;
        a.y[j] = gt[j].y;
        a.D[j] = compact ? (const void*)gt[j].compact[0] : gt[j].D;
        a.F[j] = compact ? (const void*)gt[j].compact[1] : gt[j].F;
        a.G[j] = compact ? (const void*)gt[j].compact[2] : gt[j].G;
    }
    const dpct::dim3 dots(R, n, T);
    if (compact)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class loop_gate_dots_kernel_8c8edd,
                                         dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(dots * sycl::range(1, 1, 128),
                              sycl::range(1, 1, 128)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                loop_gate_dots_kernel<true>(a, T, C, R, sc);
            });
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class loop_gate_dots_kernel_c79350,
                                         dpct_kernel_scalar<false>>>(
            sycl::nd_range<3>(dots * sycl::range(1, 1, 128),
                              sycl::range(1, 1, 128)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                loop_gate_dots_kernel<false>(a, T, C, R, sc);
            });
    }
    check("loop_gates (dots)");
    static const int tile = getenv("IK2_GATE_TILE") ? atoi(getenv("IK2_GATE_TILE")) : 0;
    const bool fixed = T == 1 && R == 64 && (tile == 32 || tile == 64 || tile == 128);
    const int threads = fixed ? tile : kBlk;
    const dpct::dim3 apply((C + threads - 1) / threads, n, T);
    if (fixed && compact)
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<
            dpct_kernel_name<class loop_gate_apply_kernel_9580fb,
                             dpct_kernel_scalar<64>, dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(apply * sycl::range(1, 1, threads),
                              sycl::range(1, 1, threads)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                loop_gate_apply_kernel<64, true>(a, T, C, R, eps, sc);
            });
    } else if (fixed)
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class loop_gate_apply_kernel_ec65e3,
                                         dpct_kernel_scalar<64>,
                                         dpct_kernel_scalar<false>>>(
            sycl::nd_range<3>(apply * sycl::range(1, 1, threads),
                              sycl::range(1, 1, threads)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                loop_gate_apply_kernel<64, false>(a, T, C, R, eps, sc);
            });
    } else if (compact)
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<
            dpct_kernel_name<class loop_gate_apply_kernel_3e4ed0,
                             dpct_kernel_scalar<0>, dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(apply * sycl::range(1, 1, threads),
                              sycl::range(1, 1, threads)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                loop_gate_apply_kernel<0, true>(a, T, C, R, eps, sc);
            });
    } else
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed
    the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<
            dpct_kernel_name<class loop_gate_apply_kernel_8452ff,
                             dpct_kernel_scalar<0>, dpct_kernel_scalar<false>>>(
            sycl::nd_range<3>(apply * sycl::range(1, 1, threads),
                              sycl::range(1, 1, threads)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                loop_gate_apply_kernel<0, false>(a, T, C, R, eps, sc);
            });
    }
    check("loop_gates (apply)");
}
void loop_gate(const float *x, const float *m, const float *D, const float *F,
               const float *G, int T, int C, int R, float eps, float *sc,
               float *y, dpct::queue_ptr s) {
    const LoopGate g{x, m, D, F, G, y};
    loop_gates(&g, 1, T, C, R, eps, sc, s);
}
constexpr int kCandPer = 8192;   // values a block of the histogram
size_t cand_scratch(int n) { return (size_t) ((n + kCandPer - 1) / kCandPer) * kCandBins; }
void cand_topk(const float *x, int n, float lo, float hi, int K, int cap,
               int *scratch, int *out, dpct::queue_ptr s) {
    if (n <= 0) return;
    const float inv_w = (float) kCandBins / (hi - lo);
    const int n_part = (n + kCandPer - 1) / kCandPer;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            auto kCandPer_ct4 = kCandPer;

            cgh.parallel_for<dpct_kernel_name<class cand_hist_kernel_7f06ae>>(
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned)n_part) *
                                      sycl::range(1, 1, kBlk),
                                  sycl::range(1, 1, kBlk)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    cand_hist_kernel(x, n, lo, inv_w, kCandPer_ct4, scratch);
                });
        });
    }
    check("cand_topk (histogram)");
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class cand_threshold_kernel_80195f>>(
            sycl::nd_range<3>(sycl::range(1, 1, kCandBins),
                              sycl::range(1, 1, kCandBins)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                cand_threshold_kernel(scratch, n_part, K, out);
            });
    }
    check("cand_topk (threshold)");
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            auto out_ct6 = out + 2;
            auto out_cap_ct7 = (float *)(out + 2 + cap);

            cgh.parallel_for<dpct_kernel_name<class cand_gather_kernel_36e977>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid((size_t)n)) *
                                      sycl::range(1, 1, kBlk),
                                  sycl::range(1, 1, kBlk)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    cand_gather_kernel(x, n, lo, inv_w, cap, out, out_ct6,
                                       out_cap_ct7);
                });
        });
    }
    check("cand_topk (gather)");
}
void loop_gate_mix(const float *x, const float *m, const float *a1,
                   const float *a2, int T, int C, int R, float eps, float *g,
                   dpct::queue_ptr s) {
    if (T <= 0) return;
    if (R > kBlk) {
        std::fprintf(stderr, "ik2::k::loop_gate_mix: rank %d\n", R);
        std::abort();
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class loop_gate_mix_kernel_30d46b>>(
            sycl::nd_range<3>(sycl::range(1, 1, (unsigned)T) *
                                  sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                loop_gate_mix_kernel(x, m, a1, a2, C, R, eps, g);
            });
    }
    check("loop_gate_mix");
}
void loop_gate_blend(const float *x, const float *m, const float *f,
                     const float *gn, size_t n, float *y, dpct::queue_ptr s) {
    if (!n) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class loop_gate_blend_kernel_4ddebb>>(
            sycl::nd_range<3>(sycl::range(1, 1, grid(n)) *
                                  sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                loop_gate_blend_kernel(x, m, f, gn, n, y);
            });
    }
    check("loop_gate_blend");
}

void conv_hist_after(const float *old, const float *x, int T, int C,
                     float *hist, dpct::queue_ptr s) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class conv_hist_after_kernel_ca6888>>(
            sycl::nd_range<3>(sycl::range(1, 1, grid(C)) *
                                  sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                conv_hist_after_kernel(old, x, T, C, hist);
            });
    }
    check("conv_hist_after");
}
void to_bf16(const float *x, size_t n, uint16_t *y, dpct::queue_ptr s) {
    if (!n) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class to_bf16_kernel_2f7ce6>>(
            sycl::nd_range<3>(sycl::range(1, 1, grid(n)) *
                                  sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                to_bf16_kernel(x, n, y);
            });
    }
    check("to_bf16");
}
void mix_rms(const float *x, int n_tap, size_t tap_stride, int rows, int C,
             const float *alpha, const float *w, float eps, float *out,
             dpct::queue_ptr s) {
    if (rows <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class mix_rms_kernel_b0455b>>(
            sycl::nd_range<3>(sycl::range(1, 1, rows) * sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                mix_rms_kernel(x, n_tap, tap_stride, C, alpha, w, eps, out);
            });
    }
    check("mix_rms");
}
void rope_neox(float *x, int rows, int heads, const int *pos, float theta,
               dpct::queue_ptr s) {
    if (rows <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class rope_neox_kernel_547d67>>(
            sycl::nd_range<3>(sycl::range(1, heads, rows) *
                                  sycl::range(1, 1, 64),
                              sycl::range(1, 1, 64)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                rope_neox_kernel(x, heads, pos, theta);
            });
    }
    check("rope_neox");
}
template <class KVT>
static void draft_attn_impl(const float *q, const KVT *kc, const KVT *vc,
                            int k_lo, int k_hi, const float *kb,
                            const float *vb, int nq, int H, int Hkv,
                            float scale, float *scratch, float *out,
                            dpct::queue_ptr s, int cap) {
    // scratch: the logits [H][S][8][per], the chunks' outputs [S][H][8][128] and (max, sum)
    if (nq > kDraftQ) {
        std::fprintf(stderr, "ik2::k::draft_attn: %d queries (at most %d)\n", nq, kDraftQ);
        std::abort();
    }
    const int nk = k_hi - k_lo + nq;
    const int S = std::max(1, std::min(64, (nk + 255) / 256)), per = (nk + S - 1) / S;
    float* lgs = scratch;
    float* part_o = lgs + (size_t) H * S * kDraftQ * per;
    sycl::float2 *part_ml =
        (sycl::float2 *)(part_o + (size_t)S * H * kDraftQ * 128);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            auto cap_cap_ct15 = cap > 0 ? cap : (1 << 30);

            cgh.parallel_for<
                dpct_kernel_name<class draft_attn_part_kernel_a3a79c, KVT>>(
                sycl::nd_range<3>(sycl::range(1, (unsigned)S, (unsigned)H) *
                                      sycl::range(1, 1, 128),
                                  sycl::range(1, 1, 128)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        draft_attn_part_kernel<KVT>(
                            q, kc, vc, k_lo, k_hi, kb, vb, nq, H, Hkv, scale,
                            per, lgs, part_o, part_ml, cap_cap_ct15);
                    });
        });
    }
    check("draft_attn (chunks)");
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class draft_attn_merge_kernel_d43020, KVT>>(
            sycl::nd_range<3>(sycl::range(1, (unsigned)nq, (unsigned)H) *
                                  sycl::range(1, 1, 128),
                              sycl::range(1, 1, 128)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                draft_attn_merge_kernel(part_o, part_ml, S, H, out);
            });
    }
    check("draft_attn (merge)");
}
void draft_attn(const float *q, const float *kc, const float *vc, int k_lo,
                int k_hi, const float *kb, const float *vb, int nq, int H,
                int Hkv, float scale, float *scratch, float *out,
                dpct::queue_ptr s, int cap) {
    draft_attn_impl<float>(q, kc, vc, k_lo, k_hi, kb, vb, nq, H, Hkv, scale, scratch, out, s, cap);
}
void draft_attn_f16(const float *q, const uint16_t *kc, const uint16_t *vc,
                    int k_lo, int k_hi, const float *kb, const float *vb,
                    int nq, int H, int Hkv, float scale, float *scratch,
                    float *out, dpct::queue_ptr s, int cap) {
    draft_attn_impl<uint16_t>(q, kc, vc, k_lo, k_hi, kb, vb, nq, H, Hkv, scale, scratch, out, s, cap);
}
void silu_mul_bf16(const float *gu, int T, int F, uint16_t *out,
                   dpct::queue_ptr s) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class silu_mul_bf16_kernel_b3e79c>>(
            sycl::nd_range<3>(sycl::range(1, 1, grid((size_t)T * F)) *
                                  sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                silu_mul_bf16_kernel(gu, T, F, out);
            });
    }
    check("silu_mul_bf16");
}
void rnn_inputs(const float *state, const uint16_t *emb, const int *prev,
                const float *hidden, const float *f, int R, int C,
                uint16_t *out, dpct::queue_ptr s) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class rnn_inputs_kernel_af5d58>>(
            sycl::nd_range<3>(sycl::range(1, 1, grid((size_t)2 * R + 2 * C)) *
                                  sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                rnn_inputs_kernel(state, emb, prev, hidden, f, R, C, out);
            });
    }
    check("rnn_inputs");
}
void rnn_gates(const float *proj, const float *bias, int R, float *state,
               float *latent, dpct::queue_ptr s) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class rnn_gates_kernel_571d0e>>(
            sycl::nd_range<3>(sycl::range(1, 1, grid(R)) *
                                  sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                rnn_gates_kernel(proj, bias, R, state, latent);
            });
    }
    check("rnn_gates");
}
void argmax_sum(const float *a, const float *b, const float *c, int n, int *out,
                dpct::queue_ptr s) {
    if (n <= 0) return;
    const int B = std::min(256, std::max(1, (n + 1023) / 1024));
    if (B == 1) {
        {
            g_argmax_v.init(*s);
            g_argmax_i.init(*s);

            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->submit([&](sycl::handler &cgh) {
                auto g_argmax_v_ptr_ct1 = g_argmax_v.get_ptr();
                auto g_argmax_i_ptr_ct1 = g_argmax_i.get_ptr();

                cgh.parallel_for<
                    dpct_kernel_name<class argmax_sum_part_kernel_4825cb,
                                     dpct_kernel_scalar<true>>>(
                    sycl::nd_range<3>(sycl::range(1, 1, 256),
                                      sycl::range(1, 1, 256)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            argmax_sum_part_kernel<true>(a, b, c, n, out,
                                                         g_argmax_v_ptr_ct1,
                                                         g_argmax_i_ptr_ct1);
                        });
            });
        }
        check("argmax_sum");
        return;
    }
    {
        g_argmax_v.init(*s);
        g_argmax_i.init(*s);

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            auto g_argmax_v_ptr_ct1 = g_argmax_v.get_ptr();
            auto g_argmax_i_ptr_ct1 = g_argmax_i.get_ptr();

            cgh.parallel_for<
                dpct_kernel_name<class argmax_sum_part_kernel_2ddd56,
                                 dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(sycl::range(1, 1, B) * sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        argmax_sum_part_kernel<false>(a, b, c, n, out,
                                                      g_argmax_v_ptr_ct1,
                                                      g_argmax_i_ptr_ct1);
                    });
        });
    }
    check("argmax_sum_part");
    {
        g_argmax_v.init(*s);
        g_argmax_i.init(*s);

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            auto g_argmax_v_ptr_ct1 = g_argmax_v.get_ptr();
            auto g_argmax_i_ptr_ct1 = g_argmax_i.get_ptr();

            cgh.parallel_for<
                dpct_kernel_name<class argmax_final_kernel_5ff2b9>>(
                sycl::nd_range<3>(sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        argmax_final_kernel(B, 1, out, g_argmax_v_ptr_ct1,
                                            g_argmax_i_ptr_ct1);
                    });
        });
    }
    check("argmax_sum_final");
}
void argmax_rows(const float *x, int ld, int n_rows, int n_cols, int *out,
                 dpct::queue_ptr s) {
    if (n_rows <= 0 || n_cols <= 0) return;
    const int max_b = std::max(1, kArgmaxMaxPartials / n_rows);
    const int B = std::min(max_b, std::min(256, std::max(1, (n_cols + 1023) / 1024)));
    if (B == 1) {
        {
            g_argmax_v.init(*s);
            g_argmax_i.init(*s);

            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->submit([&](sycl::handler &cgh) {
                auto g_argmax_v_ptr_ct1 = g_argmax_v.get_ptr();
                auto g_argmax_i_ptr_ct1 = g_argmax_i.get_ptr();

                cgh.parallel_for<
                    dpct_kernel_name<class argmax_rows_part_kernel_63f891,
                                     dpct_kernel_scalar<true>>>(
                    sycl::nd_range<3>(sycl::range(1, (unsigned)n_rows, 1) *
                                          sycl::range(1, 1, 256),
                                      sycl::range(1, 1, 256)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            argmax_rows_part_kernel<true>(x, ld, n_cols, out,
                                                          g_argmax_v_ptr_ct1,
                                                          g_argmax_i_ptr_ct1);
                        });
            });
        }
        check("argmax_rows");
        return;
    }
    {
        g_argmax_v.init(*s);
        g_argmax_i.init(*s);

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            auto g_argmax_v_ptr_ct1 = g_argmax_v.get_ptr();
            auto g_argmax_i_ptr_ct1 = g_argmax_i.get_ptr();

            cgh.parallel_for<
                dpct_kernel_name<class argmax_rows_part_kernel_26acb2,
                                 dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(
                    sycl::range(1, (unsigned)n_rows, (unsigned)B) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        argmax_rows_part_kernel<false>(x, ld, n_cols, out,
                                                       g_argmax_v_ptr_ct1,
                                                       g_argmax_i_ptr_ct1);
                    });
        });
    }
    check("argmax_rows_part");
    {
        g_argmax_v.init(*s);
        g_argmax_i.init(*s);

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            auto g_argmax_v_ptr_ct1 = g_argmax_v.get_ptr();
            auto g_argmax_i_ptr_ct1 = g_argmax_i.get_ptr();

            cgh.parallel_for<
                dpct_kernel_name<class argmax_final_kernel_b63bd9>>(
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned)n_rows) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        argmax_final_kernel(B, B, out, g_argmax_v_ptr_ct1,
                                            g_argmax_i_ptr_ct1);
                    });
        });
    }
    check("argmax_rows_final");
}
void argmax_gemv_f16(const uint16_t *W, int ld, const float *x,
                     const float *base, const uint16_t *bias_f16, int R, int n,
                     int *out, dpct::queue_ptr s) {
    if (n <= 0) return;
    const int n4 = std::max(1, n >> 2);
    const int B = std::min(512, std::max(1, (n4 + 127) / 128));
    const bool use_smem = R <= 4096;
    /*
    DPCT1083: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    const size_t smem = use_smem ? (size_t)R * sizeof(float) : 0;
    if (B == 1) {
        {
            g_argmax_v.init(*s);
            g_argmax_i.init(*s);

            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

            s->submit([&](sycl::handler &cgh) {
                auto g_argmax_v_ptr_ct1 = g_argmax_v.get_ptr();
                auto g_argmax_i_ptr_ct1 = g_argmax_i.get_ptr();

                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range(smem), cgh);

                cgh.parallel_for<
                    dpct_kernel_name<class argmax_gemv_f16_part_kernel_d170b0,
                                     dpct_kernel_scalar<true>>>(
                    sycl::nd_range<3>(sycl::range(1, 1, 128),
                                      sycl::range(1, 1, 128)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        argmax_gemv_f16_part_kernel<true>(
                            W, ld, x, base, bias_f16, R, n, use_smem, out,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            g_argmax_v_ptr_ct1, g_argmax_i_ptr_ct1);
                    });
            });
        }
        check("argmax_gemv_f16");
        return;
    }
    {
        g_argmax_v.init(*s);
        g_argmax_i.init(*s);

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto g_argmax_v_ptr_ct1 = g_argmax_v.get_ptr();
            auto g_argmax_i_ptr_ct1 = g_argmax_i.get_ptr();

            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(smem), cgh);

            cgh.parallel_for<
                dpct_kernel_name<class argmax_gemv_f16_part_kernel_9419a4,
                                 dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(sycl::range(1, 1, B) * sycl::range(1, 1, 128),
                                  sycl::range(1, 1, 128)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        argmax_gemv_f16_part_kernel<false>(
                            W, ld, x, base, bias_f16, R, n, use_smem, out,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            g_argmax_v_ptr_ct1, g_argmax_i_ptr_ct1);
                    });
        });
    }
    check("argmax_gemv_f16_part");
    {
        g_argmax_v.init(*s);
        g_argmax_i.init(*s);

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            auto g_argmax_v_ptr_ct1 = g_argmax_v.get_ptr();
            auto g_argmax_i_ptr_ct1 = g_argmax_i.get_ptr();

            cgh.parallel_for<
                dpct_kernel_name<class argmax_final_kernel_27ef33>>(
                sycl::nd_range<3>(sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        argmax_final_kernel(B, 1, out, g_argmax_v_ptr_ct1,
                                            g_argmax_i_ptr_ct1);
                    });
        });
    }
    check("argmax_gemv_f16_final");
}

int block_values(int type) {
    return type == kNVFP4 ? 64 : type == kQ8_0 ? 32
         : type == kQ5_K || type == kQ6_K || type == kIQ3_XXS || type == kIQ4_XS ? 256 : 0;
}
size_t block_bytes(int type) {
    return type == kNVFP4 ? 36 : type == kQ8_0 ? 34 : type == kQ5_K ? 176 : type == kQ6_K ? 210
         : type == kIQ3_XXS ? 98 : type == kIQ4_XS ? 136 : 0;
}

template <int WARPS, int ROWS>
void launch_experts_gate_up(const uint8_t *const *blob, const int *tok, int S,
                            int ty_gate, int ty_up, size_t eb0, const float *x,
                            int C, int F, float *gu, dpct::queue_ptr s) {
    const dpct::dim3 g((unsigned)((2 * F + WARPS * ROWS - 1) / (WARPS * ROWS)),
                       (unsigned)S);
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        d_iq3xxs_grid.init(*s);
        d_iq4nl.init(*s);

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto d_iq3xxs_grid_ptr_ct1 = d_iq3xxs_grid.get_ptr();
            auto d_iq4nl_ptr_ct1 = d_iq4nl.get_ptr();

            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(xp_floats(C) * 4), cgh);

            auto mv_row_bytes_ty_gate_C_ct8 = mv_row_bytes(ty_gate, C);
            auto mv_row_bytes_ty_up_C_ct9 = mv_row_bytes(ty_up, C);

            cgh.parallel_for<dpct_kernel_name<
                class experts_gate_up_kernel_70fdcf, dpct_kernel_scalar<WARPS>,
                dpct_kernel_scalar<ROWS>>>(
                sycl::nd_range<3>(g * sycl::range(1, 1, WARPS * 32),
                                  sycl::range(1, 1, WARPS * 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        experts_gate_up_kernel<WARPS, ROWS>(
                            blob, tok, ty_gate, ty_up, eb0, x, C, F,
                            mv_row_bytes_ty_gate_C_ct8,
                            mv_row_bytes_ty_up_C_ct9, gu,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            d_iq3xxs_grid_ptr_ct1, d_iq4nl_ptr_ct1);
                    });
        });
    }
}
template <int WARPS, int ROWS>
void launch_experts_down(const uint8_t *const *blob, int S, int ty_down,
                         size_t off_down, const uint16_t *h, int C, int F,
                         float *y, dpct::queue_ptr s) {
    const dpct::dim3 g((unsigned)((C + WARPS * ROWS - 1) / (WARPS * ROWS)),
                       (unsigned)S);
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        d_iq3xxs_grid.init(*s);
        d_iq4nl.init(*s);

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto d_iq3xxs_grid_ptr_ct1 = d_iq3xxs_grid.get_ptr();
            auto d_iq4nl_ptr_ct1 = d_iq4nl.get_ptr();

            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(xp_floats(F) * 4), cgh);

            auto mv_row_bytes_ty_down_F_ct6 = mv_row_bytes(ty_down, F);

            cgh.parallel_for<dpct_kernel_name<class experts_down_kernel_dc23b5,
                                              dpct_kernel_scalar<WARPS>,
                                              dpct_kernel_scalar<ROWS>>>(
                sycl::nd_range<3>(g * sycl::range(1, 1, WARPS * 32),
                                  sycl::range(1, 1, WARPS * 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        experts_down_kernel<WARPS, ROWS>(
                            blob, ty_down, off_down, h, C, F,
                            mv_row_bytes_ty_down_F_ct6, y,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            d_iq3xxs_grid_ptr_ct1, d_iq4nl_ptr_ct1);
                    });
        });
    }
}

void expert_limb_mv(bool gate, const uint8_t *const *blob, size_t offset, int C,
                    int F, int S, const float *x, uint16_t *h, float *y,
                    const float *sg, const float *su, float gcap, float ucap,
                    void *scratch, dpct::queue_ptr stream, bool q8,
                    bool packed) {
    auto* p = (NvInput*)scratch;
    if (gate) {
        if (!packed) {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            stream->parallel_for<dpct_kernel_name<class nv_input_9eee48,
                                                  dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, 1, (C / 16 + 31) / 32) *
                                      sycl::range(1, 1, 32),
                                  sycl::range(1, 1, 32)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    nv_input<true>(x, nullptr, C, p, q8);
                });
        }
        if (q8)
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            stream->parallel_for<dpct_kernel_name<class nv_limb_mv_974b86,
                                                  dpct_kernel_scalar<true>,
                                                  dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, S, (F + 3) / 4) *
                                      sycl::range(1, 1, 128),
                                  sycl::range(1, 1, 128)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        nv_limb_mv<true, true>(blob, p, offset, C, F, sg, su,
                                               gcap, ucap, h, nullptr);
                    });
        } else {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            stream->parallel_for<dpct_kernel_name<class nv_limb_mv_689dc8,
                                                  dpct_kernel_scalar<true>,
                                                  dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(sycl::range(1, S, (F + 3) / 4) *
                                      sycl::range(1, 1, 128),
                                  sycl::range(1, 1, 128)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        nv_limb_mv<true, false>(blob, p, offset, C, F, sg, su,
                                                gcap, ucap, h, nullptr);
                    });
        }
    } else {
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            stream->parallel_for<dpct_kernel_name<class nv_input_d41d57,
                                                  dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(sycl::range(1, S, (F / 16 + 31) / 32) *
                                      sycl::range(1, 1, 32),
                                  sycl::range(1, 1, 32)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    nv_input<false>(nullptr, h, F, p, q8);
                });
        }
        if (q8)
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            stream->parallel_for<dpct_kernel_name<class nv_limb_mv_2006c9,
                                                  dpct_kernel_scalar<false>,
                                                  dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, S, (C + 7) / 8) *
                                      sycl::range(1, 1, 128),
                                  sycl::range(1, 1, 128)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        nv_limb_mv<false, true>(blob, p, offset, C, F, nullptr,
                                                nullptr, 0, 0, nullptr, y);
                    });
        } else {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            stream->parallel_for<dpct_kernel_name<class nv_limb_mv_47f3a4,
                                                  dpct_kernel_scalar<false>,
                                                  dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(sycl::range(1, S, (C + 7) / 8) *
                                      sycl::range(1, 1, 128),
                                  sycl::range(1, 1, 128)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        nv_limb_mv<false, false>(blob, p, offset, C, F, nullptr,
                                                 nullptr, 0, 0, nullptr, y);
                    });
        }
    }
    check("nv_expert_limb");
}
bool experts_mma_fits(int ty_gate, int ty_up, int ty_down, int C, int F) {
    return ty_gate == kNVFP4 && ty_up == kNVFP4 && ty_down == kNVFP4 && C % 64 == 0 && F % 64 == 0;
}
static int experts_mma_split(bool gate) {
    static const int wg = getenv("IK2_EXPERT_MMA_SPLIT") ? atoi(getenv("IK2_EXPERT_MMA_SPLIT")) : 8;
    static const int wd = getenv("IK2_EXPERT_MMA_SPLIT_DOWN") ? atoi(getenv("IK2_EXPERT_MMA_SPLIT_DOWN")) : 4;
    return gate ? wg : wd;
}
template <bool Gate>
static void launch_experts_mma(const uint8_t *const *blob, int S, size_t off,
                               const uint16_t *x16, int x_ld, int C, int F,
                               const float *sg, const float *su, float gcap,
                               float ucap, uint16_t *h, float *y,
                               dpct::queue_ptr s) {
    const int w = experts_mma_split(Gate);
    const dpct::dim3 g((unsigned)(Gate ? F / 8 : C / 16), (unsigned)S);
#define IK2_MMA_LAUNCH(W)                                                      \
    do {                                                                          \
        auto exp_props = sycl::ext::oneapi::experimental::properties{          \
            sycl::ext::oneapi::experimental::use_root_sync};                   \
                                                                               \
        s->submit([&](sycl::handler &cgh) {                                    \
            auto blob_ct0 = blob;                                              \
            auto off_ct1 = off;                                                \
            auto C_ct2 = C;                                                    \
            auto F_ct3 = F;                                                    \
            auto x16_ct4 = x16;                                                \
            auto x_ld_ct5 = x_ld;                                              \
            auto h_ct6 = h;                                                    \
            auto y_ct7 = y;                                                    \
            auto sg_ct8 = sg;                                                  \
            auto su_ct9 = su;                                                  \
            auto gcap_ct10 = gcap;                                             \
            auto ucap_ct11 = ucap;                                             \
                                                                               \
            cgh.parallel_for<dpct_kernel_name<class experts_mma_kernel_aac73a, \
                                              dpct_kernel_scalar<Gate>,        \
                                              dpct_kernel_scalar<W>>>(         \
                sycl::nd_range<3>(g * sycl::range(1, 1, W * 32),               \
                                  sycl::range(1, 1, W * 32)),                  \
                exp_props,                                                     \
                [=](sycl::nd_item<3> item_ct1)                                 \
                    [[sycl::reqd_sub_group_size(32)]] {                        \
                        experts_mma_kernel<Gate, W>(                           \
                            blob_ct0, off_ct1, C_ct2, F_ct3, x16_ct4,          \
                            x_ld_ct5, h_ct6, y_ct7, sg_ct8, su_ct9, gcap_ct10, \
                            ucap_ct11);                                        \
                    });                                                        \
        });                                                                    \
    } while (0)
    if (w == 1) IK2_MMA_LAUNCH(1);
    else if (w == 2) IK2_MMA_LAUNCH(2);
    else if (w == 3) IK2_MMA_LAUNCH(3);
    else if (w == 6) IK2_MMA_LAUNCH(6);
    else if (w == 8) IK2_MMA_LAUNCH(8);
    else IK2_MMA_LAUNCH(4);
#undef IK2_MMA_LAUNCH
}
void experts_gate_up_swiglu_mma(const uint8_t *const *blob, int S, size_t eb0,
                                const uint16_t *x16, int C, int F,
                                const float *sg, const float *su, float gcap,
                                float ucap, uint16_t *h, dpct::queue_ptr s) {
    if (S <= 0) return;
    launch_experts_mma<true>(blob, S, eb0, x16, 0, C, F, sg, su, gcap, ucap, h, nullptr, s);
    check("experts_gate_up_swiglu_mma");
}
void experts_down_mma(const uint8_t *const *blob, int S, size_t off_down,
                      const uint16_t *h, int C, int F, float *y,
                      dpct::queue_ptr s) {
    if (S <= 0) return;
    launch_experts_mma<false>(blob, S, off_down, h, F, C, F, nullptr, nullptr, 0.0f, 0.0f, nullptr, y, s);
    check("experts_down_mma");
}
void experts_gate_up(const uint8_t *const *blob, const int *tok, int S,
                     int ty_gate, int ty_up, size_t eb0, const float *x, int C,
                     int F, float *gu, dpct::queue_ptr s) {
    if (S <= 0) return;
    if (S <= 2) launch_experts_gate_up<4, 2>(blob, tok, S, ty_gate, ty_up, eb0, x, C, F, gu, s);
    else launch_experts_gate_up<4, 4>(blob, tok, S, ty_gate, ty_up, eb0, x, C, F, gu, s);
    check("experts_gate_up");
}
void experts_down(const uint8_t *const *blob, int S, int ty_down,
                  size_t off_down, const uint16_t *h, int C, int F, float *y,
                  dpct::queue_ptr s) {

    if (S <= 0) return;
    launch_experts_down<8, 1>(blob, S, ty_down, off_down, h, C, F, y, s);
    check("experts_down");
}
void experts_gate_up_g(const uint8_t *const *blob, const int *n, const int *tok,
                       const int *row, int G, int ty_gate, int ty_up,
                       size_t eb0, const float *x, int C, int F, float *gu,
                       dpct::queue_ptr s) {
    if (G <= 0) return;
    const dpct::dim3 g((unsigned)((2 * F + kGroupWarps * kGroupRows - 1) /
                                  (kGroupWarps * kGroupRows)),
                       (unsigned)G);
    {
        d_iq3xxs_grid.init(*s);
        d_iq4nl.init(*s);

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto d_iq3xxs_grid_ptr_ct1 = d_iq3xxs_grid.get_ptr();
            auto d_iq4nl_ptr_ct1 = d_iq4nl.get_ptr();

            auto mv_row_bytes_ty_gate_C_ct10 = mv_row_bytes(ty_gate, C);
            auto mv_row_bytes_ty_up_C_ct11 = mv_row_bytes(ty_up, C);

            cgh.parallel_for<
                dpct_kernel_name<class experts_gate_up_g_kernel_3515e6>>(
                sycl::nd_range<3>(g * sycl::range(1, 1, kGroupWarps * 32),
                                  sycl::range(1, 1, kGroupWarps * 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        experts_gate_up_g_kernel(
                            blob, n, tok, row, ty_gate, ty_up, eb0, x, C, F,
                            mv_row_bytes_ty_gate_C_ct10,
                            mv_row_bytes_ty_up_C_ct11, gu,
                            d_iq3xxs_grid_ptr_ct1, d_iq4nl_ptr_ct1);
                    });
        });
    }
    check("experts_gate_up_g");
}
// The serving engine can retain the original short-prompt arithmetic and enable
// the grouped tensor-core path only in a window whose tokens are being verified.
thread_local bool grouped_tensor_products = true;
void enable_grouped_tensor_products(bool enabled) { grouped_tensor_products = enabled; }
void experts_down_g(const uint8_t *const *blob, const int *n, const int *row,
                    int G, int ty_down, size_t off_down, const uint16_t *h,
                    int C, int F, float *y, dpct::queue_ptr s) {
    if (G <= 0) return;
    static const bool use_group_mma = getenv("IK2_GROUP_MMA") && (atoi(getenv("IK2_GROUP_MMA")) & 2);
    if (grouped_tensor_products && use_group_mma && ty_down == kNVFP4 && C % 16 == 0 && F % 64 == 0) {
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->/*
  DPCT1050: The template argument of the dpct_kernel_name could not be
  deduced. You need to update this code.
  */
                parallel_for<dpct_kernel_name<
                    class grouped_expert_mma_6ba10e, dpct_kernel_scalar<false>,
                    dpct_kernel_scalar<4>,
                    dpct_kernel_scalar<false>>>(
                    sycl::nd_range<3>(sycl::range(1, G, C / 16) *
                                          sycl::range(1, 1, 128),
                                      sycl::range(1, 1, 128)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            grouped_expert_mma<false, 4>(
                                blob, nullptr, off_down, C, F, h, nullptr, y,
                                nullptr, nullptr, 0, 0, n, row, nullptr);
                        });
        }
        check("grouped_expert_mma (grouped)");
        return;
    }

    if (grouped_tensor_products && use_group_mma && ty_down == kQ8_0 && C % 16 == 0 && F % 32 == 0) {
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->parallel_for<dpct_kernel_name<
                class grouped_expert_mma_f4a088, dpct_kernel_scalar<false>,
                dpct_kernel_scalar<4>, dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, G, C / 16) *
                                      sycl::range(1, 1, 128),
                                  sycl::range(1, 1, 128)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        grouped_expert_mma<false, 4, true>(
                            blob, nullptr, off_down, C, F, h, nullptr, y,
                            nullptr, nullptr, 0, 0, n, row, nullptr);
                    });
        }
        check("q8 grouped MMA");
        return;
    }
    const dpct::dim3 g((unsigned)((C + kGroupWarps * kGroupRows - 1) /
                                  (kGroupWarps * kGroupRows)),
                       (unsigned)G);
    {
        d_iq3xxs_grid.init(*s);
        d_iq4nl.init(*s);

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto d_iq3xxs_grid_ptr_ct1 = d_iq3xxs_grid.get_ptr();
            auto d_iq4nl_ptr_ct1 = d_iq4nl.get_ptr();

            auto mv_row_bytes_ty_down_F_ct8 = mv_row_bytes(ty_down, F);

            cgh.parallel_for<
                dpct_kernel_name<class experts_down_g_kernel_328e3b>>(
                sycl::nd_range<3>(g * sycl::range(1, 1, kGroupWarps * 32),
                                  sycl::range(1, 1, kGroupWarps * 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        experts_down_g_kernel(
                            blob, n, row, ty_down, off_down, h, C, F,
                            mv_row_bytes_ty_down_F_ct8, y,
                            d_iq3xxs_grid_ptr_ct1, d_iq4nl_ptr_ct1);
                    });
        });
    }
    check("experts_down_g");
}

template <int WARPS, int ROWS>
void launch_experts_gate_up_swiglu(const uint8_t *const *blob, const int *tok,
                                   int S, int ty_gate, int ty_up, size_t eb0,
                                   const float *x, int C, int F,
                                   const float *sg, const float *su, float gcap,
                                   float ucap, uint16_t *h, dpct::queue_ptr s,
                                   const int *orow = nullptr, int h_ld = 0) {
    const dpct::dim3 g((unsigned)((F + WARPS * ROWS - 1) / (WARPS * ROWS)),
                       (unsigned)S);
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        d_iq3xxs_grid.init(*s);
        d_iq4nl.init(*s);

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto d_iq3xxs_grid_ptr_ct1 = d_iq3xxs_grid.get_ptr();
            auto d_iq4nl_ptr_ct1 = d_iq4nl.get_ptr();

            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(xp_floats(C) * 4), cgh);

            auto mv_row_bytes_ty_gate_C_ct8 = mv_row_bytes(ty_gate, C);
            auto mv_row_bytes_ty_up_C_ct9 = mv_row_bytes(ty_up, C);
            auto h_ld_h_ld_F_ct16 = h_ld > 0 ? h_ld : F;

            cgh.parallel_for<dpct_kernel_name<
                class experts_gate_up_swiglu_kernel_d039ac,
                dpct_kernel_scalar<WARPS>, dpct_kernel_scalar<ROWS>>>(
                sycl::nd_range<3>(g * sycl::range(1, 1, WARPS * 32),
                                  sycl::range(1, 1, WARPS * 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        experts_gate_up_swiglu_kernel<WARPS, ROWS>(
                            blob, tok, ty_gate, ty_up, eb0, x, C, F,
                            mv_row_bytes_ty_gate_C_ct8,
                            mv_row_bytes_ty_up_C_ct9, sg, su, gcap, ucap, h,
                            orow, h_ld_h_ld_F_ct16,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            d_iq3xxs_grid_ptr_ct1, d_iq4nl_ptr_ct1);
                    });
        });
    }
}

void experts_gate_up_swiglu_rows(const uint8_t *const *blob, const int *tok,
                                 int S, int ty_gate, int ty_up, size_t eb0,
                                 const float *x, int C, int F, const float *sg,
                                 const float *su, float gcap, float ucap,
                                 uint16_t *h, const int *orow, int h_ld,
                                 dpct::queue_ptr s) {
    if (S <= 0) return;
    launch_experts_gate_up_swiglu<8, 1>(blob, tok, S, ty_gate, ty_up, eb0, x, C, F, sg, su, gcap, ucap, h, s, orow, h_ld);
    check("experts_gate_up_swiglu_rows");
}
void experts_gate_up_swiglu(const uint8_t *const *blob, const int *tok, int S,
                            int ty_gate, int ty_up, size_t eb0, const float *x,
                            int C, int F, const float *sg, const float *su,
                            float gcap, float ucap, uint16_t *h,
                            dpct::queue_ptr s) {

    if (S <= 0) return;
    launch_experts_gate_up_swiglu<8, 1>(blob, tok, S, ty_gate, ty_up, eb0, x, C, F, sg, su, gcap, ucap, h, s);
    check("experts_gate_up_swiglu");
}

void experts_gate_up_swiglu_g(const uint8_t *const *blob, const int *n,
                              const int *tok, const int *row, int G,
                              int ty_gate, int ty_up, size_t eb0,
                              const float *x, int C, int F, const float *sg,
                              const float *su, float gcap, float ucap,
                              uint16_t *h, dpct::queue_ptr s,
                              const uint16_t *input16) {
    if (G <= 0) return;
    static const bool use_group_mma = getenv("IK2_GROUP_MMA") && (atoi(getenv("IK2_GROUP_MMA")) & 1);
    if (grouped_tensor_products && use_group_mma && input16 && ty_gate == kNVFP4 && ty_up == kNVFP4 && F % 8 == 0 && C % 64 == 0) {
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->/*
  DPCT1050: The template argument of the dpct_kernel_name could not be
  deduced. You need to update this code.
  */
                parallel_for<dpct_kernel_name<
                    class grouped_expert_mma_a169d8, dpct_kernel_scalar<true>,
                    dpct_kernel_scalar<4>,
                    dpct_kernel_scalar<false>>>(
                    sycl::nd_range<3>(sycl::range(1, G, F / 8) *
                                          sycl::range(1, 1, 128),
                                      sycl::range(1, 1, 128)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            grouped_expert_mma<true, 4>(
                                blob, tok, eb0, C, F, nullptr, h, nullptr, sg,
                                su, gcap, ucap, n, row, input16);
                        });
        }
        check("grouped_expert_mma (grouped)");
        return;
    }

    if (grouped_tensor_products && use_group_mma && input16 && ty_gate == kQ8_0 && ty_up == kQ8_0 && F % 8 == 0 && C % 32 == 0) {
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->parallel_for<dpct_kernel_name<
                class grouped_expert_mma_760e49, dpct_kernel_scalar<true>,
                dpct_kernel_scalar<4>, dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, G, F / 8) *
                                      sycl::range(1, 1, 128),
                                  sycl::range(1, 1, 128)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        grouped_expert_mma<true, 4, true>(
                            blob, tok, eb0, C, F, nullptr, h, nullptr, sg, su,
                            gcap, ucap, n, row, input16);
                    });
        }
        check("q8 grouped MMA");
        return;
    }
    const dpct::dim3 g((unsigned)((F + kGroupWarps * kGroupSwigluRows - 1) /
                                  (kGroupWarps * kGroupSwigluRows)),
                       (unsigned)G);
    {
        d_iq3xxs_grid.init(*s);
        d_iq4nl.init(*s);

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto d_iq3xxs_grid_ptr_ct1 = d_iq3xxs_grid.get_ptr();
            auto d_iq4nl_ptr_ct1 = d_iq4nl.get_ptr();

            auto mv_row_bytes_ty_gate_C_ct10 = mv_row_bytes(ty_gate, C);
            auto mv_row_bytes_ty_up_C_ct11 = mv_row_bytes(ty_up, C);

            cgh.parallel_for<
                dpct_kernel_name<class experts_gate_up_swiglu_g_kernel_f234f7>>(
                sycl::nd_range<3>(g * sycl::range(1, 1, kGroupWarps * 32),
                                  sycl::range(1, 1, kGroupWarps * 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        experts_gate_up_swiglu_g_kernel(
                            blob, n, tok, row, ty_gate, ty_up, eb0, x, C, F,
                            mv_row_bytes_ty_gate_C_ct10,
                            mv_row_bytes_ty_up_C_ct11, sg, su, gcap, ucap, h,
                            d_iq3xxs_grid_ptr_ct1, d_iq4nl_ptr_ct1);
                    });
        });
    }
    check("experts_gate_up_swiglu_g");
}

void prep_qkv_r_f16(float *q, const float *q_clip, const float *k_in,
                    uint16_t *kc, const float *v_in, uint16_t *vc, float *r,
                    const float *r_clip, int T, int H, int KV, int D, int d_rel,
                    float qk_clip, float rel_clip, float eps, const int *pos,
                    float alpha, float n_floor, int pos0, int kv_cap,
                    dpct::queue_ptr s) {
    if (T <= 0) return;
    const int total = T * (2 * H + 2 * KV);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<
            dpct_kernel_name<class prep_qkv_r_impl_kernel_d4110c, uint16_t>>(
            sycl::nd_range<3>(sycl::range(1, 1, (unsigned)total) *
                                  sycl::range(1, 1, 32),
                              sycl::range(1, 1, 32)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                prep_qkv_r_impl_kernel<uint16_t>(
                    q, q_clip, k_in, kc, v_in, vc, r, r_clip, T, H, KV, D,
                    d_rel, qk_clip, rel_clip, eps, pos, alpha, n_floor, pos0,
                    kv_cap, 1.0f, 1.0f, nullptr, nullptr, nullptr, nullptr, 0,
                    nullptr, 0, 0);
            });
    }
    check("prep_qkv_r_f16");
}

void prep_qkv_r_fp8(float *q, const float *q_clip, const float *k_in,
                    uint8_t *kc, const float *v_in, uint8_t *vc, float *r,
                    const float *r_clip, int T, int H, int KV, int D, int d_rel,
                    float qk_clip, float rel_clip, float eps, const int *pos,
                    float alpha, float n_floor, int pos0, int kv_cap,
                    float k_scale, float v_scale, dpct::queue_ptr s) {
    if (T <= 0) return;
    const int total = T * (2 * H + 2 * KV);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<
            dpct_kernel_name<class prep_qkv_r_impl_kernel_8f435e, uint8_t>>(
            sycl::nd_range<3>(sycl::range(1, 1, (unsigned)total) *
                                  sycl::range(1, 1, 32),
                              sycl::range(1, 1, 32)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                prep_qkv_r_impl_kernel<uint8_t>(
                    q, q_clip, k_in, kc, v_in, vc, r, r_clip, T, H, KV, D,
                    d_rel, qk_clip, rel_clip, eps, pos, alpha, n_floor, pos0,
                    kv_cap, k_scale, v_scale, nullptr, nullptr, nullptr,
                    nullptr, 0, nullptr, 0, 0);
            });
    }
    check("prep_qkv_r_fp8");
}

template <class KV>
void attn_multi_impl(const float *q, const float *r, const float *sink,
                     const KV *kc, const KV *vc, int kv_cap, int nq, int H,
                     int Hkv, int k0, int nk, int qpos0, float qk_scale,
                     float v_inv_scale, const float *proj, int d_rel,
                     int extent, int window, float *scratch, float *out,
                     dpct::queue_ptr s) {
    if (d_rel > 32) {
        std::fprintf(stderr, "ik2::k::attn_multi: d_rel %d\n", d_rel);
        std::abort();
    }
    const int P = std::min(128, std::max(1, (nk + 63) / 64));
    float* part = scratch;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<dpct_kernel_name<class attn_split_kernel_9a8fe8, KV>>(
            sycl::nd_range<3>(sycl::range(nq, P, H) * sycl::range(1, 1, 128),
                              sycl::range(1, 1, 128)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_split_kernel<KV>(q, r, kc, vc, kv_cap, H, Hkv, k0, nk,
                                      qpos0, qk_scale, proj, d_rel, extent,
                                      window, part, nullptr);
            });
    }
    check("attn_multi_split");
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class attn_combine_kernel_119d80, KV>>(
            sycl::nd_range<3>(sycl::range(1, nq, H) * sycl::range(1, 1, 128),
                              sycl::range(1, 1, 128)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                attn_combine_kernel(sink, part, H, P, v_inv_scale, out);
            });
    }
    check("attn_multi_combine");
}

void attn_multi(const float *q, const float *r, const float *sink,
                const float *kc, const float *vc, int nq, int H, int Hkv,
                int k0, int nk, int qpos0, float qk_scale, const float *proj,
                int d_rel, int extent, int window, float *scratch, float *out,
                dpct::queue_ptr s) {
    attn_multi_impl(q, r, sink, kc, vc, 0, nq, H, Hkv, k0, nk, qpos0, qk_scale, 1.0f, proj, d_rel, extent, window,
                    scratch, out, s);
}

void attn_multi_f16(const float *q, const float *r, const float *sink,
                    const uint16_t *kc, const uint16_t *vc, int kv_cap, int nq,
                    int H, int Hkv, int k0, int nk, int qpos0, float qk_scale,
                    const float *proj, int d_rel, int extent, int window,
                    float *scratch, float *out, dpct::queue_ptr s) {
    attn_multi_impl(q, r, sink, kc, vc, kv_cap, nq, H, Hkv, k0, nk, qpos0, qk_scale, 1.0f, proj, d_rel, extent, window,
                    scratch, out, s);
}

void attn_multi_fp8(const float *q, const float *r, const float *sink,
                    const uint8_t *kc, const uint8_t *vc, int kv_cap, int nq,
                    int H, int Hkv, int k0, int nk, int qpos0, float qk_scale,
                    float v_inv_scale, const float *proj, int d_rel, int extent,
                    int window, float *scratch, float *out, dpct::queue_ptr s) {
    attn_multi_impl(q, r, sink, kc, vc, kv_cap, nq, H, Hkv, k0, nk, qpos0, qk_scale, v_inv_scale, proj, d_rel, extent,
                    window, scratch, out, s);
}

template <int WARPS, int ROWS>
void launch_mv_blocks(int type, const uint8_t *W, int N, int K, const float *x,
                      float *y, dpct::queue_ptr s) {
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    d_iq3xxs_grid.init(*s);
    d_iq4nl.init(*s);

    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};
    dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

    s->submit([&](sycl::handler &cgh) {
        auto d_iq3xxs_grid_ptr_ct1 = d_iq3xxs_grid.get_ptr();
        auto d_iq4nl_ptr_ct1 = d_iq4nl.get_ptr();

        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range(xp_floats(K) * 4), cgh);

        auto mv_row_bytes_type_K_ct4 = mv_row_bytes(type, K);

        cgh.parallel_for<dpct_kernel_name<class mv_blocks_kernel_8410ea,
                                          dpct_kernel_scalar<WARPS>,
                                          dpct_kernel_scalar<ROWS>>>(
            sycl::nd_range<3>(
                sycl::range(
                    1, 1, (unsigned)((N + WARPS * ROWS - 1) / (WARPS * ROWS))) *
                    sycl::range(1, 1, WARPS * 32),
                sycl::range(1, 1, WARPS * 32)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                mv_blocks_kernel<WARPS, ROWS>(
                    type, W, N, K, mv_row_bytes_type_K_ct4, x, y,
                    dpct_local_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    d_iq3xxs_grid_ptr_ct1, d_iq4nl_ptr_ct1);
            });
    });
}

void mv_blocks(int type, const uint8_t *W, int N, int K, const float *x,
               float *y, dpct::queue_ptr s) {
    if (N <= 0) return;
    if (K > 8192 || !block_values(type) || K % block_values(type)) {
        std::fprintf(stderr, "ik2::k::mv_blocks: type %d, K %d\n", type, K);
        std::abort();
    }
    if (N <= 512) launch_mv_blocks<2, 1>(type, W, N, K, x, y, s);
    else if (N <= 2048) launch_mv_blocks<4, 1>(type, W, N, K, x, y, s);
    else launch_mv_blocks<4, 2>(type, W, N, K, x, y, s);
    check("mv_blocks");
}

template <class KV>
void attn_decode_impl(const float *q, const float *r, const float *sink,
                      const KV *kc, const KV *vc, int kv_cap, int H, int Hkv,
                      int D, int k0, int nk, int qpos, float qk_scale,
                      float v_inv_scale, const float *proj, int d_rel,
                      int extent, int window, float *scratch, float *out,
                      dpct::queue_ptr s) {
    if (D != 128 || d_rel > 32) {
        std::fprintf(stderr, "ik2::k::attn_decode: D %d, d_rel %d\n", D, d_rel);
        std::abort();
    }
    // Flash-Decode splits over chunks of ~64 keys, up to 128 blocks per head (partials written directly to scratch)
    const int P = std::min(128, std::max(1, (nk + 63) / 64));
    if (P == 1) {
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

            s->parallel_for<
                dpct_kernel_name<class attn_decode_kernel_40c9c7, KV>>(
                sycl::nd_range<3>(sycl::range(1, 1, H) * sycl::range(1, 1, 128),
                                  sycl::range(1, 1, 128)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        attn_decode_kernel<KV>(q, r, sink, kc, vc, kv_cap, H,
                                               Hkv, k0, nk, qpos, qk_scale,
                                               v_inv_scale, proj, d_rel, extent,
                                               window, out);
                    });
        }
        check("attn_decode");
        return;
    }
    float* part = scratch;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<dpct_kernel_name<class attn_split_kernel_d90f02, KV>>(
            sycl::nd_range<3>(sycl::range(1, P, H) * sycl::range(1, 1, 128),
                              sycl::range(1, 1, 128)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_split_kernel<KV>(q, r, kc, vc, kv_cap, H, Hkv, k0, nk,
                                      qpos, qk_scale, proj, d_rel, extent,
                                      window, part, nullptr);
            });
    }
    check("attn_decode_split");
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class attn_combine_kernel_281dce, KV>>(
            sycl::nd_range<3>(sycl::range(1, 1, H) * sycl::range(1, 1, 128),
                              sycl::range(1, 1, 128)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                attn_combine_kernel(sink, part, H, P, v_inv_scale, out);
            });
    }
    check("attn_decode_combine");
}

void attn_decode(const float *q, const float *r, const float *sink,
                 const float *kc, const float *vc, int H, int Hkv, int D,
                 int k0, int nk, int qpos, float qk_scale, const float *proj,
                 int d_rel, int extent, int window, float *scratch, float *out,
                 dpct::queue_ptr s) {
    attn_decode_impl(q, r, sink, kc, vc, 0, H, Hkv, D, k0, nk, qpos, qk_scale, 1.0f, proj, d_rel, extent, window,
                     scratch, out, s);
}

void attn_decode_f16(const float *q, const float *r, const float *sink,
                     const uint16_t *kc, const uint16_t *vc, int kv_cap, int H,
                     int Hkv, int D, int k0, int nk, int qpos, float qk_scale,
                     const float *proj, int d_rel, int extent, int window,
                     float *scratch, float *out, dpct::queue_ptr s) {
    attn_decode_impl(q, r, sink, kc, vc, kv_cap, H, Hkv, D, k0, nk, qpos, qk_scale, 1.0f, proj, d_rel, extent, window,
                     scratch, out, s);
}

void attn_decode_fp8(const float *q, const float *r, const float *sink,
                     const uint8_t *kc, const uint8_t *vc, int kv_cap, int H,
                     int Hkv, int D, int k0, int nk, int qpos, float qk_scale,
                     float v_inv_scale, const float *proj, int d_rel,
                     int extent, int window, float *scratch, float *out,
                     dpct::queue_ptr s) {
    attn_decode_impl(q, r, sink, kc, vc, kv_cap, H, Hkv, D, k0, nk, qpos, qk_scale, v_inv_scale, proj, d_rel, extent,
                     window, scratch, out, s);
}

void q8_tile(const int8_t *q, const uint16_t *d, int N, int K,
             const uint16_t *x, int T, float *y, int ldy, dpct::queue_ptr s) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<
            dpct_kernel_name<class q8_mma_tile_82711a, dpct_kernel_scalar<4>>>(
            sycl::nd_range<3>(sycl::range(1, (T + 7) / 8, (N + 15) / 16) *
                                  sycl::range(1, 1, 128),
                              sycl::range(1, 1, 128)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                q8_mma_tile<4>(q, d, N, K, x, T, y, ldy);
            });
    }
    check("q8_mma_tile");
}

void mv_q8p_conv(const int8_t *q, const uint16_t *d, int N, int K,
                 const float *x, float *y, const MvConv &c0, const MvConv &c1,
                 dpct::queue_ptr s, const MvAlt &a0, const MvAlt &a1) {
    if (N <= 0) return;
    if (K % 128 || ((uintptr_t) q & 3) || (((uintptr_t) x | (uintptr_t) a0.x | (uintptr_t) a1.x) & 15)) {
        std::fprintf(stderr, "ik2::k::mv_q8p: K %d, or q / x not aligned\n", K);
        std::abort();
    }
    static const bool specialize = getenv("IK2_Q8_UNROLL") && atoi(getenv("IK2_Q8_UNROLL"));
    if (specialize && K == 2304)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<
            dpct_kernel_name<class mv_q8p_kernel_21ef57, dpct_kernel_scalar<4>,
                             dpct_kernel_scalar<2304>>>(
            sycl::nd_range<3>(sycl::range(1, 1, (unsigned)((N + 3) / 4)) *
                                  sycl::range(1, 1, 128),
                              sycl::range(1, 1, 128)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                mv_q8p_kernel<4, 2304>(q, d, N, K, x, y, c0, c1, a0, a1);
            });
    } else if (specialize && K == 3072)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<
            dpct_kernel_name<class mv_q8p_kernel_ba6f04, dpct_kernel_scalar<4>,
                             dpct_kernel_scalar<3072>>>(
            sycl::nd_range<3>(sycl::range(1, 1, (unsigned)((N + 3) / 4)) *
                                  sycl::range(1, 1, 128),
                              sycl::range(1, 1, 128)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                mv_q8p_kernel<4, 3072>(q, d, N, K, x, y, c0, c1, a0, a1);
            });
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->/*
DPCT1050: The template argument of the dpct_kernel_name could not be
deduced. You need to update this code.
*/
            parallel_for<dpct_kernel_name<
                class mv_q8p_kernel_40eb12, dpct_kernel_scalar<4>,
                dpct_kernel_scalar<0>>>(
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned)((N + 3) / 4)) *
                                      sycl::range(1, 1, 128),
                                  sycl::range(1, 1, 128)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mv_q8p_kernel<4>(q, d, N, K, x, y, c0, c1, a0, a1);
                    });
    }
    check("mv_q8p");
}
void mv_q8p(const int8_t *q, const uint16_t *d, int N, int K, const float *x,
            float *y, dpct::queue_ptr s) {
    mv_q8p_conv(q, d, N, K, x, y, MvConv{}, MvConv{}, s);
}
void decode_q8p_f16(const int8_t *q, const uint16_t *d, size_t n, uint16_t *out,
                    dpct::queue_ptr s) {
    if (!n) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class decode_q8p_f16_kernel_d16ff3>>(
            sycl::nd_range<3>(sycl::range(1, 1, grid(n)) *
                                  sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                decode_q8p_f16_kernel(q, d, n, out);
            });
    }
    check("decode_q8p_f16");
}

void mv_f16(const uint16_t *W, int N, int K, const float *x, float *y,
            dpct::queue_ptr s) {
    if (N <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class mv_f16_kernel_fd20ea,
                                         dpct_kernel_scalar<4>>>(
            sycl::nd_range<3>(sycl::range(1, 1, (unsigned)((N + 3) / 4)) *
                                  sycl::range(1, 1, 128),
                              sycl::range(1, 1, 128)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                mv_f16_kernel<4>(W, N, K, x, y);
            });
    }
    check("mv_f16");
}

void route_gather(const int *top, const float *w, const uint8_t *const *tab,
                  const float *sg_t, const float *su_t, const float *sd_t,
                  int S, int K, const uint8_t **blob, int *tok, int *inv,
                  float *sg, float *su, float *wt, dpct::queue_ptr s) {
    if (S <= 0) return;
    if (S > 1024) {
        std::fprintf(stderr, "ik2::k::route_gather: %d slots\n", S);
        std::abort();
    }
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class route_gather_kernel_b75a5b>>(
            sycl::nd_range<3>(
                sycl::range(1, 1, (unsigned)((S + 31) / 32 * 32)),
                sycl::range(1, 1, (unsigned)((S + 31) / 32 * 32))),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                route_gather_kernel(top, w, tab, sg_t, su_t, sd_t, S, K, blob,
                                    tok, inv, sg, su, wt);
            });
    }
    check("route_gather");
}
void route_token(const float *logits, const float *bias, int E, int K,
                 float scale, const uint8_t *const *tab, const float *sg_t,
                 const float *su_t, const float *sd_t, int *top, float *w,
                 const uint8_t **blob, int *tok, int *inv, float *sg, float *su,
                 float *wt, dpct::queue_ptr s, const float *x, int width,
                 void *packed, bool q8, uint16_t *x_limbs) {
    if (E > 1024 || K > E) {   // (more experts than a block's threads: the two kernels)
        if (packed) {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->parallel_for<dpct_kernel_name<class nv_input_dbd2bd,
                                             dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, 1, (width / 16 + 31) / 32) *
                                      sycl::range(1, 1, 32),
                                  sycl::range(1, 1, 32)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    nv_input<true>(x, nullptr, width, (NvInput *)packed, q8);
                });
        }
        if (x_limbs) to_f16_limbs(x, (size_t) width, x_limbs, s);
        route(logits, bias, 1, E, K, scale, top, w, s);
        route_gather(top, w, tab, sg_t, su_t, sd_t, K, K, blob, tok, inv, sg, su, wt, s);
        return;
    }
    const unsigned nt = (unsigned) ((E + 31) / 32 * 32);
    const unsigned blocks = 1 + (x_limbs ? (width + nt - 1) / nt : packed ? (width / 16 + nt - 1) / nt : 0);
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            /*
            DPCT1083: The size of local memory in the migrated code may be
            different from the original code. Check that the allocated memory
            size in the migrated code is correct.
            */
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range((size_t)(E + nt / 32 + K + 1) * sizeof(float)),
                cgh);

            cgh.parallel_for<dpct_kernel_name<class route_token_kernel_804884>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                      sycl::range(1, 1, nt),
                                  sycl::range(1, 1, nt)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        route_token_kernel(
                            logits, bias, E, K, scale, tab, sg_t, su_t, sd_t,
                            top, w, blob, tok, inv, sg, su, wt, x, width,
                            (NvInput *)packed, q8, x_limbs,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    }
    check("route_token");
}

void prep_qkv_r_f16_at(float *q, const float *q_clip, const float *k_in,
                       uint16_t *kc, const float *v_in, uint16_t *vc, float *r,
                       const float *r_clip, int H, int KV, int D, int d_rel,
                       float qk_clip, float rel_clip, float eps, const int *pos,
                       bool log_scale, float alpha, float n_floor, int kv_cap,
                       uint16_t *q16, dpct::queue_ptr s, const uint16_t *sink_w,
                       const float *sink_x, int sink_k, float *sink_y) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<
            dpct_kernel_name<class prep_qkv_r_impl_kernel_6b9245, uint16_t>>(
            sycl::nd_range<3>(
                sycl::range(1, 1,
                            (unsigned)(2 * H + 2 * KV + (sink_w ? H : 0))) *
                    sycl::range(1, 1, 32),
                sycl::range(1, 1, 32)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                prep_qkv_r_impl_kernel<uint16_t>(
                    q, q_clip, k_in, kc, v_in, vc, r, r_clip, 1, H, KV, D,
                    d_rel, qk_clip, rel_clip, eps, log_scale ? pos : nullptr,
                    alpha, n_floor, 0, kv_cap, 1.0f, 1.0f, pos, q16, sink_w,
                    sink_x, sink_k, sink_y, 0, 0);
            });
    }
    check("prep_qkv_r_f16_at");
}
void prep_qkv_r_fp8_at(float *q, const float *q_clip, const float *k_in,
                       uint8_t *kc, const float *v_in, uint8_t *vc, float *r,
                       const float *r_clip, int H, int KV, int D, int d_rel,
                       float qk_clip, float rel_clip, float eps, const int *pos,
                       bool log_scale, float alpha, float n_floor, int kv_cap,
                       float k_scale, float v_scale, uint16_t *q16,
                       dpct::queue_ptr s, const uint16_t *sink_w,
                       const float *sink_x, int sink_k, float *sink_y) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<
            dpct_kernel_name<class prep_qkv_r_impl_kernel_70a297, uint8_t>>(
            sycl::nd_range<3>(
                sycl::range(1, 1,
                            (unsigned)(2 * H + 2 * KV + (sink_w ? H : 0))) *
                    sycl::range(1, 1, 32),
                sycl::range(1, 1, 32)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                prep_qkv_r_impl_kernel<uint8_t>(
                    q, q_clip, k_in, kc, v_in, vc, r, r_clip, 1, H, KV, D,
                    d_rel, qk_clip, rel_clip, eps, log_scale ? pos : nullptr,
                    alpha, n_floor, 0, kv_cap, k_scale, v_scale, pos, q16,
                    sink_w, sink_x, sink_k, sink_y, 0, 0);
            });
    }
    check("prep_qkv_r_fp8_at");
}

void attn_decode_f16_at(const float *q, const float *r, const float *sink,
                        const uint16_t *kc, const uint16_t *vc, int kv_cap,
                        int H, int Hkv, const int *pos, int P, float qk_scale,
                        const float *proj, int d_rel, int extent, int window,
                        float *scratch, float *out, dpct::queue_ptr s) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<
            dpct_kernel_name<class attn_split_kernel_2732de, uint16_t>>(
            sycl::nd_range<3>(sycl::range(1, P, H) * sycl::range(1, 1, 128),
                              sycl::range(1, 1, 128)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_split_kernel<uint16_t>(q, r, kc, vc, kv_cap, H, Hkv, 0, 0,
                                            0, qk_scale, proj, d_rel, extent,
                                            window, scratch, pos);
            });
    }
    check("attn_decode_f16_at (split)");
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class attn_combine_kernel_1fa4ff>>(
            sycl::nd_range<3>(sycl::range(1, 1, H) * sycl::range(1, 1, 128),
                              sycl::range(1, 1, 128)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                attn_combine_kernel(sink, scratch, H, P, 1.0f, out);
            });
    }
    check("attn_decode_f16_at (combine)");
}
void attn_decode_fp8_at(const float *q, const float *r, const float *sink,
                        const uint8_t *kc, const uint8_t *vc, int kv_cap, int H,
                        int Hkv, const int *pos, int P, float qk_scale,
                        float v_inv_scale, const float *proj, int d_rel,
                        int extent, int window, float *scratch, float *out,
                        dpct::queue_ptr s) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<
            dpct_kernel_name<class attn_split_kernel_ba8dcc, uint8_t>>(
            sycl::nd_range<3>(sycl::range(1, P, H) * sycl::range(1, 1, 128),
                              sycl::range(1, 1, 128)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_split_kernel<uint8_t>(q, r, kc, vc, kv_cap, H, Hkv, 0, 0,
                                           0, qk_scale, proj, d_rel, extent,
                                           window, scratch, pos);
            });
    }
    check("attn_decode_fp8_at (split)");
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class attn_combine_kernel_940643>>(
            sycl::nd_range<3>(sycl::range(1, 1, H) * sycl::range(1, 1, 128),
                              sycl::range(1, 1, 128)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                attn_combine_kernel(sink, scratch, H, P, v_inv_scale, out);
            });
    }
    check("attn_decode_fp8_at (combine)");
}

int attn_decode_mma_blocks(int Hkv) {
    int dev = 0, n_sm = 0, occ16 = 0, occ8 = 0;
    dev = dpct::get_current_device_id();
    n_sm = dpct::get_device(dev).get_max_compute_units();
    /*
    DPCT1111: Please verify the input arguments of
    "dpct::experimental::calculate_max_active_wg_per_xecore" base on the target
    function "attn_decode_mma_kernel<uint16_t>".
    */
    dpct::experimental::calculate_max_active_wg_per_xecore(
        &occ16, kWaNW * 32,
        0 /* SYCL: shared-memory size not queried: an upper bound */);
    /*
    DPCT1111: Please verify the input arguments of
    "dpct::experimental::calculate_max_active_wg_per_xecore" base on the target
    function "attn_decode_mma_kernel<uint8_t>".
    */
    dpct::experimental::calculate_max_active_wg_per_xecore(
        &occ8, kWaNW * 32,
        0 /* SYCL: shared-memory size not queried: an upper bound */);
    check("attn_decode_mma (occupancy)");
    return std::max(1, n_sm * std::min(occ16, occ8) / std::max(1, Hkv));
}
size_t attn_decode_mma_scratch(int H, int n_split) { return (size_t) n_split * kWaNW * H * (kFaD + 2); }
template <class KV>
void attn_decode_mma_impl(const uint16_t *q16, const float *r,
                          const float *sink, const KV *kc, const KV *vc,
                          int kv_cap, int H, int Hkv, const int *pos, int qpos,
                          int n_split, float qk_scale, float v_inv,
                          const float *proj, int d_rel, int extent, int window,
                          float *scratch, float *out, const float *gate,
                          uint16_t *out16, dpct::queue_ptr s) {
    if (H % Hkv || H / Hkv > 8 || n_split < 1 || (kv_cap > 0 && kv_cap < 16)) {
        std::fprintf(stderr, "ik2::k::attn_decode_mma: %d heads on %d, %d splits, a cache of %d\n", H, Hkv, n_split,
                     kv_cap);
        std::abort();
    }
    float* o_part = scratch;
    sycl::float2 *ml_part =
        (sycl::float2 *)(o_part + (size_t)n_split * kWaNW * H * kFaD);
    const bool rel4 = d_rel % 4 == 0 && (((uintptr_t) r | (uintptr_t) proj) & 15) == 0;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<
            dpct_kernel_name<class attn_decode_mma_kernel_c11227, KV>>(
            sycl::nd_range<3>(sycl::range(1, (unsigned)n_split, (unsigned)Hkv) *
                                  sycl::range(1, 1, kWaNW * 32),
                              sycl::range(1, 1, kWaNW * 32)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_decode_mma_kernel<KV>(
                    q16, kc, vc, kv_cap, H, Hkv, pos, qpos, qk_scale, r, proj,
                    d_rel, rel4, extent, window, n_split, o_part, ml_part);
            });
    }
    check("attn_decode_mma");
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            /*
            DPCT1083: The size of local memory in the migrated code may be
            different from the original code. Check that the allocated memory
            size in the migrated code is correct.
            */
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range((size_t)(kFaD + n_split * kWaNW) * sizeof(float)),
                cgh);

            cgh.parallel_for<
                dpct_kernel_name<class attn_decode_mma_combine_kernel_38a455, KV>>(
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned)H) *
                                      sycl::range(1, 1, kFaD),
                                  sycl::range(1, 1, kFaD)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    attn_decode_mma_combine_kernel(
                        o_part, ml_part, n_split, H, sink, v_inv, pos, qpos,
                        window, gate, out, out16,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    }
    check("attn_decode_mma (combine)");
}
void attn_decode_mma_f16(const uint16_t *q16, const float *r, const float *sink,
                         const uint16_t *kc, const uint16_t *vc, int kv_cap,
                         int H, int Hkv, const int *pos, int qpos, int n_split,
                         float qk_scale, const float *proj, int d_rel,
                         int extent, int window, float *scratch, float *out,
                         const float *gate, uint16_t *out16,
                         dpct::queue_ptr s) {
    attn_decode_mma_impl(q16, r, sink, kc, vc, kv_cap, H, Hkv, pos, qpos, n_split, qk_scale, 1.0f, proj, d_rel, extent,
                         window, scratch, out, gate, out16, s);
}
void attn_decode_mma_fp8(const uint16_t *q16, const float *r, const float *sink,
                         const uint8_t *kc, const uint8_t *vc, int kv_cap,
                         int H, int Hkv, const int *pos, int qpos, int n_split,
                         float qk_scale, float v_inv_scale, const float *proj,
                         int d_rel, int extent, int window, float *scratch,
                         float *out, const float *gate, uint16_t *out16,
                         dpct::queue_ptr s) {
    attn_decode_mma_impl(q16, r, sink, kc, vc, kv_cap, H, Hkv, pos, qpos, n_split, qk_scale, v_inv_scale, proj, d_rel,
                         extent, window, scratch, out, gate, out16, s);
}

void unpack_kv_fp8_to_f16(const uint8_t *kc, const uint8_t *vc, int kv_cap,
                          int k0, int nk, int KD, float k_inv_scale,
                          float v_inv_scale, uint16_t *k_out16,
                          uint16_t *v_out16, dpct::queue_ptr s) {
    if (nk <= 0 || KD <= 0) return;
    if ((KD & 3) == 0 && (!k_out16 || ((((uintptr_t) kc & 3) == 0) && (((uintptr_t) k_out16 & 7) == 0))) &&
        (!v_out16 || ((((uintptr_t) vc & 3) == 0) && (((uintptr_t) v_out16 & 7) == 0)))) {
        const int kd4 = KD >> 2;
        const int bx = std::min(256, kd4);
        const dpct::dim3 g((unsigned)((kd4 + bx - 1) / bx),
                           (unsigned)std::min(nk, 65535));
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

            s->parallel_for<dpct_kernel_name<
                class unpack_kv_fp8_to_f16_vec4_kernel_d5f3c8>>(
                sycl::nd_range<3>(g * sycl::range(1, 1, (unsigned)bx),
                                  sycl::range(1, 1, (unsigned)bx)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    unpack_kv_fp8_to_f16_vec4_kernel(kc, vc, kv_cap, k0, nk, KD,
                                                     k_inv_scale, v_inv_scale,
                                                     k_out16, v_out16);
                });
        }
        check("unpack_kv_fp8_to_f16");
        return;
    }
    const size_t total = (size_t) nk * KD;
    const unsigned blocks = (unsigned) std::min<size_t>((total + kBlk - 1) / kBlk, 65535u);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<
            dpct_kernel_name<class unpack_kv_fp8_to_f16_kernel_91808f>>(
            sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                  sycl::range(1, 1, kBlk),
                              sycl::range(1, 1, kBlk)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                unpack_kv_fp8_to_f16_kernel(kc, vc, kv_cap, k0, nk, KD,
                                            k_inv_scale, v_inv_scale, k_out16,
                                            v_out16);
            });
    }
    check("unpack_kv_fp8_to_f16");
}

#include "ik2_decode_lut.dp.inc"
void decode_f16(int type, const uint8_t *blocks, size_t n, uint16_t *out,
                dpct::queue_ptr s) {
    if (!n) return;
    if (type == kNVFP4 && g_decode_lut && n % 16 == 0 && (((uintptr_t) blocks & 3) == 0) && (((uintptr_t) out & 15) == 0)) {
        decode_f16_nvfp4_half(blocks, n, out, s);
        return;
    }
    if (type == kNVFP4 && n % 16 == 0 && (((uintptr_t) blocks & 3) == 0) && (((uintptr_t) out & 15) == 0)) {
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

            s->submit([&](sycl::handler &cgh) {
                auto n_ct1 = n / 16;

                cgh.parallel_for<
                    dpct_kernel_name<class decode_f16_nvfp4_kernel_4bf5ec>>(
                    sycl::nd_range<3>(sycl::range(1, 1, grid(n / 16)) *
                                          sycl::range(1, 1, kBlk),
                                      sycl::range(1, 1, kBlk)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        decode_f16_nvfp4_kernel(blocks, n_ct1, out);
                    });
            });
        }
        check("decode_f16_nvfp4");
        return;
    }
    if (type == kQ8_0 && n % 32 == 0) {
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

            s->submit([&](sycl::handler &cgh) {
                auto n_ct1 = n / 32;

                cgh.parallel_for<
                    dpct_kernel_name<class decode_f16_q8_kernel_d63b96>>(
                    sycl::nd_range<3>(
                        sycl::range(1, 1, grid(n / 32, kBlk / 8)) *
                            sycl::range(1, 1, kBlk),
                        sycl::range(1, 1, kBlk)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        decode_f16_q8_kernel(blocks, n_ct1, out);
                    });
            });
        }
        check("decode_f16_q8");
        return;
    }
    if ((type == kIQ3_XXS || type == kIQ4_XS) && n % 256 == 0 && (((uintptr_t) out & 3) == 0) &&
        (((uintptr_t) blocks & (type == kIQ4_XS ? 3 : 1)) == 0)) {
        {
            d_iq3xxs_grid.init(*s);
            d_iq4nl.init(*s);

            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->submit([&](sycl::handler &cgh) {
                auto d_iq3xxs_grid_ptr_ct1 = d_iq3xxs_grid.get_ptr();
                auto d_iq4nl_ptr_ct1 = d_iq4nl.get_ptr();

                auto n_ct2 = n / 32;

                cgh.parallel_for<
                    dpct_kernel_name<class decode_f16_iq_kernel_e0c0cd>>(
                    sycl::nd_range<3>(sycl::range(1, 1, grid(n / 32)) *
                                          sycl::range(1, 1, kBlk),
                                      sycl::range(1, 1, kBlk)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        decode_f16_iq_kernel(type, blocks, n_ct2, out,
                                             d_iq3xxs_grid_ptr_ct1,
                                             d_iq4nl_ptr_ct1);
                    });
            });
        }
        check("decode_f16_iq");
        return;
    }
    if ((type == kQ5_K || type == kQ6_K) && n % 256 == 0) {
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

            s->submit([&](sycl::handler &cgh) {
                auto n_ct2 = n / 256;

                cgh.parallel_for<
                    dpct_kernel_name<class decode_f16_k_kernel_658d4a>>(
                    sycl::nd_range<3>(
                        sycl::range(1, 1, grid(n / 256, kBlk / 32)) *
                            sycl::range(1, 1, kBlk),
                        sycl::range(1, 1, kBlk)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        decode_f16_k_kernel(type, blocks, n_ct2, out);
                    });
            });
        }
        check("decode_f16_k");
        return;
    }
    {
        d_iq3xxs_grid.init(*s);
        d_iq4nl.init(*s);

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            auto d_iq3xxs_grid_ptr_ct1 = d_iq3xxs_grid.get_ptr();
            auto d_iq4nl_ptr_ct1 = d_iq4nl.get_ptr();

            cgh.parallel_for<dpct_kernel_name<class decode_f16_kernel_e7a0f2>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid(n)) *
                                      sycl::range(1, 1, kBlk),
                                  sycl::range(1, 1, kBlk)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    decode_f16_kernel(type, blocks, n, out,
                                      d_iq3xxs_grid_ptr_ct1, d_iq4nl_ptr_ct1);
                });
        });
    }
    check("decode_f16");
}
void decode_f32(int type, const uint8_t *blocks, size_t n, float *out,
                dpct::queue_ptr s) {
    if (!n) return;
    {
        d_iq3xxs_grid.init(*s);
        d_iq4nl.init(*s);

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            auto d_iq3xxs_grid_ptr_ct1 = d_iq3xxs_grid.get_ptr();
            auto d_iq4nl_ptr_ct1 = d_iq4nl.get_ptr();

            cgh.parallel_for<dpct_kernel_name<class decode_f32_kernel_c5538a>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid(n)) *
                                      sycl::range(1, 1, kBlk),
                                  sycl::range(1, 1, kBlk)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    decode_f32_kernel(type, blocks, n, out,
                                      d_iq3xxs_grid_ptr_ct1, d_iq4nl_ptr_ct1);
                });
        });
    }
    check("decode_f32");
}
void decode_host(int type, const uint8_t* blocks, size_t n, float* out) {
    for (size_t i = 0; i < n; ++i) out[i] = decode_value(type, blocks, i, h_iq3xxs_grid, h_iq4nl);
}


// ---- the block path's launchers

template <class KV, int NT>
static void attn_block_launch(const uint16_t *q16, const float *r, int r_ld,
                              const float *sink, const KV *kc, const KV *vc,
                              int kv_cap, int H, int Hkv, const int *pos, int T,
                              int n_split, float qk_scale, float v_inv,
                              const float *proj, int d_rel, int extent,
                              int window, float *scratch, float *out,
                              const float *gate, int gate_ld, uint16_t *out16,
                              dpct::queue_ptr s) {
    float* o_part = scratch;
    sycl::float2 *ml_part =
        (sycl::float2 *)(o_part + (size_t)n_split * kWaNW * T * H * kFaD);
    const bool rel4 = d_rel % 4 == 0 && r_ld % 4 == 0 && (((uintptr_t) r | (uintptr_t) proj) & 15) == 0;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<dpct_kernel_name<class attn_block_mma_kernel_89084d, KV,
                                         dpct_kernel_scalar<NT>>>(
            sycl::nd_range<3>(sycl::range(1, (unsigned)n_split, (unsigned)Hkv) *
                                  sycl::range(1, 1, kWaNW * 32),
                              sycl::range(1, 1, kWaNW * 32)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_block_mma_kernel<KV, NT>(q16, kc, vc, kv_cap, H, Hkv, pos,
                                              T, qk_scale, r, r_ld, proj, d_rel,
                                              rel4, extent, window, n_split,
                                              o_part, ml_part);
            });
    }
    check("attn_block_mma");
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            /*
            DPCT1083: The size of local memory in the migrated code may be
            different from the original code. Check that the allocated memory
            size in the migrated code is correct.
            */
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range((size_t)(kFaD + n_split * kWaNW) * sizeof(float)),
                cgh);

            cgh.parallel_for<
                dpct_kernel_name<class attn_block_combine_kernel_8836b9, KV, dpct_kernel_scalar<NT>>>(
                sycl::nd_range<3>(sycl::range(1, (unsigned)T, (unsigned)H) *
                                      sycl::range(1, 1, kFaD),
                                  sycl::range(1, 1, kFaD)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    attn_block_combine_kernel(
                        o_part, ml_part, n_split, H, T, sink, v_inv, pos,
                        window, gate, gate_ld, out, out16,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    }
    check("attn_block_mma (combine)");
}
template <class KV>
static void attn_block_impl(const uint16_t *q16, const float *r, int r_ld,
                            const float *sink, const KV *kc, const KV *vc,
                            int kv_cap, int H, int Hkv, const int *pos, int T,
                            int n_split, float qk_scale, float v_inv,
                            const float *proj, int d_rel, int extent,
                            int window, float *scratch, float *out,
                            const float *gate, int gate_ld, uint16_t *out16,
                            dpct::queue_ptr s) {
    const int R = H / Hkv, NC = T * R;
    if (H % Hkv || T < 1 || NC > 24 || n_split < 1 || (kv_cap > 0 && kv_cap < 16 + T)) {
        std::fprintf(stderr, "ik2::k::attn_block: %d heads on %d, %d tokens, %d splits, a cache of %d\n", H, Hkv, T,
                     n_split, kv_cap);
        std::abort();
    }
    if (NC <= 8)
        attn_block_launch<KV, 1>(q16, r, r_ld, sink, kc, vc, kv_cap, H, Hkv, pos, T, n_split, qk_scale, v_inv, proj,
                                 d_rel, extent, window, scratch, out, gate, gate_ld, out16, s);
    else if (NC <= 16)
        attn_block_launch<KV, 2>(q16, r, r_ld, sink, kc, vc, kv_cap, H, Hkv, pos, T, n_split, qk_scale, v_inv, proj,
                                 d_rel, extent, window, scratch, out, gate, gate_ld, out16, s);
    else
        attn_block_launch<KV, 3>(q16, r, r_ld, sink, kc, vc, kv_cap, H, Hkv, pos, T, n_split, qk_scale, v_inv, proj,
                                 d_rel, extent, window, scratch, out, gate, gate_ld, out16, s);
}
void attn_block_f16(const uint16_t *q16, const float *r, int r_ld,
                    const float *sink, const uint16_t *kc, const uint16_t *vc,
                    int kv_cap, int H, int Hkv, const int *pos, int T,
                    int n_split, float qk_scale, const float *proj, int d_rel,
                    int extent, int window, float *scratch, float *out,
                    const float *gate, int gate_ld, uint16_t *out16,
                    dpct::queue_ptr s) {
    attn_block_impl<uint16_t>(q16, r, r_ld, sink, kc, vc, kv_cap, H, Hkv, pos, T, n_split, qk_scale, 1.0f, proj,
                              d_rel, extent, window, scratch, out, gate, gate_ld, out16, s);
}
void attn_block_fp8(const uint16_t *q16, const float *r, int r_ld,
                    const float *sink, const uint8_t *kc, const uint8_t *vc,
                    int kv_cap, int H, int Hkv, const int *pos, int T,
                    int n_split, float qk_scale, float v_inv_scale,
                    const float *proj, int d_rel, int extent, int window,
                    float *scratch, float *out, const float *gate, int gate_ld,
                    uint16_t *out16, dpct::queue_ptr s) {
    attn_block_impl<uint8_t>(q16, r, r_ld, sink, kc, vc, kv_cap, H, Hkv, pos, T, n_split, qk_scale, v_inv_scale,
                             proj, d_rel, extent, window, scratch, out, gate, gate_ld, out16, s);
}
size_t attn_block_scratch(int H, int T, int n_split) { return (size_t) n_split * kWaNW * T * H * (kFaD + 2); }

void conv_block(const float *x, int x_ld, float *hist, const float *w,
                const float *dyn, int d_ld, int heads, int T, int C,
                const float *res, int res_ld, float *y, int y_ld,
                float *rec_old, float *rec_x, dpct::queue_ptr s) {
    if (T <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            auto heads_heads_ct6 = heads > 0 ? heads : 1;

            cgh.parallel_for<dpct_kernel_name<class conv_block_kernel_18bc50>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((C + 127) / 128)) *
                        sycl::range(1, 1, 128),
                    sycl::range(1, 1, 128)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    conv_block_kernel(x, x_ld, hist, w, dyn, d_ld,
                                      heads_heads_ct6, T, C, res, res_ld, y,
                                      y_ld, rec_old, rec_x);
                });
        });
    }
    check("conv_block");
}
void rollback_block(const ConvRec *recs, int n_recs, int n, dpct::queue_ptr s) {
    if (n_recs <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class rollback_block_kernel_5f92cc>>(
            sycl::nd_range<3>(sycl::range(1, (unsigned)n_recs, 8) *
                                  sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                rollback_block_kernel(recs, n);
            });
    }
    check("rollback_block");
}
void mv_f16_rows(const uint16_t *W, int N, int K, const float *x, int x_ld,
                 int T, float *y, int y_ld, dpct::queue_ptr s) {
    if (N <= 0 || T <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class mv_f16_rows_kernel_76f883,
                                         dpct_kernel_scalar<4>>>(
            sycl::nd_range<3>(
                sycl::range(1, (unsigned)T, (unsigned)((N + 3) / 4)) *
                    sycl::range(1, 1, 128),
                sycl::range(1, 1, 128)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                mv_f16_rows_kernel<4>(W, N, K, x, x_ld, y, y_ld);
            });
    }
    check("mv_f16_rows");
}
void f16_limbs_rows(const float *x, int T, int C, uint16_t *out,
                    dpct::queue_ptr s) {
    if (T <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class f16_limbs_rows_kernel_2909f4>>(
            sycl::nd_range<3>(
                sycl::range(1, 1, (unsigned)((T * C + 255) / 256)) *
                    sycl::range(1, 1, 256),
                sycl::range(1, 1, 256)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                f16_limbs_rows_kernel(x, T, C, out);
            });
    }
    check("f16_limbs_rows");
}
void group_route(const int *top, const float *w, int S, int K,
                 const uint8_t *const *tab, const float *sg_t,
                 const float *su_t, const float *sd_t, const uint8_t **blob,
                 int *n, int *tok, int *row, float *sg, float *su, float *wt,
                 int *inv, dpct::queue_ptr s) {
    if (S <= 0 || S > 64) {
        std::fprintf(stderr, "ik2::k::group_route: %d slots (1 .. 64)\n", S);
        std::abort();
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class group_route_kernel_b0d0ee>>(
            sycl::nd_range<3>(sycl::range(1, 1, 64), sycl::range(1, 1, 64)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                group_route_kernel(top, w, S, K, tab, sg_t, su_t, sd_t, blob, n,
                                   tok, row, sg, su, wt, inv);
            });
    }
    check("group_route");
}
void experts_block_gu(const uint8_t *const *blob, const int *n, const int *tok,
                      const int *row, int G, int T, size_t off_up, int C, int F,
                      const uint16_t *limbs, const float *sg, const float *su,
                      float gcap, float ucap, uint16_t *h, dpct::queue_ptr s) {
    if (G <= 0) return;
    const dpct::dim3 g((unsigned)(F / 8), (unsigned)G);
    if (T <= 4) {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<
            dpct_kernel_name<class experts_block_gu_kernel_e0812c,
                             dpct_kernel_scalar<8>, dpct_kernel_scalar<1>>>(
            sycl::nd_range<3>(g * sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                experts_block_gu_kernel<8, 1>(blob, n, tok, row, off_up, C, F,
                                              limbs, sg, su, gcap, ucap, h);
            });
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<
            dpct_kernel_name<class experts_block_gu_kernel_60cb10,
                             dpct_kernel_scalar<8>, dpct_kernel_scalar<2>>>(
            sycl::nd_range<3>(g * sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                experts_block_gu_kernel<8, 2>(blob, n, tok, row, off_up, C, F,
                                              limbs, sg, su, gcap, ucap, h);
            });
    }
    check("experts_block_gu");
}
void experts_block_down(const uint8_t *const *blob, const int *n,
                        const int *row, int G, size_t off_down, int C, int F,
                        const uint16_t *h, float *y, dpct::queue_ptr s) {
    if (G <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<class experts_block_down_kernel_ddcd34,
                                         dpct_kernel_scalar<4>>>(
            sycl::nd_range<3>(sycl::range(1, (unsigned)G, (unsigned)(C / 16)) *
                                  sycl::range(1, 1, 128),
                              sycl::range(1, 1, 128)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                experts_block_down_kernel<4>(blob, n, row, off_down, C, F, h,
                                             y);
            });
    }
    check("experts_block_down");
}
void prep_qkv_r_fp8_block(
    float *q, int q_ld, const float *q_clip, const float *k_in, uint8_t *kc,
    const float *v_in, uint8_t *vc, float *r, int r_ld, const float *r_clip,
    int T, int H, int KV, int D, int d_rel, float qk_clip, float rel_clip,
    float eps, const int *pos, bool log_scale, float alpha, float n_floor,
    int kv_cap, float k_scale, float v_scale, uint16_t *q16, dpct::queue_ptr s,
    const uint16_t *sink_w, const float *sink_x, int sink_k, float *sink_y) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<
            dpct_kernel_name<class prep_qkv_r_impl_kernel_66f285, uint8_t>>(
            sycl::nd_range<3>(sycl::range(1, 1,
                                          (unsigned)(T * (2 * H + 2 * KV) +
                                                     (sink_w ? T * H : 0))) *
                                  sycl::range(1, 1, 32),
                              sycl::range(1, 1, 32)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                prep_qkv_r_impl_kernel<uint8_t>(
                    q, q_clip, k_in, kc, v_in, vc, r, r_clip, T, H, KV, D,
                    d_rel, qk_clip, rel_clip, eps, log_scale ? pos : nullptr,
                    alpha, n_floor, 0, kv_cap, k_scale, v_scale, pos, q16,
                    sink_w, sink_x, sink_k, sink_y, q_ld, r_ld);
            });
    }
    check("prep_qkv_r_fp8_block");
}
void prep_qkv_r_f16_block(float *q, int q_ld, const float *q_clip,
                          const float *k_in, uint16_t *kc, const float *v_in,
                          uint16_t *vc, float *r, int r_ld, const float *r_clip,
                          int T, int H, int KV, int D, int d_rel, float qk_clip,
                          float rel_clip, float eps, const int *pos,
                          bool log_scale, float alpha, float n_floor,
                          int kv_cap, uint16_t *q16, dpct::queue_ptr s,
                          const uint16_t *sink_w, const float *sink_x,
                          int sink_k, float *sink_y) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for<
            dpct_kernel_name<class prep_qkv_r_impl_kernel_161849, uint16_t>>(
            sycl::nd_range<3>(sycl::range(1, 1,
                                          (unsigned)(T * (2 * H + 2 * KV) +
                                                     (sink_w ? T * H : 0))) *
                                  sycl::range(1, 1, 32),
                              sycl::range(1, 1, 32)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                prep_qkv_r_impl_kernel<uint16_t>(
                    q, q_clip, k_in, kc, v_in, vc, r, r_clip, T, H, KV, D,
                    d_rel, qk_clip, rel_clip, eps, log_scale ? pos : nullptr,
                    alpha, n_floor, 0, kv_cap, 1.0f, 1.0f, pos, q16, sink_w,
                    sink_x, sink_k, sink_y, q_ld, r_ld);
            });
    }
    check("prep_qkv_r_f16_block");
}
void attn_res_jobs_block(const ResStates &states, int n, const float *q, int nq,
                         int jobs, int T, int C, float eps, float *out,
                         dpct::queue_ptr s) {
    launch_attn_res<false>(states, n, q, nq, T, C, eps, 0.0f, out, RmsOuts{}, s, jobs);
}

void route_rows(const float *logits, const float *bias, int T, int E, int K,
                float scale, int *top, float *w, dpct::queue_ptr s) {
    if (T <= 0) return;
    if (E > 1024 || K > E || K > kRouteMax) {
        route(logits, bias, T, E, K, scale, top, w, s);
        return;
    }
    const unsigned nt = (unsigned) ((E + 31) / 32 * 32);
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            /*
            DPCT1083: The size of local memory in the migrated code may be
            different from the original code. Check that the allocated memory
            size in the migrated code is correct.
            */
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range((size_t)(E + nt / 32 + K + 1) * sizeof(float)),
                cgh);

            cgh.parallel_for<dpct_kernel_name<class route_rows_kernel_6d8f4c>>(
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned)T) *
                                      sycl::range(1, 1, nt),
                                  sycl::range(1, 1, nt)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        route_rows_kernel(
                            logits, bias, E, K, scale, top, w,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    }
    check("route_rows");
}
void experts_block_gu_q8(const uint8_t *const *blob, const int *n,
                         const int *tok, const int *row, int G, int T,
                         size_t off_up, int C, int F, const uint16_t *limbs,
                         const float *sg, const float *su, float gcap,
                         float ucap, uint16_t *h, dpct::queue_ptr s) {
    if (G <= 0) return;
    const dpct::dim3 g((unsigned)(F / 8), (unsigned)G);
    if (T <= 4) {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<
            dpct_kernel_name<class experts_block_gu_q8_kernel_ee9bf8,
                             dpct_kernel_scalar<8>, dpct_kernel_scalar<1>>>(
            sycl::nd_range<3>(g * sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                experts_block_gu_q8_kernel<8, 1>(blob, n, tok, row, off_up, C,
                                                 F, limbs, sg, su, gcap, ucap,
                                                 h);
            });
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<
            dpct_kernel_name<class experts_block_gu_q8_kernel_a65f91,
                             dpct_kernel_scalar<8>, dpct_kernel_scalar<2>>>(
            sycl::nd_range<3>(g * sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                experts_block_gu_q8_kernel<8, 2>(blob, n, tok, row, off_up, C,
                                                 F, limbs, sg, su, gcap, ucap,
                                                 h);
            });
    }
    check("experts_block_gu_q8");
}
void experts_block_down_q8(const uint8_t *const *blob, const int *n,
                           const int *row, int G, size_t off_down, int C, int F,
                           const uint16_t *h, float *y, dpct::queue_ptr s) {
    if (G <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<dpct_kernel_name<
            class experts_block_down_q8_kernel_30453e, dpct_kernel_scalar<4>>>(
            sycl::nd_range<3>(sycl::range(1, (unsigned)G, (unsigned)(C / 16)) *
                                  sycl::range(1, 1, 128),
                              sycl::range(1, 1, 128)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                experts_block_down_q8_kernel<4>(blob, n, row, off_down, C, F, h,
                                                y);
            });
    }
    check("experts_block_down_q8");
}

}  // namespace ik2::k
