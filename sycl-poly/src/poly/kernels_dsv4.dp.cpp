// src/poly/kernels_dsv4.cu - DeepSeek V4's kernels (kernels.hpp).  The reference for every formula is llama.cpp's
// deepseek4 graph (src/models/deepseek4.cpp) and its compressed caches (src/llama-kv-cache-dsv4.cpp).  Where a kernel
// takes several tokens, blockIdx.y is the token unless it says otherwise.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/poly/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::poly {
namespace {

constexpr int WARP = 32;
constexpr int SELECT_THREADS = 1024;
constexpr int ATTEND_THREADS = 256;
constexpr int SCORE_KEYS = 32;      // dsv4_idx_score: the keys a block takes
constexpr int IDX_DIM = 128;        // an indexer key's values
constexpr int GROUP_HEADS = 8;      // attend_rows_kernel: the heads of a token a block takes, a warp each
constexpr int GROUP_KEYS = WARP;    // the keys it holds in shared memory at a time, a lane each
constexpr int GROUP_ROWS = 32;      // dsv4_attend takes attend_rows_kernel from this many tokens a call
constexpr int GROUP_DIM = 512;      // the longest head it takes

dpct::queue_ptr cs(void *s) { return strata::q_of(s); }
int grid(int n, int b) { return (n + b - 1) / b; }

void ck(const char* what) {
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (e != 0) throw std::runtime_error(std::string(what) + ": " +
                                         dpct::get_error_string_dummy(e));
}

// Lets `kernel` take as much dynamic shared memory as the current card gives a block, once a card; the most it may
// then take, in bytes.
template <class K> size_t allow_smem(K kernel, unsigned &done) try {
    int dev = 0, most = 0;
    if (DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) != 0 ||
        /*
        DPCT1019: local_mem_size in SYCL is not a complete equivalent of
        cudaDevAttrMaxSharedMemoryPerBlockOptin in CUDA. You may need to adjust
        the code.
        */
        DPCT_CHECK_ERROR(most = dpct::get_device(dev).get_local_mem_size()) !=
            0 ||
        most <= 1024)
        throw std::runtime_error("dsv4: the card's shared memory a block is unknown");
    const size_t room = size_t(most) - 1024;   // the kernels' own static shared memory
    if (dev < 32 && !(done & (1u << dev))) {
        /*
        DPCT1027: The call to cudaFuncSetAttribute was replaced with 0
        because SYCL currently does not support corresponding setting.
        */
        if (0 != 0)
            throw std::runtime_error("dsv4: a kernel cannot be given the card's shared memory");
        done |= 1u << dev;
    }
    return room;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

__dpct_inline__ float warp_sum(float x) {
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = WARP / 2; o > 0; o >>= 1) x +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffff, sycl::ext::oneapi::this_work_item::get_sub_group(), x,
            o);
    return x;
}

__dpct_inline__ float warp_max(float x) {
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = WARP / 2; o > 0; o >>= 1) x = sycl::fmax(
        x, dpct::experimental::permute_sub_group_by_xor(
               0xffffffff, sycl::ext::oneapi::this_work_item::get_sub_group(),
               x, o));
    return x;
}

// One value per thread summed (or the largest of them) over the block; `part` has a float a warp.
__dpct_inline__ float block_sum(float x, float *part) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    x = warp_sum(x);
    if ((item_ct1.get_local_id(2) & 31) == 0)
        part[item_ct1.get_local_id(2) >> 5] = x;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < int(item_ct1.get_local_range(2)) / WARP; ++k) s +=
        part[k];
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    return s;
}

__dpct_inline__ float block_max(float x, float *part) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    x = warp_max(x);
    if ((item_ct1.get_local_id(2) & 31) == 0)
        part[item_ct1.get_local_id(2) >> 5] = x;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    float m = part[0];
#pragma unroll
    for (int k = 1; k < int(item_ct1.get_local_range(2)) / WARP; ++k)
        m = sycl::fmax(m, part[k]);
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    return m;
}

__dpct_inline__ float half_at(const uint16_t *p, size_t i) {
    return sycl::vec<sycl::half, 1>(
               sycl::bit_cast<sycl::half, unsigned short>(p[i]))
        .convert<float, sycl::rounding_mode::automatic>()[0];
}

// ggml-cuda's rope_yarn() and rope_yarn_ramp(), with the pair's index i0 counted from the first turned value.
__dpct_inline__ void yarn(float theta_extrap, const Rope &r, int i0, bool back,
                          float &c, float &s) {
    const float theta_interp = r.freq_scale * theta_extrap;
    float theta = theta_interp, mscale = r.attn_factor;
    if (r.ext_factor != 0.0f) {
        const float y =
            (i0 / 2 - r.corr0) / sycl::fmax(0.001f, r.corr1 - r.corr0);
        const float ramp_mix =
            (1.0f - sycl::fmin(1.0f, sycl::fmax(0.0f, y))) * r.ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * sycl::log(1.0f / r.freq_scale);
    }
    c = sycl::cos(theta) * mscale;
    s = sycl::sin(theta) * mscale;
    if (back) s = -s;
}

// Token t's position: what the card holds at `at` for it when the launch runs (a recorded launch), else
// pos + t * step.
__dpct_inline__ long long position(const long long *at, long long pos, int step,
                                   int t) {
    return at ? at[2 * t] : pos + (long long) t * step;
}

// A block a head of a token; a thread a pair of values (dim <= 2 * blockDim.x).
__dpct_inline__ void rope_kernel(const float *__restrict__ x, float *y, int dim,
                                 int off, int n_rot, Rope r, long long pos,
                                 int step, bool norm, float eps, bool back,
                                 const long long *at) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const size_t base =
        (size_t(item_ct1.get_group(1)) * item_ct1.get_group_range(2) +
         item_ct1.get_group(2)) *
        dim;
    const int i0 = 2 * item_ct1.get_local_id(2);
    float x0 = 0.0f, x1 = 0.0f;
    if (i0 < dim) {
        x0 = x[base + i0];
        x1 = x[base + i0 + 1];
    }
    float scale = 1.0f;
    if (norm) scale =
        1.0f /
        sycl::sqrt(block_sum(x0 * x0 + x1 * x1, part) / float(dim) + eps);
    if (i0 >= dim) return;
    x0 *= scale;
    x1 *= scale;
    if (i0 >= off && i0 < off + n_rot) {
        const int iw = i0 - off;
        const float p = float(position(at, pos, step, item_ct1.get_group(1)));
        float c, s;
        yarn(p * dpct::pow(r.theta_scale, iw / 2.0f), r, iw, back, c, s);
        const float a = x0, b = x1;
        x0 = a * c - b * s;
        x1 = a * s + b * c;
    }
    y[base + i0] = x0;
    y[base + i0 + 1] = x1;
}

// A thread a value of a token's row.
__dpct_inline__ void
state_kernel(const float *__restrict__ kv, const float *__restrict__ score,
             const float *__restrict__ ape, float *__restrict__ ring_kv,
             float *__restrict__ ring_score, int ring, int width, int ratio,
             long long pos, const long long *at) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= width) return;
    const long long p = position(at, pos, 1, item_ct1.get_group(1));
    const size_t src = size_t(item_ct1.get_group(1)) * width + i,
                 dst = size_t(p % ring) * width + i;
    ring_kv[dst] = kv[src];
    ring_score[dst] = score[src] + ape[size_t(p % ratio) * width + i];
}

// A block a token, a thread a value: nothing unless the token ends a block of ratio positions, else the block's
// compressed row.  Its values come in the order llama.cpp concatenates them, with overlap the rows before the block
// first; the weights are the softmax's, then the sum of weight times value in that order, as ggml's soft_max, mul
// and sum_rows give it.  Then the row is RMS-normed with its weights, turned by the block's first position on its
// last n_rot values, and stored as fp16 at the block's index.
/*
DPCT1110: The total declared local variable size in device function
squeeze_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void squeeze_kernel(const float *__restrict__ ring_kv,
                                    const float *__restrict__ ring_score,
                                    int ring, int dim, int ratio, bool overlap,
                                    const float *__restrict__ norm, float eps,
                                    Rope r, int n_rot, long long pos,
                                    const long long *at,
                                    uint16_t *__restrict__ cache) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &vals =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1024]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const long long p = position(at, pos, 1, item_ct1.get_group(2));
    if ((p + 1) % ratio != 0) return;
    const int d = item_ct1.get_local_id(2);
    const long long start = p + 1 - ratio;
    const int width = overlap ? 2 * dim : dim, n = overlap ? 2 * ratio : ratio;
    // the j-th value: a ring row and a column, or nothing for a row before the sequence
    auto src = [&](int j, size_t& idx) {
        long long row;
        int col = d;
        if (!overlap) {
            row = start + j;
        } else if (j < ratio) {
            row = start - ratio + j;
        } else {
            row = start + j - ratio;
            col = dim + d;
        }
        if (row < 0) return false;
        idx = size_t(row % ring) * width + col;
        return true;
    };
    float mx = -INFINITY;
    for (int j = 0; j < n; ++j) {
        size_t idx;
        if (src(j, idx)) mx = sycl::fmax(mx, ring_score[idx]);
    }
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) {
        size_t idx;
        if (src(j, idx)) sum += sycl::native::exp(ring_score[idx] - mx);
    }
    const float inv = 1.0f / sum;
    float acc = 0.0f;
    for (int j = 0; j < n; ++j) {
        size_t idx;
        if (src(j, idx)) acc +=
            sycl::native::exp(ring_score[idx] - mx) * inv * ring_kv[idx];
    }
    const float scale =
        1.0f / sycl::sqrt(block_sum(acc * acc, part) / float(dim) + eps);
    vals[d] = acc * scale * norm[d];
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int off = dim - n_rot;
    float v = vals[d];
    if (d >= off) {
        const int iw = (d - off) & ~1;
        float c, s;
        yarn(float(start) * dpct::pow(r.theta_scale, iw / 2.0f), r, iw, false,
             c, s);
        const float a = vals[off + iw], b = vals[off + iw + 1];
        v = (d - off) & 1 ? a * s + b * c : a * c - b * s;
    }
    cache[size_t(start / ratio) * dim + d] =
        sycl::bit_cast<unsigned short, sycl::half>(
            sycl::vec<float, 1>(v)
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0]);
}

// A block a token and SCORE_KEYS keys; a warp a key at a time, a lane two heads of it.  The queries are in shared
// memory a head a row, one value longer than a row so the lanes' rows lie in different banks.
__dpct_inline__ void
idx_score_kernel(const float *__restrict__ q, const float *__restrict__ w,
                 const uint16_t *__restrict__ keys, float *__restrict__ score,
                 int score_stride, int n_head, int ratio, long long pos,
                 float scale, const long long *at, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sq = (float *)dpct_local; // n_head * (IDX_DIM + 1) queries, n_head
                                   // weights, SCORE_KEYS * IDX_DIM keys
    float* sw = sq + n_head * (IDX_DIM + 1);
    float* sk = sw + n_head;
    const int t = item_ct1.get_group(1);
    const long long visible = (position(at, pos, 1, t) + 1) / ratio;
    const long long k0 = (long long)item_ct1.get_group(2) * SCORE_KEYS;
    if (k0 >= visible) return;
    const float* qt = q + size_t(t) * n_head * IDX_DIM;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < n_head * IDX_DIM;
         i += item_ct1.get_local_range(2))
        sq[(i / IDX_DIM) * (IDX_DIM + 1) + i % IDX_DIM] = qt[i];
#pragma unroll
    for (int h = item_ct1.get_local_id(2); h < n_head;
         h += item_ct1.get_local_range(2)) sw[h] = w[size_t(t) * n_head + h];
    const int nk = (int)sycl::min((long long)SCORE_KEYS, visible - k0);
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < nk * IDX_DIM;
         i += item_ct1.get_local_range(2))
        sk[i] = half_at(keys, size_t(k0) * IDX_DIM + i);
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const int lane = item_ct1.get_local_id(2) % WARP,
              wid = item_ct1.get_local_id(2) / WARP,
              warps = item_ct1.get_local_range(2) / WARP;
    for (int k = wid; k < nk; k += warps) {
        const float* key = sk + k * IDX_DIM;
        float acc = 0.0f;
        for (int h = lane; h < n_head; h += WARP) {
            const float* qh = sq + h * (IDX_DIM + 1);
            float dot = 0.0f;
#pragma unroll 8
            for (int i = 0; i < IDX_DIM; ++i) dot += qh[i] * key[i];
            acc += sycl::fmax(dot, 0.0f) * sw[h];
        }
        acc = warp_sum(acc);
        if (lane == 0) score[size_t(t) * score_stride + k0 + k] = acc * scale;
    }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void idx_score_kernel_wrapper(const float *__restrict q,
                              const float *__restrict w,
                              const uint16_t *__restrict keys,
                              float *__restrict score, int score_stride,
                              int n_head, int ratio, long long pos, float scale,
                              const long long *at) {
    sycl::queue queue = *dpct::kernel_launcher::_que;
    unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range(localMemSize), cgh);

        cgh.parallel_for<dpct_kernel_name<class idx_score_kernel_>>(
            nr, exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                idx_score_kernel(
                    q, w, keys, score, score_stride, n_head, ratio, pos, scale,
                    at,
                    dpct_local_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
}

__dpct_inline__ uint32_t order_key(float s) {
    const uint32_t b = sycl::bit_cast<unsigned int>(s + 0.0f);
    return b & 0x80000000u ? ~b : b | 0x80000000u;
}

// x summed over the threads before this one, and over all of them in *all.  part has a slot a warp.
__dpct_inline__ int block_before(int x, int *part, int *all) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) % WARP,
              wid = item_ct1.get_local_id(2) / WARP,
              warps = item_ct1.get_local_range(2) / WARP;
    int in = x;
#pragma unroll
    for (int o = 1; o < WARP; o <<= 1) {
        /*
        DPCT1108: '__shfl_up_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        const int v = dpct::experimental::shift_sub_group_right(
            0xffffffff, sycl::ext::oneapi::this_work_item::get_sub_group(), in,
            o);
        if (lane >= o) in += v;
    }
    if (lane == WARP - 1) part[wid] = in;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    int before = 0, total = 0;
#pragma unroll
    for (int k = 0; k < warps; ++k) {
        if (k < wid) before += part[k];
        total += part[k];
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    *all = total;
    return before + in - x;
}

// A block a token.  The n_top blocks with the highest scores by a radix search for the n_top-th score, eight bits a
// round, then the blocks above it and of those on it the lowest, written in the order of the blocks.  A token that
// sees n_top blocks or fewer takes them all.
/*
DPCT1110: The total declared local variable size in device function
select_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void select_kernel(const float *__restrict__ score,
                                   int score_stride, int32_t *__restrict__ sel,
                                   int sel_stride, int n_top, int ratio,
                                   long long pos, const long long *at) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &hist = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    int[SELECT_THREADS / WARP][256]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        int[SELECT_THREADS / WARP]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &pick = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[2]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int n =
        int((position(at, pos, 1, item_ct1.get_group(2)) + 1) / ratio);
    int32_t *out = sel + size_t(item_ct1.get_group(2)) * sel_stride;
    if (n <= n_top) {
#pragma unroll
        for (int i = item_ct1.get_local_id(2); i < n;
             i += item_ct1.get_local_range(2)) out[i] = i;
        return;
    }
    const float *s = score + size_t(item_ct1.get_group(2)) * score_stride;
    const int lane = item_ct1.get_local_id(2) % WARP,
              wid = item_ct1.get_local_id(2) / WARP;
    uint32_t cut = 0, mask = 0;
    int want = n_top;
    for (int shift = 24; shift >= 0; shift -= 8) {
#pragma unroll
        for (int d = lane; d < 256; d += WARP) hist[wid][d] = 0;
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
        for (int i = item_ct1.get_local_id(2); i < n;
             i += item_ct1.get_local_range(2)) {
            const uint32_t k = order_key(s[i]);
            if ((k & mask) == cut) dpct::atomic_fetch_add<
                sycl::access::address_space::generic_space>(
                &hist[wid][(k >> shift) & 255], 1);
        }
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
        if (item_ct1.get_local_id(2) < 256) {
            int t = 0;
#pragma unroll
            for (int k = 0; k < SELECT_THREADS / WARP; ++k) t +=
                hist[k][item_ct1.get_local_id(2)];
            hist[0][item_ct1.get_local_id(2)] = t;
        }
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
        if (item_ct1.get_local_id(2) == 0) {
            int d = 255, above = 0;
#pragma unroll
            for (; d > 0 && above + hist[0][d] < want; --d) above += hist[0][d];
            pick[0] = d;
            pick[1] = above;
        }
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
        cut |= uint32_t(pick[0]) << shift;
        mask |= 255u << shift;
        want -= pick[1];
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
    const int per = (n + int(item_ct1.get_local_range(2)) - 1) /
                    int(item_ct1.get_local_range(2));
    const int lo = item_ct1.get_local_id(2) * per, hi = sycl::min(lo + per, n);
    int g = 0, e = 0;
    for (int i = lo; i < hi; ++i) {
        const uint32_t k = order_key(s[i]);
        g += k > cut;
        e += k == cut;
    }
    int g_all, e_all;
    const int g0 = block_before(g, part, &g_all);
    int e0 = block_before(e, part, &e_all);
    const int need = n_top - g_all;
    int o = g0 + sycl::min(e0, need);
    for (int i = lo; i < hi; ++i) {
        const uint32_t k = order_key(s[i]);
        if (k > cut || (k == cut && e0++ < need)) out[o++] = i;
    }
}

// A block a head of a token.  Its keys are the raw rows of its window, oldest first, then its compressed rows; a
// key is its value too.  Scores, the softmax with the head's sink, then the weighted sum of the keys.
/*
DPCT1110: The total declared local variable size in device function
attend_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
attend_kernel(const float *__restrict__ q, const uint16_t *__restrict__ raw,
              int ring, int window, const uint16_t *__restrict__ comp,
              int ratio, const int32_t *__restrict__ sel, int sel_stride,
              int top, const float *__restrict__ sinks, float *__restrict__ out,
              int dim, int n_head, float scale, long long pos,
              const long long *at, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto smem = (float *)
        dpct_local; // dim query values, then a score a key, then a key's row
    auto &part =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    float* sq = smem;
    float* sc = smem + dim;
    const int h = item_ct1.get_group(2), t = item_ct1.get_group(1);
    const long long p = position(at, pos, 1, t);
    const int n_raw = (int)sycl::min((long long)window, p + 1);
    const long long first_raw = p + 1 - n_raw;
    int n_comp = 0;
    if (comp) {
        const long long visible = (p + 1) / ratio;
        n_comp = sel ? (int)sycl::min(visible, (long long)top) : (int)visible;
    }
    const int n_key = n_raw + n_comp;
    int* row = (int*) (sc + n_key);
    const float* qh = q + (size_t(t) * n_head + h) * dim;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < dim;
         i += item_ct1.get_local_range(2)) sq[i] = qh[i];
#pragma unroll
    for (int k = item_ct1.get_local_id(2); k < n_key;
         k += item_ct1.get_local_range(2))
        row[k] = k < n_raw ? int((first_raw + k) % ring) : sel ? sel[size_t(t) * sel_stride + k - n_raw] : k - n_raw;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    auto key = [&](int k) { return (k < n_raw ? raw : comp) + size_t(row[k]) * dim; };
    const int lane = item_ct1.get_local_id(2) % WARP,
              wid = item_ct1.get_local_id(2) / WARP,
              warps = item_ct1.get_local_range(2) / WARP;
    for (int k = wid; k < n_key; k += warps) {
        const uint16_t* kr = key(k);
        float dot = 0.0f;
#pragma unroll
        for (int i = lane; i < dim; i += WARP) dot += sq[i] * half_at(kr, i);
        dot = warp_sum(dot);
        if (lane == 0) sc[k] = dot * scale;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const float sink = sinks[h];
    float mx = sink;
#pragma unroll
    for (int k = item_ct1.get_local_id(2); k < n_key;
         k += item_ct1.get_local_range(2)) mx = sycl::fmax(mx, sc[k]);
    mx = block_max(mx, part);
    float sum = 0.0f;
    for (int k = item_ct1.get_local_id(2); k < n_key;
         k += item_ct1.get_local_range(2)) {
        const float e = sycl::native::exp(sc[k] - mx);
        sc[k] = e;
        sum += e;
    }
    sum = block_sum(sum, part) + sycl::native::exp(sink - mx);
    const float inv = 1.0f / sum;
    float* oh = out + (size_t(t) * n_head + h) * dim;
    for (int i = item_ct1.get_local_id(2); i < dim;
         i += item_ct1.get_local_range(2)) {
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < n_key; ++k) acc += sc[k] * half_at(key(k), i);
        oh[i] = acc * inv;
    }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void attend_kernel_wrapper(const float *__restrict q,
                           const uint16_t *__restrict raw, int ring, int window,
                           const uint16_t *__restrict comp, int ratio,
                           const int32_t *__restrict sel, int sel_stride,
                           int top, const float *__restrict sinks,
                           float *__restrict out, int dim, int n_head,
                           float scale, long long pos, const long long *at) {
    sycl::queue queue = *dpct::kernel_launcher::_que;
    unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range(localMemSize), cgh);

        cgh.parallel_for<dpct_kernel_name<class attend_kernel_>>(
            nr, exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attend_kernel(q, raw, ring, window, comp, ratio, sel,
                              sel_stride, top, sinks, out, dim, n_head, scale,
                              pos, at,
                              dpct_local_acc_ct1
                                  .get_multi_ptr<sycl::access::decorated::no>()
                                  .get());
            });
    });
}

// attend_kernel for a batch of tokens.  The heads share their one key row, so a block takes GROUP_HEADS heads of a
// token, a warp each, and reads the keys GROUP_KEYS at a time into shared memory once for all of them; the softmax
// runs along the keys, from the sink.  attend_kernel read every key twice for every head.
/*
DPCT1110: The total declared local variable size in device function
attend_rows_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
attend_rows_kernel(const float *__restrict__ q,
                   const uint16_t *__restrict__ raw, int ring, int window,
                   const uint16_t *__restrict__ comp, int ratio,
                   const int32_t *__restrict__ sel, int sel_stride, int top,
                   const float *__restrict__ sinks, float *__restrict__ out,
                   int dim, int n_head, float scale, long long pos,
                   const long long *at, uint8_t *dpct_local) {
    // GROUP_HEADS rows of query values; GROUP_KEYS rows of keys, a word a pair of halves and four words more a row:
    // read 16 bytes at a time, the 8 lanes of a quarter warp then reach every bank once, along a row or down a
    // column; a probability a head and key
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto smem = (float *)dpct_local;
    auto &row =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[GROUP_KEYS]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int pitch = dim / 2 + 4;
    float* sq = smem;
    uint32_t* sk = (uint32_t*) (sq + GROUP_HEADS * dim);
    float* sp = (float*) (sk + GROUP_KEYS * pitch);
    const int lane = item_ct1.get_local_id(2) % WARP,
              w = item_ct1.get_local_id(2) / WARP;
    const int h = item_ct1.get_group(2) * GROUP_HEADS + w,
              t = item_ct1.get_group(1);
    const long long p = position(at, pos, 1, t);
    const int n_raw = (int)sycl::min((long long)window, p + 1);
    const long long first_raw = p + 1 - n_raw;
    int n_comp = 0;
    if (comp) {
        const long long visible = (p + 1) / ratio;
        n_comp = sel ? (int)sycl::min(visible, (long long)top) : (int)visible;
    }
    const int n_key = n_raw + n_comp;
    for (int i = item_ct1.get_local_id(2); i < GROUP_HEADS * dim;
         i += item_ct1.get_local_range(2)) {
        const int hh = item_ct1.get_group(2) * GROUP_HEADS + i / dim;
        sq[i] = hh < n_head ? q[(size_t(t) * n_head + hh) * dim + i % dim] : 0.0f;
    }
    // this lane's part of the head's output: values 8 lane to 8 lane + 7 of every 256
    float o[GROUP_DIM / WARP] = {};
    const int chunks = dim / (8 * WARP);
    float m = h < n_head ? sinks[h] : 0.0f, l = 1.0f;
    for (int k0 = 0; k0 < n_key; k0 += GROUP_KEYS) {
        const int nk = sycl::min(GROUP_KEYS, n_key - k0);
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(
            sycl::access::fence_space::local_space); // the keys before are read
        if (item_ct1.get_local_id(2) < GROUP_KEYS) {
            const int k = k0 + item_ct1.get_local_id(2);
            row[item_ct1.get_local_id(2)] =
                k >= k0 + nk ? 0
                : k < n_raw  ? int((first_raw + k) % ring)
                : sel        ? sel[size_t(t) * sel_stride + k - n_raw]
                             : k - n_raw;
        }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        for (int i = item_ct1.get_local_id(2); i < GROUP_KEYS * (dim / 8);
             i += item_ct1.get_local_range(2)) {
            const int kk = i / (dim / 8), c = i % (dim / 8);
            sycl::uint4 v = sycl::uint4(0, 0, 0, 0);
            if (kk < nk) v =
                ((const sycl::uint4 *)((k0 + kk < n_raw ? raw : comp) +
                                       size_t(row[kk]) * dim))[c];
            *(sycl::uint4 *)&sk[kk * pitch + 4 * c] = v;
        }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        float s = -INFINITY;
        if (lane < nk) {
            const sycl::uint4 *kr = (const sycl::uint4 *)(sk + lane * pitch);
            const sycl::float4 *qh = (const sycl::float4 *)(sq + w * dim);
            float d0 = 0.0f, d1 = 0.0f;
            for (int c = 0; c < dim / 8; ++c) {
                const sycl::uint4 kv = kr[c];
                const sycl::float4 qa = qh[2 * c], qb = qh[2 * c + 1];
                const sycl::float2
                    a = (*(const sycl::half2 *)&kv.x())
                            .template convert<float,
                                              sycl::rounding_mode::automatic>(),
                    b = (*(const sycl::half2 *)&kv.y())
                            .template convert<float,
                                              sycl::rounding_mode::automatic>();
                const sycl::float2
                    e = (*(const sycl::half2 *)&kv.z())
                            .template convert<float,
                                              sycl::rounding_mode::automatic>(),
                    f = (*(const sycl::half2 *)&kv.w())
                            .template convert<float,
                                              sycl::rounding_mode::automatic>();
                d0 += qa.x() * a.x() + qa.y() * a.y() + qa.z() * b.x() +
                      qa.w() * b.y();
                d1 += qb.x() * e.x() + qb.y() * e.y() + qb.z() * f.x() +
                      qb.w() * f.y();
            }
            s = (d0 + d1) * scale;
        }
        const float mn = sycl::fmax(m, warp_max(s));
        const float alpha = sycl::native::exp(m - mn),
                    e = lane < nk ? sycl::native::exp(s - mn) : 0.0f;
        l = l * alpha + warp_sum(e);
        m = mn;
        sp[w * GROUP_KEYS + lane] = e;
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
#pragma unroll
        for (int j = 0; j < GROUP_DIM / WARP; ++j) o[j] *= alpha;
        for (int k = 0; k < nk; ++k) {
            const float pk = sp[w * GROUP_KEYS + k];
            const sycl::uint4 *kr =
                (const sycl::uint4 *)(sk + k * pitch) + lane;
#pragma unroll
            for (int j = 0; j < GROUP_DIM / (8 * WARP); ++j)   // whole, so that o stays in registers
                if (j < chunks) {
                    const sycl::uint4 v = kr[WARP * j];
                    const sycl::float2
                        a = (*(const sycl::half2 *)&v.x())
                                .template convert<
                                    float, sycl::rounding_mode::automatic>(),
                        b = (*(const sycl::half2 *)&v.y())
                                .template convert<
                                    float, sycl::rounding_mode::automatic>();
                    const sycl::float2
                        c = (*(const sycl::half2 *)&v.z())
                                .template convert<
                                    float, sycl::rounding_mode::automatic>(),
                        d = (*(const sycl::half2 *)&v.w())
                                .template convert<
                                    float, sycl::rounding_mode::automatic>();
                    float* oj = o + 8 * j;
                    oj[0] += pk * a.x(), oj[1] += pk * a.y(),
                        oj[2] += pk * b.x(), oj[3] += pk * b.y();
                    oj[4] += pk * c.x(), oj[5] += pk * c.y(),
                        oj[6] += pk * d.x(), oj[7] += pk * d.y();
                }
        }
        sycl::group_barrier(
            sycl::ext::oneapi::this_work_item::
                get_sub_group()); // sp is written again for the next keys
    }
    if (h >= n_head) return;
    float* oh = out + (size_t(t) * n_head + h) * dim + 8 * lane;
    const float inv = 1.0f / l;
#pragma unroll
    for (int j = 0; j < GROUP_DIM / (8 * WARP); ++j)
        if (j < chunks) {
            const float* oj = o + 8 * j;
            *(sycl::float4 *)&oh[8 * WARP * j] = sycl::float4(
                oj[0] * inv, oj[1] * inv, oj[2] * inv, oj[3] * inv);
            *(sycl::float4 *)&oh[8 * WARP * j + 4] = sycl::float4(
                oj[4] * inv, oj[5] * inv, oj[6] * inv, oj[7] * inv);
        }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void attend_rows_kernel_wrapper(
    const float *__restrict q, const uint16_t *__restrict raw, int ring,
    int window, const uint16_t *__restrict comp, int ratio,
    const int32_t *__restrict sel, int sel_stride, int top,
    const float *__restrict sinks, float *__restrict out, int dim, int n_head,
    float scale, long long pos, const long long *at) {
    sycl::queue queue = *dpct::kernel_launcher::_que;
    unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};
    dpct::has_capability_or_fail(queue.get_device(), {sycl::aspect::fp16});

    queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range(localMemSize), cgh);

        cgh.parallel_for<dpct_kernel_name<class attend_rows_kernel_>>(
            nr, exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attend_rows_kernel(
                    q, raw, ring, window, comp, ratio, sel, sel_stride, top,
                    sinks, out, dim, n_head, scale, pos, at,
                    dpct_local_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
}

__dpct_inline__ void permute_kernel(const float *__restrict__ x,
                                    float *__restrict__ y, int n, int groups,
                                    int rows, bool back) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= n) return;
    const int g = item_ct1.get_group(1) % groups,
              t = item_ct1.get_group(1) / groups;
    const size_t a = (size_t(t) * groups + g) * n + i, b = (size_t(g) * rows + t) * n + i;
    if (back) y[a] = x[b];
    else y[b] = x[a];
}

}  // namespace

void dsv4_rope(const float* x, float* y, int dim, int heads, int off, int n_rot, const Rope& r, int64_t pos,
               int step, int rows, bool norm, float eps, bool back, void* stream, const long long* at) {
    if (rows <= 0) return;
    if (dim % 2 || dim > 2048 || off % 2 || off + n_rot > dim)
        throw std::invalid_argument("dsv4_rope: the heads are not dim <= 2048 with the turned values inside");
    const int threads = std::max(WARP, (dim / 2 + WARP - 1) / WARP * WARP);
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)->parallel_for<dpct_kernel_name<class rope_kernel_6b6add>>(
            sycl::nd_range<3>(sycl::range(1, rows, heads) *
                                  sycl::range(1, 1, threads),
                              sycl::range(1, 1, threads)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                rope_kernel(x, y, dim, off, n_rot, r, (long long)pos, step,
                            norm, eps, back, at);
            });
    }
    ck("dsv4_rope");
}

void dsv4_state(const float* kv, const float* score, const float* ape, float* ring_kv, float* ring_score, int ring,
                int width, int ratio, int64_t pos, int rows, void* stream, const long long* at) {
    if (rows <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)->parallel_for<dpct_kernel_name<class state_kernel_a82ae6>>(
            sycl::nd_range<3>(sycl::range(1, rows, grid(width, 256)) *
                                  sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                state_kernel(kv, score, ape, ring_kv, ring_score, ring, width,
                             ratio, (long long)pos, at);
            });
    }
    ck("dsv4_state");
}

void dsv4_squeeze(const float* ring_kv, const float* ring_score, int ring, int dim, int ratio, bool overlap,
                  const float* norm, float eps, const Rope& r, int n_rot, int64_t pos, int rows, uint16_t* cache,
                  void* stream, const long long* at) {
    if (rows <= 0) return;
    if (dim % WARP || dim > 1024 || n_rot % 2 || n_rot > dim)
        throw std::invalid_argument("dsv4_squeeze: a row of 32 to 1024 values, in 32s");
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)->parallel_for<dpct_kernel_name<class squeeze_kernel_57cf99>>(
            sycl::nd_range<3>(sycl::range(1, 1, rows) * sycl::range(1, 1, dim),
                              sycl::range(1, 1, dim)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                squeeze_kernel(ring_kv, ring_score, ring, dim, ratio, overlap,
                               norm, eps, r, n_rot, (long long)pos, at, cache);
            });
    }
    ck("dsv4_squeeze");
}

void dsv4_idx_score(const float* q, const float* w, const uint16_t* keys, float* score, int score_stride, int dim,
                    int n_head, int ratio, int64_t pos, int rows, float scale, void* stream, const long long* at,
                    int room) {
    if (dim != IDX_DIM)
        throw std::invalid_argument("dsv4_idx_score: the indexer's keys are not 128 values");
    const long long visible = (at ? room : pos + rows) / ratio;   // the last token sees the most
    if (rows <= 0 || visible == 0) return;
    /*
    DPCT1083: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    const size_t smem =
        (size_t(n_head) * (IDX_DIM + 1) + n_head + SCORE_KEYS * IDX_DIM) *
        sizeof(float);
    static unsigned done = 0;
    if (smem > allow_smem(idx_score_kernel_wrapper, done))
        throw std::runtime_error("dsv4_idx_score: the indexer's queries do not fit the card's shared memory");
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(smem), cgh);

            cgh.parallel_for<dpct_kernel_name<class idx_score_kernel_6aa1de>>(
                sycl::nd_range<3>(
                    sycl::range(1, rows, grid((int)visible, SCORE_KEYS)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        idx_score_kernel(
                            q, w, keys, score, score_stride, n_head, ratio,
                            (long long)pos, scale, at,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    }
    ck("dsv4_idx_score");
}

void dsv4_select(const float* score, int score_stride, int32_t* sel, int sel_stride, int n_top, int ratio,
                 int64_t pos, int rows, void* stream, const long long* at) {
    if (rows <= 0) return;
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)->parallel_for<dpct_kernel_name<class select_kernel_528439>>(
            sycl::nd_range<3>(sycl::range(1, 1, rows) *
                                  sycl::range(1, 1, SELECT_THREADS),
                              sycl::range(1, 1, SELECT_THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                select_kernel(score, score_stride, sel, sel_stride, n_top,
                              ratio, (long long)pos, at);
            });
    }
    ck("dsv4_select");
}

void dsv4_attend(const float* q, const uint16_t* raw, int ring, int window, const uint16_t* comp, int ratio,
                 const int32_t* sel, int sel_stride, int top, const float* sinks, float* out, int dim, int n_head,
                 float scale, int64_t pos, int rows, void* stream, const long long* at, int room) {
    if (rows <= 0) return;
    // a batch: its shared memory does not grow with the keys.  A few tokens keep a block a head, which spreads a
    // token over more of the card.
    if (rows >= GROUP_ROWS && dim % (8 * WARP) == 0 && dim <= GROUP_DIM) {
        /*
        DPCT1083: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        const size_t smem =
            (size_t(GROUP_HEADS) * dim + GROUP_HEADS * GROUP_KEYS) *
                sizeof(float) +
            size_t(GROUP_KEYS) * (dim / 2 + 4) * sizeof(uint32_t);
        static unsigned done = 0;
        if (smem <= allow_smem(attend_rows_kernel_wrapper, done)) {
            {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};
                dpct::has_capability_or_fail(cs(stream)->get_device(),
                                             {sycl::aspect::fp16});

                cs(stream)->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range(smem), cgh);

                    cgh.parallel_for<
                        dpct_kernel_name<class attend_rows_kernel_4809a7>>(
                        sycl::nd_range<3>(
                            sycl::range(1, rows, grid(n_head, GROUP_HEADS)) *
                                sycl::range(1, 1, GROUP_HEADS * WARP),
                            sycl::range(1, 1, GROUP_HEADS * WARP)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                attend_rows_kernel(
                                    q, raw, ring, window, comp, ratio, sel,
                                    sel_stride, top, sinks, out, dim, n_head,
                                    scale, (long long)pos, at,
                                    dpct_local_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get());
                            });
                });
            }
            ck("dsv4_attend");
            return;
        }
    }
    long long keys = window;
    if (comp) keys += sel ? top : (at ? room : pos + rows) / ratio;
    /*
    DPCT1083: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    const size_t smem = (size_t(dim) + 2 * size_t(keys)) * sizeof(float);
    static unsigned done = 0;
    if (smem > allow_smem(attend_kernel_wrapper, done))
        throw std::runtime_error("dsv4_attend: " + std::to_string(keys) + " keys a token do not fit the card's shared "
                                 "memory; lower --max-context");
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(smem), cgh);

            cgh.parallel_for<dpct_kernel_name<class attend_kernel_a8a5fe>>(
                sycl::nd_range<3>(sycl::range(1, rows, n_head) *
                                      sycl::range(1, 1, ATTEND_THREADS),
                                  sycl::range(1, 1, ATTEND_THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        attend_kernel(
                            q, raw, ring, window, comp, ratio, sel, sel_stride,
                            top, sinks, out, dim, n_head, scale, (long long)pos,
                            at,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    }
    ck("dsv4_attend");
}

void permute_groups(const float* x, float* y, int n, int groups, int rows, bool back, void* stream) {
    if (rows <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)->parallel_for<dpct_kernel_name<class permute_kernel_939f92>>(
            sycl::nd_range<3>(sycl::range(1, groups * rows, grid(n, 256)) *
                                  sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                permute_kernel(x, y, n, groups, rows, back);
            });
    }
    ck("permute_groups");
}

}  // namespace strata::poly
