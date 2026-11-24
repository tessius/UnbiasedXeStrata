// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// (written against, and calling, XeStrata's iq_mmq.hpp; its interface and its dpct-free design are XeStrata's: LGPL.)
// Mmq (src/poly/mmq.hpp) on XeStrata's SYCL port of llama.cpp's quantized matrix tiles (src/xe/iq_mmq.cpp: dp4a, and
// Intel XMX where the GPU reports the matrix shapes).  The activations are quantized in that file's own layout
// (int8 rows, then the block scales, then the block sums), which is all Mmq's `xq` ever is.
#include "mmq.hpp"

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <array>
#include <stdexcept>

#include "iq_mmq.hpp"

namespace strata::poly {
namespace {

sycl::queue& Q(void* stream) { return *static_cast<sycl::queue*>(stream); }

// the identity row map and the [0, rows] bounds that a single matrix's call leaves out, and the table of a pointer-table call
struct Ctx {
    int32_t* iota = nullptr;
    int64_t iota_n = 0;
    int32_t* bounds = nullptr;
    const uint8_t** tab = nullptr;
    sycl::queue* q = nullptr;
    ~Ctx() {
        if (!q) return;
        try {
            q->wait();
            sycl::free(iota, *q); sycl::free(bounds, *q); sycl::free(tab, *q);
        } catch (...) {}
    }
};

bool device_ok(sycl::queue& q) { return kernels::iq_mmq_usable(q) || kernels::iq_mmq_dp4a_usable(q); }

}  // namespace

Mmq::Mmq() { ctx_ = new Ctx(); }
Mmq::~Mmq() { delete static_cast<Ctx*>(ctx_); }

bool Mmq::fits(int ggml_type, int64_t w_rows, int64_t w_cols) {
    // iq_mmq's tiles are up to 128 weight rows high and read K in steps of 32 values (a K that is no multiple of 128 has its
    // own tail form); Mmq's callers keep 256 zero-padded columns behind every matrix
    if (!kernels::iq_mmq_type_ok(ggml_type) || w_rows <= 0 || w_rows % 128 != 0 || w_cols <= 0 || w_cols % 256 != 0) return false;
    static const bool ok = [] { return device_ok(dpct::get_in_order_queue()); }();
    return ok;
}

size_t Mmq::q8_bytes(int64_t rows, int64_t cols) {
    // Q2_K reads every activation block twice (iq_mmq_type_dup), as a row of twice the columns
    return kernels::iq_mmq_act_bytes(rows, 2 * cols);
}

void Mmq::quantize(const float* x, const int32_t* ids, void* xq, int ggml_type, int64_t cols, int64_t ld, int64_t rows,
                   void* stream) {
    if (rows > 0)
        kernels::iq_mmq_quantize(Q(stream), x, ids, xq, cols, ld, rows, kernels::iq_mmq_type_sums(ggml_type),
                                 kernels::iq_mmq_type_dup(ggml_type));
}

void Mmq::mul(const void* w, int ggml_type, int64_t w_rows, int64_t w_cols, size_t stride, int n, const void* xq,
              const int32_t* bounds, const int32_t* ids, int64_t total_rows, int64_t max_rows, float* dst, int64_t ld_dst,
              void* stream) {
    if (n <= 0 || max_rows <= 0 || total_rows <= 0) return;
    sycl::queue& q = Q(stream);
    auto* c = static_cast<Ctx*>(ctx_);
    c->q = &q;
    if (!bounds || !ids) {   // one matrix over every row, row i to row i
        if (n != 1) throw std::runtime_error("Mmq::mul: no bounds for more than one matrix");
        if (c->iota_n < total_rows) {
            q.wait();
            sycl::free(c->iota, q);
            c->iota_n = std::max<int64_t>(total_rows, 8192);
            c->iota = sycl::malloc_device<int32_t>((size_t) c->iota_n, q);
            if (!c->iota) throw std::runtime_error("Mmq: no memory for the row map");
            int32_t* p = c->iota;
            q.parallel_for(sycl::range<1>((size_t) c->iota_n), [=](sycl::id<1> i) { p[i[0]] = (int32_t) i[0]; });
        }
        if (!c->bounds) {
            c->bounds = sycl::malloc_device<int32_t>(2, q);
            if (!c->bounds) throw std::runtime_error("Mmq: no memory for the bounds");
        }
        int32_t* b = c->bounds;
        const int32_t rows = (int32_t) total_rows;
        q.single_task([=] { b[0] = 0; b[1] = rows; });   // in order with the product that reads it
        bounds = c->bounds;
        ids = c->iota;
    }
    const int64_t cols = kernels::iq_mmq_type_dup(ggml_type) ? 2 * w_cols : w_cols;
    kernels::iq_mmq_grouped(q, ggml_type, w, stride, w_rows, cols, n, xq, total_rows, bounds, max_rows, ids, dst, ld_dst,
                            nullptr, 0, total_rows);
}

bool Mmq::tab(int ggml_type) { return kernels::iq_mmq_type_ok(ggml_type); }

bool Mmq::mul_tab(const void* const* w, int ggml_type, int64_t w_rows, int64_t w_cols, int n, const void* xq,
                  const int32_t* bounds, const int32_t* ids, int64_t total_rows, int64_t max_rows, float* dst,
                  int64_t ld_dst, void* stream) {
    if (n <= 0 || n > kTab || !kernels::iq_mmq_type_ok(ggml_type) || w_rows % 128 != 0) return false;
    if (max_rows <= 0 || total_rows <= 0) return true;
    sycl::queue& q = Q(stream);
    auto* c = static_cast<Ctx*>(ctx_);
    c->q = &q;
    if (!c->tab) {
        c->tab = sycl::malloc_device<const uint8_t*>((size_t) kTab, q);
        if (!c->tab) return false;
    }
    std::array<const uint8_t*, kTab> t{};
    for (int i = 0; i < n; ++i) t[(size_t) i] = static_cast<const uint8_t*>(w[i]);
    const uint8_t** dev = c->tab;
    q.single_task([=] { for (int i = 0; i < kTab; ++i) dev[i] = t[(size_t) i]; });   // by value: the host array goes away
    const int64_t cols = kernels::iq_mmq_type_dup(ggml_type) ? 2 * w_cols : w_cols;
    kernels::iq_mmq_grouped(q, ggml_type, nullptr, 0, w_rows, cols, n, xq, total_rows, bounds, max_rows, ids, dst, ld_dst,
                            c->tab, 0, total_rows);
    return true;
}

}  // namespace strata::poly
