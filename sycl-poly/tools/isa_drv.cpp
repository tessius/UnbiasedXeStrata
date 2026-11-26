// SPDX-License-Identifier: MIT
// Launches the int8 Q4_K expert products once so that the kernels are built (tools/isa_mix.sh disassembles them).
#include "iq_mmq.hpp"
#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>
using namespace strata::kernels;
int main() {   // one launch of the int8 Q4_K products at gate/up's shape: enough to have the kernels built
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    const int ty = 12, NE = 4, ROWS = 64; const int64_t R = 1024, C = 2048, total = NE * ROWS;
    const size_t eb = iq_mmq_row_bytes(ty, C) * R;
    auto* dw = sycl::malloc_device<uint8_t>(eb * NE, q); q.memset(dw, 0, eb * NE);
    auto* dxq = sycl::malloc_device<uint8_t>(iq_mmq_act_bytes(total, C) + 4096, q); q.memset(dxq, 0, iq_mmq_act_bytes(total, C));
    std::vector<int32_t> bnd(NE + 1), ids(total); for (int i = 0; i <= NE; ++i) bnd[i] = i * ROWS; for (int i = 0; i < total; ++i) ids[i] = i;
    auto* db = sycl::malloc_device<int32_t>(NE + 1, q); q.memcpy(db, bnd.data(), (NE + 1) * 4);
    auto* di = sycl::malloc_device<int32_t>(total, q); q.memcpy(di, ids.data(), total * 4);
    auto* dd = sycl::malloc_device<float>(total * R, q); q.wait();
    iq_mmq_grouped(q, ty, dw, eb, R, C, NE, dxq, total, db, ROWS, di, dd, R, nullptr, 0, total).wait();
    std::printf("ok\n");
}
