// include/strata/kernels/pdl.hpp for SYCL: programmatic dependent launch is an NVIDIA sm_90 feature. Like the HIP
// build, every launch is an ordinary one, the device functions are no-ops and the kernels keep their plain loops.
#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

namespace strata::kernels {

inline bool& pdl_scope() { static thread_local bool v = false; return v; }
inline bool pdl_supported() { return false; }

#define STRATA_PDL_RESTRICT __restrict__
inline constexpr bool kPdlPrefetch = false;

__dpct_inline__ void pdl_wait() {}
__dpct_inline__ void pdl_trigger() {}

/// An ordinary launch through dpct's kernel-wrapper launcher (what dpct made of `kernel<<<...>>>` in launch_pdl's callers).
template <typename F, typename... Args>
inline void launch_pdl(F* wrapper, dpct::dim3 grid, dpct::dim3 block, size_t smem, dpct::queue_ptr q, Args... args) {
    dpct::kernel_launcher::launch(wrapper, grid, block, (unsigned) smem, q, args...);
}

}  // namespace strata::kernels
