#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
# SPDX-License-Identifier: MIT
"""The hand fixes to dpct's output for src/ik2, as code (idempotent).  usage: fixups_ik2.py DIR   (DIR has ik2_*.dp.cpp)
Each fix says why.  A fix whose pattern is gone prints 'skipped'; one that fails to apply an expected pattern stops."""
import re, sys, os
d = sys.argv[1]
def edit(name, fn):
    p = os.path.join(d, name)
    if not os.path.exists(p): return
    s = open(p).read(); t = fn(s)
    open(p, "w").write(t)
    print(f"{name}: {'changed' if t != s else 'unchanged'}")

CP_ASYNC = '''// SYCL: cp.async is NVIDIA's asynchronous global-to-shared copy.  Here it is a plain copy that finishes before it returns (the
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

'''
def kernels(s):
    # FP8 (E4M3): the sm_89 hardware conversions do not exist here; the file has a software path for older GPUs.  dpct turned
    # `__CUDA_ARCH__ >= 890` into DPCT_COMPATIBILITY_TEMP >= 890 (always true: 900) and `!(... >= 890)` into a constant-false mess.
    s = s.replace("#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 890", "#if 0   // SYCL: no sm_89 FP8 hardware conversions; the software path is the one")
    s = s.replace("#if !defined(DPCT_COMPATIBILITY_TEMP) && !DPCT_COMPATIBILITY_TEMP >= 890", "#if 0   // SYCL: the host copy of the function: its `__CUDA_ARCH__ >= 890` is false, so the software else-branch (dpct mangled the test)")
    # Hopper's register fence: an empty asm that only orders the compiler's use of an accumulator; nothing to do here
    s = s.replace('asm volatile("" : "+f"(r)::"memory");', '(void) r;   // SYCL: no asm; the wgmma (sm_90a) path is not built')
    # IK2_MMA_LAUNCH(W) is used as `if (w == 1) IK2_MMA_LAUNCH(1); else ...`: dpct made it a bare {block}, whose `;` ends the if.
    s = re.sub(r"(#define IK2_MMA_LAUNCH\(W\)[ ]*\\\n[ ]*)\{", r"\1do {", s)
    s = re.sub(r"(\\\n)    \}\n(    if \(w == 1\) IK2_MMA_LAUNCH)", r"\1    } while (0)\n\2", s)
    # kernel names dpct could not complete (DPCT1050): the template argument it left out is the kernel's defaulted one
    def name(m):
        cls = re.findall(r"class (\w+?)_[0-9a-f]{6}", s[max(0, m.start() - 300):m.start()])
        c = cls[-1] if cls else ""
        return {"grouped_expert_mma": "dpct_kernel_scalar<false>", "mv_q8p_kernel": "dpct_kernel_scalar<0>"}[c]
    s = re.sub(r"dpct_placeholder /\*Fix the type mannually\*/", name, s)
    # occupancy (DPCT1111): how many work-groups fit a core is a tuning input (the number of splits); without the kernels' shared-memory
    # size this is an upper bound (more splits than resident groups: slower at worst, never wrong)
    s = s.replace("0 + dpct_placeholder /* total share local memory size */", "0 /* SYCL: shared-memory size not queried: an upper bound */")
    # the host-only decode: dpct threaded the codebook tables through as parameters; the host has its own copies of them
    s = s.replace("out[i] = decode_value(type, blocks, i);", "out[i] = decode_value(type, blocks, i, h_iq3xxs_grid, h_iq4nl);")
    # cp.async (see CP_ASYNC)
    a = s.find("__dpct_inline__ void cp_async16(void *dst, const void *src, bool valid) {")
    b = s.find("// one 64-key tile of K and V (rows kt .. kt+63")
    if a >= 0 and b > a: s = s[:a] + CP_ASYNC + s[b:]
    # NVIDIA tensor-core primitives -> shuffle emulation (MMA_EMUL); they sit just before the cp.async helpers
    a = s.find("__dpct_inline__ void mma16816(")
    b = s.find("// SYCL: cp.async is NVIDIA's")
    if a >= 0 and b > a: s = s[:a] + '#include "ik2_mma_emul.dp.inc"   // SYCL: the tensor-core primitives, emulated (see that file)\n' + s[b:]
    # __byte_perm -> dpct's byte_level_permute (same selector semantics)
    s = s.replace("__byte_perm(", "dpct::byte_level_permute(")
    # ldmatrix's shared-window address: the NVPTX asm that used it is not built; give the dpct fallback a plain address
    s = re.sub(r"__cvta_generic_to_shared\(([^)]*)\)", r"(uintptr_t)(\1)", s)
    return s
edit("ik2_kernels.dp.cpp", kernels)

def main_cpp(s):
    # the leftovers header (cudaMemcpyAsync with a kind, events, host functions, graph destroy): dpct left them in macros
    if "ik2_leftovers.dp.hpp" not in s:
        s = s.replace('#include "ik2_kernels.dp.hpp"', '#include "ik2_kernels.dp.hpp"\n#include "ik2_leftovers.dp.hpp"', 1)
    # cublasGemmEx that dpct could not migrate (DPCT1020): the same call as its neighbours, minus the algorithm argument
    s = re.sub(r"CB\(cublasGemmEx\(([^;]*?dpct::compute_type::f32_standard), -1\)\);", r"CB(DPCT_CHECK_ERROR(dpct::blas::gemm(\1)));", s, flags=re.S)
    return s
edit("ik2_main.dp.cpp", main_cpp)
