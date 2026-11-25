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
    # Intel's graphics compiler crashes (a segfault inside IGC, offline too) on experts_gate_up_swiglu_g_kernel, where row_dot_multi and
    # row_dot_pair_multi - a runtime switch over every quant type, each branch unrolled over 8 tokens - are inlined for 1, 2, 4 and 8 tokens.
    # Only -cl-opt-disable compiles it.  Keeping those two out of line makes each compile unit a fraction of the size.
    s = re.sub(r"\ninline void (row_dot_multi|row_dot_pair_multi)\(", r"\n__attribute__((noinline)) void \1(", s)
    # IK2_SYNC_CHECK=1 (diagnostic): every launch's check("name") waits for the card and prints the name, so a hang or a device error
    # shows which kernel it was (CUDA's cudaGetLastError, which check() was written around, has no equivalent in SYCL).
    # IK2_SYNC_CHECK=2: no printing; the time from one check to the next (launch + the card's work, as the card is waited for) is added up
    # per name and the table is printed at exit: where the time goes, per kernel.
    if "IK2_SYNC_CHECK" not in s:
        s = s.replace("void check(const char* what) {\n", """struct SyncStats {
    std::map<std::string, std::pair<double, long>> t;
    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
    ~SyncStats() {
        std::vector<std::pair<double, std::string>> v;
        double all = 0; for (auto &kv : t) { v.push_back({kv.second.first, kv.first}); all += kv.second.first; }
        std::sort(v.rbegin(), v.rend());
        if (v.empty()) return;
        std::fprintf(stderr, "[ik2 time per launch name, %.3f s in all]\\n", all);
        for (size_t i = 0; i < v.size() && i < 24; ++i) std::fprintf(stderr, "  %7.1f ms %5.1f%% %6ld x  %s\\n", v[i].first * 1e3, 100 * v[i].first / all, t[v[i].second].second, v[i].second.c_str());
    }
};
void check(const char* what) {
    static const char *mode = std::getenv("IK2_SYNC_CHECK");
    static SyncStats stats;
    if (mode) {
        try {
            dpct::get_current_device().queues_wait_and_throw();
            if (mode[0] == '2') { const auto now = std::chrono::steady_clock::now(); auto &e = stats.t[what]; e.first += std::chrono::duration<double>(now - stats.last).count(); ++e.second; stats.last = now; }
            else std::fprintf(stderr, "[ik2 done] %s\\n", what);
        } catch (const sycl::exception &ex) { std::fprintf(stderr, "[ik2 FAILED] %s: %s\\n", what, ex.what()); std::exit(2); }
    }
""", 1)
    # dpct maps __shfl_*_sync(0xffffffff, ...) to its MASKED sub-group operations (DPCT1108, "experimental"); the full-mask forms are the
    # standard ones, built on select_from_group.  (Same replacement as sycl-poly's.)
    s = re.sub(r"dpct::experimental::(permute_sub_group_by_xor|shift_sub_group_left|shift_sub_group_right|select_from_sub_group)\(\s*0xffffffffu?,\s*", r"dpct::\1(", s)
    if "#include <chrono>" not in s: s = s.replace("#include <algorithm>\n", "#include <algorithm>\n#include <chrono>\n#include <map>\n#include <string>\n#include <vector>\n", 1)   # (for check()'s statistics)
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

def draft_cpp(s):
    return re.sub(r"dpct::experimental::(permute_sub_group_by_xor|shift_sub_group_left|shift_sub_group_right|select_from_sub_group)\(\s*0xffffffffu?,\s*", r"dpct::\1(", s)
edit("ik2_draft.dp.cpp", draft_cpp)

# CUDA's cudaMemcpy / cudaMemset (no "Async") are synchronous with respect to the host: the caller may free or reuse the host buffer
# at once (dev_f32(host_f32(...)) frees a temporary vector).  dpct turned them into get_in_order_queue().memcpy(...), which does not
# wait (DPCT1124): the GPU then copied from freed memory (NaN norm weights, 1.9e28).  Wait after each.
def sync_default_queue(s):
    out, i, n = [], 0, 0
    pat = re.compile(r"get_in_order_queue\(\)\s*\.(memcpy|memset)\(")
    while True:
        m = pat.search(s, i)
        if not m: out.append(s[i:]); break
        depth, j = 1, m.end()
        while depth and j < len(s):
            depth += (s[j] == "(") - (s[j] == ")"); j += 1
        out.append(s[i:j])
        if not re.match(r"\s*\.wait\(\)", s[j:]): out.append(".wait()"); n += 1
        i = j
    return re.sub(r"\.wait\(\)(\s*)\.wait\(\)", r".wait()", "".join(out))   # (a first version of this fix doubled some)
for _name in ("ik2_main.dp.cpp", "ik2_draft.dp.cpp", "ik2_kernels.dp.cpp"):
    edit(_name, sync_default_queue)

# On Intel the tensor-core MMA is emulated with sub-group shuffles (src/ik2_mma_emul.dp.inc), which loses badly to the plain
# dp4a-style grouped expert kernels (measured on the Arc Pro B60: 9.4 vs 26.6 tok/s, same tokens).  Default IK2_EXPERT_MMA to 0.
edit("ik2_main.dp.cpp", lambda s: s.replace('getenv("IK2_EXPERT_MMA") ? atoi(getenv("IK2_EXPERT_MMA")) : 1;', 'getenv("IK2_EXPERT_MMA") ? atoi(getenv("IK2_EXPERT_MMA")) : 0;'))

# Intel: dpct::byte_level_permute is four variable 64-bit shifts per call and sat in the NVFP4 inner loop (e2m1_pair_lo/hi).
# Decode a nibble with a packed table in a constant instead: |value|*2 in {0,1,2,3,4,6,8,12} is nibble `mag` of 0xC8643210.
def fast_e2m1(s):
    if "ik2_e2m1" in s: return s
    pat = re.compile(r"__dpct_inline__ sycl::float2 e2m1_pair_lo\(uint32_t s\) \{.*?\n\}\n__dpct_inline__ sycl::float2 e2m1_pair_hi\(uint32_t s\) \{.*?\n\}\n", re.S)
    new = """__dpct_inline__ float ik2_e2m1(uint32_t nib) {   // (Intel) one E2M1 nibble, doubled, as a float
    const float v = (float) ((0xC8643210u >> ((nib & 7u) * 4u)) & 15u);
    return sycl::bit_cast<float>(sycl::bit_cast<uint32_t>(v) | ((nib & 8u) << 28));
}
__dpct_inline__ sycl::float2 e2m1_pair_lo(uint32_t s) { return sycl::float2(ik2_e2m1(s), ik2_e2m1(s >> 8)); }        // nibbles 0, 2
__dpct_inline__ sycl::float2 e2m1_pair_hi(uint32_t s) { return sycl::float2(ik2_e2m1(s >> 4), ik2_e2m1(s >> 12)); }   // nibbles 1, 3
"""
    s2, n = pat.subn(lambda m: new, s, count=1)
    assert n == 1, "e2m1_pair_lo/hi not found"
    return s2
edit("ik2_kernels.dp.cpp", fast_e2m1)

# Expert-kernel tiling for the Arc Pro B60 (measured, decode of Inkling-2-Mini): 8 warps x 1 row per work-group for gate/up and for down
# beat the CUDA defaults (4x2 / 4x4): 26.8 -> 33.4 tok/s.  Same for every S.
def tile_experts(s):
    s = re.sub(r"if \(S <= 8\) launch_experts_gate_up_swiglu<4, 2>\((.*?)\);\s*else launch_experts_gate_up_swiglu<4, 4>\(.*?\);",
               r"launch_experts_gate_up_swiglu<8, 1>(\1);", s, flags=re.S)
    s = re.sub(r"if \(S <= 4\) launch_experts_down<4, 2>\((.*?)\);\s*else launch_experts_down<4, 4>\(.*?\);",
               r"launch_experts_down<8, 1>(\1);", s, flags=re.S)
    return s
edit("ik2_kernels.dp.cpp", tile_experts)
