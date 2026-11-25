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

# Prefill attention on the Intel matrix engine (XMX, joint_matrix): src/ik2_flash_xmx.dp.inc replaces the FP16 attn_flash path (the CUDA kernel's mma.sync
# emulated with shuffles is ~50x slower).  FP8 caches keep the emulated kernel.
def flash_xmx(s):
    if "ik2_flash_xmx.dp.inc" in s: return s
    a = "void attn_flash_f16(const uint16_t *q16, int q_ld, const uint16_t *kc,"
    assert a in s
    s = s.replace(a, '#include "ik2_flash_xmx.dp.inc"\n' + a, 1)
    b = "    if (fa_wgmma())\n        attn_flash_launch<kFaNW, true, uint16_t>"
    assert b in s
    s = s.replace(b, "    if (g_flash_xmx) attn_flash_xmx_launch(q16, q_ld, kc, vc, kv_cap, H, KV, Tq, qpos0, qk_scale, bias, extent, window, n_split, o_part, ml_part, s);\n    else " + b.lstrip(), 1)
    return s
edit("ik2_kernels.dp.cpp", flash_xmx)
edit("ik2_kernels.dp.hpp", lambda s: s if "set_flash_xmx" in s else s.replace("void attn_flash_f16(", "void set_flash_xmx(bool on);   // (Intel) the prefill attention on XMX (default) or the emulated CUDA kernel\nvoid attn_flash_f16(", 1))

# Prefill MoE on XMX: src/ik2_moe_xmx.dp.inc (grouped, NVFP4 dequantized into SLM; gate/up + SwiGLU fused) and its declarations.
def moe_xmx_cpp(s):
    if "ik2_moe_xmx.dp.inc" in s: return s
    a = '#include "ik2_flash_xmx.dp.inc"\n'
    assert a in s
    return s.replace(a, a + '#include "ik2_moe_xmx.dp.inc"\n', 1)
edit("ik2_kernels.dp.cpp", moe_xmx_cpp)
def moe_xmx_hpp(s):
    if "MoeXmxExp" in s: return s
    a = "void set_flash_xmx(bool on);"
    assert a in s
    decl = """// (Intel) prefill MoE products on the matrix engine, straight from NVFP4 expert weights; all hit experts of a layer in one launch
struct MoeXmxExp { const uint8_t *blob; int start, cnt; };   // the expert's weights; its rows in the sorted input / output
void moe_gate_up_swiglu_xmx(const MoeXmxExp *tab, int n_exp, int max_cnt, const uint16_t *A, int C, int F, size_t eb_up, const float *sg, const float *su,
                            float gcap, float ucap, uint16_t *h, dpct::queue_ptr s);
void moe_down_xmx(const MoeXmxExp *tab, int n_exp, int max_cnt, const uint16_t *A, int C, int F, size_t off_down, float *y, dpct::queue_ptr s);
"""
    return s.replace(a, decl + a, 1)
edit("ik2_kernels.dp.hpp", moe_xmx_hpp)

# Prefill MoE (batched path) on XMX when the layer's three expert matrices are NVFP4: one grouped launch for gate/up + SwiGLU, one for down, instead of
# decode_f16 + gemm16 per expert.  IK2_MOE_XMX=0 keeps the per-expert path (always used for other quant types).
def moe_xmx_main(s):
    if "moe_gate_up_swiglu_xmx" in s: return s
    old = """            for (int e : X.hit_exps) {
                const size_t offset = W.start[e];
                gate_up_product(e, sorted_input + offset * C, X.gu32 + offset * 2 * F);
            }
            k::swiglu_cap_f32(X.gu32, S, F, X.sg, X.su, c.gate_cap, c.up_cap, sorted_input, g_s);
            for (int e : X.hit_exps) {
                const size_t offset = W.start[e];
                down_product(e, sorted_input + offset * F, sorted_out + offset * C);
            }
"""
    new = """            static const bool g_moe_xmx = !(getenv("IK2_MOE_XMX") && !atoi(getenv("IK2_MOE_XMX")));
            if (g_moe_xmx && Ly.ty[0] == k::kNVFP4 && Ly.ty[1] == k::kNVFP4 && Ly.ty[2] == k::kNVFP4 && C % 64 == 0 && F % 64 == 0) {
                static k::MoeXmxExp* d_tab = nullptr;   // (the hit experts: weights, rows)
                if (!d_tab) d_tab = (k::MoeXmxExp*) sycl::malloc_device(sizeof(k::MoeXmxExp) * c.E, g_s->get_device(), g_s->get_context());
                std::vector<k::MoeXmxExp> tab;
                int max_cnt = 0;
                for (int e : X.hit_exps) { tab.push_back({X.slot_ptrs[e], W.start[e], W.cnt[e]}); max_cnt = std::max(max_cnt, (int) W.cnt[e]); }
                g_s->memcpy(d_tab, tab.data(), tab.size() * sizeof(k::MoeXmxExp)).wait();   // (the vector dies here: wait)
                uint16_t* hid = (uint16_t*) X.gu32;   // (the hidden, F16 [S][F]: the gate/up output's room, free now; not the input's, other tiles still read it)
                k::moe_gate_up_swiglu_xmx(d_tab, (int) tab.size(), max_cnt, sorted_input, C, F, Ly.eb[0], X.sg, X.su, c.gate_cap, c.up_cap, hid, g_s);
                k::moe_down_xmx(d_tab, (int) tab.size(), max_cnt, hid, C, F, Ly.eb[0] + Ly.eb[1], sorted_out, g_s);
            } else {
""" + "\n".join(("    " + l if l else l) for l in old.rstrip("\n").split("\n")) + "\n            }\n"
    assert old in s, "batched MoE loops not found"
    return s.replace(old, new, 1)
edit("ik2_main.dp.cpp", moe_xmx_main)

# The XMX MoE path needs the batched prefill's sorted layout (tokens grouped by expert) but not its scratch (the attention's logits buffer, empty with
# flash attention): it brings its own sorted input / output buffers, and turns the batched path on by itself (IK2_MOE_XMX=0: the old behaviour).
def moe_xmx_batch(s):
    if "xmx_moe" in s: return s
    old = """        const bool batch = batch_prefill && d == -1 && !Ly.tab_h.empty() && n_miss == 0 && T >= 64 && C % 8 == 0 &&
                           (size_t)S * (2 * std::max(C, F) + 4 * C) <= X.lg_cap * 4 &&
                           (size_t)S * 2 * F * sizeof(float) <= X.gu32_bytes;
"""
    new = """        static const bool moe_xmx_on = getenv("IK2_MOE_XMX") && atoi(getenv("IK2_MOE_XMX"));   // (opt-in: measured slower end to end than decode_f16 + gemm16)
        static const bool batch_own = !(getenv("IK2_BATCH_PREFILL") && !atoi(getenv("IK2_BATCH_PREFILL")));   // (the batched path, with its own buffers)
        const bool xmx_moe = moe_xmx_on && Ly.ty[0] == k::kNVFP4 && Ly.ty[1] == k::kNVFP4 && Ly.ty[2] == k::kNVFP4 && C % 64 == 0 && F % 64 == 0;
        const bool batch = d == -1 && !Ly.tab_h.empty() && n_miss == 0 && T >= 64 && C % 8 == 0 &&
                           (size_t)S * 2 * F * sizeof(float) <= X.gu32_bytes &&
                           (xmx_moe || batch_own || (batch_prefill && (size_t)S * (2 * std::max(C, F) + 4 * C) <= X.lg_cap * 4));
"""
    assert old in s
    s = s.replace(old, new, 1)
    old2 = """            uint16_t* sorted_input = (uint16_t*)X.lg;
            float* sorted_out = (float*)(sorted_input + (size_t)S * std::max(C, F));
"""
    new2 = """            uint16_t* sorted_input = (uint16_t*)X.lg;
            float* sorted_out = (float*)(sorted_input + (size_t)S * std::max(C, F));
            if (xmx_moe || batch_own) {   // (own buffers, grown when a bigger chunk comes: [S][C] F16 in, [S][C] F32 out)
                static uint16_t* si = nullptr; static float* so = nullptr; static size_t cap = 0;
                if ((size_t) S * C > cap) {
                    g_s->wait();
                    if (si) sycl::free(si, g_s->get_context()), sycl::free(so, g_s->get_context());
                    cap = (size_t) S * C;
                    si = (uint16_t*) sycl::malloc_device(cap * 2, g_s->get_device(), g_s->get_context());
                    so = (float*) sycl::malloc_device(cap * 4, g_s->get_device(), g_s->get_context());
                }
                sorted_input = si, sorted_out = so;
            }
"""
    assert old2 in s
    s = s.replace(old2, new2, 1)
    # the earlier fixup's own decision becomes the same flag
    s = s.replace("""            static const bool g_moe_xmx = !(getenv("IK2_MOE_XMX") && !atoi(getenv("IK2_MOE_XMX")));
            if (g_moe_xmx && Ly.ty[0] == k::kNVFP4 && Ly.ty[1] == k::kNVFP4 && Ly.ty[2] == k::kNVFP4 && C % 64 == 0 && F % 64 == 0) {""", """            if (xmx_moe) {""", 1)
    return s
edit("ik2_main.dp.cpp", moe_xmx_batch)

# The XMX MoE path takes its activations in the "pair" column order (see ik2_moe_xmx.dp.inc: position 2q = column q, 2q+1 = column q+8 of each 16): the
# gather that builds its sorted input permutes.
def gather_perm(s):
    if "gather_perm16_kernel" in s: return s
    a = "__dpct_inline__ void gather_f16_16_kernel("
    assert a in s
    kern = """// (Intel) the XMX MoE path's sorted input: rows gathered, columns in the pair order (word i = columns (q, q + 8) of group i / 8, q = i % 8)
__dpct_inline__ void gather_perm16_kernel(const uint16_t *__restrict__ x16, const int *__restrict__ idx, int n, int C, uint16_t *__restrict__ y16) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2);
    if (row >= n) return;
    const uint16_t *src = x16 + (size_t) idx[row] * C;
    uint32_t *dst = (uint32_t *) (y16 + (size_t) row * C);
    for (int i = item_ct1.get_local_id(2); i < C / 2; i += item_ct1.get_local_range(2)) {
        const int g = i >> 3, q = i & 7;
        dst[i] = (uint32_t) src[g * 16 + q] | ((uint32_t) src[g * 16 + q + 8] << 16);
    }
}

"""
    s = s.replace(a, kern + a, 1)
    pat = re.compile(r"gather_f16_16_kernel\(xin, X_gidx_ct1, S, C,\s*sorted_input\);")
    s, n = pat.subn("if (xmx_moe) gather_perm16_kernel(xin, X_gidx_ct1, S, C, sorted_input);\n                            else gather_f16_16_kernel(xin, X_gidx_ct1, S, C, sorted_input);", s, count=1)
    assert n == 1, "batched gather call not found"
    return s
edit("ik2_main.dp.cpp", gather_perm)

# (Intel) the 256-register kernel property used by the XMX MoE kernels
def grf_include(s):
    if "grf_size_properties.hpp" in s: return s
    return s.replace("#include <algorithm>\n", "#include <algorithm>\n#include <sycl/ext/intel/experimental/grf_size_properties.hpp>\n", 1)
edit("ik2_kernels.dp.cpp", grf_include)

# NVFP4 weight decode, store-coalesced (src/ik2_decode_lut.dp.inc: a thread takes half a sub-block, one 16-byte store); opt-in with IK2_DECODE_LUT=1.
def decode_lut(s):
    if "ik2_decode_lut.dp.inc" in s: return s
    a = "void decode_f16(int type, const uint8_t *blocks, size_t n, uint16_t *out,\n                dpct::queue_ptr s) {\n    if (!n) return;\n"
    assert a in s
    s = s.replace(a, '#include "ik2_decode_lut.dp.inc"\n' + a + "    if (type == kNVFP4 && g_decode_lut && n % 16 == 0 && (((uintptr_t) blocks & 3) == 0) && (((uintptr_t) out & 15) == 0)) {\n        decode_f16_nvfp4_half(blocks, n, out, s);\n        return;\n    }\n", 1)
    return s
edit("ik2_kernels.dp.cpp", decode_lut)
edit("ik2_kernels.dp.hpp", lambda s: s if "set_decode_lut" in s else s.replace("void set_flash_xmx(bool on);", "void set_decode_lut(bool on);   // (Intel) NVFP4 -> FP16 decode, store-coalesced (opt-in: IK2_DECODE_LUT=1) or the original kernel (default)\nvoid set_flash_xmx(bool on);", 1))

# The batched prefill's per-expert decode launches (2 a hit expert) become one launch a group of experts, each decoded into its own FP16 buffer, then each
# expert's GEMM.  IK2_DECODE_GROUP=N groups N experts a launch (default 1, 16 at most): measured slower, the decoded matrices of a group leave the L2 before their GEMMs read them.  Only for NVFP4 layers (the others keep the per-expert path).
def decode_group_hpp(s):
    if "decode_f16_nvfp4_multi" in s: return s
    return s.replace("void set_decode_lut(bool on);", "void decode_f16_nvfp4_multi(const uint8_t *const *blobs, int ne, size_t n, uint16_t *out, size_t out_stride, dpct::queue_ptr s);   // (Intel) ne matrices in one launch (ne <= 16)\nvoid set_decode_lut(bool on);", 1)
edit("ik2_kernels.dp.hpp", decode_group_hpp)
def decode_group_main(s):
    if "dec_group" in s: return s
    old = """            } else {
                for (int e : X.hit_exps) {
                    const size_t offset = W.start[e];
                    gate_up_product(e, sorted_input + offset * C, X.gu32 + offset * 2 * F);
                }
                k::swiglu_cap_f32(X.gu32, S, F, X.sg, X.su, c.gate_cap, c.up_cap, sorted_input, g_s);
                for (int e : X.hit_exps) {
                    const size_t offset = W.start[e];
                    down_product(e, sorted_input + offset * F, sorted_out + offset * C);
                }
            }
"""
    new = """            } else {
                static const int dec_group = std::min(16, std::max(1, getenv("IK2_DECODE_GROUP") ? atoi(getenv("IK2_DECODE_GROUP")) : 1));   // (opt-in: the decoded weights then leave the L2: slower)
                const bool grouped = dec_group > 1 && Ly.ty[0] == k::kNVFP4 && Ly.ty[1] == k::kNVFP4 && Ly.ty[2] == k::kNVFP4 && Ly.ty[0] == Ly.ty[1];
                static uint16_t *gw = nullptr, *dw = nullptr;   // (a group's decoded matrices: gate+up [G][2F][C], down [G][C][F])
                if (grouped && !gw) {
                    gw = (uint16_t*) sycl::malloc_device((size_t) dec_group * 2 * F * C * 2, g_s->get_device(), g_s->get_context());
                    dw = (uint16_t*) sycl::malloc_device((size_t) dec_group * C * F * 2, g_s->get_device(), g_s->get_context());
                }
                const int nhit = (int) X.hit_exps.size();
                if (grouped) {
                    for (int i = 0; i < nhit; i += dec_group) {
                        const int g = std::min(dec_group, nhit - i);
                        const uint8_t* bl[16];
                        for (int j = 0; j < g; ++j) bl[j] = X.slot_ptrs[X.hit_exps[i + j]];
                        k::decode_f16_nvfp4_multi(bl, g, (size_t) 2 * F * C, gw, (size_t) 2 * F * C, g_s);
                        for (int j = 0; j < g; ++j) {
                            const int e = X.hit_exps[i + j];
                            const size_t offset = W.start[e];
                            gemm16(gw + (size_t) j * 2 * F * C, 2 * F, C, sorted_input + offset * C, W.cnt[e], X.gu32 + offset * 2 * F, 2 * F);
                        }
                    }
                } else {
                    for (int e : X.hit_exps) {
                        const size_t offset = W.start[e];
                        gate_up_product(e, sorted_input + offset * C, X.gu32 + offset * 2 * F);
                    }
                }
                k::swiglu_cap_f32(X.gu32, S, F, X.sg, X.su, c.gate_cap, c.up_cap, sorted_input, g_s);
                if (grouped) {
                    for (int i = 0; i < nhit; i += dec_group) {
                        const int g = std::min(dec_group, nhit - i);
                        const uint8_t* bl[16];
                        for (int j = 0; j < g; ++j) bl[j] = X.slot_ptrs[X.hit_exps[i + j]] + Ly.eb[0] + Ly.eb[1];
                        k::decode_f16_nvfp4_multi(bl, g, (size_t) C * F, dw, (size_t) C * F, g_s);
                        for (int j = 0; j < g; ++j) {
                            const int e = X.hit_exps[i + j];
                            const size_t offset = W.start[e];
                            gemm16(dw + (size_t) j * C * F, C, F, sorted_input + offset * F, W.cnt[e], sorted_out + offset * C, C);
                        }
                    }
                } else {
                    for (int e : X.hit_exps) {
                        const size_t offset = W.start[e];
                        down_product(e, sorted_input + offset * F, sorted_out + offset * C);
                    }
                }
            }
"""
    assert old in s, "batched expert loops not found"
    return s.replace(old, new, 1)
edit("ik2_main.dp.cpp", decode_group_main)

# (Intel) a bigger prefill chunk by default: every expert is decoded once a layer visit per chunk, so 4096 instead of 2048 tokens a chunk takes ~10% off a long
# prompt's prefill (12,000 tokens: 9.4 -> 8.5 s) for ~0.3 GiB more at a 131k context.
def chunk_default(s):
    s = s.replace("int gen = 16, chunk = 2048,", "int gen = 16, chunk = 4096,", 1)
    s = s.replace("[--chunk 2048]", "[--chunk 4096]", 1)
    return s
edit("ik2_main.dp.cpp", chunk_default)

# Two-stream pipeline for the batched prefill's experts: even-indexed hit experts decode + GEMM on the main queue, odd ones on a second queue (own FP16 buffer, own BLAS
# handle), so one expert's bandwidth-bound decode overlaps another's GEMM and small GEMMs share the matrix engine.  Fork / join with barriers.  IK2_EXPERT_PIPE=1: on (default off: measured slower, the two decoded buffers do not fit the L2 together).
def pipe_gemm16(s):
    if "gemm16_on" in s: return s
    a = "uint8_t* g_stage = nullptr;   // a raw tensor on its way to its FP16 decode"
    assert a in s
    fn = """// gemm16 on another BLAS handle (its own queue)
void gemm16_on(dpct::blas::descriptor_ptr h, const uint16_t *W, int N, int Kd, const uint16_t *X, int T, float *Y, int ldy) {
    const float one = 1.0f, zero = 0.0f;
    CB(DPCT_CHECK_ERROR(dpct::blas::gemm(
        h, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, N, T, Kd, &one, W, dpct::library_data_t::real_half, Kd, X,
        dpct::library_data_t::real_half, Kd, &zero, Y, dpct::library_data_t::real_float, ldy, dpct::compute_type::f32)));
}

"""
    return s.replace(a, fn + a, 1)
edit("ik2_main.dp.cpp", pipe_gemm16)
def pipe_main(s):
    if "expert_pipe" in s: return s
    old_a = """                } else {
                    for (int e : X.hit_exps) {
                        const size_t offset = W.start[e];
                        gate_up_product(e, sorted_input + offset * C, X.gu32 + offset * 2 * F);
                    }
                }
"""
    new_a = """                } else if (expert_pipe) {
                    run_pipe([&](dpct::queue_ptr q, dpct::blas::descriptor_ptr hb, uint16_t* wb, int e) {
                        const size_t offset = W.start[e];
                        const uint8_t* blob = X.slot_ptrs[e];
                        if (Ly.ty[0] == Ly.ty[1]) k::decode_f16(Ly.ty[0], blob, (size_t) 2 * F * C, wb, q);
                        else {
                            k::decode_f16(Ly.ty[0], blob, (size_t) F * C, wb, q);
                            k::decode_f16(Ly.ty[1], blob + Ly.eb[0], (size_t) F * C, wb + (size_t) F * C, q);
                        }
                        gemm16_on(hb, wb, 2 * F, C, sorted_input + offset * C, W.cnt[e], X.gu32 + offset * 2 * F, 2 * F);
                    });
                } else {
                    for (int e : X.hit_exps) {
                        const size_t offset = W.start[e];
                        gate_up_product(e, sorted_input + offset * C, X.gu32 + offset * 2 * F);
                    }
                }
"""
    assert old_a in s, "gate/up loop not found"
    s = s.replace(old_a, new_a, 1)
    old_b = """                } else {
                    for (int e : X.hit_exps) {
                        const size_t offset = W.start[e];
                        down_product(e, sorted_input + offset * F, sorted_out + offset * C);
                    }
                }
"""
    new_b = """                } else if (expert_pipe) {
                    run_pipe([&](dpct::queue_ptr q, dpct::blas::descriptor_ptr hb, uint16_t* wb, int e) {
                        const size_t offset = W.start[e];
                        k::decode_f16(Ly.ty[2], X.slot_ptrs[e] + Ly.eb[0] + Ly.eb[1], (size_t) C * F, wb, q);
                        gemm16_on(hb, wb, C, F, sorted_input + offset * F, W.cnt[e], sorted_out + offset * C, C);
                    });
                } else {
                    for (int e : X.hit_exps) {
                        const size_t offset = W.start[e];
                        down_product(e, sorted_input + offset * F, sorted_out + offset * C);
                    }
                }
"""
    assert old_b in s, "down loop not found"
    s = s.replace(old_b, new_b, 1)
    old_c = "                const int nhit = (int) X.hit_exps.size();\n"
    new_c = """                const int nhit = (int) X.hit_exps.size();
                static const bool expert_pipe_on = getenv("IK2_EXPERT_PIPE") && atoi(getenv("IK2_EXPERT_PIPE"));   // opt-in: two streams double the FP16 working set past the L2 (slower)
                const bool expert_pipe = expert_pipe_on && !grouped;
                static dpct::queue_ptr pq = nullptr; static dpct::blas::descriptor_ptr pblas = nullptr; static uint16_t* pw16 = nullptr;
                if (expert_pipe && !pq) {   // (the second stream: its queue, BLAS handle and FP16 buffer, made on first use)
                    pq = dpct::get_current_device().create_queue(true);
                    pblas = new dpct::blas::descriptor();
                    pblas->set_queue(pq);
                    pblas->set_math_mode(dpct::blas::math_mode::mm_default);
                    pw16 = (uint16_t*) sycl::malloc_device((size_t) 2 * F * C * 2, g_s->get_device(), g_s->get_context());
                }
                // the hit experts, alternating between the main stream and the second: fork after the main queue's work, join before its next kernel
                auto run_pipe = [&](auto&& one) {
                    const sycl::event fork = g_s->ext_oneapi_submit_barrier();
                    pq->ext_oneapi_submit_barrier({fork});
                    int idx = 0;
                    for (int e : X.hit_exps) {
                        if (idx++ & 1) one(pq, pblas, pw16, e); else one(g_s, g_blas, X.w16, e);
                    }
                    const sycl::event done = pq->ext_oneapi_submit_barrier();
                    g_s->ext_oneapi_submit_barrier({done});
                };
"""
    assert old_c in s, "nhit line not found"
    s = s.replace(old_c, new_c, 1)
    return s
edit("ik2_main.dp.cpp", pipe_main)
