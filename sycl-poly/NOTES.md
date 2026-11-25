# Unbiased XeStrata: strata-poly on Intel GPUs (SYCL) - experimental

See README.md (overview), DEVLOG.md (how it was made), CREDITS.md (who wrote what, licenses).

Builds `strata-poly` (the Qwen3.5/3.6, GLM, Gemma, DeepSeek, LFM2 engine) for SYCL. Separate from `../sycl/` (the Strata
`strata` engine port, which is stale against this tree: ~700 merge conflicts).

    source /opt/intel/oneapi/setvars.sh   # icpx 2026.0 and oneMKL
    git submodule update --init third_party/llama.cpp
    # ahead-of-time for an Arc Pro B60 / B580 (bmg-g21): starts in seconds.  Without STRATA_SYCL_AOT the GPU code is JIT-compiled
    # at run time: 2-4 minutes at every start for the MMQ kernels.
    cmake -S sycl-poly -B build-poly -G Ninja -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DSTRATA_SYCL_AOT=bmg-g21
    ninja -C build-poly strata-poly
    ONEAPI_DEVICE_SELECTOR=level_zero:gpu build-poly/strata-poly --model Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --tokens ... --no-graphs

## Status (2026-11-23, Arc Pro B60 24 GB, Level Zero V2, icpx 2026.0)
Qwen3.6-35B-A3B UD-Q4_K_XL (Unsloth), all 10,240 experts in VRAM (18.3 GiB), `--no-graphs`:
- **Correct** (checked token for token): a teacher-forced counting sequence at 10 positions from 50 to 190; "capital of France";
  numbered-facts retrieval at 333 and 2,709 tokens ("What is Fact 42?" gives Fact 42 exactly); "count from 1 to 200" gives 1..200 and stops.
- **Prefill 447 tok/s** (2,710-token prompt; 178 with the stub `Mmq` before, dp4a tiles now).
- **Decode** 82 tok/s on short answers; 49 tok/s on a 472-token story (`--no-draft`), 69 with the draft head, ~100 on counting.
- Only Qwen3.6 was run. GLM / Gemma / DeepSeek / LFM2 compile, are untested.

## Status (2026-11-25): clean build from a fresh clone, and the Inkling engine
- `git clone` of the committed branch + `make intel` (AOT for the Arc Pro B60, about 20 minutes with `-j8`) builds the front end, `strata-poly` and `strata-ik2` into `build-intel/bin/`; `make intel-test` passes
  (the three poly tests and the six `sycl-ik2` tests).  Logs: `docs/logs/clean-clone-*-2026-11-25.log`.
- From those binaries: Inkling-2-Mini through `--engine strata-ik2` (by name): 10 of 10 retrievals at ~4k and ~7.5k tokens, prefill ~1,900 tok/s, decode 47-57 tok/s; Qwen3.6 through `--engine strata-poly`: 3 of 3 retrievals at 2.6k tokens.
- **Not reconciled:** that Qwen run measured prefill ~380 tok/s and decode 33 tok/s (80-token answer, no draft head) *through the front end*, against the 447 tok/s and 49-82 tok/s recorded on 2026-11-23 (engine run directly).  The card's clocks drift ~3% between
  sessions and the front end adds work, but that does not explain 15%; not investigated.  Treat the 2026-11-23 numbers as unverified until re-measured the same way.
- Inkling status, switches and measurements: [`../sycl-ik2/README.md`](../sycl-ik2/README.md).  The other UnbiasedStrata engines (`strata` Flash-Next, `strata-dsv41`, `hy4flash`) are not ported.

## Unit tests (build targets, run by hand; each compares a kernel with a CPU reference)
`poly_gdn_q_test` (gated delta rule, 300 stateful steps), `poly_attend_q_test` (attend(), head 256, 16 context lengths),
`poly_rope_q_test` (head_norm_rope, 700 positions).  The gdn test is what found the bug below.

## Bugs found in dpct's output (all fixed)
- `__fmul_rn(__fsub_rn(a, b), c)` became `a - b * c` (DPCT1013): GDN's `d = (va - sk) * sa`.  The recurrent state was wrong from the
  second token, so every answer broke after ~90 tokens.  Checked every other DPCT1013 site: no other nesting error.
- `dpct::experimental::*_sub_group*(0xffffffff, ...)` (masked shuffles) replaced by the unmasked `dpct::*` forms.
- Level Zero physical memory must be a whole number of 2 MiB although `get_mem_granularity` reports 64 KiB.
- dpct names a templated wrapper's kernel without its template arguments (`tools/fix_kernel_names.py`), drops constant-table
  arguments (DPCT1084; private `alignas(16)` tables instead) and mis-expands macros (DPCT1064).

## `Mmq` (prefill)
`src/poly/mmq_sycl.cpp` on XeStrata's SYCL port of llama.cpp's quantized matrix tiles (`src/xe/`, LGPL-3.0-or-later AND MIT, see
src/xe/README.md): dp4a today.  The Intel XMX path is built but rejected at start ("the GPU's int8 XMX accumulators are laid out
otherwise"): the accumulator layout probe fails with icpx 2026.0 on the B60; XeStrata builds with intel/llvm 7.2 and may
differ.  XMX is the next prefill gain.  The AOT build drops the Xe-HPG and generic matrix variants (`STRATA_MMQ_NO_XE_HPG_SHAPE`).

## What is not there
1. XMX for `Mmq` (above); `tensor_cores()` is false (the NVIDIA mma/ldmatrix/cp.async kernels are not ported).
2. CUDA graphs (`record()` finalizes a SYCL graph, never exercised): run with `--no-graphs`.
3. `iq_kernels.dp.cpp`: the Strata engine's grouped-expert kernels are `#if 0`; the NVFP4 dequant/embed launchers pass nullptr
   for the codebook (NVFP4 files need it threaded).
4. `pdl.hpp` is a no-op (NVIDIA programmatic dependent launch).  Multi-GPU, peer copies, images: not looked at.
5. One `UR_RESULT_ERROR_DEVICE_LOST` every few runs, usually in the XMX layout probe at start or the first launch of a run (the
   process then hangs until its timeout).  Not understood; a retry works.
6. Hand fixes were made to dpct output: re-migrating needs them re-applied (`tools/migrate_poly.sh`, `tools/fix_kernel_names.py`).
