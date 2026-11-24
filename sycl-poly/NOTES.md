# strata-poly on Intel GPUs (SYCL) - experimental

Builds `strata-poly` (the Qwen3.5/3.6, GLM, Gemma, DeepSeek, LFM2 engine) for SYCL. Separate from `../sycl/` (the Strata
`strata` engine port, which is stale against this tree: ~700 merge conflicts).

    source /opt/intel/oneapi/setvars.sh   # icpx 2026.0 and oneMKL
    git submodule update --init third_party/llama.cpp
    cmake -S sycl-poly -B build-poly -G Ninja -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx && ninja -C build-poly strata-poly
    ONEAPI_DEVICE_SELECTOR=level_zero:gpu build-poly/strata-poly --model Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --tokens ... --no-graphs

## Status (2026-11-23, Arc Pro B60 24 GB, Level Zero V2, icpx 2026.0)
- Qwen3.6-35B-A3B UD-Q4_K_XL (Unsloth): all 10,240 experts in VRAM (18.3 GiB); the answer to "capital of France" is right
  (token-for-token "The capital of France is Paris." + <|im_end|>).
- Decode 60-61 tok/s (`--no-draft --no-graphs`, 226 tokens, 3 runs); 49 tok/s with the draft head (83/125 guessed right: the
  window-8 verify is the slow part here, see gaps). Prompt: 35-90 tok/s on 24-29 tokens (not a prefill benchmark).
- One `UR_RESULT_ERROR_DEVICE_LOST` in ~9 runs (first run of a config, not reproduced in 3 retries): open.
- Only Qwen3.6 was run. GLM / Gemma / DeepSeek / LFM2 compile, are untested.

## What is not there (the work left)
1. `Mmq` (src/poly/mmq.hpp, llama.cpp's ggml-cuda tiles) is `mmq_sycl.cpp`: fits() is false, so the prompt path takes its other
   products. A real one: XeStrata's iq_mmq / dp4a_gemm / xmx_gemm are the model. This is the prefill speed.
2. `tensor_cores()` is false: the mma/ldmatrix/cp.async kernels (NVIDIA PTX) do not run; joint_matrix (XMX) versions missing.
3. CUDA graphs (`record()` finalizes a SYCL graph, never exercised): run with `--no-graphs`.
4. `iq_kernels.dp.cpp`: the grouped-expert kernels of the Strata engine (S26/S27/expdec/native_expert_*) are `#if 0`;
   IQ4_NL / MXFP4 / NVFP4 codebooks are private constants; the NVFP4 dequant/embed launchers pass nullptr (TODO: NVFP4 files).
5. `pdl.hpp` is a no-op (NVIDIA programmatic dependent launch).
6. Multi-GPU (`--layer-split`), peer copies, the image/vision path: not looked at.
7. Hand fixes were made to dpct output (not generated): see git history of sycl-poly/; re-migrating needs them re-applied
   (`tools/migrate_poly.sh`, `tools/fix_kernel_names.py` cover the mechanical part).

## Findings worth keeping
- Level Zero virtual memory works (dpct maps cuMem* to sycl_ext_oneapi_virtual_mem) but physical allocations must be a whole
  number of 2 MiB although `get_mem_granularity` says 64 KiB.
- dpct's DPCT1084 ("could not unify") drops constant-table arguments; the cure here is a private `alignas(16) const int8_t[16]`.
- dpct names the kernel of a templated `*_wrapper` without its template arguments: integration-header "redefinition of
  KernelInfo" errors. `tools/fix_kernel_names.py` fixes it.
- Merge base of the older `../sycl/` port is Strata v0.1.39 (not 0.1.40.2).
