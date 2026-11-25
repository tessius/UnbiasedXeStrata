# sycl-ik2: the Inkling 2 Mini engine on Intel GPUs

UnbiasedStrata's `strata-ik2` (Project Maya's Inkling engine, CUDA) migrated to SYCL: it runs **Inkling-2-Mini** from Unsloth's GGUF on an Intel Arc GPU
with all 3,072 experts (14.7 GiB) resident in VRAM.  Same front end, same OpenAI-style API as UnbiasedStrata's `--engine strata-ik2`.

> **Experimental.**  Written and tested on one machine on **2026-11-24/25**: an Intel Arc Pro B60 (24 GB), one model (`Inkling-2-Mini-UD-Q4_K_XL`).
> Other Intel GPUs, other quantizations, multi-GPU, the speculative drafter, image input and CUDA-graph replay are untested.

## Measured (Arc Pro B60, AOT build, 2026-11-25)

| | |
|---|---|
| Prefill | ~1,900-2,000 tok/s (4,000-12,000-token prompts) |
| Decode | ~56 tok/s on a short context, ~47 after a 4,000-token prompt, ~44 after 12,000 |
| Correctness | 10 of 10 needle-in-a-haystack answers right at ~4k and ~7.5k tokens; ten varied prompts at temperature 0 give the same answers as the reference build made before the decode-kernel tuning (`../sycl-poly/docs/validate/ab_greedy.py`); kernel tests against references pass |

Numbers drift about 3% between sessions on this card; the DEVLOG always compares against a baseline from the same session.

## Model files

Use Unsloth's GGUF from the **`Old_Support/`** folder of [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF): the engine reads the `inkling2.*`
tensor layout of those files (they are stored as `*.gguf_file`: rename them to `*.gguf`).  The files in the repository's root use llama.cpp master's newer layout and are **not**
loadable.  `Inkling-2-Mini-UD-Q4_K_XL.gguf` (17.6 GB) is the one tested; `mmproj-BF16.gguf` is the vision projector (untested here).

## Build and run

You need Intel oneAPI (`icpx`, oneMKL), the Intel GPU driver (Level Zero) and CMake.  No CUDA.  From the repository root:

```
make intel          # builds the front end, strata-poly and strata-ik2 into build-intel/bin/ (AOT for the card it detects)
build-intel/bin/unbiased-strata --engine strata-ik2 -m Inkling-2-Mini-UD-Q4_K_XL.gguf --alias inkling-2-mini --port 8080 \
    -- --model Inkling-2-Mini-UD-Q4_K_XL.gguf --max-context 8192
```

Then use the API on port 8080 as with UnbiasedStrata (set `"reasoning_effort":"none"` for a direct answer).  A start reads the 17.6 GB file and copies the experts to VRAM (the copy itself is a few seconds; the file read depends on whether it is in the disk cache).
The defaults are the measured best for this card; the switches that matter are listed below.  `make intel-test` runs the kernel tests.

## What is here

| | |
|---|---|
| `src/ik2_*.dp.cpp`, `.dp.hpp` | the engine as dpct migrated it (from `../src/ik2/`), with our fixes applied by `tools/fixups_ik2.py` (idempotent: regenerate with `tools/migrate_ik2.sh`) |
| `src/ik2_mma_emul.dp.inc` | NVIDIA's `mma.sync` / `ldmatrix` emulated with sub-group shuffles (decode attention and the engine's other tensor-core paths; slow, so the fast paths below replace them) |
| `src/ik2_flash_xmx.dp.inc` | prefill attention on the Intel matrix engine (`joint_matrix`), 27 sub-groups a work-group, 256 registers |
| `src/ik2_moe_xmx.dp.inc` | a grouped NVFP4 expert GEMM for prefill: correct, but slower than the default (opt-in `IK2_MOE_XMX=1`) |
| `src/ik2_limb_rp.dp.inc` | the NVFP4 expert kernel for decode with 4 row pairs a warp |
| `src/ik2_small.dp.inc`, `src/ik2_decode_lut.dp.inc` | small decode kernels rewritten for latency; an opt-in decode kernel |
| `tests/` | `flash_xmx_test`, `moe_xmx_test`, `small_bench`, `rms_test`, `mma_emul_test`, `experts_down_test` (against references), `decode_experts_bench`, `decode_bench` (timings) |

## Switches (environment variables)

| | default | |
|---|---|---|
| `IK2_EXPERT_LIMB` | `3` | NVFP4 experts by int8 dot products over a two-limb copy of the activations (1 gate/up, 2 down); `0` = the generic float kernels |
| `IK2_LIMB_RP` | `4` | row pairs a warp in that kernel (1, 2, 4) |
| `IK2_ATTN_MMA` | `0` | decode attention: `0` the scalar kernels (faster here), `1` the emulated tensor-core kernel |
| `IK2_GATE_TILE` | `64` | the loop gates' fixed-rank apply kernel |
| `IK2_FLASH_XMX` | `1` | prefill attention on the matrix engine; `0` the emulated kernel |
| `IK2_BATCH_PREFILL` | `1` | the batched expert prefill with its own buffers |
| `IK2_SMALL_WIDE` | `1` | `mv_f16` with 16-byte loads |
| `--chunk` | `4096` | prefill tokens a pass |
| `IK2_SYNC_CHECK` | off | `1` prints each kernel as it finishes; `2` prints a per-kernel time table at exit (it syncs after each kernel: small kernels look slower than they are) |

## Licenses

MIT, like the rest of the tree: the engine is UnbiasedStrata's / Project Maya's (MIT) migrated mechanically; the kernels and notes we added carry `SPDX-License-Identifier: MIT`.  Nothing in `sycl-ik2/` is
XeStrata's code (the LGPL parts live in `../sycl-poly/src/xe/` and are not linked here).  The vendored dpct headers are Apache-2.0 WITH LLVM-exception.  See [`../sycl-poly/CREDITS.md`](../sycl-poly/CREDITS.md).

## Notes

The development log (what was tried, what was measured, what failed) is in [`../sycl-poly/DEVLOG.md`](../sycl-poly/DEVLOG.md), sections 10 and 11-27.
