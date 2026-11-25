# Unbiased XeStrata: development log

How the Intel (SYCL) port of UnbiasedStrata's `strata-poly` engine was made, in the order it happened, with the versions,
commands, numbers, dead ends and mistakes.  Date of all work: 2026-11-23 (one session).  Raw outputs: `docs/logs/`.
Where a number was later found to come from a broken engine, it says so; those numbers are kept because they are part of the
record, not because they are good.

## 0. The goal and the name

Goal: run UnbiasedStrata (https://github.com/robertkarl/UnbiasedStrata, a CUDA/Metal fork of PolyStrata/Strata) on an Intel Arc
GPU, using the Intel support that XeStrata (https://github.com/MistVVK/XeStrata, a SYCL port of Strata) already has.  The
project is called **Unbiased XeStrata**.  Credits and licenses: `CREDITS.md`.

## 1. Environment

| | |
|---|---|
| GPU | Intel Arc Pro B60, 24 GB (Battlemage, `bmg-g21`), PCIe Gen3 x8 |
| CPU / RAM | AMD Ryzen 7 3700X (AVX2, no AVX-512), 31 GB |
| OS | CachyOS (Arch), Linux 7.3.5-2-cachyos |
| GPU stack | Level Zero V2 adapter, compute-runtime (NEO) 26.40.40196, `sycl-ls` reports `Level-Zero V2 ... Arc(TM) Pro B60 Graphics 20.1.0 [1.18.40196]` |
| Compiler used for sycl-poly | Intel oneAPI DPC++/C++ `icpx` 2026.0.0.20260331, oneMKL 2026.0, `ocloc` (system) |
| Migration tool | Intel DPC++ Compatibility Tool (dpct), package `intel-oneapi-dpcpp-ct-2025.3-2025.3.1-18_amd64.deb` from Intel's apt repository, extracted with `ar`+`tar` (no install); `dpct --version` prints `2025.3.0 ... clang 21.0.0`; helper headers vendored in `../sycl/include/dpct` |
| CUDA headers for dpct to parse | the pip wheels `nvidia-cuda-runtime-cu12==12.8.90 nvidia-cuda-nvcc-cu12==12.8.93 nvidia-cublas-cu12==12.8.4.1 nvidia-cuda-cccl-cu12==12.8.90 nvidia-curand-cu12==10.3.9.90`, unpacked by `../sycl/tools/get-cuda-headers.sh` (headers only; no GPU or toolkit) |
| Compiler XeStrata requires | intel/llvm v7.2.0 (`71767974d44acf0464ebf4b26c2059781eb7d6ee`), built by XeStrata's `tools/intel_llvm_build.py --yes --jobs 8` |
| Model | Qwen3.6-35B-A3B, Unsloth `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (22 GB; repo `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`) |

## 2. Sources (exact revisions)

| What | Revision |
|---|---|
| UnbiasedStrata | `9f75192287c2abf88c77da3b4b6f54ccee01db83` (2026-11-22) |
| its llama.cpp submodule | `db1f382bb25f9cc481893105dbead09c86a52f01` (robertkarl/llama.cpp, branch `UnbiasedStrata`) |
| PolyStrata (imported into UnbiasedStrata) | v0.1.45-poly.1 `d0cd84dc6bd009d1d9e2191e4e28cc76b80f1ee3` |
| Strata, base of the older `sycl/` port | v0.1.39 `a1641e9f77aacad4d201b53c8a7ae8fa21059ebb` (also looked at: v0.1.40.2 `e8ca9afd03d839d4f8dbbe82dffce7f8a3bafd7a`) |
| XeStrata | `721afe51f223fe5e60470a06a22c9961c7625fa8` (2026-10-30; integrates upstream Strata 0.2.8) |
| this work | branch `intel-poly-sycl` of the UnbiasedStrata tree |

## 3. What UnbiasedStrata had, and what it did not

- UnbiasedStrata keeps an **unbuilt** `sycl/` directory (the community Intel port, Strata PR #423 by maxfridbe), 239 files, migrated
  from CUDA with dpct at Strata 0.1.39/0.1.40.2.  Its `PLAN.md` says "kept, unbuilt".  It ports only the `strata` engine
  (Qwen3.8-Flash-Next), not `strata-poly`.
- Built as it stood, with icpx 2026.0, `sycl/` configured and the `strata` target linked after two compiler-flag fixes: `-mbmi2`
  (a CPU kernel added since 0.1.39 uses `_pdep_u64`) and `-mavx512vpopcntdq` plus the other AVX-512 flags (`_mm256_popcnt_epi32`).
  **It was never run**: the model it needs (Qwen3.8-Flash-Next, 60-110 GB) was not on the machine.
- XeStrata is not the same code: a separate SYCL/Level Zero rewrite with its own runtime (`core::Runtime`), 54 Xe kernels, and
  its own API.  46 of the 52 kernel headers differ from UnbiasedStrata's, so its kernels cannot be dropped in.

## 4. Attempt 1: refresh the old `sycl/` port to the current tree (abandoned)

Followed the port's own maintenance procedure (`docs/INTEL.md`, "Keeping up with upstream"): dpct the old tree and the new tree,
normalize, `git merge-file` each migrated file into `sycl/`.

1. Merge base.  The port's docs say 0.1.40.2.  Measuring which Strata tag the migrated files match best (identifiers in each
   CUDA original that are missing from the port's copy, summed over the 44 migrated files; `score.py`): v0.1.39 **1,093**, v0.1.40.2
   2,049, v0.1.41 2,541.  So the base is **v0.1.39**.
2. Drift since then: 20,686 changed lines across 44 migrated files (`drift.txt`).
3. dpct 2025.3 over the old tree and over UnbiasedStrata HEAD (`migrate.sh`, about 3 minutes each), `normalize.sh`, then a
   hash-canonicalizing 3-way merge (`tools/merge_upstream.py` adapted to directories): **23 files clean, 46 conflicted, 698
   conflict hunks**; `generate.cpp` alone 127.  74 more files would be new copies of other engines' code (dsv41, ik2, poly).
4. Decision: stopped.  698 hunks cannot be validated without Qwen3.8-Flash-Next weights, and that engine does not run the models
   on this machine.  Nothing from this attempt is in the tree.  (`sycl-poly` re-migrates poly from scratch instead, which needed no merge.)

## 5. Getting XeStrata running as a baseline

- XeStrata refuses oneAPI's icpx ("contrib-icpx mode was removed"); it needs intel/llvm >= 7 (libsycl 9).  Built with
  `tools/intel_llvm_build.py --yes --jobs 8`: about an hour on 8 threads (its doc says 13 minutes on 28), 3.5 GB kept.
  (The script downloads and builds a compiler; read it before you run it.)
- `./setup.sh --family qwen36 --model UD-Q4_K_XL --gguf-dir <dir> --intel-llvm <install> --license free --yes --no-start`: creates a venv,
  builds the engine (206 steps), compiles the GPU code for the model (490 s once), writes a start script.  Running `setup.py`
  directly fails on Arch (PEP 668); `setup.sh` makes the venv.  Its `--inspect` rejects a local Q4_K_M GGUF ("not
  Qwen3.8-Flash-Next"); the file it supports for Qwen3.6 is Unsloth's UD-Q4_K_XL, downloaded for this (22 GB).
- Result: serving worked; a 300-token story request gave **44.1 tok/s** (first, 8 s) and **63.6 tok/s** (second, 5 s) with 98-99% of
  expert lookups hitting VRAM, context 131072.  The model "thought" (all 300 tokens were reasoning tokens), so the text was not read.
  This is XeStrata's own engine, not Unbiased XeStrata; it shows the card and model work.
- fish-speech (a TTS server on :8081 that holds the GPU) was stopped before any engine ran.

## 6. Attempt 2: port `strata-poly` (what became Unbiased XeStrata)

`strata-poly` is self-contained: `src/poly/{main,engine,qwen35,glm5next,gemma4,deepseek4,lfm2moe,cpu_experts}.cpp`, its own CUDA kernels
(`kernels.cu`, 66 `__global__`; `kernels_dsv4.cu`), a wrapper `Mmq` around ggml-cuda's quantized matrix tiles (`mmq.cu`), and one
dependency on Strata's `native_mmvq` kernels.  dpct had already migrated those files in the run of section 4.

### 6.1 Scaffold and compile (commit 84a8329)
- `sycl-poly/CMakeLists.txt`: sources are `../src` unchanged where nothing uses CUDA; dpct's output for the rest; ggml-cpu/base from
  the submodule; `-fsycl -fsycl-default-sub-group-size=32 -fp-model=precise` (the kernels assume 32-lane warps);
  `-qmkl=sequential` (cuBLAS became oneMKL through `dpct::blas`).
- Include order matters: `sycl-poly/include`, then the repo's `include`, and `../sycl/include` last (its `strata/kernels/*` are the
  old port's stale copies and shadowed the current headers; only its `dpct/` and `strata/sycl_queue.hpp` are wanted).
- Hand fixes to dpct's output, in the order they were met:
  - `pdl.hpp` (NVIDIA programmatic dependent launch): a no-op SYCL version, plus `launch_pdl` over `dpct::kernel_launcher`.
  - `tensor_cores()` returns false (the `mma`/`ldmatrix`/`cp.async` PTX kernels do not exist for SYCL; poly has scalar fallbacks);
    `__cvta_generic_to_shared`, `ptxVersion`, `uint4` removed.
  - `native_mmvq.dp.cpp`: dpct could not unify `F::load` across traits (DPCT1084) and dropped the IQ4/MXFP4 codebook arguments:
    the codebooks became private `alignas(16)` constants with default arguments; `trait_load<F>` helper; two `dpct_placeholder`
    kernel names; a macro (`STRATA_Q8P_LOAD`) that dpct expanded wrongly (DPCT1064); a bare-name kernel wrapper.
  - Kernel names: dpct names a templated `*_wrapper`'s kernel without its template arguments, giving "redefinition of KernelInfo"
    errors.  `tools/fix_kernel_names.py` qualifies them.
  - `iq_kernels.dp.cpp`: the Strata engine's grouped-expert kernels (S26, S27, expdec, `native_expert_*`) are `#if 0`: `strata-poly`
    does not call them and dpct's output for them was unusable; NVFP4 dequant/embed launchers pass `nullptr` for the codebook.
  - `runtime.hpp`: a shim for CUDA runtime calls dpct left inside lambdas and macros (`cudaMemcpy*`, `cudaFree*`, events, `cudaSetDevice`).
  - `engine.cpp`: virtual-memory leftovers (`cuMemUnmap`/`cuMemRelease`), the `Mapped` handle/pointer casts, events,
    queue-reference temporaries, graph `finalize()`.
  - The five model files (`qwen35.cpp` ...) call CUDA only through `runtime.hpp`, so dpct skipped them; they were migrated
    explicitly with `MIGRATE_ONLY` (`tools/migrate_poly.sh`).
  - `Mmq`: stubbed (`fits()` false) so the build could link.
- Link: needs the SPIR-V extension for `__dp4a`: `-Xspirv-translator --spirv-ext=+SPV_KHR_integer_dot_product`.

### 6.2 First runs (commit 60fe8d9)
- First run: `UR_RESULT_ERROR_INVALID_VALUE` then `DEVICE_LOST` in `urPhysicalMemCreate` for a 45,088,768-byte (43 MiB) allocation.
  Level Zero reports a 64 KiB granularity but needs whole **2 MiB** multiples for physical memory: `grain_` is rounded up to 2 MiB.
- Then it ran: Qwen3.6-35B-A3B, all 10,240 experts in VRAM (18.3 GiB), loaded in about 4-6 s.
  Numbers measured then (**engine later found broken, see 6.5; short outputs were right, long ones were not**): "capital of France"
  prompt, 8 tokens at 47 tok/s; a 226-token story 61 tok/s without the draft head (3 runs, 60.4-61.5), 49.3 with it.
- The output was checked only against short prompts here.  That was not enough (6.5).

### 6.3 A real `Mmq` for prefill (commit d2960a7)
- Baseline with the stub: **178 tok/s** prefill (1,692-token prompt, 2 runs: 177.4, 178.9).
- XeStrata's `moe_mmq`/`iq_mmq` has the same interface as poly's `Mmq` (quantize, grouped product with per-matrix row bounds and
  a scatter map, optional pointer table).  `iq_mmq.cpp` and its small headers were lifted into `src/xe/` (provenance and the
  exact edits: `src/xe/README.md`); `src/poly/mmq_sycl.cpp` implements `Mmq` over `iq_mmq_quantize` / `iq_mmq_grouped`, with a small
  context for the identity row map and `[0, rows]` bounds that single-matrix calls leave out, and the pointer-table path
  (`tab()` true; the table is written by a one-work-item kernel because the host array goes away).
- JIT (`spir64`) build: works, but the first start compiles the tiles for 2-4 minutes at every start (`loaded in 246 s`, then 171-203 s
  even with `SYCL_CACHE_PERSISTENT=1`); warm prefill 270.8 tok/s.
- AOT for the B60 (`-DSTRATA_SYCL_AOT=bmg-g21`, `-fsycl-targets=spir64_gen`): the backend refused three variants,
  "Unsupported JointMatrix operation" for the Xe-HPG 8x8 shape, the generic (NVIDIA/AMD) matrix form, and the layout probes
  that instantiate them.  `STRATA_MMQ_NO_XE_HPG_SHAPE` (set for `bmg`/`lnl` targets) compiles only the Xe2 shape and dp4a.
  Result: start in 5.6 s warm (23 s and 142 s on first starts while files and caches were cold), prefill 356.6-434.1 tok/s.
- At start the layout probe prints "the GPU's int8 XMX accumulators are laid out otherwise ...": the **XMX path is rejected**
  with icpx 2026.0 on this card and the engine runs the tiles in **dp4a**.  XMX is the main untried speedup (XeStrata builds with
  intel/llvm 7.2, which may differ).  Not investigated further.

### 6.4 Things that went wrong with the process, not the code
- Several runs hung until `timeout` or died with `DEVICE_LOST`, most often in the XMX layout probe at start or the first launch of a
  run, then worked on retry.  Cause not found.  Scripts retry.
- A run of the validation script was corrupted because the script file was edited while bash was reading it; its output was
  discarded and the run repeated from a clean file.
- zsh does not word-split `$extra`; several option sweeps silently printed the usage text until `${=extra}` was used.

### 6.5 The long-context bug (found after 6.3; fixed)
After 6.3 a retrieval test (100 numbered facts, then "What is Fact N?") gave an immediate `<|im_end|>`.  Findings, in order:
1. The stub-era binary failed the same way, so it was not `Mmq`.  Everything I had said about "correct output" rested on prompts
   under ~90 tokens.  **That claim was wrong and was corrected.**
2. Teacher forcing (feed the true counting sequence, ask for the next token): right to position 90-93, wrong from 93/94, whatever the
   `--prompt-batch` (1, 16, 64 ... 1024), `--max-context` (256 ... 8192), `--kv-whole` (no virtual-memory KV) or `--no-gpu-experts`
   (experts on the CPU).  Identical garbage from the JIT and AOT builds: deterministic, so not a race.
3. Unit tests with CPU references (`src/poly/*_q_test.cpp`): `attend()` for head 256 / 16 heads / 2 KV heads at 16 context lengths:
   max error 6e-8..4e-7 (**passes**); `head_norm_rope` at 700 positions: 1e-5..9e-5, growing with the angle as float `pow` does
   (**passes**); `gdn_step` (gated delta rule, 128-wide heads, 32 value heads over 16 key heads) for 300 stateful steps:
   **wrong from step 1**.  Step 0 matched only because the state starts at zero.
4. Reading the migrated `gdn_state_reg_kernel` against the CUDA original: `d = __fmul_rn(__fsub_rn(va, sk), sa)` became
   `const float d = va - sk * sa;` (DPCT1013, "the rounding mode could not be specified": the parentheses were dropped).
   Fixed to `(va - sk) * sa`.  The recurrent state was corrupted from the second token on, so answers fell apart after roughly
   a hundred tokens.  Every other DPCT1013 site (11 in `kernels.dp.cpp`, 6 in `native_mmvq`, 10 in `iq_kernels`) was read: no other
   nesting error.
5. Also changed while hunting this (not the cause, kept as safer): the 29 masked sub-group shuffles
   (`dpct::experimental::permute_sub_group_by_xor(0xffffffff, ...)`, `shift_sub_group_*`, `select_from_sub_group`) became the
   unmasked `dpct::` forms, which are built on standard `select_from_group`.
6. After the fix: GDN test max error 1e-6..5.6e-6 over 300 steps (stable); teacher forcing 10/10 correct at positions 50-190; retrieval
   exact at 333 and 2,709 tokens; "count 1 to 200" exactly 1..200 then stop.

### 6.6 Final numbers (AOT bmg-g21 build, `--no-graphs`, Qwen3.6-35B-A3B UD-Q4_K_XL, 24 GB B60, all experts in VRAM)
| Measure | Result |
|---|---|
| Prefill, 2,710-token prompt | 444.8-447.2 tok/s (validation log, 6 runs) |
| Decode, 27 tokens (retrieval answer), validation log | 68.95-70.11 tok/s without the draft head, 92.57-93.15 with it (3 runs each) |
| Decode, same, an earlier run the same hour | 81.7-81.8 tok/s without, 91.4 with (not reproduced in the validation run; the card's clocks vary: treat short-answer decode as 70-82) |
| Decode, 472-token story | 49.4 tok/s without the draft head, 69.2 with it |
| Decode, 692 tokens (counting 1..200) | 99.9 tok/s with the draft head |
| Start (warm) | about 5 s |
Re-run by `docs/validate/run_validation.sh`; its output for the released build is in `docs/logs/`.  For comparison XeStrata's own
engine on the same card and model measured 44-64 tok/s decode on a thinking-mode story (section 5); different prompt and mode, so
not a like-for-like comparison.

## 7. Known problems and what is not done
See `NOTES.md` ("What is not there").  In short: XMX not used by `Mmq`; CUDA graphs untested (`--no-graphs`); NVFP4 codebook not
threaded; multi-GPU, image path and every model except Qwen3.6 untested; an intermittent `DEVICE_LOST` around the first launches;
the older `sycl/` port still stale.

## 8. Reproduce
```
git clone --recursive https://github.com/<this repo>   # includes third_party/llama.cpp
source /opt/intel/oneapi/setvars.sh
cmake -S sycl-poly -B build-poly -G Ninja -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DSTRATA_SYCL_AOT=bmg-g21
ninja -C build-poly strata-poly poly_gdn_q_test poly_attend_q_test poly_rope_q_test
BIN=build-poly/strata-poly BUILD=build-poly MODEL=<Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf> TOKENIZE=<llama.cpp>/llama-tokenize \
  sycl-poly/docs/validate/run_validation.sh | tee sycl-poly/docs/logs/validation-$(date +%F).log
```
For other Intel GPUs leave `STRATA_SYCL_AOT` empty (JIT: slow first start) or give your device (`ocloc compile --help`); the
Xe-HPG/Alchemist shape of `Mmq` is built only in that case.

## 9. 2026-11-24: the front end, so that it is used the way UnbiasedStrata is used

Goal stated by the project owner: Unbiased XeStrata must be as usable as the original: clone, build, run `unbiased-strata`, get the
OpenAI-style API with every supported model.  UnbiasedStrata is a C++ front end (`unbiased-strata`, built from llama.cpp's server,
common and mtmd code) that starts an engine binary (`--engine strata-poly`) and talks to it over stdin/stdout (`GEN ...` in, `T <id>`
lines out).  The front end is CPU code; it needed nothing from us except an engine that speaks the protocol.

- `make frontend` (no CUDA, g++ 16.2) built it; `make test` (against the fake engine in `frontend/tests/`) passed: "e2e: all checks passed".
- The SYCL `strata-poly` already had `--serve`, so the front end ran it unchanged:
  `unbiased-strata --engine strata-poly -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --mmproj mmproj-F16.gguf --threads 8 -- --model ... --max-context 8192 --no-graphs`.
  The engine took about 3.5 minutes to become ready on the first start (reading the 22 GB file), then answered.
- `docs/validate/frontend_checks.py` (+ `run_frontend_validation.sh`) drives the running server like a user or a coding agent.  Result,
  `docs/logs/frontend-validation-2026-11-24.log`: chat (right answer); SSE streaming (33 events, `[DONE]`); retrieval of one fact among 100 at
  2,710 tokens **exact** (prefill 451 tok/s, decode 92 tok/s); a 459-token story (57 tok/s); thinking mode (`reasoning_content` separate from the
  answer); a tool call (`get_weather {"city":"Paris"}`, `finish_reason: tool_calls`); and an image (the model described a green/red
  image with a white bar correctly, the projector run on the CPU by the front end).  **PASS.**
- Nothing in the front end or the engine had to change for this.  What was missing was only the build and the documentation:
- `make intel` (new, in the top-level `Makefile`): builds the front end and the SYCL engine into `build-intel/bin/`, detects the card with
  `sycl-ls` (B50/B60/B570/B580 -> `bmg-g21`, B65/B70 -> `bmg-g31`, anything else -> JIT), no manual cmake.  `make intel-test` runs the kernel unit tests.
- The mmproj file for Qwen3.6 (`mmproj-F16.gguf`, 899 MB, Unsloth) was downloaded for the vision check.

Engines of UnbiasedStrata still missing on Intel: `strata-ik2` / `strata-inkling` (Inkling 2 Mini), `strata-dsv41` (DeepSeek V4.1), `strata-hy4`
(Hy4-Flash) and `strata` (Qwen3.8-Flash-Next).  The next steps are in this order: Inkling (the one asked for), then the others as far as a 24 GB card allows.

## 10. 2026-11-24: Inkling-2-Mini engine (`strata-ik2`) -> `sycl-ik2/` (in progress)

The owner asked for the same usability as UnbiasedStrata, which includes Inkling 2 Mini.  UnbiasedStrata's Inkling engine for NVIDIA is
`src/ik2` (Project Maya's engine, ~14,000 lines: `ik2_kernels.cu` 6,261, `ik2_main.cu` 6,082, `ik2_draft.cu` 537, a CPU engine `ik2_cpu*.cpp`).

**Which file the engine reads.**  The first model downloaded (ggml-org's `Inkling-2-Mini-IQ4_NL.gguf`, 17.6 GB) is the llama.cpp-master layout
(`general.architecture = inkling2_loop`, `block_count` = 24 layers, `loop.start/end/count`).  The engine reads PR #32187's layout (`inkling2.*`,
`block_count` = 48 layer visits, `num_loops`, `loop.core_start/end`): neither `strata-poly` nor llama.cpp v1.0.0 nor `strata-ik2` loads the
master layout.  Unsloth publishes both: its repo root is now the master layout and `Old_Support/` is the engine's.  Downloaded
`Old_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file` (17,629,759,456 bytes, verified against the Hub's size) and `Old_Support/mmproj-BF16.gguf_file`;
`engine-layout/` has symlinks with the names the engine expects.

**Migration.**  `sycl-ik2/tools/migrate_ik2.sh` (dpct 2025.3 over the three GPU files, then `fixups_ik2.py`, `fix_kernel_names_ik2.py`).  Counts: zero
DPCT1013 (the precedence bug class of section 6.5 does not occur here), 3 unmigrated asm sites, 188 DPCT1098.  Fixes, in the order met:
- **`__ldg`**: dpct rewrites `__ldg(p)` to `*(p)` and in 6 places dropped the vector cast of `__ldg((const float4*) p)`.  Cure: the sources get
  `ik2_ldg(p)` (a wrapper, `src/ik2_ldg.dp.hpp`) before dpct runs, so no cast can be lost (212 calls; DPCT1098 count 0).
- **FP8 (E4M3)**: the sm_89 hardware conversions do not exist; the file has a software path under `#else`.  dpct turned `__CUDA_ARCH__ >= 890` into
  `DPCT_COMPATIBILITY_TEMP >= 890` (always true) and `!(...)` into `!X >= 890` (always false); both mangled tests are replaced (`#if 0` / and for the host
  copies of those functions `#if 0`, which selects the software `#else` branch; a first attempt used `#if 1` there and was wrong).
- **`cp.async`** (4/16-byte global->shared copies): plain synchronous copies; commit/wait are no-ops.  Loses the overlap of the next tile's loads, not correctness.
- **Wrapper-macro**: `IK2_MMA_LAUNCH(W)` became a bare `{}` block used as `if (...) M(1); else ...`: wrapped in `do { } while (0)`.
- Kernel names without template arguments (as in section 6): `fix_kernel_names_ik2.py` qualifies names in any enclosing template (5 sites).
- Unnamed template arguments (DPCT1050): `grouped_expert_mma`'s defaulted `Q8 = false`, `mv_q8p_kernel`'s `StaticK = 0`.
- `calculate_max_active_wg_per_xecore` (DPCT1111) without the kernels' shared-memory size: 0 (an upper bound; only the number of attention splits depends on it).
- The host decode helper called `decode_value` without the IQ tables dpct had added as parameters: it has host copies (`h_iq3xxs_grid`, `h_iq4nl`).
- `ik2_main.cu`'s CUDA runtime leftovers (events, host callbacks, pinned allocation, graph destroy, one `cublasGemmEx`): `src/ik2_leftovers.dp.hpp`
  and one rewritten call.
- **Tensor-core primitives** (`mma.sync.m16n8k16`, `ldmatrix.x4`, plain and `.trans`): dpct maps them to a 16x8x16 joint_matrix that Intel's
  engine does not have.  Emulated with sub-group shuffles from PTX's register layouts (`src/ik2_mma_emul.dp.inc`, ~32 shuffles per mma); the unit
  test `tests/mma_emul_test.cpp` (random operands laid out per the PTX layouts, compared with a CPU product and transpose) passes (20 trials, error < 1e-4).
  Correct and slow: the XMX version is an open item.

**Debugging the hang, in the order it went (2026-11-24 / 25).**  Each step is something that looked like the cause and was or was not:
1. IGC (Intel's graphics compiler) segfaulted on `experts_gate_up_swiglu_g_kernel`, in JIT and in AOT (offline `ocloc`, 14 s to crash).  With
   `-fsycl-device-code-split=per_kernel` (a module per kernel) exactly **1 of 202** modules crashed; only `-cl-opt-disable` compiled it.  Cause:
   `row_dot_multi` / `row_dot_pair_multi` are a runtime switch over every quant type, each branch unrolled over 8 tokens, inlined for 1, 2, 4 and 8
   tokens.  Fix: `__attribute__((noinline))` on those two: all 202 modules compile (fixups_ik2.py).
2. `GPU HANG detected` in the expert kernels.  The 29 masked sub-group shuffles (`dpct::experimental::permute_sub_group_by_xor(0xffffffff, ...)`)
   became the unmasked `dpct::` forms (as in sycl-poly): the gate/up kernel then finished.  (Not the whole story, see 4.)
3. `IK2_SYNC_CHECK=1` (a diagnostic added to `check()`: wait for the card after each launch and print the name) showed the failing kernel was
   *random*: a different layer each run, once the very first trivial kernel after a hang (a hung GPU needs ~20-30 s to come back; wait between runs).
   A synthetic reproducer (`tests/experts_down_test.cpp`: 500 launches x 8 configurations of `experts_down_g` on random NVFP4/Q8_0 data) never hung: the
   kernel was not the problem.
4. Printing the engine's real arguments (the groups the host builds for the GPU) showed `top[] = 128` for all 24 slots: expert id 128 does not exist
   (0..127), so the host indexed its pointer table out of bounds and sent garbage pointers: that was the hang.  The routing kernel returns 128 when all
   router scores are NaN, so the layer's input was bad.  A NaN scan of each stage of layer 0 (`--dump`, then temporary taps) found the first bad tensor:
   the **normalization weights** (`attn_norm`, 31 NaN, max 1.9e28).  `rms_f16` itself is correct (`tests/rms_test.cpp`).
5. Root cause: `dev_f32(host_f32(...))` copies from a **temporary** `std::vector` with `dpct::get_in_order_queue().memcpy(...)` and does not wait.  CUDA's
   plain `cudaMemcpy` (no `Async`) is synchronous for the host; dpct's translation is not (DPCT1124 says so).  The vector was freed before the GPU
   read it.  Fix: `.wait()` after every default-queue `memcpy`/`memset` (`sync_default_queue` in fixups_ik2.py; 15 sites).  Every symptom above followed from it.

**Result (2026-11-25), Arc Pro B60, Unsloth `Inkling-2-Mini-UD-Q4_K_XL.gguf` (the engine's layout), every expert in VRAM (3,072 slots, 14.7 GiB):**
through the front end: "What is the capital of France? Answer in one sentence." -> "The capital of France is Paris."; "17 * 23, just the number" -> "391";
a coherent haiku.  Decode ~8-9 tok/s (graphs on or off, token-identical), which is slow: the next item.  The kernel pieces are tested
(`tests/mma_emul_test.cpp`, `rms_test.cpp`, `experts_down_test.cpp`); the engine's `--dump` comparison against its reference (`tools/ik2_compare.py`) is not yet run.

## 11. Inkling decode speed (2026-11-25), Arc Pro B60, Inkling-2-Mini UD-Q4_K_XL, 64-token greedy decode, identical tokens throughout

| change | decode tok/s |
|---|---|
| first working build | 9.4 |
| `IK2_EXPERT_MMA=0` default: the shuffle-emulated tensor-core MMA experts lose to the plain grouped kernels | 26.8 |
| expert tiling 8 warps x 1 row (gate/up and down), was 4x2 / 4x4 | 33.4 |
| E2M1 decode without `dpct::byte_level_permute` (four variable 64-bit shifts a call, in the NVFP4 inner loop) | **41.4** |

Through the front end: 38-40 tok/s decode on short prompts, 32 tok/s after a 4k-token prompt; prefill 65 tok/s (short) to 94 tok/s (4k tokens), which is
still slow (the prefill attention and expert paths are not tuned).  Per token now ~24 ms: experts ~10 ms (floor at memory bandwidth ~4 ms), dense
`mv_q8p` ~6 ms, the rest launch overhead.  `IK2_SYNC_CHECK=2` prints a per-kernel time table; its first row ("rms", ~2.7 s) is start-up work, not a kernel.
Next: prefill (grouped expert path with real GEMM tiles), fusing the small launches, `mv_q8p`.  `make intel` now builds `strata-ik2` next to `strata-poly`.

## 12. Inkling prefill speed (2026-11-25), Arc Pro B60

Profile of a 1024-token prefill (160 tok/s): `attn_flash` was 5.5 s of 8.3 s.  It is the CUDA kernel with `mma.sync` / `ldmatrix` emulated with sub-group shuffles
(~32 shuffles + 128 conversions per 16x8x16 product), ~56 GFLOP/s.  The non-flash fallback (`IK2_FLASH=0`) fails on oneMKL's stride-0 batched GEMM
(GQA broadcast), so it is not an alternative.

New: `src/ik2_flash_xmx.dp.inc`, the FP16 prefill attention on the Xe matrix engine (SYCL `joint_matrix`, 8x16x16 half -> float, sub-group 16; JIT and AOT
for bmg-g21 both give exact results in a probe).  A work-group of 4 sub-groups = 32 queries of one head against one split of its keys, 64-key tiles; K / V
tiles staged once per work-group in SLM in VNNI form; S = QK^T by 32 mads per tile and sub-group, stored to SLM, online softmax by the lanes (same mask /
window / relative-bias / log2 arithmetic as the CUDA kernel), P (FP16) in SLM, O += PV by 32 mads, O rescaled in place.  It writes the same `o_part` /
`ml_part`, so `attn_flash_combine` is unchanged.  `tests/flash_xmx_test.cpp` compares it with the emulated kernel (causal, window, bias, GQA, splits, ragged
Tq, ring cache): 7/7 within 1.2e-4 (FP16 noise).  `IK2_FLASH_XMX=0` selects the old path; FP8 caches keep it.

| | prefill tok/s |
|---|---|
| before (emulated MMA attention) | 160 (1024 tokens), 94 (4k) |
| XMX attention | **735** (1024 tokens), **642** (4k: a 3,988-token prompt in 6.4 s, was 44 s) |

Retrieval over the 4k prompt is correct (3 of 3 items).  Attention is now 0.65 s of a 1.4 s prefill; the rest is the experts (NVFP4 -> FP16 decode + oneMKL GEMM
per expert), the dense GEMMs and small kernels.  Next: skip fully masked key tiles per sub-group, fuse the expert decode, XMX for the expert GEMMs.

## 13. Inkling prefill experts (2026-11-25), Arc Pro B60

Where the time is: in a 1024-token prefill (1.36 s) the engine's expert work is ~0.2-0.25 s (about 15%); at 4000 tokens (6 s) attention is ~4 s and the
experts ~0.4 s.  So the experts cannot give more than ~15-20% however fast they get.

**Shipped (default): the batched expert path with its own buffers.**  The engine had a batched prefill (tokens sorted by expert: one gather, one SwiGLU, one
combine per layer instead of a gather / scatter-add per expert) that borrowed the attention's logits scratch, which is empty with flash attention, so it never
ran.  It now allocates its own sorted buffers (`IK2_BATCH_PREFILL=0` turns it off).  Same greedy tokens; 740 -> 767 tok/s (1024 tokens), 668 -> 684 (4000).

**Built, correct, not faster, so opt-in (`IK2_MOE_XMX=1`): a grouped XMX expert GEMM straight from NVFP4** (`src/ik2_moe_xmx.dp.inc`, `tests/moe_xmx_test.cpp`
against decode_f16 + a CPU F32 product: gate/up + SwiGLU and down, ragged counts, max error at FP16 rounding).  All hit experts in one launch per product,
weights dequantized into SLM (a 256-entry byte -> half2 table; activations in a "pair" column order so one byte is one VNNI pair), double-buffered, gate/up with
the SwiGLU fused.  A layer visit (128 experts x 64 tokens) 17.8 -> 13.4 ms, but the old decode_f16 + oneMKL gemm16 path is still cheaper end to end
(1024 tokens: 1.68 s vs 1.36 s; 4000: 7.4 s vs 6.0 s).  What the ablations showed (all on a layer visit):
- dequant ~3.6 ms, matrix part ~5 ms, fixed ~1.4 ms; with one A fragment reused from a register the 32 x 16 DPAS cost only ~0.3 ms: the matrix part is the
  A-fragment loads (`joint_matrix_load` from global, 8 rows x 32 B), not the matrix engine.
- A staged in SLM: slower (occupancy).  A in a fragment-contiguous layout: slower.  `joint_matrix_load_checked` (2D block load): ~8% faster, and I never got
  its coordinates validated.  A 64x128 tile with 4x4 register blocking and 256 GRFs: correct but no faster; prefetching the next fragments with
  `joint_matrix_copy`: 2x slower (spills).
- Next, if someone continues: a proper register-blocked kernel with the A/B fragments in registers (cutlass-sycl style), verified against `tests/moe_xmx_test`.

Where the prefill time is now: attention at long contexts (it grows with the context: ~4 s of a 4000-token prompt).  The next real gain is there (skip fully
masked key tiles per sub-group; the sliding-window layers only need their window).

## 14. Inkling prefill attention (2026-11-25), Arc Pro B60

At 4000 tokens the XMX flash attention (section 12) was still ~4 of 6 s: the 320 sliding-window calls (window 1024) ran at ~1 TFLOP/s, the 64 full-attention
calls at ~1.7.  `tests/flash_xmx_test.cpp perf` times Inkling's shapes (H = 24, KV = 8, a chunk of 1024 queries into a 4000-token context, relative bias 1024);
in-kernel ablations (skip the tile load / S products / softmax / PV products) said where it went.  What changed, in order of effect (sliding / full, JIT):

| step | sliding ms | full ms |
|---|---|---|
| section 12 kernel (4 sub-groups a work-group, 55 KB of SLM: 2 work-groups a core = one hardware thread an execution unit, nothing hides a stall) | 13.9 | 50.5 |
| logits / probability staging padded (the softmax read 8-way bank-conflicted), K / V tile with 16-byte global loads and conflict-free SLM writes | 9.3 | 28.9 |
| 16 sub-groups a work-group (128 queries): the K / V tile load is shared by 4x the queries, and more threads per core | 5.0 | 11.1 |
| fast softmax path for fully visible 32-column runs (no masks, the bias as vector loads) | 5.3 | 10.3 |
| (AOT build) | **3.4** | **11.0** |

Dead ends: forcing `#pragma unroll` (already unrolled); the 4000-token numbers did not need more.  The relative bias costs only ~1 ms of 9; DPAS itself is nearly free
(the cost is the fragment loads from SLM and the softmax between the two products, both latency, not throughput).

Engine, 4000-token prompt: attention ~4 s -> ~0.85 s; prefill 684 -> **1325-1410 tok/s** (5.85 s -> 2.83 s, through the front end 6.4 s -> 3.1 s); 1024 tokens
767 -> 1143 tok/s.  Same greedy tokens as before; 7 of 7 retrievals correct at 4k.  `tests/flash_xmx_test` against the emulated kernel: 7/7.
What is left in a 4000-token prefill: the expert weight decode (decode_f16_nvfp4, ~1.1 s), attention ~0.85 s, small kernels.

## 15. Inkling prefill expert weight decode (2026-11-25), Arc Pro B60

`decode_f16_nvfp4` showed as ~1.1 s of a 4000-token prefill in the per-kernel table, but that table waits for the card after every launch.  Measured properly:
skipping every NVFP4 decode (garbage weights, timing only) takes the prefill from 2.82 s to 2.40 s, so the decode costs ~0.42 s (15%) in all, ~13,500 launches
of ~30 us each: mostly launch and dependency latency, not bandwidth.

`tests/decode_bench.cpp` (one expert's gate+up, 5.3M weights, 10.6 MB out): the original kernel 28-29 us; a plain memset of the same output 7.8 us; reads alone ~11 us;
the original's structure with trivial ALU ~17 us.  Findings: (a) the original writes two 16-byte stores 32 bytes apart per lane, so each store instruction fills half of
every cache line; (b) the conversion arithmetic is the other ~12 us.  Tried:
- a 256-entry byte -> half2 table in SLM, one 36-byte block a thread, packed half2 multiply by the scale (bit-exact): *slower* (35-65 us: unaligned 36-byte
  strides, register pressure);
- a thread takes half a sub-block (low or high nibbles) and writes one 16-byte store, lanes contiguous (`src/ik2_decode_lut.dp.inc`, bit-exact): 21.5 us in a loop,
  but 21-36 us from run to run (JIT and AOT), and no change in the 4000-token prefill (2.81 s either way); shifting the word instead of selecting between the
  two nibble decodes was slower.
It ships as opt-in (`IK2_DECODE_LUT=1`); the default is the original kernel.

What would help: fewer launches, not faster kernels: decode several experts' weights per launch into a few FP16 buffers and batch the GEMMs (oneMKL group
`gemm_batch` accepts a different M per group), or the fused-dequant GEMM of section 13 with proper register blocking.  Together the expert path is
~0.4 s (decode) + ~0.3 s (GEMMs and glue) of a 2.8 s prefill; attention ~0.85 s; the rest is the dense GEMMs, loop gates and small kernels.

## 16. Fewer expert decodes in prefill (2026-11-25), Arc Pro B60

**Correction to section 15.**  I called the ~13,500 decode launches "launch latency".  A micro-test (`use_root_sync` or not, 16 to 4096 work-groups) puts a back-to-back
launch at ~3 us, so the 0.42 s is GPU time: ~28 us a decode x 13,500.  The lever is how many decodes there are, not how fast each launches.

**Tried: several experts' matrices per launch** (`decode_f16_nvfp4_multi`, `IK2_DECODE_GROUP=N`, up to 16, each into its own FP16 buffer, then the experts' GEMMs).  Fewer
launches (-75% at N = 4, -88% at N = 8) but *slower*: 4000-token prefill 2.81 s -> 3.24 s (N = 4), 3.30 s (N = 8, 16).  The single reused FP16 buffer of the per-expert path
stays in the L2 between the decode and its GEMM; a group's 40-170 MB of decoded weights spills to DRAM and is read back.  Kept as opt-in, default 1.  Same tokens.

**Shipped: a bigger prefill chunk.**  The engine decodes every hit expert once a layer visit *per chunk*, so the chunk size (`--chunk`, default 2048; a looped model cannot use the
engine's layer-major super-chunk mode, which is for n_pass == 1) sets the number of decodes.  Default now 4096.  Greedy tokens identical in every run.

| prompt | chunk 1024 | 2048 (old default) | 4096 (new) | 8192 |
|---|---|---|---|---|
| 4000 tokens | 3.32 s | 2.80 s | **2.49 s** | |
| 12000 tokens | | 9.39 s | **8.46 s** | 8.08 s |

(4096 vs 2048: -11% and -10%; 8192 saves 4% more for ~1.2 GiB of VRAM, so not the default.)  At a 131,072-token context the 4096 chunk costs 0.3 GiB more K / V cache (the
sliding layers keep window + chunk).  Through the front end: 3988-token prompts prefill at ~1490-1510 tok/s (was ~1325), a 7453-token prompt (two chunks) at ~1410-1470; 10 of 10
retrievals correct, including items from before and after the chunk boundary; decode unchanged (41.8 tok/s).

## 17. Inkling prefill dense GEMMs (2026-11-25), Arc Pro B60: already at the matrix engine's speed, left alone

A temporary profiler around `gemm16` (wait, time, key by N / K / T; not committed) on a 4000-token prefill (one 4096-token chunk), 2.73 s in all, GEMM time 0.83 s:

| products | calls | ms | TFLOP/s |
|---|---|---|---|
| dense (q/k/v/o, gate, projections), T = 4000, N >= 1000 | 486 | 261 | **74.5** |
| dense, router (N 128) and sink (N 24) | 112 | 21 | 6.9 |
| experts of the batched layers, T 64..2999 | 3,408 | 380 | 38.9 |
| tiny T (the decode step, cold experts) | 4,424 | 163 | 3.4 |

The dense GEMMs are oneMKL's: N8576 x K2304 x T4000 in 2.5 ms (63 TFLOP/s), N2304 x K3072 in 0.71 ms (79), N3456 in 0.82 ms (77): about three quarters of the card's FP16 matrix peak.
Their FP16 weight decode (Q8 -> FP16 before each GEMM) is 24 ms in all (caching the FP16 weights would cost ~2.2 GiB of VRAM for that).  The only slow dense shapes are the router
and sink products (21 ms, 0.8% of the prefill).  Nothing worth a change; nothing was committed to the engine.

Mistake avoided: the per-kernel `IK2_SYNC_CHECK=2` table charges the time since the previous launch to a kernel, so the element-wise kernels after a GEMM (swiglu_cap_f32, conv4_res,
attn_res) look slow because they include the GEMM before them.  Read GEMM cost from a GEMM-level profile.

Where a 4000-token prefill's 2.7 s goes now (approximately): GEMMs 0.83 s (dense 0.28, experts 0.54), expert NVFP4 weight decode 0.4, attention 0.85 (section 14), the rest small kernels.
The expert GEMMs (39 TFLOP/s on small, ragged token counts, plus the decode that feeds them) are the largest remaining piece; the fused-dequant kernel of section 13 is the way in.

## 18. Inkling prefill expert GEMMs (2026-11-25), Arc Pro B60: measured, nothing faster found

The expert GEMMs of a 4000-token prefill (one 4096-token chunk, ~250 tokens an expert on average): 3,408 calls, 380 ms, 39 TFLOP/s (dense GEMMs: 74.5).  oneMKL's FP16 GEMM
(`gemm16`: m = N, n = T, K) swept over T on the expert shapes (16 distinct weight matrices, so the weights come from memory like the engine's):

| N x K | T 64 | 128 | 250 | 256 | 384 | 512 | 1024 | 4000 |
|---|---|---|---|---|---|---|---|---|
| 2304 x 2304 (gate+up) | 34 us / 20 TFLOP/s | 38 / 35 | 57 / 46 | 58 / 47 | 53 / 78 | 67 / 81 | 132 / 82 | 514 / 83 |
| 2304 x 1152 (down) | 14 / 24 | 14 / 47 | 20 / 65 | 28 / 49 | 29 / 71 | 36 / 75 | 70 / 77 | 280 / 76 |

The time is a staircase in T (tiles of 128), with dips and bumps (K = 2304: T = 254 takes 60 us, T = 262 takes 51), so a GEMM only reaches the dense speed from T ~ 384 up;
the engine's experts sit around 250.  What I tried or measured:
- **Pad T to a faster nearby size** (T..T+128 over the engine's T range 100..450): -1% (+8), -3% (+64), -6% (+128) at best.  Not worth rows of slack and ordering constraints.
- **Concurrent queues** for the GEMMs alone (`mkl_conc`): 57 -> 51 us at T = 256 (2 queues), 25 -> 21 at T = 128; no more with 4.
- **A two-stream pipeline in the engine** (even experts on the main queue, odd on a second queue with its own FP16 buffer and BLAS handle, fork / join with barriers; decode of one
  overlapping the GEMM of another): *slower*, 2.50 s -> 2.64 s for the 4000-token prefill, same tokens.  Same reason as the grouped decode in section 16: the decode -> GEMM pair
  is L2-resident with one buffer; two doubles the working set past the L2.  Kept opt-in (`IK2_EXPERT_PIPE=1`), default off.
- A bigger chunk raises T and so the speed (section 16: 8192 tokens: 12,000-token prefill 8.46 -> 8.08 s for +1.2 GiB of VRAM).

What would still help: a GEMM that dequantizes NVFP4 in SLM and keeps a *whole expert's* tokens in one work-group (T ~ 256 x K of FP16 activations does not fit registers; it needs the
A-fragment loads of section 13 solved).  The ceiling is ~0.25 s of a 2.5 s prefill (expert GEMMs 0.38 s at 39 TFLOP/s -> 0.2 s at ~75).

## 19. Inkling prefill attention, round 2 (2026-11-25), Arc Pro B60

The engine's prefill chunk is now 4096 (section 16), so `tests/flash_xmx_test.cpp perf` times 4096-query chunks (first chunk / second chunk, sliding window 1024 / full, H = 24, KV = 8, bias 1024).  JIT:

| change | sliding, chunk 1 | full, chunk 1 | sliding, chunk 2 | full, chunk 2 |
|---|---|---|---|---|
| section 14 kernel (16 sub-groups a work-group) | 11.9 ms | 26.0 | 13.5 | 77.4 |
| Qs aliased with the logits staging (3.4 KB a sub-group instead of 5.2), 27 sub-groups a work-group (SLM-limited, 125 KB) | 9.2 | 17.3 | 10.4 | 52.5 |
| (sub-groups skip tiles none of their 8 queries can see: no change, the kernel is not bound by that work) | 9.1 | 17.4 | 10.4 | 51.0 |
| **256 registers a thread** (`grf_size<256>`) | **5.3** | **8.4** | **6.2** | **22.2** |

The last row is the finding: ablating the kernel showed tile loads + barriers ~1-2 ms but the per-sub-group compute ~8 of 9.8, and per softmax element ~280 cycles, far more than the
arithmetic.  The Q fragments (8 x 4 registers), 8 output accumulators (64) and the 32-element logit array do not fit the default 128 registers: the kernel was spilling to scratch.  256
registers halve the threads an execution unit can hold, but 27 sub-groups a core is under 4 threads an EU anyway (SLM-bound), so occupancy is unchanged: 1.8-2.5x.  (Check any
hand-written XMX kernel with several accumulators for spills first; the section-13 MoE kernel already set 256.)

Remaining split of the 21 ms case (ablation at 256 registers): S products ~3 ms, PV products ~4 ms (one SLM fragment load a product: a sub-group has one 8-query block, so a fragment
is used once; 2 blocks a sub-group would reuse it but need ~190 registers of accumulators + Q fragments), exponent / probability pass ~1 ms, logits + stores + rescale ~6 ms, tile loads ~4 ms.
AOT: 6.0 / 10.2 / 6.9 / 27.6 ms (8.6-11.2 TFLOP/s, was ~4-5).

Engine: 4000-token prefill 2.43 -> 2.10 s; 12,000 tokens 8.16 -> 6.44 s (-21%); same greedy tokens.  Through the front end 3988-token prompts prefill at ~1,710 tok/s (was ~1,490-1,510) and
7,453-token prompts at ~1,655 (was ~1,410-1,470); 10 of 10 retrievals correct, decode 41.8 tok/s.  `tests/flash_xmx_test` against the emulated kernel: 7/7 (JIT and AOT).

## 20. Inkling prefill attention: the logits pass (2026-11-25), Arc Pro B60

Ablation of the flash kernel's softmax section at 256 registers (JIT, 4096-query shapes): skipping the *logits loop* (SLM read of S, scale, relative bias, mask, row max) saves 2.5-7 ms of 6.0 / 9.3 /
6.2 / 24 ms, ~40% of the kernel; the S store to SLM, the exponent / probability stores and the O rescale are each 0.1-1 ms.  Tried, none faster (all kept the 7/7 comparison against the emulated
kernel): 16-byte SLM loads of S and 16-byte stores of P (instead of 32 scalar loads and 2-byte stores), the 8 row factors held in registers once, the rescale skipped when no row's max moved
(`any_of_group`), `native::exp2`, a tree reduction for the row max instead of a 32-long `fmax` chain, and the bias loads hoisted above the S products (+32 registers: slightly slower).  The loop is
latency-bound (SLM, then the global bias loads), not instruction-bound, with < 4 threads an execution unit to cover it.

What matters more: the real call size.  The engine does prefill attention in sub-calls of `Work::kRelbRows` queries, each after a GEMM that builds the relative-bias table ([rows * H][extent] floats, the
kernel's biggest input); it was 512 rows, so the flash kernel (216-query blocks) saw 3 x 24 = 72 work-groups on 20 cores.  `tests/flash_xmx_test chunk` times a chunk's 8 sub-calls: the current
kernel (27 sub-groups, no key split) is best of {8, 12, 16, 27} x {1..4 splits}: 10.4 ms a sliding layer visit, 16.9 ms a full one (a single 4096-query call: 5.3 ms sliding).  `kRelbRows` 2048
(table 50 -> 200 MB): 12,000-token prefill 6.43 -> 6.23 s, 4000 tokens unchanged; 4096 is no better (400 MB).  Shipped: 2048.

The ceiling: skipping the flash attention launches entirely (timing only) takes the 4000-token prefill from 2.07 to 1.84 s and the 12,000-token one from 6.25 to 5.29 s, i.e. the whole kernel is 11-15% of a
prefill and the logits pass ~5%.  Only an in-kernel bias (bias = r . proj[dist], d_rel = 16: 5 extra products a tile on the matrix engine + an SLM gather, which also removes the table, its GEMM and the
global loads) could remove most of it; not done.  Through the front end now: 3988-token prompts ~1,760 tok/s, 7,453-token ~1,717 (was ~1,710 / ~1,655); 10 of 10 retrievals correct, decode 41.5 tok/s.
