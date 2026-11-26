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

## 21. Inkling decode (2026-11-25), Arc Pro B60

Decode was 41.5 tok/s (24 ms a token).  In-situ cost of each group (temporary skip switches in the kernel launchers: outputs go stale but stay finite; 25.5 ms a token on the JIT build):

| group | ms a token | share |
|---|---|---|
| expert kernels (gate/up + SwiGLU, down) | 11.4 | 45% |
| attention decode (emulated-MMA kernel) | 4.0 | 16% |
| `mv_q8p` (dense Q8 matvecs, ~113 a token) | 2.9 | 11% (~1 GB of weights a token: at the bandwidth floor) |
| `mv_f16` (64 small matvecs a token) | 2.4 | 9% |
| everything else (~800 small launches) | ~8.9 | the rest |

**Shipped: the scalar decode attention by default** (`IK2_ATTN_MMA=0`; the engine's tensor-core path is the mma16816 emulation, slower here).  AOT, same tokens: decode 41.4 -> 45.0 tok/s at a short
context, 33.1 -> 39.1 at a 4000-token context (+18%), 29.0 -> 36.6 at 12,000 (+26%).  Through the front end: 45 tok/s on short prompts, 38-39 after 4k-7.5k-token prompts; 10 of 10 retrievals correct.
(Graph replay on or off (`IK2_GRAPH`) is 41.4 vs 41.2: decode is GPU-bound, not host-bound; a launch costs ~3 us.)

**Tried on the expert kernels, none faster** (`tests/decode_experts_bench.cpp`: 8 distinct experts a launch in rotation, NVFP4, C = 2304, F = 1152; the generic kernels: gate/up 118 us = 203 GB/s,
down 71 us = 168 GB/s, 9.07 ms a token, the card should do 400+ GB/s):
- with the NVFP4 dot stubbed out (same loads, no decode arithmetic) they reach 89 / 51 us (269 / 233 GB/s): the arithmetic is ~25%, the loads' pattern most of the rest;
- loads first (all 5 sub-blocks' weight words of a lane before using any): gate/up 157 us, slower (a first version of this change had a bug: the `else` covered only a declaration, so the old loop also ran
  and the kernel did double work; after fixing it the result above stands);
- wide loads (a lane owns 4 blocks = 144 contiguous bytes, 16-byte loads, 9 lanes a row, 3 rows a warp; down: 2 blocks, 8-byte loads) with a masked shuffle tree: correct (max diff at rounding level),
  192 us (141 us with 256 registers) for gate/up, 75 us for down: slower.
The floor for decode is ~4 ms of expert reads a token (1.7 GB); reaching it needs something other than a better load pattern (likely an int8 dot product with quantized activations, which changes the
numerics: Inkling's router is full of near-ties).  Left alone.

What else is in the 8.9 ms: ~800 small dependent launches (router, loop gates, norms, residual mixes, combine) at ~10 us each; fusing them is the lever, one kernel at a time.

## 22. Inkling decode: small launches (2026-11-25), Arc Pro B60

One decode token is **806 launches** (`IK2_SYNC_CHECK=1` prints the sequence).  A looped visit is 18-19: `attn_res -> loop_gates (dots) -> loop_gates (apply) -> rms(_multi)` (before attention) ->
`mv_q8p` (qkv) -> `prep_qkv_r_f16_at` -> attention split -> combine -> `gate_mul_f16` -> `mv_q8p` (o) -> `attn_res -> dots -> apply -> rms` (before the MoE) -> `mv_f16` (router) -> `route_token` -> experts gate/up
-> experts down -> `moe_combine_conv`.  Per token: 113 `mv_q8p`, 64 `mv_f16`, 50 `attn_res`, 48 each of prep / split / combine / gate_mul / attn_res_rms / route / gate-up / down / combine_conv / loop-gate dots and apply, ~49 rms.

**Launches are cheap here; the kernels are what cost.**  A launch is ~3 us back to back (section 16), and *removing 48 of them* (below) changed nothing measurable: even deleting every launch gap would be
bounded at ~1.5 ms of a 21 ms token.  The ~9 ms the small kernels add up to is their own GPU time: real cost (sync table minus ~10 us of sync) of `loop_gates (apply)` ~26 us, `attn_res` ~18, `attn_res_rms_f16` ~17,
`prep_qkv_r_f16_at` ~16, `mv_f16` ~13, `route_token` ~12, `loop_gates (dots)` ~9, each launched with one or a few work-groups (1 of 20 cores busy): latency-bound chains of load -> reduce -> barrier -> load.
`attn_res` is already one cross-warp reduction with its loads batched four at a time; its two passes over the states are inherent.

**Shipped:**
- `IK2_GATE_TILE` default 64 (the engine's own option: a fixed-rank-64 `apply` kernel for one token): 45.0 -> 47.2 tok/s (+5%), same tokens.  The other engine knobs (`IK2_Q8_UNROLL`, `IK2_GATE_BF16`) do nothing here.
- The attention combine applies the output gate and writes the FP16 copy itself (`attn_decode_f16_at_gate`), so `gate_mul_f16` is gone from the scalar path: 806 -> 758 launches a token, bit-identical output, no speed change.
AOT: decode 47.3 tok/s at a short context, 40.8 after 4,000 tokens, 38.0 after 12,000; through the front end 47.5 tok/s short, 39-41 after 4k-7.5k-token prompts; 10 of 10 retrievals correct.

**Not done** (the two bigger fusions): the router matvec into `route_token` (a multi-block kernel that also packs the expert inputs, so a different kernel to extend), and anything around the loop gates
(`attn_res -> dots -> apply -> rms` are separated by grid-wide dependencies).  Given the measurement above, neither would gain much through the launch count; making the single-block kernels use more of the
card (split a reduction across blocks, overlap the independent q / k / v gates) is the direction that could.

## 23. Inkling decode: the small kernels (2026-11-25), Arc Pro B60

Section 22 found the ~9 ms of small kernels is their own latency, not launches.  The pattern in the slow ones: strided *scalar* loads in a serial chain, one warp or block on a card of 20 cores.  Real cost a call (sync table minus
~10 us), before -> after, and a token's saving (ms):

| kernel (calls a token) | what it was | change | before -> after |
|---|---|---|---|
| `mv_f16` (61: router 128 x 2304, small dense) | a lane: 72 scalar loads (2-byte weight, 4-byte x) | 16-byte weight loads, two float4 of x, 9 chunks a lane | 13.3 -> ~3 us (`tests/small_bench`: 19.9 -> 3.4 us back to back, max diff 1.4e-6 on |y| ~ 6) |
| `attn_res` / `attn_res_rms_f16` (49 / 46) | states loaded 4 at a time between reductions | all 12 states in flight (`kResGroup` 12, 256 registers); the sums keep their order | 18.1 -> 12.9, 17.0 -> 14.2 us |
| `prep_qkv_r_f16_at` (46): the H sink dots, K = 2304 | the same 72-load chain | the same wide loads | 16.5 us -> less (the sink warps were its longest) |
| `loop_gates (dots)` (46) | 4 scalar loads x 18 iterations | float4 loads of x, m and the weight rows (8-byte for the BF16 form) | 8.9 us -> less |
| `route_token` (46) | rank over E = 128 values in SLM, one load each | float4 loads (a count: bit-identical) | 11.6 us -> less |

Decode (AOT): 47.3 -> **52.5 tok/s** at a short context, 40.8 -> **44.8** after 4,000 tokens, 38.0 -> **41.5** after 12,000.  Through the front end 52-53 tok/s short, 44-45 after 4k-7.5k-token prompts; 10 of 10 retrievals correct.

**Numerics.**  `attn_res` (batched loads only) and `route_token` (a count) are bit-identical; `mv_f16`, the sink dots and the gate dots add in a different order, so their sums differ in the last bits (~1e-6 relative).  With the
nearly flat distribution of the `--tokens 1,2,3` smoke prompt that flips the second greedy token (16 -> 930), so that prompt is no longer a regression check across such changes.  `docs/validate/ab_greedy.py` is the gate:
ten varied prompts, temperature 0, ~120 tokens each, reference build against new build through the front end: **9 of 10 answers identical**; the tenth diverges ~80 tokens in at a near-tie ("cream or milk" vs "cream or coconut milk"), fluent both ways.
(Run it before and after any change that reorders sums; identical or near-tie divergence is the expected outcome, repeated garbage is not.)

Not done: `attn_decode` split (7 us), `loop_gates (apply)` (6), `moe_combine_conv`, `rms_multi` are small already; `prep_qkv`'s head warps and `attn_res`' two passes are the next-longest chains.

## 24. Inkling decode: the expert kernels (2026-11-25), Arc Pro B60

The generic expert kernels (gate/up + SwiGLU 118 us, down 71 us, 9.07 ms a token at ~200 GB/s: `tests/decode_experts_bench`) resisted three restructurings (section 21: loads first, wide loads, ...).  What worked is a path the
engine already had and nobody had run here: **`IK2_EXPERT_LIMB`**, the NVFP4 experts as *int8 dot products* (dp4a) over a two-limb int8 copy of the activations (`nv_input` / `pack_limbs`: two signed bytes and one FP32 scale a
16 values, ~14 bits).  As ported it was *slower* than the generic kernels (LIMB=1: 38.7 tok/s, 2: 43.7, 3: 33.6 vs 52.5): `nv_limb_dot` expanded each 4-nibble group to int8x4 with three `dpct::byte_level_permute` calls, which on
Intel are four variable 64-bit shifts each.  The activations' byte order already follows the nibbles, so the expansion is **two lookups in a 256-entry SLM table** (byte -> its two nibbles as signed int8, the doubled E2M1 values
{0, 1, 2, 3, 4, 6, 8, 12}), filled once a work-group.  Then (same session, AOT):

| `IK2_EXPERT_LIMB` | short ctx | 4000-token ctx | 12,000-token ctx |
|---|---|---|---|
| 0 (generic kernels) | 52.7 tok/s | 44.7 | 41.6 |
| **3 (gate/up + down on int8 dots; the new default)** | **56.8** | **47.7** | **44.0** |

(+7.8%, +6.6%, +6%.)  The 4 visits with Q8_0 experts keep the generic kernels.  **Numerics are the better news**: the limb path reproduces the *original* reference build's greedy answers on 10 of 10 prompts of
`docs/validate/ab_greedy.py` (the reordered-sum small-kernel changes of section 23 had moved one answer at a near-tie: "cream or milk" -> "cream or coconut milk"; limb brings it back), and the `--tokens 1,2,3` smoke test is back
to `90 16 92 1 ...`.  `IK2_EXPERT_LIMB=0` restores the generic kernels.  Retrieval 10 of 10; through the front end ~55-56 tok/s on short prompts, 45-47 after 4k-7.5k-token prompts.

**Tried on the limb kernels, no gain** (always compared with a baseline from the *same session*: the card's clocks drift ~3% between sessions, which made two experiments look slower than they were until the baseline was re-run):
the activation records staged in SLM (neutral), the 36-byte record padded to 48 / 16-byte aligned for vector loads (neutral), 4 copies of the table at staggered banks (slower: 51.3 vs 54.9, the fill), 8 or 16 warps a work-group
(54.4 / 54.1 vs 54.8), the table as a compile-time constant array (much slower: 44.5, random 2-byte constant loads).  The kernel is per sub-block lane-strided like the generic ones; the table lookups and the dp4a are no longer its cost.

What is left in the experts (~7 ms of a ~18 ms token): the limb kernels at ~84 us a launch.  The ideas that remain are the int8 *activation* side (one limb instead of two: half the dp4a, ~7 bits: a numerics change) and
splitting rows over more of the card; neither done.

## 25. Inkling decode: the expert kernels, round 2 (2026-11-25), Arc Pro B60

`tests/decode_experts_bench` now times the limb path too (`expert_limb_mv`, each call with its activation-packing launch, 8 distinct experts in rotation): generic 118 / 72 us (9.1 ms a token), limb 109 / 62 us (8.2 ms), i.e. weights read at
190-220 GB/s of a card that should do 400+.  The limb kernel's lane iteration loads the 36-byte activation record (9 scalar words) and 3 words per weight row for just two dot products; that record is 60% of the loads.

**`nv_limb_mv_rp<Gate, RP>`** (`src/ik2_limb_rp.dp.inc`, `IK2_LIMB_RP`, NVFP4 only): a warp takes RP row pairs (gate/up) or 2 RP rows (down), so the record is loaded once for 2 RP dots and 6 x 2 RP weight words are in flight.  Each row's sum is accumulated in
the same order as before, so the outputs are **bit-identical** (same difference from the generic kernels in the bench; same greedy answers).  Bench, a token's experts: RP 1 8.17 ms, 2 7.91, **4 7.32**, 8 8.56 (too few warps).
Engine, same session (AOT): decode +2.2% at a short context (56.1 vs 54.9 tok/s), +2.0% after 4000 tokens (47.3 vs 46.4), +1.7% after 12,000 (43.7 vs 42.9).  Default RP 4.

No gain: 2 / 8 warps a work-group (7.33 / 7.39 vs 7.35 ms), `#pragma unroll 2` / 4 on the sub-block loop (7.7 / 8.0 ms: registers).

Greedy A/B (`ab_greedy.py`): 10 of 10 answers identical to the original reference build and to the previous build.  Retrieval 10 of 10; through the front end 56.6 tok/s on short prompts, 46-47 after 4k-7.5k-token prompts.

The experts are now ~7 ms of a ~17.5 ms token.  Left: the activation-side (one int8 limb instead of two: half the dp4a and half the record, ~7 bits instead of ~14: a numerics change) and wider weight loads feeding the dp4a (the generic kernels' wide-load
attempt failed on their heavy arithmetic; the limb kernels' is light).

## 26. Inkling decode: one int8 limb for the experts (2026-11-25): no gain, not kept

The idea (section 25's "left"): the activation record carries two int8 limbs (~14 bits); the high limb alone is already `round(v * 127 / max)` per 16 values (`qh = (q + 64) >> 7` of the 15-bit `q`), so a kernel can use `128 * hi * scale` and skip the low limb's dp4a and its
four loads, with no change to the packing.  Built as `IK2_LIMB_ONE` on the 4-row-pair kernel and measured:

| | two limbs (default) | one limb |
|---|---|---|
| bench gate/up + swiglu | 96.0 us | 99.6 us |
| bench down | 56.6 us | 57.4 us |
| engine decode (same session, JIT) | 56.1 tok/s | 55.9 tok/s |
| synthetic accuracy (max diff from the generic kernels, saturated SwiGLU outputs) | 0.11 | 15 |

Slightly *slower* and visibly less accurate.  The limb kernels are bound by the weight traffic and the load latency of a lane-strided pass over the rows, not by the integer arithmetic or the activation record; halving the dp4a work moves nothing.  Backed out (no
dead numerics-changing path in the tree).  What would still move the experts is a different weight access pattern (wider loads per lane feeding the light dp4a; the generic kernels' attempt failed on their heavy arithmetic, this kernel's is light), not the activation side.

## 27. Inkling decode: wide weight loads in the limb kernel (2026-11-25): slower in every form, not kept

The limb kernels (section 25) read each weight row lane-strided: three 4-byte loads per 16-weight sub-block.  Three ways of reading the NVFP4 rows with wide loads, all producing the same outputs as the strided kernels
(bench: a token's experts for 8 distinct experts a launch, `tests/decode_experts_bench`; the strided limb kernel with 4 row pairs a warp: gate/up 96 us, down 57 us, 7.34 ms a token):

| form | gate/up | down | per token |
|---|---|---|---|
| a lane owns a contiguous 144-byte unit (4 blocks) = 16-byte loads, 9 lanes a row, 3 rows a warp, 256 registers; activation records from global | 149.5 us | 75.9 us | 10.8 ms |
| the same with the records staged in SLM (3 vector loads a record) | 157 us | 85 us | 11.6 ms |
| coalesced: a warp loads each row as adjacent 16-byte pieces into SLM, the strided arithmetic reads its words there (2 rows pairs a warp, 20 KB of SLM a work-group) | 146 us | 89 us | 11.3 ms |
| the same with 4 row pairs a warp (41 KB of SLM) | 214 us | 145 us | 17.3 ms |

Why: with a contiguous unit a lane the 16-byte load instruction touches one cache line per lane (27 lines, 16 of 64 bytes used each), where the strided kernel's lanes read adjacent sub-blocks that share lines, so the load count falls but the line traffic rises; the records a lane then
needs are 4x as many (16 sub-blocks, not 4.5) and, staged or not, cost more than they save; the coalesced SLM staging gets the line traffic right but takes 20-41 KB of SLM a work-group, which cuts the warps in flight more than the 5x fewer global load instructions help.

With the activation side (section 26) and the weight side both measured, the NVFP4 limb kernels are where this card's load path and occupancy put them (~250 GB/s of weights, 62% of a 400 GB/s ceiling) for any lane/row mapping I could write in SYCL; what remains is outside this kernel
(fewer, larger work per launch, fusing the pack launch into the previous kernel, or a different quantization of what the experts read).

## 28. Clean-clone build and docs (2026-11-25)

`git clone --branch intel-poly-sycl` of the repository (the `third_party/llama.cpp` submodule's files copied in, no network), then `make intel -j8` in the clone: **built all three binaries** (`unbiased-strata`, `strata-poly`, `strata-ik2`), AOT for bmg-g21 (the log is
`docs/logs/clean-clone-make-intel-2026-11-25.log`).  `make intel-test` there: exit 0, the poly tests and the six sycl-ik2 tests (results: `docs/logs/clean-clone-intel-test-results-2026-11-25.log`).  `make intel-test` now builds and runs the sycl-ik2 tests too.
Run from those binaries, one model at a time: Inkling 10 of 10 retrievals, ~1,900 tok/s prefill, 47-57 tok/s decode; Qwen 3 of 3 retrievals (decode 33 tok/s, prefill ~380 tok/s: below the 2026-11-23 figures, see NOTES.md).

A mistake worth keeping: a first Qwen check ran *while the Inkling server was still up* (my cleanup `kill` pattern did not match its process), the two shared the 24 GB card, and Qwen decoded at 2.6 tok/s.  Servers are now stopped by exact PID, and "no servers running" is
checked with `ps`, not by trusting that the `kill` matched.

Docs: `sycl-ik2/README.md` (new), Inkling in the root README (build, model files from `unsloth/Inkling-2-Mini-GGUF` `Old_Support/`, measurements, what was and was not tested), Project Maya and the Inkling model in CREDITS / CONTRIBUTORS.

## 29. Qwen prefill 380 vs 447 (2026-11-25)

**The kernels did not get slower.**  The clean-clone binary and the old Nov-23 binary both give 442-446 tok/s on the 2,710-token facts prompt run directly on the engine
(`strata-poly --tokens <ids> --max-new 60 --no-graphs --no-draft --max-context 4096`) whenever the run completes (446.3 clean; 445.8, 446.1, 436.8 old; 442.7 with `SYCL_UR_TRACE=2`).
Two other direct runs read 348 and 357 (one batch, both binaries).  The ~380 seen through the front end (server mode, 2.6k-token prompts) was not re-measured in a good state, so whether the server path costs ~15% is **open**.

**Start-up hang, much more frequent on 2026-11-25.**  Most `strata-poly` starts never reach "loaded in ...": stderr shows "the int8 XMX layout check did not run (... UR_RESULT_ERROR_DEVICE_LOST)", then the process sits in
`Engine::alloc()` -> `event::wait` -> `urEventWait` (gdb stack).  Each hung run is followed in `journalctl -k` by `xe ... Engine memory CAT error: class=bcs` (the copy engine), "Fault response: Unsuccessful -ENOENT", an engine reset and "Timedout job ... in strata-poly".
Ruled out: page cache/RAM (Qwen file fully cached, 28 GiB free), `STRATA_NO_XMX=1`, `--kv-whole`, `UR_L0_USE_COPY_ENGINE*=0`.  Standalone repros do not hang (10/10 each): a first-kernel + device-to-host copy, mmap + touching every page of the model file, reserving 22 GiB of GPU virtual address space (with and without mapping physical memory), spaced runs.
One run under `SYCL_UR_TRACE=2` completed normally.  Not fixed; the hang is the "intermittent device-lost at start" of NOTES.md, now far more frequent, so the card/driver state is the main suspect (a GPU reset or reboot would tell; not done: the card drives the displays).
Throughput numbers taken in this state should be repeated after a reboot.

**After a reboot (same day, 0 engine resets at boot, no other GPU process running):** 6 of 6 direct starts hung the same way (3 clean-clone binary, 3 old binary; 5 engine resets logged).  So the hang is **not** left-over driver state: it reproduces from a fresh boot with both binaries.  Cause still open (the standalone repros in this section do not hang); next step is bisecting the engine's start-up (what runs before the XMX layout probe) against the passing standalone programs.

**Root cause and fix (2026-11-25).**  The kernel log's fault record named the same two non-present GPU-side read addresses every run (copy engine, `bcs`).  `Engine::upload` converts weights into
temporary vectors (`q`, `f`, `h`) and calls `Engine::h2d`, which dpct had made an asynchronous `memcpy` (DPCT1124/1114) with no wait: the vector was freed on return while the copy engine had not yet read it.
Whether it faulted depended on whether the freed pages were already unmapped (large vectors are `munmap`ed at once), so it came and went with memory layout and timing, then reproduced every time.
It surfaced at the first synchronisation (the int8 layout probe: `DEVICE_LOST`), then waited forever in `Engine::alloc()`.  Same class of bug as sycl-ik2's early hang (section 10).  Fix: `.wait()` in `h2d`.
After the fix (rebuilt in the clean clone): 4 of 4 direct starts, 0 engine resets, prompt 459.7-461.5 tok/s, decode 71.5-71.8 tok/s (the 2,710-token facts prompt; earlier "no regression" runs read 442-446).
The standalone repros (pinned memory, mmap, VA reservation, copy volume) did not reproduce it because they never freed a source buffer before the copy ran.
Front end re-check after the fix (server mode, `--engine strata-poly`, same 2,754-token prompt, three requests with 30 new tokens): 6.54 / 6.19 / 6.21 s each, i.e. ~5.7 s of prefill = ~480 tok/s.  The earlier ~380 through the front end was measured while the start-up bug was
intermittently resetting the copy engine; there is no front-end-specific slowdown.

## 30. Qwen prefill: the matrix engine was never on (2026-11-25), Arc Pro B60: 460 -> ~1,175 tok/s

**Profile first.**  VTune does not support this CPU, so a scratch copy of the tree (never committed) had every `queue.submit` and oneMKL `gemm` wrapped by a script that records the events (queues with `DPCT_PROFILING_ENABLED`),
and `STRATA_PROF=1` printed GPU time a launch site at exit.  For the 2,710-token prompt: **98% of the GPU time (4.9 of 5.0 s) is the experts' products** (`iq_mmq`, 1,470 launches of ~3.3 ms), about 1.1 int8 TOPS on a card
with ~200; attention GEMMs 0.7%, everything else under 1%.  So: the prefill is the experts' tiles, and they were on dp4a (the engine had always printed "the GPU's int8 XMX accumulators are laid out otherwise ...", sections 6.3 and NOTES).

**Cause.**  icpx 2026.0 ignores a kernel's `sub_group_size` *property* (a `get(properties_tag)` member) when the build passes `-fsycl-default-sub-group-size=32`: the kernel is compiled at 32 lanes.  A 8x16 int32 joint_matrix
then hands each lane 4 elements (rows 0, 2, 4, 6) instead of 8 (column = lane, element i = row i), so the layout probe failed and rejected XMX.  (A standalone copy built without that flag passed, which is why the first
look found nothing.)  The same property-only declaration was on the XMX tile kernels themselves (`IS != 0`): enabling them by fixing only the probe gave 1,520 tok/s and **wrong answers** (retrieval and counting wrong; every
product but the first 16 columns).  Fix: `[[sycl::reqd_sub_group_size(...)]]` as an attribute on the probe and on the XMX kernels' `operator()`, `grf_size<256>` stays a property (both are accepted together by Intel's device compile).

**Verification.**  New `poly_mmq_xmx_test` (in `make intel-test`): the same grouped expert products for Q4_K, Q5_K, Q6_K, Q8_0 and IQ4_XS on synthetic weights, XMX against dp4a in two processes, relative difference 0 - 4.5e-7;
it fails unless the path is "XMX".  `run_validation.sh` PASS (log `docs/logs/validation-xmx-2026-11-25.log`): teacher-forced counting 10/10, numbered-facts retrieval exact at 334 and 2,710 tokens, counting 1..200 exact, 0 engine resets.
**Result (2,710-token prompt, `--no-draft`):** prompt 1,175-1,182 tok/s (460 before), decode 71.5 tok/s unchanged; with the draft head 94.8-95.2 tok/s on the 27-token answer.
The profile showed the matrix-engine kernels are far from the card's int8 peak still; further tile tuning (XeStrata's tiles are shaped for B70) is untried.

## 31. Qwen prefill, round 2 (2026-11-25/26): 1,175 -> ~1,395 tok/s

With the matrix engine on, the per-launch profile (same scratch instrumentation, now also wrapping `queue->parallel_for`) of the 2,710-token prompt, GPU time 2.18 s of 2.31 s wall:
experts' XMX products 59% (1.28 s, 874 us a launch), the Gated-DeltaNet state kernel 18% (0.39 s, 3.3 ms a launch), `gemm_small` 6.8%, attention batched GEMMs 5%, everything else under 2% each.  Three changes, each measured against a same-session baseline:
1. **Prompt batch 2,048** (stock 1,024; engine.hpp is shared, so the SYCL engine applies it when the option is the stock value, the card has >= 20 GiB and the context >= 2,048): 1,175 -> 1,253 tok/s at 2.7k tokens, 1,247 -> 1,377 at 7.9k;
   nothing more at 3,072.  A 1,024-token batch gives ~32 rows an expert, so the 64-row XMX tiles ran half empty.  All 10,240 experts still fit in VRAM at 8k context.  `--prompt-batch 1000` gives a smaller batch.
2. **Hardware sub-group reduction in `gdn_state_reg_kernel`** (`sycl::reduce_over_group` for its two per-token sums instead of the xor-shuffle tree: the sums are that kernel's dependency chain): 1,253 -> 1,327 (+5.8%).
   The summation order differs, so the results differ in the last bits (the kernel's old comment claimed bit-equality with the CUDA kernel); the validation below is exact on retrieval and counting.
3. **oneMKL for the small-output projections** instead of the hand-written `gemm_small` (written for cuBLAS's slow small shapes): 1,336 -> 1,404 (+5%).  `STRATA_GEMM_SMALL=1` brings the old kernel back.
`run_validation.sh` PASS (`docs/logs/validation-xmx-round2-2026-11-26.log`): teacher-forced 10/10, retrieval exact at 334 and 2,710 tokens, counting exact; prompt 1,390-1,400 tok/s, decode 72 tok/s (96 with the draft head).
Not done: the experts' XMX tile (59% of the time, ~4 int8 TOPS against ~200 peak): XeStrata's tiles are shaped for the B70; a retune for this card is the largest remaining item.  The state kernel is still 15% (one warp a column, two reductions a token).

## 32. Qwen prefill, round 3: four state columns a warp (2026-11-26): 1,395 -> ~1,510 tok/s

`gdn_state_reg_kernel` (one warp a state column, two reductions a token: a latency chain) now has `gdn_state_regn_kernel<PER, NC>`: NC consecutive columns a warp, their chains independent so the compiler overlaps them, and
the token's query / key reads shared.  Per column the arithmetic and order are the 1-column kernel's.  Measured on the 2,710-token prompt, same session: NC=1 1,389-1,406, NC=2 1,510-1,512, **NC=4 1,521-1,529**, NC=8 1,376-1,383, NC=16 1,191-1,199
(fewer warps than the card can hide latency with, and more registers).  4 is the default (`STRATA_GDN_NC=1` restores the old kernel; the 1,2,4,8,16 are built).  `run_validation.sh` PASS (`docs/logs/validation-xmx-round3-2026-11-26.log`):
teacher-forced 10/10, retrieval exact, counting exact; prefill 1,493-1,520, decode 72 (96 with the draft head, unchanged).
The 128-row XMX expert tile: `STRATA_MMQ_BIG_ROWS` (mean rows an expert above which it is taken; default 64, as before) at 16 / 32 / 64: 1,341 / 1,389 / 1,388 tok/s, 4,096-token batch no better: the tile shape is not the lever.

## 33. Qwen prefill, round 4: the weights' loads were scattered (2026-11-26): 1,510 -> ~1,765 tok/s

Ablations of the XMX expert tile on a bench at Qwen's shapes (gate/up R=1024 K=2048, down R=2048 K=512; 64 experts, 64 rows each; real `iq_mmq_grouped`, scratch copy with the parts compiled out): Q4_K gate/up 1,492 us (11.5 int8 TOPS),
**without the weights' decode 789 us** (-47%), without the products (`joint_matrix_mad`) 1,171 us (-22%), without the scale epilogue 1,312 us (-12%).  The weights are only ~50 GB/s of 456: not bandwidth, the decode's *loads*:
for a 64- or 128-row tile each work-item decoded the block of a different output row (neighbouring lanes = neighbouring rows, `by_block` false: that mapping was only for the 16-row tile), so every load instruction of the decode
(a Q4_K block is ~10 small loads) touched 16 cache lines.  `by_block` for every tile (neighbouring lanes = the four blocks of one output, then the next output: 4 rows an instruction): gate/up 1,489 -> 995 us, down (Q5_K) 657 -> 489 / 973 -> 614 us.
The model's experts are Q4_K (gate, up), Q5_K (down; Q6_K in 3 layers).  Tried and dropped: staging step k+1's activations and decoded weights in registers while step k multiplies, with the local tiles double-buffered (no gain, down slower: 657 -> 800 us);
a 256-wide K step (1,000 -> 1,800 us).  Engine: 1,510 -> 1,730-1,770 tok/s; `poly_mmq_xmx_test` unchanged (relative difference <= 4.5e-7); `run_validation.sh` PASS (`docs/logs/validation-xmx-round4-2026-11-26.log`): teacher-forced 10/10,
retrieval exact, counting exact, decode 71-72 (96-100 with the draft head).  XeStrata's note for the B70 (the 64-row tile 3% slower by block) did not hold on the B60.
Also tried on the bench and not kept: Q4_K / Q5_K block decode with vector loads (two 16-byte loads for the nibbles, the scale bytes as three words, d / dmin as one word): Q4_K gate/up 1,110-1,150 -> 983-1,001 us, down and Q5_K unchanged within the card's
clock noise (+-5%): not worth the changed decode.  What is left in the tile (bench, Q4_K gate/up, 64 rows an expert: ~17 int8 TOPS of ~200 peak): decode ALU, the products (22%) and the scale epilogue (12%); the next step would be a redesign
(decoding the weights once for several row tiles, scale application off the critical path), not tuning.

## 34. Dense Qwen3.8-27B on strata-poly (2026-11-26)

`strata-poly` refused the family on the card ("is a 'qwen35' model, which this engine does not run"): Qwen3.8-27B is architecture `qwen35`, the recurrent + attention stack of Qwen3.6-35B-A3B (`qwen35moe`) with a dense FFN in every layer
(64 layers, 5,120 wide, FFN 17,408, full attention every 4th layer, 48 value / 16 key heads of 128).  The rest of UnbiasedStrata runs these models on its `strata` engine (31,700 lines of CUDA in `src/kernels`), which is not ported; the cheaper route was to teach the engine one more shape:
- `Engine::load_shape`: a file without `expert_count` but with `feed_forward_length` is dense: `n_expert_ = n_used_ = n_ff_exp_ = 0`, `n_ff_ = n_ff_shexp_ = feed_forward_length`; the block that guesses ahead is turned off (it is built around a layer's experts).
- `Qwen35::load_layer`: for `n_expert_ == 0` the FFN goes into the shared expert's places (`Layer::sh_gate/up/down`, no header change); `layer_head()` runs it (SwiGLU, no sigmoid gate) and adds it to the residual, as Gemma4 does for its dense layers.  `layer_tail()` is not reached (no layer is `moe`).
- the engine factory takes `qwen35` as well; `tiles = true` for a dense model, so that `lin()`'s prompt batches take the XMX tiles (without it the dense matrices ran through the per-token kernel: **prefill 47 tok/s -> 350**).
Result on the 24 GB B60, `Qwen3.8-27B-Uncensored-Q4_K_M` (16.8 GB, nothing offloaded): prefill 349-351 tok/s (2,710 tokens), decode 20.9 tok/s (the weights read once a token: ~350 GB/s of the card's 456).  `run_validation.sh` PASS
(`docs/logs/validation-qwen38-27b-dense-2026-11-26.log`).  Against llama.cpp's SYCL server on the same file (ten greedy chat answers through the front end, `ab_greedy.py`): 6 identical, 2 diverging only in the last ~20 characters, 2 diverging early at a fluent near-tie
("quick, comforting" / "quick, easy").  `Q5_K_M`, `UD-IQ4_XS` and the two Jack models load and answer "Paris" to the same one-line prompt.  Qwen3.6-35B-A3B afterwards: validation PASS, prefill 1,760-1,800, decode 72 / 96 with the draft head (`validation-qwen36-35b-after-dense-2026-11-26.log`).
Not done for dense models: the draft head (see below for later sections), tuning (prefill is the same XMX tile as section 33: ~19 int8 TOPS).

## 35. Dense Qwen3.8-27B: long context, q8_0 KV, vision (2026-11-26)
`Qwen3.8-27B-Uncensored-Q4_K_M`, numbered-facts retrieval (the answer must be the exact fact), `--max-context 32768`: 8,483 tokens exact (prefill 344 tok/s, decode 24), 17,172 exact (352 / 16), 30,290 exact (341 / 13.9); the prefill rate holds,
the decode rate falls with the context (the attention reads the whole cache a token).  `--kv q8_0` was refused for `qwen35` by an architecture check only (the Qwen35 class's q8_0 cache is the same for both shapes): allowed; 8.5k and 17k exact with it (1,088 bytes a KV row).
Vision: `Jack-3.8-27B-V2-VL-16GB-VRAM.gguf` with `mmproj-Qwen3.8-27B-BF16.gguf` through the front end (`run_frontend_validation.sh`, log `docs/logs/frontend-validation-jack-vl-dense-2026-11-26.log`): chat, streaming, retrieval (4.2k tokens, prefill 393 tok/s,
decode 13 tok/s), long generation, thinking mode, tool call and the image check (green left, red right, white bar) all pass.  That fine-tune answers in a `<workspace_state>` style: its own behaviour.  0 engine resets throughout.

## 36. Dense Qwen3.8-27B: the draft head (2026-11-26): decode 20.9 -> 30 tok/s
The dense files carry the model's next-token block (`blk.64`: one attention layer, a dense FFN and the `nextn.*` projection).  Section 34 switched it off for dense models because `draft_step()` ran that block's router and routed experts.
Now `draft_step()` runs the router and the experts only when the model has experts (`n_expert_ > 0`) and otherwise just the block's FFN (`moe_shared()`, which skips the shared expert's sigmoid gate for a dense model); the switch-off is gone.
`Qwen3.8-27B-Uncensored-Q4_K_M`, 2,710-token prompt, 27-token answer: decode **20.9 -> 30.0 tok/s** (+43%; 19 of 23 guesses right), prefill unchanged (349).  Output: the ten greedy chat answers through the front end are identical with and without the draft head (10 of 10),
`run_validation.sh` PASS with it on (teacher-forced 10/10, retrieval exact, counting 1..200 exact; `docs/logs/validation-qwen38-27b-dense-draft-2026-11-26.log`).  Qwen3.6-35B-A3B afterwards: PASS, decode 72 / 96 with the draft head, prefill 1,650-1,770 (the same session's spread).

## 37. The XMX expert / dense tile: where its time goes, and the 128-register mode (2026-11-26)
Bench at Qwen3.6's shapes (real `iq_mmq_grouped`, Q4_K gate/up R=1024 K=2048 and down R=2048 K=512, 64 experts, 64 rows each; scratch copy with parts compiled out; after section 33's by-block change: gate/up ~805 us, ~21 int8 TOPS, down ~490-545 us, ~16-18):
without the weights' decode 670 us (-17%), without the SLM stores of the decoded weights 721 (-10%), without the products (`joint_matrix_mad`) 606 (-25%), without the scale epilogue 768 (-5%), without the output stores 788, without the activation loads 724.
**All of those removed together: 669 us.**  The rest is not any one thing: time grows linearly with the K steps (16 -> 8 -> 2 -> 1 steps: 806 / 448 / 178 / 131 us: ~48 us a step across the launch plus ~85 us fixed) and with the work-groups (64 / 128 / 256 / 512 / 1,024 groups: 201 / 279 / 486 / 805 / 1,609 us),
i.e. a work-group takes ~3.7 us a K step at full occupancy where its 16 sub-groups' matrix instructions need ~0.2 us.  The step is a chain of loads, local-memory stores, two barriers and the products, and nothing in it dominates: it is latency / issue bound, not bandwidth (50-100 GB/s of 456) and not matrix-engine bound.
Tried: staging step k+1 in registers while step k multiplies (section 33: no gain), a 256-wide K step (slower), 128-row tiles at lower thresholds (no gain: section 32).  **The 128-register mode** (`grf_size<128>`, 8 threads a vector engine; the 256 mode gives 4) for the 64-row tile:
gate/up 804 -> 714 us, down 488 -> 415-433 us on the bench; Qwen3.6-35B prefill ~1,770 -> ~1,820 tok/s.  The 128-row tile (dense 27B's large batches) keeps 256: with 128 the dense model's prefill fell 350 -> 309.  `STRATA_MMQ_GRF` (default 128, the B70's note in the source says 256 was 3.1x faster there: build with -DSTRATA_MMQ_GRF=256).
Both models validated after it (`validation-qwen36-35b-grf128-2026-11-26.log`, `validation-qwen38-27b-grf128-2026-11-26.log`); `poly_mmq_xmx_test` unchanged.  What would still change the picture is a different structure (a producer sub-group decoding ahead of consumer sub-groups through two local buffers with one-way barriers, or
a persistent kernel keeping its A tile resident across the n-tiles of an expert): a rewrite of `intel_body`, not tuning; not attempted.

## 38. The XMX tile, second pass: what the hardware does, and what did not help (2026-11-26)
**Correction to section 37.**  Its "all of those removed together: 669 us" floor is not a reliable measurement: builds with several ablations at once kept the same 669 us even with the kernel returning before its K loop, while each ablation alone with that return gave 14-17 us,
so the combined builds were not measuring what their flags said (dead-code effects of the ablation macros).  The single-flag numbers stand (decode -17%, SLM stores -10%, products -25%, epilogue -5%, activation loads -10%, output stores -2%); they do not add up to 100% and nothing dominates.
**What the hardware does.**  A register-only loop of `joint_matrix_mad` (int8 8x16x32, 8 independent accumulators, 16-lane sub-groups, 320 work-groups of 256 threads) reaches **~190 int8 TOPS in the 128-register mode** (84-110 in the 256 one, fewer threads): the card and the instruction are not the limit; the tile's 21 TOPS is ~11% of it.
An empty kernel of the tile's launch shape (512 groups of 16 sub-groups, 44 KB local memory, either register mode) takes ~9 us: dispatch is not it either.  A work-group's time grows linearly with its K steps (~3.7 us a step where its matrix instructions take ~0.2) and its work-groups with their count.
**Tried on the bench (Q4_K gate/up, 64 rows an expert, base 805 us) and not kept:** prefetching the weights' superblock and the activations' pieces 1, 2 or 4 steps ahead (`sycl::ext::oneapi::experimental::prefetch`): +8% slower (memory latency is not the limit);
the 8 row scales read as two vector loads: 0%; all of a block's tiles' products issued before their epilogues (independent accumulators): +1.5% slower; staging the next step in registers behind a double-buffered local tile (section 33): slower; a 256-wide K step: slower; 128-row tile thresholds: 0%.
**Reading:** per 16-cycle matrix instruction the scale epilogue (convert, multiply, fma for each of 8 results: ~24 vector operations on 16 lanes) and the weight decode need more vector-ALU cycles than the matrix engine needs, and the instruction issue of a vector engine is shared by its 4-8 threads: the tile is issue-bound on the per-block scales, which Q4_K / Q5_K force
(a different scale for every 32 of K on both operands).  What would change that is fewer vector operations a block (e.g. the scales applied to packed halves, a block's scales merged for pairs of blocks where the type allows it, or half-precision tiles with the weights pre-scaled: a decode that costs about what the epilogue saves, as `sycl-ik2`'s NVFP4 XMX GEMM found),
which is a new kernel with a different numerical contract, to be validated as one, not a restructuring of this one.  Not started.
Two more, same bench: the Q4_K / Q5_K minimum term of the epilogue (`acc -= wmn * xu`, 16 more operations a tile) costs ~7% (806 -> 753 us); with it gone the main scale epilogue costs nothing measurable (753 -> 756 with it removed too), so the epilogue is not the limit either.
Trimming the local memory a group asks for from 44 KB to 36 KB (the Intel body never uses the 8 KB staging tile the other bodies share) to fit a third group per core: 0% (806 -> 805 / 808 us).  The tile is balanced across the decode, the local-memory traffic, the barriers, the products and the epilogue, each worth 5-25%, none
more: a gain of a factor needs a different structure (and numerical contract), not a change to one of them.
