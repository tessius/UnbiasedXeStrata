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
