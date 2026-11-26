# UnbiasedXeStrata

**[UnbiasedStrata](https://github.com/robertkarl/UnbiasedStrata) with Intel GPU support**, taken from
[XeStrata](https://github.com/MistVVK/XeStrata).  Runs two of UnbiasedStrata's engines on Intel Arc GPUs through SYCL: `strata-poly` (Qwen3.6 and other models) and `strata-ik2` (**Inkling-2-Mini**).

> **Experimental.**  Written and tested on one machine, **2026-11-23 to 2026-11-25**: an Intel Arc Pro B60 (24 GB), two models (Qwen3.6-35B-A3B `UD-Q4_K_XL`
> on `strata-poly`; Inkling-2-Mini `UD-Q4_K_XL` on `strata-ik2`).  Other models, other Intel GPUs, multi-GPU and CUDA graphs are untested.  UnbiasedStrata's other engines
> (`strata` Flash-Next, `strata-dsv41`, `hy4flash`, the Metal ones) are **not ported**.
> The original UnbiasedStrata README (NVIDIA/CUDA build) is kept, unchanged, as [`README.upstream.md`](README.upstream.md).
> Not affiliated with or endorsed by the authors of UnbiasedStrata, XeStrata, Strata, Project Maya, llama.cpp or Intel.

| Qwen3.6-35B-A3B on an Arc Pro B60, all experts in VRAM | |
|---|---|
| Prefill | ~1,395 tok/s (2,700-token prompt; 447 before the matrix engine was enabled) |
| Decode | ~72 tok/s without the draft head, ~95 with it (short answers; varies with the answer's length) |
| Correctness | exact on retrieval at 2.7k tokens and on counting 1..200; kernel unit tests against CPU references pass |

| Inkling-2-Mini on an Arc Pro B60, all experts in VRAM (`strata-ik2`) | |
|---|---|
| Prefill | ~1,900-2,000 tok/s (4,000-12,000-token prompts) |
| Decode | ~56 tok/s on a short context, ~47 after 4,000 tokens, ~44 after 12,000 |
| Correctness | 10 of 10 needle-in-a-haystack answers right at ~4k and ~7.5k tokens; greedy answers match the reference build; kernel tests pass |

## Build and run (Intel GPUs)

You need Intel oneAPI (`icpx`, oneMKL), the Intel GPU driver (Level Zero), CMake and a C++20 compiler.  No CUDA.

```
git clone --recursive https://github.com/tessius/UnbiasedXeStrata && cd UnbiasedXeStrata
make intel                       # front end + SYCL engine into build-intel/bin/ (detects your card; first build takes a while)
build-intel/bin/unbiased-strata --engine strata-poly -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --port 8080 \
    -- --model Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --max-context 32768 --no-graphs
```
**Inkling-2-Mini** (`strata-ik2`, 24 GB card): use Unsloth's GGUF from the `Old_Support/` folder of
[`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) (rename `*.gguf_file` to `*.gguf`; the files in the repository root use a newer layout the engine cannot read):
```
build-intel/bin/unbiased-strata --engine strata-ik2 -m Inkling-2-Mini-UD-Q4_K_XL.gguf --alias inkling-2-mini --port 8080 \
    -- --model Inkling-2-Mini-UD-Q4_K_XL.gguf --max-context 8192
```
Details, switches and what was tried: [`sycl-ik2/README.md`](sycl-ik2/README.md).  Tested: chat answers and needle-in-a-haystack retrieval up to 12k tokens.  Untested here: streaming, tool calls, image input, the speculative drafter.

Then use the OpenAI-style API on port 8080 as you would with UnbiasedStrata (chat, streaming, thinking mode, tool calls, images with
`--mmproj`).  `make intel-test` runs the Intel kernels' unit tests.  Everything Intel-specific is in [`sycl-poly/`](sycl-poly/) and [`sycl-ik2/`](sycl-ik2/).  **No existing UnbiasedStrata file was changed** except moving this
README to `README.upstream.md`.  Build and run: [`sycl-poly/README.md`](sycl-poly/README.md).

## Why this exists, and what we found (2026-11-23)

**The starting point.**  UnbiasedStrata already contains a `sycl/` directory: a community Intel port (Strata PR #423) that
its own plan lists as "kept, unbuilt".  It ports only the `strata` engine (Qwen3.8-Flash-Next).  It had not been built or run against this tree
(its plan says so),
and the models people run on a 24 GB card (Qwen3.6 and similar) go through `strata-poly`, which it does not cover.  XeStrata has
Intel support, but it is a separate rewrite with its own runtime; 46 of its 52 kernel headers differ from UnbiasedStrata's, so it
cannot be dropped in.

**What we tried first: refresh the old `sycl/` port.**  Following its own maintenance procedure (migrate the old and the new
tree with Intel's dpct, then 3-way merge) gave **698 conflicting hunks in 46 files**, on top of a merge base that turned out to be
Strata v0.1.39, not the documented 0.1.40.2.  It could not be validated either: the engine it ports needs model weights that were
not available.  **Conclusion: not worth it.**  We dropped it (nothing from that attempt is in the tree).

**What we did instead: migrate `strata-poly` from scratch.**  dpct over `src/poly` and the few Strata kernels it calls, then by-hand fixes to
what dpct got wrong; a real SYCL implementation of the prompt-path matrix product (`Mmq`) built on XeStrata's tiles; an ahead-of-time
build for Battlemage so it starts in seconds.  Prefill went from 178 tok/s (stub `Mmq`) to ~447 on dp4a, then to ~1,395 with the Intel matrix engine (XMX) enabled and tuned (2026-11-25, `sycl-poly/DEVLOG.md` sections 30-31).

**The bug that mattered.**  The first working build gave right answers to short prompts and garbage after about 90 tokens.
It was not in the new `Mmq`.  CPU-reference unit tests for the attention, RoPE and gated-delta-rule kernels found it: dpct had
turned `__fmul_rn(__fsub_rn(a, b), c)` into `a - b * c` (it dropped the parentheses, diagnostic DPCT1013), which corrupted the
recurrent state from the second token on.  **If you migrate CUDA code with dpct, read every DPCT1013 and test every kernel
against a CPU reference over many steps, not one.**  Other dpct problems we hit (kernel names without template arguments,
dropped constant-table arguments, masked sub-group shuffles, Level Zero needing 2 MiB physical-memory granularity) are in
[`sycl-poly/DEVLOG.md`](sycl-poly/DEVLOG.md).

## If UnbiasedStrata's author wants Intel support upstream

This is meant to be usable that way.  Everything is additive, in one directory, and documented:

- `sycl-poly/` builds `strata-poly` for SYCL without touching `src/`.  It compiles the repo's own sources where nothing uses
  CUDA, and carries dpct's migrated copies of the rest.  Merging it is adding a directory (plus the README move).
- Two pieces are **LGPL-3.0-or-later** because they are XeStrata's code: `sycl-poly/src/xe/` and `sycl-poly/src/poly/mmq_sycl.cpp`
  (kept in a separate static library, `sp_mmq`).  Everything else we wrote is MIT.  See [`sycl-poly/CREDITS.md`](sycl-poly/CREDITS.md).
  If the LGPL is a problem, the alternative is to write a `Mmq` of your own; the interface is `src/poly/mmq.hpp` and XeStrata's tiles
  are the reference.
- The old `sycl/` port is still stale against this tree and still unbuilt; this work does not depend on it except for its vendored dpct
  helper headers and `strata/sycl_queue.hpp`.  Whether to keep, refresh or remove it is your call.
- Not done, in priority order: CUDA graphs, the other models, other GPUs, an intermittent device-lost at start.  Details: [`sycl-poly/NOTES.md`](sycl-poly/NOTES.md).
- To check any change: `sycl-poly/docs/validate/run_validation.sh` (prints PASS/FAIL; our PASS log is in `sycl-poly/docs/logs/`).

## Documents

| | |
|---|---|
| [`sycl-poly/README.md`](sycl-poly/README.md) | how to build and run `strata-poly` (Qwen3.6 and others) |
| [`sycl-ik2/README.md`](sycl-ik2/README.md) | how to build and run `strata-ik2` (Inkling-2-Mini), its switches and measurements |
| [`sycl-poly/DEVLOG.md`](sycl-poly/DEVLOG.md) | the full development log: versions, commands, numbers, dead ends, mistakes |
| [`sycl-poly/CREDITS.md`](sycl-poly/CREDITS.md) | who wrote what, licenses, what to keep if you redistribute |
| [`sycl-poly/NOTES.md`](sycl-poly/NOTES.md) | status and what is missing |
| [`README.upstream.md`](README.upstream.md) | UnbiasedStrata's original README (NVIDIA build) |

## License

MIT ([`LICENSE`](LICENSE), "Copyright (c) 2026 Niko1221 and the Strata contributors"), with these exceptions: `sycl-poly/src/xe/` and
`sycl-poly/src/poly/mmq_sycl.cpp` are LGPL-3.0-or-later (some parts MIT), the vendored dpct headers are Apache-2.0 WITH
LLVM-exception, and every model has its own license.  `sycl-ik2/` has no LGPL parts.  Texts: `sycl-poly/LICENSES/`, `sycl-poly/src/xe/COPYING*`.
