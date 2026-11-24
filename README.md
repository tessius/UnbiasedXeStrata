# UnbiasedXeStrata

**[UnbiasedStrata](https://github.com/robertkarl/UnbiasedStrata) with Intel GPU support**, taken from
[XeStrata](https://github.com/MistVVK/XeStrata).  Runs UnbiasedStrata's `strata-poly` engine on Intel Arc GPUs through SYCL.

> **Experimental.**  Written and tested on one machine on **2026-11-23**: an Intel Arc Pro B60 (24 GB), one model (Qwen3.6-35B-A3B,
> Unsloth `UD-Q4_K_XL`).  Other models, other Intel GPUs, multi-GPU and CUDA graphs are untested.
> The original UnbiasedStrata README (NVIDIA/CUDA build) is kept, unchanged, as [`README.upstream.md`](README.upstream.md).
> Not affiliated with or endorsed by the authors of UnbiasedStrata, XeStrata, Strata, llama.cpp or Intel.

| Qwen3.6-35B-A3B on an Arc Pro B60, all experts in VRAM | |
|---|---|
| Prefill | ~447 tok/s (2,700-token prompt) |
| Decode | 49-93 tok/s depending on the answer's length and whether the draft head is on |
| Correctness | exact on retrieval at 2.7k tokens and on counting 1..200; kernel unit tests against CPU references pass |

Everything Intel-specific is in [`sycl-poly/`](sycl-poly/).  **No existing UnbiasedStrata file was changed** except moving this
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
build for Battlemage so it starts in seconds.  Prefill went from 178 tok/s (stub `Mmq`) to ~447.

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
- Not done, in priority order: Intel XMX for `Mmq` (the layout probe rejects it with icpx 2026.0, so the tiles run on dp4a),
  CUDA graphs, the other models, other GPUs, an intermittent device-lost at start.  Details: [`sycl-poly/NOTES.md`](sycl-poly/NOTES.md).
- To check any change: `sycl-poly/docs/validate/run_validation.sh` (prints PASS/FAIL; our PASS log is in `sycl-poly/docs/logs/`).

## Documents

| | |
|---|---|
| [`sycl-poly/README.md`](sycl-poly/README.md) | how to build and run |
| [`sycl-poly/DEVLOG.md`](sycl-poly/DEVLOG.md) | the full development log: versions, commands, numbers, dead ends, mistakes |
| [`sycl-poly/CREDITS.md`](sycl-poly/CREDITS.md) | who wrote what, licenses, what to keep if you redistribute |
| [`sycl-poly/NOTES.md`](sycl-poly/NOTES.md) | status and what is missing |
| [`README.upstream.md`](README.upstream.md) | UnbiasedStrata's original README (NVIDIA build) |

## License

MIT ([`LICENSE`](LICENSE), "Copyright (c) 2026 Niko1221 and the Strata contributors"), with these exceptions: `sycl-poly/src/xe/` and
`sycl-poly/src/poly/mmq_sycl.cpp` are LGPL-3.0-or-later (some parts MIT), the vendored dpct headers are Apache-2.0 WITH
LLVM-exception, and every model has its own license.  Texts: `sycl-poly/LICENSES/`, `sycl-poly/src/xe/COPYING*`.
