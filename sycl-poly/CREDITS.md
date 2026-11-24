# Credits, licenses and provenance of Unbiased XeStrata

**Unbiased XeStrata** is [UnbiasedStrata](https://github.com/robertkarl/UnbiasedStrata) with Intel GPU support taken from
[XeStrata](https://github.com/MistVVK/XeStrata).  The Intel part lives in `sycl-poly/` and builds UnbiasedStrata's
`strata-poly` engine for Intel GPUs through SYCL.  It stands on other people's work; this file names it, says what we took,
and where each license text is.  The name only says what the project is made of: it is not affiliated with, or endorsed by,
the authors of UnbiasedStrata, XeStrata or Strata.  If a credit is missing or wrong, that is our mistake:
please open an issue.

**Licenses at a glance.** The repository is MIT (`../LICENSE`).  `src/xe/` and `src/poly/mmq_sycl.cpp` are
**LGPL-3.0-or-later** (parts MIT), because they are XeStrata's code.  The vendored dpct helper headers
(`../sycl/include/dpct/`) are Apache-2.0 WITH LLVM-exception.  A binary that links `src/xe/` is therefore a combined work
under the LGPL for those files: keep them in their own library (`sp_mmq`), ship their source and the license texts, and let
users relink.  All texts are in `LICENSES/` and `src/xe/COPYING*`.

## What we took from whom

| Component | Author / project | License | Where it is here | Source and version |
|---|---|---|---|---|
| **Strata**: the engine, kernels, GGUF reader, everything `strata-poly` is built from | Niko1221 and the Strata contributors | MIT | `../src/` (compiled, not copied); dpct's migration of it in `src/poly/*.dp.cpp`, `src/poly/engine.cpp`, `src/poly/{qwen35,glm5next,gemma4,deepseek4,lfm2moe}.cpp`, `src/kernels/cuda/{native_mmvq,iq_kernels}.dp.cpp` | https://github.com/Niko1221/Strata (v0.1.39 `a1641e9f77aacad4d201b53c8a7ae8fa21059ebb` is the base of the older `../sycl/` port; the migrated sources are of UnbiasedStrata `9f75192287c2abf88c77da3b4b6f54ccee01db83`) |
| **PolyStrata**: the `strata-poly` multi-model engine (Qwen3.5/3.6, GLM, Gemma, DeepSeek, LFM2) | VecSzn and contributors | MIT | `../src/poly/` | https://github.com/VecSzn/PolyStrata v0.1.45-poly.1 `d0cd84dc6bd009d1d9e2191e4e28cc76b80f1ee3` |
| **UnbiasedStrata**: the tree this is part of | Robert Karl and contributors | MIT | everything outside `sycl-poly/` | https://github.com/robertkarl/UnbiasedStrata |
| **XeStrata**: the SYCL port of the quantized matrix tiles (`iq_mmq`, dp4a and Intel XMX), used as poly's `Mmq` | MistVVK and the XeStrata contributors | LGPL-3.0-or-later AND MIT (file by file, see headers) | `src/xe/` (lifted, include paths changed, see `src/xe/README.md`); `src/poly/mmq_sycl.cpp` is written against its interface | https://github.com/MistVVK/XeStrata commit `721afe51f223fe5e60470a06a22c9961c7625fa8` (2026-10-30) |
| **ggml / llama.cpp**: the quantized-matrix tile design that `iq_mmq.cpp` ports, the quant formats, `ggml-cpu`/`ggml-base`, `ggml-common.h`, the `llama-tokenize` the validation uses | The ggml authors (Georgi Gerganov and contributors) | MIT | `src/xe/iq_mmq.cpp` (transcribed parts), `../third_party/llama.cpp` (submodule, built as ggml-cpu/ggml-base) | https://github.com/ggml-org/llama.cpp; the submodule is robertkarl/llama.cpp branch `UnbiasedStrata` `db1f382bb25f9cc481893105dbead09c86a52f01` |
| **The older SYCL port** (`../sycl/`): `strata/sycl_queue.hpp` (`q_of`), the vendored dpct helper library, the migration/fixup tooling and the approach (dpct + 3-way merge), whose notes taught most of what is in `docs/INTEL.md` | maxfridbe (Strata PR #423) and the people who tested it | MIT (Strata); dpct headers Apache-2.0 WITH LLVM-exception | `../sycl/include/`, `../sycl/tools/`, `../docs/INTEL.md`, `../docs/INTEL_ARC.md` | https://github.com/Niko1221/Strata/pull/423 |
| **dpct / SYCLomatic helper headers** (`dpct/*.hpp`) | Intel Corporation | Apache-2.0 WITH LLVM-exception | `../sycl/include/dpct/` (vendored; `sycl-poly` includes them from there) | Intel DPC++ Compatibility Tool 2025.3 (SYCLomatic) |
| **Intel DPC++/C++ compiler (icpx), oneMKL, Level Zero runtime** | Intel Corporation | Intel's own license terms (oneAPI); we did not check them in detail | not distributed; the build uses what is installed | icpx 2026.0.0.20260331, oneMKL 2026.0 |
| **Qwen3.6-35B-A3B** (the model used to validate) | Qwen team (Alibaba); GGUF quantization and the MTP layer by Unsloth | the model's own license (see the model card; we did not verify it) | not distributed | https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF, file `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` |

`../docs/HOW_IT_WORKS.md#credits` and `../docs/DETAILS.md#credits-and-licenses` credit the rest of the Strata family
(Project Maya, Splash, ninfer, HyperQwen, eddoursul/Strata, the model makers).  XeStrata's own credits (oneMath, intel/llvm,
the Debian rocm-llvm packaging, AMD's device libraries, KaTeX, stb, Outfit, Project Maya) are in `src/xe/NOTICE.XeStrata`;
none of those parts are used here.

## What is ours

- `CMakeLists.txt`, `src/poly/mmq_sycl.cpp` (the `Mmq` class over XeStrata's tiles; LGPL because it is written against XeStrata's
  interface), `src/xe/xe_types.hpp`.
- The hand fixes to dpct's output (listed in `DEVLOG.md`), the `#if 0` cuts of the Strata engine's grouped-expert kernels, the
  private-constant codebooks, `src/poly/engine.cpp`'s virtual-memory/events/graph leftovers and `runtime.hpp`'s shim.
- `tools/fix_kernel_names.py`, `tools/migrate_poly.sh` (adapted from `../sycl/tools/merge_upstream.py` and `migrate.sh`).
- The kernel unit tests (`src/poly/{gdn,attend,rope}_q_test.cpp`) and `docs/validate/` (MIT).
- This documentation and `DEVLOG.md`.

The claims in `DEVLOG.md` are backed by output kept in `docs/logs/`
(`docs/logs/history/measurements-by-hand.txt` marks the few numbers that were noted from a terminal).

## If you redistribute

1. Keep `LICENSE` (MIT), `LICENSES/`, `src/xe/COPYING`, `src/xe/COPYING.LESSER`, `src/xe/NOTICE.XeStrata`, and this file.
2. Keep the SPDX headers in `src/xe/*` and `src/poly/mmq_sycl.cpp`.  Mark any change you make to them (LGPL-3.0 section 4).
3. Offer the source of `src/xe/` with any binary, and keep it linkable (`sp_mmq` is a static library: a user can relink).
4. Do not remove XeStrata's, ggml's, Strata's or Intel's copyright notices.
5. The model files are not ours and not covered here.

*We are not lawyers.  This file is our good-faith account of the licenses involved; check the license headers and texts yourself before you redistribute.*
