# Unbiased XeStrata

[UnbiasedStrata](https://github.com/robertkarl/UnbiasedStrata)'s `strata-poly` engine on **Intel GPUs**, through SYCL, with the
Intel support taken from [XeStrata](https://github.com/MistVVK/XeStrata).

**Status: experimental.**  Tested on one machine: an Intel Arc Pro B60 (24 GB, Battlemage), one model (Qwen3.6-35B-A3B,
Unsloth `UD-Q4_K_XL`).  On it, with every expert in VRAM:

| | |
|---|---|
| prefill | ~1,395 tok/s (2,700-token prompt; the experts' products run on the Intel matrix engine) |
| decode | ~72 tok/s without the draft head, ~95 with it on short answers (varies with the answer's length) |
| correctness | exact on retrieval at 2.7k tokens and on counting 1..200; kernel unit tests against CPU references pass |

Other models (GLM, Gemma, DeepSeek, LFM2), other Intel GPUs, multi-GPU, CUDA graphs and the images path are **untested**.
Read [`NOTES.md`](NOTES.md) for what is missing and [`DEVLOG.md`](DEVLOG.md) for how it was made, what failed, and every number.

## Build and run

You need Intel's oneAPI (DPC++ `icpx` and oneMKL; built with 2026.0), the Intel GPU driver (Level Zero), CMake, Ninja, and the
git submodule.

```
git clone --recursive <this repository>
source /opt/intel/oneapi/setvars.sh
# ahead-of-time for an Arc Pro B60 / B580 (bmg-g21): starts in seconds.  Leave STRATA_SYCL_AOT empty for the JIT build,
# which compiles the GPU code at run time and takes minutes at every start.
cmake -S sycl-poly -B build-poly -G Ninja -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DSTRATA_SYCL_AOT=bmg-g21
ninja -C build-poly strata-poly
ONEAPI_DEVICE_SELECTOR=level_zero:gpu build-poly/strata-poly --model <model.gguf> --tokens <ids> --no-graphs
```
`--serve` is the server mode of `strata-poly` (see its `--help`).  Stop other programs that hold the GPU first.

## Check it yourself

`docs/validate/run_validation.sh` runs the kernel unit tests, a teacher-forced check, exact-retrieval checks, a counting check and
a speed measurement, and prints PASS or FAIL.  Our run is in `docs/logs/validation-2026-11-23.log`.  If you change a kernel, run it:
the bug that took the longest to find was a one-character precedence error in migrated code that only showed after ~90 tokens.

## Licenses and credits

Mixed: MIT for most of it; **`src/xe/` and `src/poly/mmq_sycl.cpp` are LGPL-3.0-or-later (parts MIT)** because they are
XeStrata's code; the vendored dpct helper headers are Apache-2.0 WITH LLVM-exception.  Who wrote what, which versions we took, and
what you must keep if you redistribute: [`CREDITS.md`](CREDITS.md).  License texts: [`LICENSES/`](LICENSES/) and `src/xe/COPYING*`.
The models are not included and have their own licenses.

This project is not affiliated with or endorsed by the authors of UnbiasedStrata, XeStrata, Strata, llama.cpp or Intel.

## The Inkling engine

The same dpct approach was applied to UnbiasedStrata's `strata-ik2` (Inkling-2-Mini): build, run, measurements and switches are in [`../sycl-ik2/README.md`](../sycl-ik2/README.md); `make intel` builds both engines.
