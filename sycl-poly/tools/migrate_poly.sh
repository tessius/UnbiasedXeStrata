#!/usr/bin/env bash
# Re-creates sycl-poly's dpct input: dpct 2025.3 over the poly sources of a tree, using ../../sycl/tools/migrate.sh.
#   migrate_poly.sh TREE OUT     TREE = a `git archive` of the repo; OUT = an empty directory
# Then: take src/poly/{kernels,kernels_dsv4}.dp.cpp, engine.cpp, runtime.hpp, X.cpp.dp.cpp -> src/poly/X.cpp
# (qwen35 glm5next gemma4 deepseek4 lfm2moe), src/kernels/cuda/{native_mmvq,iq_kernels}.dp.cpp and the headers they
# need from OUT, strip the DPCT message serials (sed -E 's/DPCT([0-9]{4}):[0-9]+:/DPCT\1:/'), run tools/fix_kernel_names.py,
# and re-apply the hand fixes listed in NOTES.md.
# dpct: intel-oneapi-dpcpp-ct 2025.3 (extract Intel's .deb; no install needed), CUDA headers: ../../sycl/tools/get-cuda-headers.sh
set -euo pipefail
tree=$(cd "$1" && pwd); out=$2
here=$(cd "$(dirname "$0")" && pwd)
export MIGRATE_ONLY="src/poly/qwen35.cpp src/poly/glm5next.cpp src/poly/gemma4.cpp src/poly/deepseek4.cpp src/poly/lfm2moe.cpp src/poly/main.cpp src/poly/cpu_experts.cpp"
# (the .cu files and the .cpp files that include a CUDA header themselves, engine.cpp among them, are picked up by migrate.sh)
bash "$here/../../sycl/tools/migrate.sh" "$tree" "$out"
