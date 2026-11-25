#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
# SPDX-License-Identifier: MIT
# Re-creates sycl-ik2's dpct input and output.  usage: migrate_ik2.sh TREE OUT     (TREE: a `git archive` of the repo)
#  1. `__ldg(` becomes `ik2_ldg(` in the tree's ik2 sources (and a CUDA-side ik2_ldg.cuh is added): dpct rewrites __ldg to a bare
#     dereference and in 6 places dropped the vector cast in `__ldg((const float4*) p)` - with a wrapper it cannot.
#  2. dpct 2025.3 over ik2_kernels.cu, ik2_main.cu, ik2_draft.cu (../../sycl/tools/migrate.sh).
#  3. fixups_ik2.py on the result (the hand fixes, as code).
# Needs dpct on PATH and CUDA_HEADERS (see ../../sycl/tools/get-cuda-headers.sh).
set -euo pipefail
tree=$(cd "$1" && pwd); out=$2; here=$(cd "$(dirname "$0")" && pwd)
cd "$tree/src/ik2"
sed -i 's/__ldg(/ik2_ldg(/g' ik2_kernels.cu ik2_kernels.cuh ik2_main.cu ik2_draft.cu
cat > ik2_ldg.cuh <<'CU'
#pragma once
template <class T> __device__ __forceinline__ T ik2_ldg(const T* p) { return __ldg(p); }
CU
sed -i '0,/^#pragma once/s//#pragma once\n#include "ik2_ldg.cuh"/' ik2_kernels.cuh
cd "$tree"
MIGRATE_ONLY="src/ik2/ik2_kernels.cu src/ik2/ik2_main.cu src/ik2/ik2_draft.cu" bash "$here/../../sycl/tools/migrate.sh" "$tree" "$out"
sed -i -E 's/DPCT([0-9]{4}):[0-9]+:/DPCT\1:/' "$out"/src/ik2/*.dp.cpp "$out"/src/ik2/*.hpp
python3 "$here/fixups_ik2.py" "$out/src/ik2"
python3 "$here/fix_kernel_names_ik2.py" "$out"/src/ik2/ik2_kernels.dp.cpp "$out"/src/ik2/ik2_main.dp.cpp "$out"/src/ik2/ik2_draft.dp.cpp
