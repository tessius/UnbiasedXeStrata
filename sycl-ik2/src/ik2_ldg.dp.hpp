// SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
// SPDX-License-Identifier: MIT
// ik2_ldg: what the CUDA source's __ldg(p) is, here a plain load (dpct's own rewrite of __ldg dropped vector casts).
#pragma once
template <class T> inline T ik2_ldg(const T* p) { return *p; }
