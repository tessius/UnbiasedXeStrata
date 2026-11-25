// SPDX-FileCopyrightText: 2026 the UnbiasedStrata SYCL contributors
// SPDX-License-Identifier: MIT
// CUDA runtime calls that dpct left in ik2_main.cu (in macros and lambdas), mapped by hand.  Every function returns 0 on success.
#pragma once
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdlib>

enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };
constexpr unsigned cudaHostAllocPortable = 1;

// pinned host memory: what the transfers between the card and the host go through
template <class T> inline int cudaHostAlloc(T **p, size_t n, unsigned) {   // (CUDA's C++ header takes any T**, as here)
    try { *p = static_cast<T *>(sycl::malloc_host(n ? n : 1, dpct::get_in_order_queue())); return *p ? 0 : 1; }
    catch (const sycl::exception &) { *p = nullptr; return 1; }
}
inline int cudaMemcpyAsync(void *dst, const void *src, size_t n, cudaMemcpyKind, dpct::queue_ptr q) {
    try { q->memcpy(dst, src, n); return 0; } catch (const sycl::exception &) { return 1; }
}
inline int cudaStreamSynchronize(dpct::queue_ptr q) {
    try { q->wait(); return 0; } catch (const sycl::exception &) { return 1; }
}
// cudaEventRecord(e, stream): the event completes when the work already in the stream has.
inline int cudaEventRecord(dpct::event_ptr e, dpct::queue_ptr q) {
    try { dpct::sync_barrier(e, q); return 0; } catch (const sycl::exception &) { return 1; }
}
// cudaLaunchHostFunc(stream, fn, user): fn(user) runs on the host after the work already in the stream (a host task in an in-order queue).
inline int cudaLaunchHostFunc(dpct::queue_ptr q, void (*fn)(void *), void *user) {
    try { q->submit([=](sycl::handler &h) { h.host_task([=] { fn(user); }); }); return 0; } catch (const sycl::exception &) { return 1; }
}
inline int cudaGraphExecDestroy(dpct::experimental::command_graph_exec_ptr e) { delete e; return 0; }
inline int cudaGraphDestroy(dpct::experimental::command_graph_ptr g) { delete g; return 0; }
