# strata-poly for Intel GPUs (SYCL). Sources are dpct's migration of ../src/poly (tools/migrate_poly.sh) plus the
# kernels it calls; everything else is ../src unchanged. Experimental: see NOTES.md.
#   cmake -S sycl-poly -B build-poly -G Ninja -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx && ninja -C build-poly strata-poly
cmake_minimum_required(VERSION 3.24)
project(strata_poly_sycl LANGUAGES C CXX)
set(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_definitions(STRATA_VERSION="poly-sycl")
# kernels are written for 32-lane warps (see ../sycl/CMakeLists.txt); fp32 divide/sqrt must round like nvcc's
set(STRATA_SYCL_AOT "" CACHE STRING "AOT device for the GPU code, e.g. bmg-g21 (Arc Pro B60); empty: SPIR-V, JIT-compiled at run time (minutes at first start)")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fsycl;-fsycl-default-sub-group-size=32>" -fp-model=precise -w)
# CUDA's __dp4a (src/xe/cuda_intrinsics.hpp) is SPV_KHR_integer_dot_product
add_link_options("SHELL:-Xspirv-translator --spirv-ext=+SPV_KHR_integer_dot_product")
if(STRATA_SYCL_AOT)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fsycl-targets=spir64_gen>")
  add_link_options(-fsycl -fsycl-default-sub-group-size=32 -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen
                   "-device ${STRATA_SYCL_AOT} -options -cl-fp32-correctly-rounded-divide-sqrt")
else()
  add_link_options(-fsycl -fsycl-default-sub-group-size=32 -Xsycl-target-backend=spir64
                   "-cl-fp32-correctly-rounded-divide-sqrt -ze-opt-greater-than-4GB-buffer-required")
endif()

# ggml-cpu / ggml-base from the llama.cpp submodule (CPU experts, GGUF types)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
foreach(o GGML_CUDA GGML_SYCL GGML_OPENMP GGML_BACKEND_DL GGML_BUILD_TESTS GGML_BUILD_EXAMPLES)
  set(${o} OFF CACHE BOOL "" FORCE)
endforeach()
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
add_subdirectory(${ROOT}/third_party/llama.cpp/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)

find_package(Threads REQUIRED)
# migrated headers and dpct's helper library first, then the repo's
# (../sycl/include goes last: only its dpct/ helpers and strata/sycl_*.hpp are wanted, its strata/kernels are the old port's)
set(INC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda ${CMAKE_CURRENT_SOURCE_DIR}/src/poly
        ${ROOT}/include ${ROOT}/src/poly ${ROOT}/third_party/ggml ${ROOT}/sycl/include)

# CPU experts (cpu_experts.cpp) and the speculative-decoding helpers: unchanged
add_library(sp_cpu STATIC ${ROOT}/src/kernels/cpu/expert.cpp ${ROOT}/src/kernels/cpu/pool.cpp
        ${ROOT}/src/kernels/cpu/expert_layout.cpp ${ROOT}/src/kernels/cpu/q2_avx2.cpp
        ${ROOT}/src/kernels/cpu/native_expert.cpp ${ROOT}/src/kernels/cpu/iq_avx512.cpp ${ROOT}/src/kernels/cpu/iq_avx2.cpp
        ${ROOT}/src/kernels/cpu/kq_avx2.cpp ${ROOT}/src/kernels/cpu/kq_avx1.cpp ${ROOT}/src/kernels/cpu/q2_avx1.cpp
        ${ROOT}/src/kernels/cpu/iq_avx1.cpp ${ROOT}/src/kernels/cpu/nvfp4_avx512.cpp ${ROOT}/src/kernels/cpu/nvfp4_avx2.cpp)
target_include_directories(sp_cpu PUBLIC ${INC})
target_compile_definitions(sp_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)
set_source_files_properties(${ROOT}/src/kernels/cpu/expert.cpp ${ROOT}/src/kernels/cpu/iq_avx512.cpp ${ROOT}/src/kernels/cpu/nvfp4_avx512.cpp
  PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mavx512vpopcntdq;-mbmi2;-mf16c;-mfma")
set_source_files_properties(${ROOT}/src/kernels/cpu/q2_avx2.cpp ${ROOT}/src/kernels/cpu/iq_avx2.cpp ${ROOT}/src/kernels/cpu/kq_avx2.cpp
  ${ROOT}/src/kernels/cpu/nvfp4_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
set_source_files_properties(${ROOT}/src/kernels/cpu/kq_avx1.cpp PROPERTIES COMPILE_OPTIONS "-mavx")
set_source_files_properties(${ROOT}/src/kernels/cpu/q2_avx1.cpp ${ROOT}/src/kernels/cpu/iq_avx1.cpp
  PROPERTIES COMPILE_OPTIONS "-msse4.2;-mavx;-mno-avx2;-mno-fma;-mno-f16c")
target_link_libraries(sp_cpu PUBLIC ggml-cpu ggml-base Threads::Threads)

add_library(sp_spec STATIC ${ROOT}/src/spec/suffix_drafter.cpp ${ROOT}/src/spec/controller.cpp
        ${ROOT}/src/spec/draft_policy.cpp ${ROOT}/src/spec/draft_source.cpp)
target_include_directories(sp_spec PUBLIC ${INC})

# XeStrata's SYCL port of ggml's quantized matrix tiles (LGPL-3.0-or-later AND MIT, src/xe/README.md): poly's Mmq
add_library(sp_mmq STATIC src/xe/iq_mmq.cpp src/xe/matrix_report.cpp)
if(STRATA_SYCL_AOT MATCHES "^(bmg|lnl)")
  target_compile_definitions(sp_mmq PRIVATE STRATA_MMQ_NO_XE_HPG_SHAPE=1)
endif()
target_include_directories(sp_mmq PUBLIC src/xe ${ROOT}/third_party/llama.cpp/ggml/src ${ROOT}/third_party/llama.cpp/ggml/include)

# the one strata kernel file poly calls (native_mmvq, quantize, add+RMSnorm): freshly migrated
add_library(sp_kernels STATIC src/kernels/cuda/native_mmvq.dp.cpp src/kernels/cuda/iq_kernels.dp.cpp)
target_include_directories(sp_kernels PUBLIC ${INC})

# These (dpct left them as they are) include "runtime.hpp", and a quoted include looks beside the includer first: build the copies from the build
# directory, so that the migrated runtime.hpp (src/poly, on the include path) is the one they get.
set(_poly_plain main cpu_experts)
set(_poly_copies "")
foreach(f IN LISTS _poly_plain)
  configure_file(${ROOT}/src/poly/${f}.cpp ${CMAKE_BINARY_DIR}/poly/${f}.cpp COPYONLY)
  list(APPEND _poly_copies ${CMAKE_BINARY_DIR}/poly/${f}.cpp)
endforeach()
add_executable(strata-poly ${_poly_copies} src/poly/engine.cpp src/poly/glm5next.cpp src/poly/qwen35.cpp
        src/poly/gemma4.cpp src/poly/deepseek4.cpp src/poly/lfm2moe.cpp
        src/poly/kernels.dp.cpp src/poly/kernels_dsv4.dp.cpp src/poly/mmq_sycl.cpp ${ROOT}/src/core/conversation_memory.cpp)
target_include_directories(strata-poly PRIVATE ${INC})
target_compile_options(strata-poly PRIVATE -qmkl=sequential)
target_link_options(strata-poly PRIVATE -qmkl=sequential)
target_link_libraries(strata-poly PRIVATE sp_mmq sp_kernels sp_spec sp_cpu ggml-cpu ggml-base Threads::Threads)

# attend() against a CPU reference (Qwen3.6's head shape); run by hand: poly_attend_q_test
# the experts' prompt products on the matrix engine against dp4a (runs itself twice); run by hand: poly_mmq_xmx_test
add_executable(poly_mmq_xmx_test src/poly/mmq_xmx_test.cpp)
target_link_libraries(poly_mmq_xmx_test PRIVATE sp_mmq)
add_executable(poly_attend_q_test src/poly/attend_q_test.cpp src/poly/kernels.dp.cpp)
target_include_directories(poly_attend_q_test PRIVATE ${INC})
target_compile_options(poly_attend_q_test PRIVATE -qmkl=sequential)
target_link_options(poly_attend_q_test PRIVATE -qmkl=sequential)
target_link_libraries(poly_attend_q_test PRIVATE sp_kernels Threads::Threads)
add_executable(poly_rope_q_test src/poly/rope_q_test.cpp src/poly/kernels.dp.cpp)
target_include_directories(poly_rope_q_test PRIVATE ${INC})
target_compile_options(poly_rope_q_test PRIVATE -qmkl=sequential)
target_link_options(poly_rope_q_test PRIVATE -qmkl=sequential)
target_link_libraries(poly_rope_q_test PRIVATE sp_kernels Threads::Threads)
add_executable(poly_gdn_q_test src/poly/gdn_q_test.cpp src/poly/kernels.dp.cpp)
target_include_directories(poly_gdn_q_test PRIVATE ${INC})
target_compile_options(poly_gdn_q_test PRIVATE -qmkl=sequential)
target_link_options(poly_gdn_q_test PRIVATE -qmkl=sequential)
target_link_libraries(poly_gdn_q_test PRIVATE sp_kernels Threads::Threads)
