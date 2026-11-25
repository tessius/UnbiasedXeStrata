# strata-ik2 (Inkling 2 / Inkling-2-Mini) for Intel GPUs, through SYCL.  The sources are dpct's migration of ../src/ik2
# (tools/migrate_ik2.sh) with hand fixes; the CPU parts (ik2_cpu*.cpp) are ../src/ik2 unchanged.  Experimental: see NOTES.md.
#   cmake -S sycl-ik2 -B build-ik2 -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx [-DSTRATA_SYCL_AOT=bmg-g21] && cmake --build build-ik2
cmake_minimum_required(VERSION 3.24)
project(strata_ik2_sycl LANGUAGES C CXX)
set(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_definitions(STRATA_VERSION="ik2-sycl")
set(STRATA_SYCL_AOT "" CACHE STRING "AOT device for the GPU code, e.g. bmg-g21 (empty: SPIR-V, JIT at run time)")
# One device module per kernel: a launch compiles only its own kernel (startup is faster, and a kernel the GPU compiler cannot
# build only fails when it is used, with the launcher in the backtrace).
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fsycl;-fsycl-default-sub-group-size=32;-fsycl-device-code-split=per_kernel>" -fp-model=precise -w)
add_link_options("SHELL:-Xspirv-translator --spirv-ext=+SPV_KHR_integer_dot_product")
if(STRATA_SYCL_AOT)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fsycl-targets=spir64_gen>")
  add_link_options(-fsycl -fsycl-default-sub-group-size=32 -fsycl-device-code-split=per_kernel -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen
                   "-device ${STRATA_SYCL_AOT} -options -cl-fp32-correctly-rounded-divide-sqrt")
else()
  add_link_options(-fsycl -fsycl-default-sub-group-size=32 -fsycl-device-code-split=per_kernel -Xsycl-target-backend=spir64
                   "-cl-fp32-correctly-rounded-divide-sqrt -ze-opt-greater-than-4GB-buffer-required")
endif()
find_package(Threads REQUIRED)
find_package(OpenMP QUIET COMPONENTS CXX)
# the migrated headers first, then the repo's, and ../sycl/include last (only its dpct/ helpers are wanted)
set(INC ${CMAKE_CURRENT_SOURCE_DIR}/src ${ROOT}/src/ik2 ${ROOT}/include ${ROOT}/sycl/include)

add_library(ik2_kernels STATIC src/ik2_kernels.dp.cpp)
target_include_directories(ik2_kernels PUBLIC ${INC})

add_executable(strata-ik2 src/ik2_main.dp.cpp src/ik2_draft.dp.cpp ${ROOT}/src/ik2/ik2_cpu.cpp)
target_include_directories(strata-ik2 PRIVATE ${INC})
target_compile_options(strata-ik2 PRIVATE -qmkl=sequential)
target_link_options(strata-ik2 PRIVATE -qmkl=sequential)
target_link_libraries(strata-ik2 PRIVATE ik2_kernels Threads::Threads)
if(OpenMP_CXX_FOUND)
  target_sources(strata-ik2 PRIVATE ${ROOT}/src/ik2/ik2_cpu_deep.cpp)
  set_source_files_properties(${ROOT}/src/ik2/ik2_cpu_deep.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-ffp-contract=off")
  target_compile_definitions(strata-ik2 PRIVATE IK2_HOST_AVX2=1)
  target_link_libraries(strata-ik2 PRIVATE OpenMP::OpenMP_CXX)
endif()

# the emulated tensor-core primitives against a CPU reference (needs the GPU): ik2_mma_emul_test
add_executable(ik2_mma_emul_test tests/mma_emul_test.cpp)
target_include_directories(ik2_mma_emul_test PRIVATE src)
add_executable(ik2_experts_down_test tests/experts_down_test.cpp)
target_include_directories(ik2_experts_down_test PRIVATE ${INC})
target_link_libraries(ik2_experts_down_test PRIVATE ik2_kernels)
add_executable(ik2_rms_test tests/rms_test.cpp)
target_include_directories(ik2_rms_test PRIVATE ${INC})
target_link_libraries(ik2_rms_test PRIVATE ik2_kernels)
add_executable(ik2_flash_xmx_test tests/flash_xmx_test.cpp)
target_include_directories(ik2_flash_xmx_test PRIVATE ${INC})
target_link_libraries(ik2_flash_xmx_test PRIVATE ik2_kernels)
