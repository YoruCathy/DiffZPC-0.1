cmake_minimum_required(VERSION 3.18)

option(ZS_ENABLE_INSTALL "Enable install rule" OFF)
option(ZS_ENABLE_OPENGL "Enable opengl" OFF)
option(ZS_ENABLE_OPENVDB "Enable openvdb" OFF)
option(ZS_ENABLE_CUDA "Enable cuda" OFF)
option(ZS_ENABLE_SYCL_COMPUTECPP "Enable sycl[computecpp]" OFF)
option(ZS_ENABLE_SYCL_DPCPP "Enable SYCL[Intel-DPC++] backend" OFF)
option(ZS_ENABLE_OPENCL "Enable OpenCL backend" OFF)
option(ZS_ENABLE_OPENMP "Enable openmp" OFF)
option(ZS_ENABLE_PTHREADS "Enable pthreads" OFF)
option(ZS_ENABLE_PARTIO "Enable partio" ON)
option(ZS_ENABLE_MSHIO "Enable .msh file io" ON)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake ${CMAKE_CURRENT_SOURCE_DIR}/CMake ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/CMake ${CMAKE_CURRENT_SOURCE_DIR}/CMake ${CMAKE_CURRENT_BINARY_DIR})

add_library(zpc_deps INTERFACE)
add_library(zsproj_deps INTERFACE)
target_link_libraries(zpc_deps INTERFACE zsproj_deps)

# opengl
if (ZS_ENABLE_OPENGL)
    find_package(OpenGL)
    if (TARGET OpenGL::GL)
        target_link_libraries(zsproj_deps INTERFACE OpenGL::GL)
    else()
        set(ZS_ENABLE_OPENGL OFF)
        # message(FATAL_ERROR "OpenGL not found!")
    endif()
endif(ZS_ENABLE_OPENGL)
set(ZS_ENABLE_OPENGL ${ZS_ENABLE_OPENGL} PARENT_SCOPE)

# cuda
if (ZS_ENABLE_CUDA)
    find_package(CUDAToolkit)
    if (TARGET CUDA::cuda_driver)
        add_library(zpc_cuda_deps INTERFACE)
        target_link_libraries(zpc_deps INTERFACE zpc_cuda_deps)
        target_compile_features(zpc_cuda_deps INTERFACE cuda_std_17)
        target_link_libraries(zpc_cuda_deps INTERFACE CUDA::cuda_driver CUDA::nvrtc)
        #  CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::cufft
        if (ZS_BUILD_SHARED_LIBS)
            target_link_libraries(zpc_cuda_deps INTERFACE CUDA::cudart)
        else()
            target_link_libraries(zpc_cuda_deps INTERFACE CUDA::cudart_static)
        endif()
    else()
        set(ZS_ENABLE_CUDA OFF)
    endif()
endif(ZS_ENABLE_CUDA)
set(ZS_ENABLE_CUDA ${ZS_ENABLE_CUDA} PARENT_SCOPE)

# sycl-computecpp
if (ZS_ENABLE_SYCL_COMPUTECPP)
    if (NOT DEFINED ComputeCpp_DIR OR NOT DEFINED COMPUTECPP_BITCODE)
        message(WARNING "SYCL enabled but ComputeCpp_DIR and COMPUTECPP_BITCODE not all set!")
        set(ZS_ENABLE_SYCL_COMPUTECPP OFF)
    else(FIND_DIR_MACRO_DEFINED)
        find_package(ComputeCpp)
        if (ComputeCpp_FOUND)
            add_library(zpc_sycl_deps INTERFACE)
            target_link_libraries(zpc_deps INTERFACE zpc_sycl_deps)
            target_compile_features(zpc_sycl_deps INTERFACE cxx_std_17)
            set_target_properties(ComputeCpp::ComputeCpp PROPERTIES IMPORTED_GLOBAL TRUE)
            target_link_libraries(zpc_sycl_deps INTERFACE ComputeCpp::ComputeCpp)

            list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -std=c++17)
            set(COMPUTECPP_DEVICE_COMPILER_FLAGS ${COMPUTECPP_DEVICE_COMPILER_FLAGS} PARENT_SCOPE)
        else(ComputeCpp_NOTFOUND)
            set(ZS_ENABLE_SYCL_COMPUTECPP OFF)
        endif(ComputeCpp_FOUND)
    endif()
endif(ZS_ENABLE_SYCL_COMPUTECPP)
set(ZS_ENABLE_SYCL_COMPUTECPP ${ZS_ENABLE_SYCL_COMPUTECPP} PARENT_SCOPE)

# sycl-dpcpp
if (ZS_ENABLE_SYCL_DPCPP)
  get_filename_component(ZS_CXX_COMPILER_BINARY ${CMAKE_CXX_COMPILER} NAME)
  if (${ZS_CXX_COMPILER_BINARY} STREQUAL "dpcpp")
    add_library(zpc_sycl_deps INTERFACE)
    target_link_libraries(zpc_deps INTERFACE zpc_sycl_deps)
    target_compile_features(zpc_sycl_deps INTERFACE cxx_std_17)
    target_link_libraries(zpc_sycl_deps INTERFACE sycl)
  else()
    set(ZS_ENABLE_SYCL_DPCPP OFF)
  endif()
endif(ZS_ENABLE_SYCL_DPCPP)
set(ZS_ENABLE_SYCL_DPCPP ${ZS_ENABLE_SYCL_DPCPP} PARENT_SCOPE)

# sycl-clang
if (ZS_ENABLE_SYCL)
  if (${CMAKE_CXX_COMPILER_ID} MATCHES ".*Clang")
    add_library(zpc_sycl_deps INTERFACE)
    target_link_libraries(zpc_deps INTERFACE zpc_sycl_deps)
    target_compile_features(zpc_sycl_deps INTERFACE cxx_std_17)
    target_compile_options(zpc_sycl_deps INTERFACE 
        $<$<COMPILE_LANGUAGE:CXX>: -fsycl -fsycl-targets=nvptx64>
    )
  else()
    set(ZS_ENABLE_SYCL OFF)
  endif()
endif(ZS_ENABLE_SYCL)
set(ZS_ENABLE_SYCL ${ZS_ENABLE_SYCL} PARENT_SCOPE)

# opencl
if (ZS_ENABLE_OPENCL)
    find_package(OpenCL)
    if (OpenCL_FOUND)
        add_library(zpc_cl_deps INTERFACE)
        target_link_libraries(zpc_deps INTERFACE zpc_cl_deps)
        #target_compile_features(zpc_cl_deps INTERFACE cxx_std_17)
        #target_include_directories(zpc_cl_deps INTERFACE ${OpenCL_INCLUDE_DIRS})
        #target_link_libraries(zpc_cl_deps INTERFACE ${OpenCL_LIBRARIES})
        target_link_libraries(zpc_cl_deps INTERFACE OpenCL::OpenCL)
        message(INFO "opencl directories: ${OpenCL_INCLUDE_DIRS}")
        message(INFO "opencl libraries: ${OpenCL_LIBRARIES}")
    else()
        set(ZS_ENABLE_OPENCL OFF)
    endif()
endif(ZS_ENABLE_OPENCL)
set(ZS_ENABLE_OPENCL ${ZS_ENABLE_OPENCL} PARENT_SCOPE)

# threads
if (ZS_ENABLE_PTHREADS)
  find_package(Threads)
  if (NOT TARGET Threads::Threads)
    set(ZS_ENABLE_PTHREADS OFF)
  endif()
  target_link_libraries(zsproj_deps INTERFACE Threads::Threads)
endif(ZS_ENABLE_PTHREADS)
set(ZS_ENABLE_PTHREADS ${ZS_ENABLE_PTHREADS} PARENT_SCOPE)

# openmp
if (ZS_ENABLE_OPENMP)
    find_package(OpenMP)
    if (TARGET OpenMP::OpenMP_CXX)
        add_library(zpc_omp_deps INTERFACE)
        target_link_libraries(zpc_deps INTERFACE zpc_omp_deps)
        # https://gitlab.kitware.com/cmake/cmake/-/issues/17256
        # this target already imported in zen
        target_link_libraries(zpc_omp_deps INTERFACE OpenMP::OpenMP_CXX)
    else()
        set(ZS_ENABLE_OPENMP OFF)
        # message(FATAL_ERROR "OpenMP not found!")
    endif()
endif(ZS_ENABLE_OPENMP)
set(ZS_ENABLE_OPENMP ${ZS_ENABLE_OPENMP} PARENT_SCOPE)

# openvdb
if (ZS_ENABLE_OPENVDB)
    find_package(OpenVDB)
    if (ZS_BUILD_SHARED_LIBS)
        if (TARGET OpenVDB::openvdb_shared)
            add_library(zsopenvdb INTERFACE)
            target_link_libraries(zpc_deps INTERFACE zsopenvdb)
            target_link_libraries(zsopenvdb INTERFACE OpenVDB::openvdb_shared)
        elseif(TARGET OpenVDB::openvdb)
            add_library(zsopenvdb INTERFACE)
            target_link_libraries(zpc_deps INTERFACE zsopenvdb)
            target_link_libraries(zsopenvdb INTERFACE OpenVDB::openvdb)
        else()
            set(ZS_ENABLE_OPENVDB OFF)
        endif()
    else()
        if (TARGET OpenVDB::openvdb_static)
            add_library(zsopenvdb INTERFACE)
            target_link_libraries(zpc_deps INTERFACE zsopenvdb)
            target_link_libraries(zsopenvdb INTERFACE OpenVDB::openvdb_static)
        elseif(TARGET OpenVDB::openvdb)
            add_library(zsopenvdb INTERFACE)
            target_link_libraries(zpc_deps INTERFACE zsopenvdb)
            target_link_libraries(zsopenvdb INTERFACE OpenVDB::openvdb)
        else()
            set(ZS_ENABLE_OPENVDB OFF)
        endif()
    endif()
endif(ZS_ENABLE_OPENVDB)
set(ZS_ENABLE_OPENVDB ${ZS_ENABLE_OPENVDB} PARENT_SCOPE)

# tpl that needs compilation
# partio
if (ZS_ENABLE_PARTIO)
    if (ZS_BUILD_SHARED_LIBS)
        set(PARTIO_BUILD_SHARED_LIBS ON)
    else()
        set(PARTIO_BUILD_SHARED_LIBS OFF)
    endif()
    add_subdirectory(partio EXCLUDE_FROM_ALL)
endif(ZS_ENABLE_PARTIO)
set(ZS_ENABLE_PARTIO ${ZS_ENABLE_PARTIO} PARENT_SCOPE)

# mshio
if (ZS_ENABLE_MSHIO)
    if (NOT TARGET mshio)
        set(MSHIO_PYTHON OFF)
        set(MSHIO_BUILD_TESTS OFF)
        set(MSHIO_BUILD_EXAMPLES OFF)
        set(MSHIO_EXT_NANOSPLINE OFF)
        add_subdirectory(mshio EXCLUDE_FROM_ALL)
    endif()
endif(ZS_ENABLE_MSHIO)
set(ZS_ENABLE_MSHIO ${ZS_ENABLE_MSHIO} PARENT_SCOPE)

# plog
target_include_directories(zsproj_deps INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# other local tpls
# fmt
target_compile_definitions(zsproj_deps INTERFACE FMT_HEADER_ONLY)
# Catch2
# gcem
# function_ref
# jitify
# rapidjson
# cxxopts
# magic_enum
