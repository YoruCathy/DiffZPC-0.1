set(ZENSIM_LIBRARY_CORE_SOURCE_FILES
    Port.cpp
    math/probability/Probability.cpp
    math/matrix/MatrixTransform.cpp
    memory/MemOps.cpp
    memory/Allocator.cpp
    profile/CppTimers.cpp
    execution/Stacktrace.cpp
    execution/ExecutionPolicy.cpp
    execution/ConcurrencyPrimitive.cpp
    Logger.cpp
    # simulation
)
set(ZENSIM_LIBRARY_IO_SOURCE_FILES
    io/ParticleIO.cpp
    io/IO.cpp
)
set(ZENSIM_LIBRARY_TOOL_SOURCE_FILES
    geometry/VdbLevelSet_Point.cpp
    geometry/VdbLevelSet_LoadPhi.cpp
    geometry/VdbLevelSet_LoadPhiVel.cpp
    geometry/VdbLevelSet_Conversion.cpp
    geometry/VdbSampler.cpp
    # simulation
    simulation/init/Scene.cpp
)
set(ZENSIM_LIBRARY_CUDA_SOURCE_FILES
    cuda/Port.cu
    cuda/Cuda.cu
	  cuda/profile/CudaTimers.cu
    cuda/memory/MemOps.cpp
    cuda/memory/Allocator.cpp
    cuda/simulation/wrangler/Wrangler.cpp
)
set(ZENSIM_LIBRARY_OMP_SOURCE_FILES
    omp/Port.cpp
    omp/math/matrix/MatrixTransform.cpp
    omp/execution/ExecutionPolicy.cpp
)
set(ZENSIM_LIBRARY_ZPC_SOURCE_FILES
# ref: https://stackoverflow.com/questions/2351148/explicit-template-instantiation-when-is-it-used
    container/HashTable.cpp
    container/Vector.cpp
    container/TileVector.cpp
    math/matrix/Matrix.cpp
    simulation/mpm/Simulator.cpp
    simulation/sparsity/SparsityCompute.cpp
    resource/Resource.cpp
)
set(ZENSIM_LIBRARY_ZPC_CUDA_SOURCE_FILES 
  # cuda/math/matrix/Matrix.cpp
  cuda/simulation/sparsity/SparsityCompute.cpp
)

## headers
set(ZENSIM_LIBRARY_CORE_INCLUDE_FILES
    Port.hpp
    # container
    container/DenseGrid.hpp
    container/RingBuffer.hpp
    container/TileVector.hpp
    container/HashTable.hpp
    container/Vector.hpp
    container/Bvh.hpp
    container/Bvtt.hpp
    container/IndexBuckets.hpp
    geometry/PointDataGrid.hpp
    geometry/OverlayGrid.hpp
    geometry/AdaptiveGrid.hpp
    geometry/Structure.hpp
    geometry/Structurefree.hpp
    # execution
    execution/Concurrency.h
    execution/ExecutionPolicy.hpp
    execution/Stacktrace.hpp
    execution/Atomics.hpp
    execution/Intrinsics.hpp
    # geometry
    geometry/AnalyticLevelSet.h
    geometry/BoundingVolumeInterface.hpp
    geometry/Collider.h
    geometry/GenericLevelSet.h
    geometry/LevelSet.h
    geometry/LevelSetInterface.h
    geometry/LevelSetSequence.hpp
    geometry/Mesh.hpp
    geometry/PoissonDisk.hpp
    geometry/SparseLevelSet.hpp
    geometry/LevelSetUtils.hpp
    # math
    math/bit/Bits.h
    math/curve/InterpolationKernel.hpp
    math/linear/ConjugateResidual.hpp
    math/linear/ConjugateGradient.hpp
    math/linear/MinimumResidual.hpp
    math/linear/LinearOperators.hpp
    math/matrix/MatrixUtils.h
    math/matrix/Utility.h
    math/matrix/MatrixTransform.hpp
    math/matrix/MatrixCtor.hpp
    math/matrix/Transform.hpp
    math/matrix/Matrix.hpp
    math/matrix/Givens.hpp
    math/matrix/QRSVD.hpp
    math/matrix/SVD.hpp
    math/probability/Probability.h
    math/Hash.hpp
    math/MathUtils.h
    math/RandomNumber.hpp
    math/Rotation.hpp
    math/VecInterface.hpp
    math/Vec.h
    math/Tensor.hpp
    # memory
    memory/MemOps.hpp
    memory/Allocator.h
    memory/MemoryResource.h
    # meta
    meta/ControlFlow.h
    meta/Functional.h
    meta/Meta.h
    meta/Relationship.h
    meta/Sequence.h
    # profile
    profile/CppTimers.hpp
    # resource
    resource/Resource.h
    # types
    types/Pointers.hpp
    types/Function.h
    types/Iterator.h
    types/Optional.h
    types/Polymorphism.h
    types/Property.h
    types/Tuple.h
    types/BuilderBase.hpp
    types/SmallVector.hpp
    types/SourceLocation.hpp
    types/View.h
    Logger.hpp
    Platform.hpp
    Reflection.h
    Singleton.h
    TypeAlias.hpp
    # simulation
    physics/constitutive_models/AnisotropicArap.hpp
    physics/constitutive_models/EquationOfState.hpp
    physics/constitutive_models/StvkWithHencky.hpp
    physics/constitutive_models/FixedCorotated.h
    physics/constitutive_models/NeoHookean.hpp
    physics/constitutive_models/DirichletDamping.hpp
    physics/plasticity_models/SnowPlasticity.hpp
    physics/plasticity_models/VonMisesCapped.hpp
    physics/plasticity_models/NonAssociativeDruckerPrager.hpp
    physics/ConstitutiveModelHelper.hpp
    physics/ConstitutiveModel_Vol_dP.hpp
    physics/ConstitutiveModel.hpp
    physics/SoundSpeedCfl.hpp
    simulation/fem/ElementToDof.hpp
    simulation/mpm/Simulator.hpp
    simulation/transfer/P2G.hpp
    simulation/transfer/G2P.hpp
    simulation/transfer/G2P2G.hpp
    simulation/transfer/P2C2G.hpp
    simulation/transfer/G2C2P.hpp
    simulation/grid/GridOp.hpp
    simulation/particle/ParticleOp.hpp
    simulation/sparsity/SparsityOp.hpp
    simulation/sparsity/SparsityCompute.hpp
    simulation/Utils.hpp
    # tpp
    simulation/sparsity/SparsityCompute.tpp
    container/BvhImpl.tpp
    # autodiff
    autodiff/forward/dual.hpp
    autodiff/forward/real.hpp
    autodiff/forward/dual/dual.hpp
    autodiff/forward/utils/derivative.hpp
    autodiff/forward/utils/taylorseries.hpp
    autodiff/forward/real/real.hpp
    autodiff/reverse/var.hpp
    autodiff/reverse/var/var.hpp
    autodiff/common/numbertraits.hpp
    autodiff/common/meta.hpp
    autodiff/common/vectortraits.hpp
    autodiff/common/binomialcoefficient.hpp
)
set(ZENSIM_LIBRARY_IO_INCLUDE_FILES
    io/IO.h
    io/MeshIO.hpp
    io/ParticleIO.hpp
    # simulation
    simulation/init/Scene.hpp
)
set(ZENSIM_LIBRARY_TOOL_INCLUDE_FILES
    geometry/VdbLevelSet.h
    geometry/VdbSampler.h
)
set(ZENSIM_LIBRARY_CUDA_INCLUDE_FILES
    cuda/Port.hpp
    cuda/Cuda.h
    cuda/DeviceUtils.cuh
    cuda/HostUtils.hpp
    cuda/CudaLaunchConfig.cuh
    cuda/profile/CudaTimers.cuh
    cuda/memory/MemOps.hpp
    cuda/memory/Allocator.h
    cuda/math/matrix/EigenDecomposition.cuh
    cuda/math/matrix/svd.cuh
    cuda/execution/ExecutionPolicy.cuh
    # container
    cuda/geometry/Structurefree.hpp # recent
    cuda/algorithm/MappingKernels.cuh
    cuda/algorithm/SortKernels.cuh
    # simulation
    cuda/simulation/transfer/P2G.hpp
    # physics
    cuda/physics/ConstitutiveModel.hpp
    # wrangler
    cuda/simulation/wrangler/Wrangler.hpp
)
set(ZENSIM_LIBRARY_OMP_INCLUDE_FILES
    omp/Port.hpp
    omp/execution/ExecutionPolicy.hpp
    omp/math/matrix/MatrixTransform.hpp
)
set(ZENSIM_LIBRARY_ZPC_INCLUDE_FILES
)
if (ZS_ENABLE_CUDA)
set(ZENSIM_LIBRARY_ZPC_INCLUDE_FILES ${ZENSIM_LIBRARY_ZPC_INCLUDE_FILES} 
  cuda/math/matrix/Matrix.hpp
)
endif ()
if (ZS_ENABLE_OMP)
set(ZENSIM_LIBRARY_ZPC_INCLUDE_FILES ${ZENSIM_LIBRARY_ZPC_INCLUDE_FILES} 
)
endif ()
# wrangler
set(ZENSIM_LIBRARY_CUDA_WRANGLER_INCLUDE_FILES
)
set(ZENSIM_LIBRARY_CUDA_WRANGLER_SOURCE_FILES
    cuda/simulation/wrangler/VolumeWrangler.cpp
    cuda/simulation/wrangler/ParticleWrangler.cpp
)

#set(CMAKE_CXX_VISIBILITY_PRESET     hidden)
#set(CMAKE_VISIBILITY_INLINES_HIDDEN NO)
#include(GenerateExportHeader)
#generate_export_header(zensim EXPORT_FILE_NAME ZensimExport.hpp)
#message("building shared?(${BUILD_SHARED_LIBS}) library")

add_library(zensim INTERFACE)

#############
# zpc: core #
#############
if (ZS_BUILD_SHARED_LIBS)
  add_library(zpccore SHARED)
else ()
  add_library(zpccore STATIC)
endif ()

target_sources(zpccore
  PRIVATE 
    ${ZENSIM_LIBRARY_CORE_SOURCE_FILES} ${ZENSIM_LIBRARY_CORE_INCLUDE_FILES}
)
if (ENABLE_PCH)
  target_precompile_headers(zpccore INTERFACE 
    ${ZENSIM_LIBRARY_CORE_INCLUDE_FILES}
  )
endif()
target_link_libraries(zpccore PUBLIC zpc_cxx_deps)
target_compile_options(zpccore
  PUBLIC        $<$<COMPILE_LANGUAGE:CXX>: $<IF:$<CXX_COMPILER_ID:MSVC>, /O2 /EHsc, -O3 -fPIC $<IF:$<CXX_COMPILER_ID:Clang>, , > >># -fuse-ld=lld -fvisibility=hidden># -flto=thin -fsanitize=cfi
)
target_link_options(zpccore
  PUBLIC        $<HOST_LINK:
    $<IF:$<CXX_COMPILER_ID:MSVC>, /MACHINE:X64 /OPT:REF /OPT:ICF, 
    $<IF:$<CXX_COMPILER_ID:GNU>, "LINKER:--no-as-needed", > 
    $<IF:$<CXX_COMPILER_ID:Clang>, ,>
    -O3 -m64>>
)
set_target_properties(zpccore
  PROPERTIES  POSITION_INDEPENDENT_CODE ON
              # CXX_VISIBILITY_PRESET default
)

target_link_libraries(zpc_cxx_deps INTERFACE $<IF:$<CXX_COMPILER_ID:MSVC>, synchronization.lib, >)
if (ZS_BUILD_SHARED_LIBS)
  target_compile_definitions(zpc_cxx_deps INTERFACE ZPC_PRIVATE=1)
else()
  target_compile_definitions(zpc_cxx_deps INTERFACE ZPC_PRIVATE=0)
endif()
if (ZS_ENABLE_OPENMP)
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_OPENMP=1)
else()
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_OPENMP=0)
endif(ZS_ENABLE_OPENMP)

if (ZS_ENABLE_CUDA)
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_CUDA=1)
else()
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_CUDA=0)
endif(ZS_ENABLE_CUDA)

if (ZS_ENABLE_SYCL_COMPUTECPP)
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_SYCL_COMPUTECPP=1)
else()
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_SYCL_COMPUTECPP=0)
endif(ZS_ENABLE_SYCL_COMPUTECPP)

if (ZS_ENABLE_SYCL_DPCPP)
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_SYCL_DPCPP=1)
else()
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_SYCL_DPCPP=0)
endif(ZS_ENABLE_SYCL_DPCPP)

if (ZS_ENABLE_SYCL)
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_SYCL=1)
else()
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_SYCL=0)
endif(ZS_ENABLE_SYCL)

###############
# zpc: openmp #
###############
if (ZS_ENABLE_OPENMP)

if (ZS_BUILD_SHARED_LIBS)
  add_library(zpcomp SHARED)
else ()
  add_library(zpcomp STATIC)
endif ()

target_sources(zpcomp 
  PRIVATE   ${ZENSIM_LIBRARY_OMP_SOURCE_FILES} ${ZENSIM_LIBRARY_OMP_INCLUDE_FILES}
)
if (ENABLE_PCH)
  target_precompile_headers(zpcomp INTERFACE ${ZENSIM_LIBRARY_OMP_INCLUDE_FILES})
endif()
target_link_libraries(zpcomp PUBLIC zpc_omp_deps zpccore)
target_compile_options(zpcomp # MSVC, GNU, Clang, Intel
  PUBLIC   $<$<COMPILE_LANGUAGE:CXX>: $<IF:$<CXX_COMPILER_ID:MSVC>, /openmp:llvm /openmp:experimental, -fopenmp $<IF:$<CXX_COMPILER_ID:Clang>, , > >>
)

endif(ZS_ENABLE_OPENMP)

#############
# zpc: cuda #
#############
if (ZS_ENABLE_CUDA)

if (ZS_BUILD_SHARED_LIBS)
  add_library(zpccuda SHARED)
else ()
  add_library(zpccuda STATIC)
endif ()
target_sources(zpccuda 
  PRIVATE   ${ZENSIM_LIBRARY_CUDA_SOURCE_FILES} #${ZENSIM_LIBRARY_CUDA_INCLUDE_FILES}
)
if (ENABLE_PCH)
  target_precompile_headers(zpccuda INTERFACE ${ZENSIM_LIBRARY_CUDA_INCLUDE_FILES})
endif()
set_source_files_properties(${ZENSIM_LIBRARY_CUDA_SOURCE_FILES} ${ZENSIM_LIBRARY_CUDA_INCLUDE_FILES} PROPERTIES  LANGUAGE CUDA)
target_link_libraries(zpccuda PUBLIC zpc_cuda_deps zpccore) # cuda_runtime cuda cudadevrt nvrtc cusolver cublas cusparse
target_compile_options(zpccuda 
  PRIVATE       $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS} -lineinfo --ptxas-options=-allow-expensive-optimizations=true -Xcudafe "--display_error_number --diag_suppress=implicit_return_from_non_void_function --diag_suppress=2642">
# https://stackoverflow.com/questions/64523302/cuda-missing-return-statement-at-end-of-non-void-function-in-constexpr-if-fun
  PUBLIC        $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda --expt-relaxed-constexpr --default-stream=per-thread>
)
target_link_options(zpccuda 
  PRIVATE       $<DEVICE_LINK:-dlto>
)
set_target_properties(zpccuda
        PROPERTIES  POSITION_INDEPENDENT_CODE ON
        )

set_target_properties(zpccuda
  PROPERTIES  #LINKER_LANGUAGE CUDA
              CUDA_EXTENSIONS ON
              CUDA_SEPARABLE_COMPILATION ON
              CUDA_RESOLVE_DEVICE_SYMBOLS ON # https://devblogs.nvidia.com/building-cuda-applications-cmake/
              CUDA_ARCHITECTURES OFF
              #CUDA_VISIBILITY_PRESET default
)
target_compile_definitions(zpc_cuda_deps INTERFACE CMAKE_GENERATOR_PLATFORM=x64)

endif(ZS_ENABLE_CUDA)

#############
# zpc: sycl #
#############
if (ZS_ENABLE_SYCL_COMPUTECPP)

add_library(zpcsycl INTERFACE)
target_link_libraries(zpcsycl INTERFACE zpc_sycl_deps zpc_cxx_deps)

elseif(ZS_ENABLE_SYCL_DPCPP)

add_library(zpcsycl INTERFACE)
target_link_libraries(zpcsycl INTERFACE zpc_sycl_deps zpc_cxx_deps)

elseif(ZS_ENABLE_SYCL)

add_library(zpcsycl INTERFACE)
target_link_libraries(zpcsycl INTERFACE zpc_sycl_deps zpc_cxx_deps)

endif(ZS_ENABLE_SYCL_COMPUTECPP)

#############
# zpc: cl   #
#############
if (ZS_ENABLE_OPENCL)

add_library(zpccl INTERFACE)
target_link_libraries(zpccl INTERFACE zpc_cl_deps zpc_cxx_deps)

endif(ZS_ENABLE_OPENCL)

#######
# zpc #
#######
if (ZS_BUILD_SHARED_LIBS)
  add_library(zpc SHARED)
else ()
  add_library(zpc STATIC)
endif ()

target_sources(zpc
  PRIVATE 
    ${ZENSIM_LIBRARY_ZPC_SOURCE_FILES}
    ${ZENSIM_LIBRARY_ZPC_INCLUDE_FILES}
)
target_link_libraries(zpc PUBLIC zpccore)

if (ZS_ENABLE_OPENMP)
  target_link_libraries(zpc PUBLIC zpcomp)
  # target_sources(zpc PUBLIC $<TARGET_OBJECTS:zpcomp>)
  # target_link_libraries(zpc PUBLIC zpc_omp_deps)
endif(ZS_ENABLE_OPENMP)

if (ZS_ENABLE_CUDA)
  #set_source_files_properties(${ZENSIM_LIBRARY_ZPC_CUDA_SOURCE_FILES} PROPERTIES LANGUAGE CUDA)
  target_link_libraries(zpc PUBLIC zpccuda)
  # target_sources(zpc PRIVATE $<TARGET_OBJECTS:zpccuda>)
  # target_link_libraries(zpc PUBLIC zpc_cuda_deps)
endif(ZS_ENABLE_CUDA)

if (ZS_ENABLE_SYCL_COMPUTECPP)
  target_link_libraries(zpc PUBLIC zpcsycl)
elseif (ZS_ENABLE_SYCL_DPCPP)
  target_link_libraries(zpc PUBLIC zpcsycl)
elseif (ZS_ENABLE_SYCL)
  target_link_libraries(zpc PUBLIC zpcsycl)
endif(ZS_ENABLE_SYCL_COMPUTECPP)

if (ZS_ENABLE_OPENCL)
  target_link_libraries(zpc PUBLIC zpccl)
endif(ZS_ENABLE_OPENCL)

target_link_libraries(zensim INTERFACE zpc)  # zpc -> zensim

# openvdb, partio
###########
# zpctool #
###########
if (ZS_BUILD_SHARED_LIBS)
  add_library(zpctool SHARED)
else ()
  add_library(zpctool STATIC)
endif ()

target_sources(zpctool
  PRIVATE 
    ${ZENSIM_LIBRARY_IO_SOURCE_FILES}
    ${ZENSIM_LIBRARY_IO_INCLUDE_FILES}
)
if (ENABLE_PCH)
  target_precompile_headers(zpctool INTERFACE ${ZENSIM_LIBRARY_IO_INCLUDE_FILES})
endif(ENABLE_PCH)

target_link_libraries(zpctool PUBLIC zpc)

set_target_properties(zpctool
  PROPERTIES  POSITION_INDEPENDENT_CODE ON
)

if (ZS_ENABLE_PARTIO)
  target_link_libraries(zpctool PUBLIC zspartio)
endif(ZS_ENABLE_PARTIO)

if (ZS_ENABLE_MSHIO)
  target_link_libraries(zpctool PUBLIC mshio::mshio)
endif(ZS_ENABLE_MSHIO)
# set_property(TARGET zpctool APPEND PROPERTY PUBLIC_HEADER "${ZENSIM_LIBRARY_TOOL_INCLUDE_FILES} ${ZENSIM_LIBRARY_IO_INCLUDE_FILES}")

if (ZS_ENABLE_OPENVDB)
  if (ENABLE_PCH)
    target_precompile_headers(zpctool INTERFACE ${ZENSIM_LIBRARY_TOOL_INCLUDE_FILES})
  endif(ENABLE_PCH)

  target_link_libraries(zpctool PRIVATE zsopenvdb)
  target_sources(zpctool
    PRIVATE 
      ${ZENSIM_LIBRARY_TOOL_SOURCE_FILES} 
      ${ZENSIM_LIBRARY_TOOL_INCLUDE_FILES} 
  )

endif(ZS_ENABLE_OPENVDB)

target_link_libraries(zensim INTERFACE zpctool)  # zpctool -> zensim

##################
# zs_cu_wrangler #
##################

if (ZS_ENABLE_CUDA)

if (MSVC)
target_compile_definitions(zpc_cuda_deps INTERFACE ZS_PTX_INCLUDE_DIR="")

else()

add_library(zs_cu_wrangler OBJECT)

target_sources(zs_cu_wrangler
  PRIVATE ${ZENSIM_LIBRARY_CUDA_WRANGLER_SOURCE_FILES}
)
target_compile_options(zs_cu_wrangler 
  PRIVATE       $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS} -ptx -lineinfo --ptxas-options=-allow-expensive-optimizations=true>
)
set_source_files_properties(${ZENSIM_LIBRARY_CUDA_WRANGLER_INCLUDE_FILES} ${ZENSIM_LIBRARY_CUDA_WRANGLER_SOURCE_FILES} PROPERTIES  LANGUAGE CUDA)

target_link_libraries(zs_cu_wrangler PRIVATE zpc)
set(ZS_PTX_SOURCE_DIR cuda/simulation/wrangler)
set(ZS_PTX_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/zs_cu_wrangler.dir/${ZS_PTX_SOURCE_DIR})
set_target_properties(zs_cu_wrangler
  PROPERTIES  LINKER_LANGUAGE CUDA
              CUDA_EXTENSIONS ON
              CUDA_SEPARABLE_COMPILATION ON
              CUDA_RESOLVE_DEVICE_SYMBOLS ON # https://devblogs.nvidia.com/building-cuda-applications-cmake/
              CUDA_ARCHITECTURES OFF
              CUDA_PTX_COMPILATION ON # https://stackoverflow.com/questions/55993745/how-should-i-get-cmake-to-also-create-ptx-files-for-my-kernels
              #CUDA_VISIBILITY_PRESET default
)
target_compile_definitions(zpc_cuda_deps INTERFACE ZS_PTX_INCLUDE_DIR="${ZS_PTX_INCLUDE_DIR}")

# $<TARGET_OBJECTS:zs_cu_wrangler>
target_link_libraries(zensim INTERFACE zs_cu_wrangler)  # zs_cu_wrangler -> zensim
endif(MSVC)

endif (ZS_ENABLE_CUDA)

# for header installation
# list_transform_prepend(ZENSIM_INTERFACE_FILES include/zensim/)
# message("${ZENSIM_INTERFACE_FILES}")
# PUBLIC_HEADER not working as expected.
# set_target_properties(zensim PROPERTIES PUBLIC_HEADER "${ZENSIM_INTERFACE_FILES}")
