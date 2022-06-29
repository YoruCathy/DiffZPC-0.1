cmake_minimum_required(VERSION 3.19)

# assume cuda
project(ZPC CXX CUDA)
set(ZS_ENABLE_CUDA ON)
set(ZS_ENABLE_OPENVDB ON)
#set(ZENSIM_INSTALL ON)
#set(ZS_BUILD_SHARED_LIBS OFF)
#set(CMAKE_PREFIX_PATH "/home/cathy/zpc/build")
#find_package(zensim)
find_package(CUDAToolkit)


add_executable(zeno_cutest main.cu)
target_compile_features(zeno_cutest PRIVATE cxx_std_17 cuda_std_17)

set(ZS_BUILD_SHARED_LIBS OFF)
add_subdirectory(zpc)
target_link_libraries(zeno_cutest PRIVATE zensim)

set_target_properties(zeno_cutest
        PROPERTIES  #CUDA_EXTENSIONS ON
        CUDA_SEPARABLE_COMPILATION ON
        LINKER_LANGUAGE CUDA
        POSITION_INDEPENDENT_CODE ON
        CUDA_ARCHITECTURES OFF
        )




