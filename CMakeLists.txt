cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(poc-ggml
    LANGUAGES C CXX
)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LINK_DEPENDS_NO_SHARED ON)

set(CMAKE_CUDA_ARCHITECTURES 52) # 89

# will enable cuda language and find cuda if available
find_package(Torch REQUIRED)

#################
# subdirs/targets

# check for the only var which is defined on all platforms if torch cuda is 
# available
if(TORCH_CUDA_LIBRARIES)
    # cuda toolit was found by pytorch
    # enable it in ggml
    set(GGML_CUBLAS ON CACHE BOOL "" FORCE)
    add_compile_definitions(GGML_USE_CUBLAS)
endif()

set(GGML_PLUGIN ON)
add_subdirectory(ggml)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_subdirectory(scratch)
endif()
