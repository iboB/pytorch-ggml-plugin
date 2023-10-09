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

# This is a dangerous option if one has a cuda version installed which is
# different from the pytorch version.
# Normally, find_package(Torch) will also find cuda and enable it as a language
# However ggml will not link...
option(ENABLE_CUDA "Enable Cuda (May be dangerous. Check comments)" OFF)

# will enable cuda language and find cuda if available
find_package(Torch REQUIRED)

#################
# subdirs/targets
add_subdirectory(ggml)
add_subdirectory(scratch)
