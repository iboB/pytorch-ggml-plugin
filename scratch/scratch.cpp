// Copyright (c) Borislav Stanimirov
// SPDX-License-Identifier: MIT
//
#include <ggml.h>
#include <ggml/ggml-alloc.h>
#include <ggml-backend.h>

#include <stdexcept>
#include <iostream>
#include <cassert>
#include <vector>

#ifdef GGML_USE_CUBLAS
#include <ggml-cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

int main() {
    return 0;
}
