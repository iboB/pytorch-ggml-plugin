// Copyright (c) Borislav Stanimirov
// SPDX-License-Identifier: MIT
//
#pragma once
#include <cstdint>

struct ggml_tensor;
struct ggml_cgraph;
typedef struct ggml_backend* ggml_backend_t;
struct ggml_context;
enum ggml_type;

struct Model {
    struct Tensors {
        ggml_tensor* input = nullptr;
        ggml_tensor* weights = nullptr;
        ggml_tensor* output = nullptr;
    } tensors;

    ggml_cgraph* graph = nullptr;

    ggml_backend_t backend = nullptr;

    ggml_context* ctx = nullptr;

    const int64_t size;
    const ggml_type type;

    Model(ggml_backend_t be, int64_t s, ggml_type t, void* weights);
    ~Model();

    void compute(void* output, void* input);
};
