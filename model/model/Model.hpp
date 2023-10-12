// Copyright (c) Borislav Stanimirov
// SPDX-License-Identifier: MIT
//
#pragma once
#include <ggml.h>
#include <cstdint>

typedef struct ggml_backend* ggml_backend_t;
struct ggml_backend_buffer;
struct ggml_allocr;

struct Model {
    struct Weights {
        ggml_tensor* w = nullptr;
    } weights;

    ggml_backend_t backend = nullptr;

    ggml_context* wctx = nullptr;
    ggml_backend_buffer* wbuf = nullptr; // weights buffer

    ggml_backend_buffer* cbuf = nullptr; // compute buffer
    ggml_allocr* callocr = nullptr; // compute allocator

    const int64_t size;
    const ggml_type type;

    Model(ggml_backend_t be, int64_t s, ggml_type t, void* weightsData);
    ~Model();

    void compute(void* output, void* input);
};
