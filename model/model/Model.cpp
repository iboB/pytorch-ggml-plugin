// Copyright (c) Borislav Stanimirov
// SPDX-License-Identifier: MIT
//
#include "Model.hpp"

#include <ggml.h>
#include <ggml-backend.h>

#include <cassert>

Model::Model(ggml_backend_t be, int64_t s, ggml_type t, void* weights)
        : backend(be)
        , size(s)
        , type(t)
{
    assert(weights);
    static constexpr int64_t num_tensors = sizeof(Tensors) / sizeof(ggml_tensor*);
    struct ggml_init_params init_params {
        /*.mem_size   =*/ ggml_tensor_overhead()* num_tensors + ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
    };
    ctx = ggml_init(init_params);

    tensors.input = ggml_new_tensor_1d(ctx, type, size);
    tensors.weights = ggml_new_tensor_1d(ctx, type, size);
    ggml_backend_set_tensor_external_data(backend, tensors.weights, weights);

    tensors.output = ggml_add(ctx, tensors.input, tensors.weights);

    graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, tensors.output);
}

Model::~Model() {
    ggml_free(ctx);
}

void Model::compute(void* output, void* input) {
    assert(input);
    assert(output);
    ggml_backend_set_tensor_external_data(backend, tensors.output, output);
    ggml_backend_set_tensor_external_data(backend, tensors.input, input);
    ggml_backend_graph_compute(backend, graph);
}
