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

    Model(ggml_backend_t be, int64_t s, ggml_type t, void* weights, void* output)
        : backend(be)
        , size(s)
        , type(t)
    {
        assert(weights);
        assert(output);
        static constexpr int64_t num_tensors = sizeof(Tensors) / sizeof(ggml_tensor*);
        struct ggml_init_params init_params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors + ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(init_params);

        tensors.input = ggml_new_tensor_1d(ctx, type, size);
        tensors.weights = ggml_new_tensor_1d(ctx, type, size);
        ggml_backend_set_tensor_external_data(backend, tensors.weights, weights);

        tensors.output = ggml_add(ctx, tensors.input, tensors.weights);
        ggml_backend_set_tensor_external_data(backend, tensors.output, output);

        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, tensors.output);
    }

    ~Model() {
        ggml_free(ctx);
    }

    void compute(void* input) {
        ggml_backend_set_tensor_external_data(backend, tensors.input, input);
        ggml_backend_graph_compute(backend, graph);
    }
};

template <typename T>
size_t data_size(const std::vector<T>& vec) {
    return vec.size() * sizeof(T);
}

int main() {
    std::vector<float> weights;
    for (int i = 0; i < 10; ++i) {
        weights.push_back(float(i));
    }
    std::vector<float> output(weights.size());

    std::vector<float> input;
    for (size_t i = 0; i < weights.size(); ++i) {
        input.push_back(float(i) / 10);
    }

#if !GGML_USE_CUBLAS
    auto backend = ggml_backend_cpu_init();
    Model model(backend, weights.size(), GGML_TYPE_F32, weights.data(), output.data());

    model.compute(input.data());

    ggml_backend_free(backend);
#else
    int deviceId = 0;
    cudaSetDevice(deviceId);
    cublasHandle_t cublasHandle = nullptr;
    cublasCreate(&cublasHandle);
    cudaStream_t cudaStream = nullptr;
    cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking);
    auto backend = ggml_backend_cuda_init_plugin(deviceId, cublasHandle, cudaStream);

    void* cu_weights = nullptr;
    cudaMallocAsync(&cu_weights, data_size(weights), cudaStream);
    cudaMemcpyAsync(cu_weights, weights.data(), data_size(weights), cudaMemcpyHostToDevice, cudaStream);

    void* cu_output = nullptr;
    cudaMallocAsync(&cu_output, data_size(output), cudaStream);

    Model model(backend, weights.size(), GGML_TYPE_F32, cu_weights, cu_output);

    void* cu_input = nullptr;
    cudaMallocAsync(&cu_input, data_size(input), cudaStream);
    cudaMemcpyAsync(cu_input, input.data(), data_size(input), cudaMemcpyHostToDevice, cudaStream);
    model.compute(cu_input);

    // cudaMemcpyAsync(output.data(), cu_output, data_size(output), cudaMemcpyDeviceToHost, cudaStream);
    // cudaStreamSynchronize(cudaStream);
    ggml_backend_tensor_get(model.tensors.output, output.data(), 0, data_size(output));

    ggml_backend_free(backend);
#endif

    for (auto o : output) {
        std::cout << o << '\n';
    }

    return 0;
}
