// Copyright (c) Borislav Stanimirov
// SPDX-License-Identifier: MIT
//
#include <model/Model.hpp>

#include <vector>
#include <iostream>

#ifdef GGML_USE_CUBLAS
#include <ggml-cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

template <typename T>
size_t data_size(const std::vector<T>& vec) {
    return vec.size() * sizeof(T);
}

int main() {
    std::vector<float> weightsData;
    for (int i = 0; i < 10; ++i) {
        weightsData.push_back(float(i));
    }

#if GGML_USE_CUBLAS
    int deviceId = 0;
    cudaSetDevice(deviceId);
    cublasHandle_t cublasHandle = nullptr;
    cublasCreate(&cublasHandle);
    cudaStream_t cudaStream = nullptr;
    cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking);
    auto backend = ggml_backend_cuda_init_plugin(deviceId, cublasHandle, cudaStream);

    void* weights = nullptr;
    cudaMallocAsync(&weights, data_size(weightsData), cudaStream);
    cudaMemcpyAsync(weights, weightsData.data(), data_size(weightsData), cudaMemcpyHostToDevice, cudaStream);
#else
    auto backend = ggml_backend_cpu_init();
    void* weights = weightsData.data();
#endif

    Model model(backend, weightsData.size(), GGML_TYPE_F32, weights);

    std::vector<float> inputData;
    for (size_t i = 0; i < weightsData.size(); ++i) {
        inputData.push_back(float(i) / 10);
    }

    std::vector<float> outputData(inputData.size());

#if GGML_USE_CUBLAS
    void* input = nullptr;
    cudaMallocAsync(&input, data_size(inputData), cudaStream);
    cudaMemcpyAsync(input, inputData.data(), data_size(inputData), cudaMemcpyHostToDevice, cudaStream);

    void* output = nullptr;
    cudaMallocAsync(&output, data_size(outputData), cudaStream);
#else
    void* input = inputData.data();
    void* output = outputData.data();
#endif

    model.compute(output, input);

#if GGML_USE_CUBLAS
    cudaMemcpyAsync(outputData.data(), output, data_size(outputData), cudaMemcpyDeviceToHost, cudaStream);
    cudaStreamSynchronize(cudaStream);
#endif

    ggml_backend_free(backend);

    std::cout << "[";
    for (auto o : outputData) {
        std::cout << o << ", ";
    }
    std::cout << "]\n";

    return 0;
}
