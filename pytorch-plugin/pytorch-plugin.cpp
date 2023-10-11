// Copyright (c) Borislav Stanimirov
// SPDX-License-Identifier: MIT
//
#include <model/Model.hpp>

#include <ggml.h>
#include <ggml-backend.h>

#include <torch/custom_class.h>
#include <torch/script.h>
#include <torch/types.h>

#ifdef GGML_USE_CUBLAS
#include <ggml-cuda.h>
#include <ATen/cuda/CUDAContext.h>
#endif


ggml_type torchTypeToGGML(torch::ScalarType t) {
    switch (t) {
    case torch::ScalarType::Float: return GGML_TYPE_F32;
    case torch::ScalarType::Half: return GGML_TYPE_F16;
    case torch::ScalarType::Int: return GGML_TYPE_I32;
    default:
        throw std::runtime_error("unsupported torch type");
    }
}

class Plugin : public torch::jit::CustomClassHolder {
    ggml_backend_t m_backend;
    std::unique_ptr<Model> m_model;
    torch::Tensor m_weights;

public:
    Plugin(torch::Tensor weights)
        : m_weights(weights)
    {
        if (m_weights.is_cuda()) {
#ifdef GGML_USE_CUBLAS
            auto deviceId = at::cuda::current_device();
            auto cublasHandle = at::cuda::getCurrentCUDABlasHandle();
            auto cudaStream = at::cuda::getStreamFromPool().stream();
            m_backend = ggml_backend_cuda_init_plugin(deviceId, cublasHandle, cudaStream);
#else
            throw std::runtime_error("No ggml cuda support");
#endif
        }
        else {
            m_backend = ggml_backend_cpu_init();
        }

        if (m_weights.dim() != 1) throw std::runtime_error("weights shape must be 1");
        m_model.reset(new Model(m_backend, m_weights.size(0), torchTypeToGGML(m_weights.dtype().toScalarType()), m_weights.data_ptr()));
    }

    ~Plugin() {
        m_model.reset();
        ggml_backend_free(m_backend);
    }

    torch::Tensor forward(torch::Tensor input) {
        if (input.size(0) < m_model->size) throw std::runtime_error("tensor size mismatch");
        if (torchTypeToGGML(input.dtype().toScalarType()) != m_model->type) throw std::runtime_error("tensor dtype mismatch");
        if (input.is_cuda() != m_weights.is_cuda()) throw std::runtime_error("input/weights device mismatch");

        auto ret = torch::empty({input.size(0)},
            torch::dtype(input.dtype())
            .device(input.device())
            .requires_grad(false)
        );

        m_model->compute(ret.data_ptr(), input.data_ptr());

        return ret;
    }
};

static auto _exports =
    torch::jit::class_<Plugin>("GGMLPlugin", "Model")
        .def(torch::jit::init<torch::Tensor>())
        .def("forward", &Plugin::forward)
;
