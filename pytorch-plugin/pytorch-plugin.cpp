// Copyright (c) Borislav Stanimirov
// SPDX-License-Identifier: MIT
//
#include <ggml.h>
#include <ggml-backend.h>

#include <torch/custom_class.h>
#include <torch/script.h>
#include <torch/types.h>

#ifdef GGML_USE_CUBLAS
#include <ggml-cuda.h>
#include <ATen/cuda/CUDAContext.h>
#endif

#include <cstdint>

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

    Model(ggml_backend_t be, int64_t s, ggml_type t, void* weights)
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

    ~Model() {
        ggml_free(ctx);
    }

    void compute(void* output, void* input) {
        assert(input);
        assert(output);
        ggml_backend_set_tensor_external_data(backend, tensors.output, output);
        ggml_backend_set_tensor_external_data(backend, tensors.input, input);
        ggml_backend_graph_compute(backend, graph);
    }
};

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
