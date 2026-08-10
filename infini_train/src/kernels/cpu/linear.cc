#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <numeric>
#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    CHECK_GE(input->Dims().size(), 2);
    CHECK_EQ(input->Dims().size(), other->Dims().size());
    CHECK(input->GetDevice().Type() == DeviceType::kCPU);
    CHECK(other->GetDevice().Type() == DeviceType::kCPU);
    CHECK(input->Dtype() == DataType::kFLOAT32);
    CHECK(other->Dtype() == DataType::kFLOAT32);

    const auto &a_dims = input->Dims();
    const auto &b_dims = other->Dims();
    const int64_t rank = a_dims.size();
    const int64_t m = a_dims[rank - 2];
    const int64_t k = a_dims[rank - 1];
    const int64_t n = b_dims[rank - 1];
    CHECK_EQ(k, b_dims[rank - 2]);
    for (int64_t i = 0; i < rank - 2; ++i) CHECK_EQ(a_dims[i], b_dims[i]);

    std::vector<int64_t> out_dims(a_dims.begin(), a_dims.end());
    out_dims[rank - 1] = n;
    auto output = std::make_shared<Tensor>(out_dims, DataType::kFLOAT32, input->GetDevice());
    const int64_t batch = std::accumulate(a_dims.begin(), a_dims.end() - 2, 1LL, std::multiplies<int64_t>());
    const float *a = static_cast<const float *>(input->DataPtr());
    const float *b = static_cast<const float *>(other->DataPtr());
    float *out = static_cast<float *>(output->DataPtr());
    for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
        const float *a_batch = a + batch_idx * m * k;
        const float *b_batch = b + batch_idx * k * n;
        float *out_batch = out + batch_idx * m * n;
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int64_t p = 0; p < k; ++p) sum += a_batch[i * k + p] * b_batch[p * n + j];
                out_batch[i * n + j] = sum;
            }
        }
    }
    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    CHECK_EQ(input->Dims().size(), other->Dims().size());
    CHECK_EQ(input->Dims().size(), grad_output->Dims().size());
    const auto &a_dims = input->Dims();
    const auto &b_dims = other->Dims();
    const int64_t rank = a_dims.size();
    const int64_t m = a_dims[rank - 2];
    const int64_t k = a_dims[rank - 1];
    const int64_t n = b_dims[rank - 1];
    CHECK_EQ(k, b_dims[rank - 2]);
    CHECK_EQ(grad_output->Dims()[rank - 2], m);
    CHECK_EQ(grad_output->Dims()[rank - 1], n);
    auto grad_input = std::make_shared<Tensor>(a_dims, DataType::kFLOAT32, input->GetDevice());
    auto grad_other = std::make_shared<Tensor>(b_dims, DataType::kFLOAT32, input->GetDevice());
    const int64_t batch = std::accumulate(a_dims.begin(), a_dims.end() - 2, 1LL, std::multiplies<int64_t>());
    const float *a = static_cast<const float *>(input->DataPtr());
    const float *b = static_cast<const float *>(other->DataPtr());
    const float *g = static_cast<const float *>(grad_output->DataPtr());
    float *ga = static_cast<float *>(grad_input->DataPtr());
    float *gb = static_cast<float *>(grad_other->DataPtr());
    for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
        const float *a_batch = a + batch_idx * m * k;
        const float *b_batch = b + batch_idx * k * n;
        const float *g_batch = g + batch_idx * m * n;
        float *ga_batch = ga + batch_idx * m * k;
        float *gb_batch = gb + batch_idx * k * n;
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t p = 0; p < k; ++p) {
                float sum = 0.0f;
                for (int64_t j = 0; j < n; ++j) sum += g_batch[i * n + j] * b_batch[p * n + j];
                ga_batch[i * k + p] = sum;
            }
        }
        for (int64_t p = 0; p < k; ++p) {
            for (int64_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int64_t i = 0; i < m; ++i) sum += a_batch[i * k + p] * g_batch[i * n + j];
                gb_batch[p * n + j] = sum;
            }
        }
    }
    return {grad_input, grad_other};
}

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {
    /*
    transpose:  output = input * weight^T + bias
    output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]

    !transpose: output = input * weight + bias
    output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    const int out_features = weight_dims[transpose ? 0 : 1];

    if (bias) {
        const auto &bias_dims = bias->Dims();
        CHECK_EQ(bias_dims.size(), 1);
        CHECK_EQ(bias_dims[0], out_features);
    }

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    if (transpose) {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix().transpose();
    } else {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix();
    }

    if (bias) {
        output->EigenMatrix().rowwise() += bias->EigenVector();
    }

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<Tensor> &grad_output, const bool bias) {
    /*
    transpose: grad_input = grad_output * weight
    grad_input[*, in_features] = grad_output[*, out_features] * weight[out_features, in_features]
    grad_weight[out_features, in_features] = grad_output[*, out_features]^T * input[*, in_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)

    !transpose: grad_input = grad_output * weight^T
    grad_input[*, in_features] = grad_output[_, out_features] * weight[in_features, out_features]^T
    grad_weight[in_features, out_features] = input[*, in_features]^T * grad_output[*, out_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    CHECK_EQ(out_features, weight_dims[transpose ? 0 : 1]);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32);
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32);
    }

    if (transpose) {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix();
        grad_weight->EigenMatrix() = grad_output->EigenMatrix().transpose() * input->EigenMatrix();
    } else {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix().transpose();
        grad_weight->EigenMatrix() = input->EigenMatrix().transpose() * grad_output->EigenMatrix();
    }
    if (bias) {
        grad_bias->EigenVector() = grad_output->EigenMatrix().colwise().sum();
    }

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_LINEAR_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_LINEAR_KERNEL(MatmulForward)
REGISTER_CPU_LINEAR_KERNEL(MatmulBackward)
REGISTER_CPU_LINEAR_KERNEL(LinearForward)
REGISTER_CPU_LINEAR_KERNEL(LinearBackward)

#undef REGISTER_CPU_LINEAR_KERNEL
