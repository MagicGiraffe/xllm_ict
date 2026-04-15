/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "qwen3_next_rms_norm.h"

#include <glog/logging.h>

#include "kernels/npu/npu_ops_api.h"
#include "platform/device.h"

namespace xllm {
namespace layer {

Qwen3NextRMSNormImpl::Qwen3NextRMSNormImpl(int64_t dim,
                                           double eps,
                                           const torch::TensorOptions& options)
    : norm_dim_(dim), eps_(eps) {
  weight_ = register_parameter("weight", torch::empty({dim}, options), false);
}

torch::Tensor Qwen3NextRMSNormImpl::forward(torch::Tensor& input) {
  auto input_dtype = input.dtype();
  auto gamma = this->gamma();
  if (Device::type_str() == "npu") {
    auto org_shape = input.sizes().vec();
    auto reshaped = input.reshape({-1, norm_dim_});
    auto output = xllm::kernel::npu::rms_norm(reshaped, gamma, eps_, "rmsnorm");
    return output.view(org_shape).to(input_dtype);
  }

  input = input.to(torch::kFloat32);

  // Calculate RMS
  auto variance = torch::mean(torch::pow(input, 2), -1, true);
  auto normalized = input * torch::rsqrt(variance + eps_);

  // Apply weight and convert back to original dtype
  return (normalized * gamma.to(torch::kFloat32)).to(input_dtype);
}

torch::Tensor Qwen3NextRMSNormImpl::gamma() const { return 1.0f + weight_; }

void Qwen3NextRMSNormImpl::load_state_dict(const StateDict& state_dict) {
  LOAD_WEIGHT(weight);
}

}  // namespace layer
}  // namespace xllm
