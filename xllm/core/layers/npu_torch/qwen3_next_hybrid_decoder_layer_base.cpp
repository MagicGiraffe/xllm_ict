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

#include "qwen3_next_hybrid_decoder_layer_base.h"

#include <algorithm>

#ifdef USE_NEO_FUSED_OPS
#include "kernels/npu/npu_ops_api.h"
#endif

namespace xllm {
namespace layer {

Qwen3HybridDecoderLayerImplBase::Qwen3HybridDecoderLayerImplBase(
    const ModelContext& context,
    int32_t layer_id,
    std::shared_ptr<Qwen3GatedDeltaNetBaseImpl> linear_attention_module) {
  const auto& model_args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();
  const bool use_full_attention = is_full_attention_layer(model_args, layer_id);

  // Initialize attention layers
  if (use_full_attention) {
    attention_ = register_module(
        "self_attn",
        Qwen3NextAttention(
            model_args, quant_args, parallel_args, options, layer_id));
  } else {
    linear_attention_ =
        register_module("linear_attn", std::move(linear_attention_module));
  }

  // Initialize norm layers
  input_norm_ = register_module(
      "input_layernorm",
      Qwen3NextRMSNorm(
          model_args.hidden_size(), model_args.rms_norm_eps(), options));

  post_norm_ = register_module(
      "post_attention_layernorm",
      Qwen3NextRMSNorm(
          model_args.hidden_size(), model_args.rms_norm_eps(), options));

  // Initialize mlp
  auto mlp_only_layers = model_args.mlp_only_layers();
  if ((std::count(mlp_only_layers.begin(), mlp_only_layers.end(), layer_id) ==
       0) &&
      model_args.n_routed_experts() > 0 &&
      (layer_id + 1) % model_args.decoder_sparse_step() == 0) {
    moe_mlp_ = register_module("mlp",
                               FusedMoE(model_args,
                                        FusedMoEArgs{.is_gated = true},
                                        quant_args,
                                        parallel_args,
                                        options));
  } else {
    mlp_ = register_module("mlp",
                           DenseMLP(model_args.hidden_size(),
                                    model_args.intermediate_size(),
                                    true,
                                    false,
                                    model_args.hidden_act(),
                                    /*enable_result_reduction=*/true,
                                    quant_args,
                                    parallel_args.tp_group_,
                                    options));
  }
}

void Qwen3HybridDecoderLayerImplBase::load_state_dict(
    const StateDict& state_dict) {
  if (attention_) {
    attention_->load_state_dict(state_dict.get_dict_with_prefix("self_attn."));
  } else {
    linear_attention_->load_state_dict(
        state_dict.get_dict_with_prefix("linear_attn."));
  }
  input_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("input_layernorm."));
  post_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("post_attention_layernorm."));
  if (moe_mlp_) {
    moe_mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
  } else {
    mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
  }
}

void Qwen3HybridDecoderLayerImplBase::verify_loaded_weights(
    const std::string& prefix) const {
  if (linear_attention_) {
    linear_attention_->verify_loaded_weights(prefix + "linear_attn.");
  }
}

torch::Tensor Qwen3HybridDecoderLayerImplBase::forward(
    torch::Tensor& x,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  // Pre-attention norm
  torch::Tensor residual = x;
  x = input_norm_(x);

  // Attention
  if (attention_) {
    x = attention_->forward(positions, x, attn_metadata, kv_cache);
  } else {
    x = linear_attention_->forward(x, attn_metadata, kv_cache, input_params);
  }

#ifdef USE_NEO_FUSED_OPS
  // 融合优化 - FusedAddRmsNorm: 将 Residual Add + Post-Attention Norm 融合
  // 等价性分析:
  //   原始路径 (4 步):
  //     1. x = x.to(kFloat32)           // Cast BF16→FP32
  //     2. residual = residual.to(kFloat32)  // Cast BF16→FP32
  //     3. x = x + residual             // Add FP32
  //     4. residual = x; x = x.to(orig_dtype); x = post_norm_(x)  // Norm
  //   这产生了 2x Cast + 1x Add + 1x RmsNorm = 4 次 Kernel 下发
  //   且中间 FP32 张量需要额外 HBM 写回 (hidden_size * 2 * sizeof(float))
  //
  //   融合路径 (1 步): npu::add_rms_norm(x, residual, gamma, eps)
  //     - 在单个 Vector Core Kernel 中完成:
  //       y = x + residual
  //       rms = sqrt(mean(y^2) + eps)
  //       output = gamma * y / rms
  //     - 中间结果留存在 UB 中，不写回 HBM
  //     - 参考 ops-nn/norm/add_rms_norm 的 AscendC 实现:
  //       KernelAddRmsNorm<T, MODE> 支持 fp16/bf16/fp32
  //     - ATB 中 RmsNormOperation 的 RMS_NORM_PRENORM 模式也提供相同语义
  //   数学等价: output = RmsNorm(x + residual, gamma, eps)
  //   性能优势: 4次 Kernel → 1次，消除 2x Cast 和中间 FP32 张量的 HBM 读写
  //   对应 op_statistic: 减少 Add (5万次) 和 Cast 相关的 ViewCopy (2.6万次)
  {
    auto norm_weight = post_norm_->named_parameters()["weight"];
    double eps = 1e-6;  // default Qwen3 RMS norm eps
    auto [normed, residual_out, rstd] =
        xllm::kernel::npu::add_rms_norm(x, residual, norm_weight, eps);
    x = normed;
    residual = residual_out;
  }
#else
  auto orig_dtype = x.dtype();
  if (orig_dtype == torch::kBFloat16) {
    x = x.to(torch::kFloat32);
    residual = residual.to(torch::kFloat32);
  }
  x = x + residual;

  // Post-attention norm
  residual = x;
  x = x.to(orig_dtype);
  x = post_norm_(x);
#endif

  // MLP forward
  if (moe_mlp_) {
    x = moe_mlp_(x, input_params);
  } else {
    x = mlp_(x);
  }

#ifdef USE_NEO_FUSED_OPS
  // 融合优化 - MLP 后的 Residual Add
  // 同理, 将 MLP 输出与 residual 的加法也使用融合方式
  // 但由于这是最后一步且没有后续 norm，直接做 add 即可
  {
    auto orig_dtype2 = x.dtype();
    if (orig_dtype2 == torch::kBFloat16) {
      x = x.to(torch::kFloat32);
      residual = residual.to(torch::kFloat32);
    }
    x = x + residual;
    x = x.to(orig_dtype2);
  }
#else
  orig_dtype = x.dtype();
  if (orig_dtype == torch::kBFloat16) {
    x = x.to(torch::kFloat32);
    residual = residual.to(torch::kFloat32);
  }
  x = x + residual;
  x = x.to(orig_dtype);
#endif
  return x;
}

}  // namespace layer
}  // namespace xllm
