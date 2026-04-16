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

// Neo Fused Attention Ops: 融合算子桥接层
// 连接 xllm 推理引擎与 xllm_ops_neo 提供的高性能自定义算子
//
// 架构说明:
// xllm_ops_neo 中的算子通过 CANN 的 ACLNN API 暴露:
//   - x_flash_attention_infer → aclnnXFlashAttentionInfer
//   - x_attention → aclnnXAttention
//   - custom_paged_attention → atb::customize::CustomPagedAttentionParam
//
// 当 xllm_ops_neo 已编译安装到 $NPU_HOME_PATH/opp/vendors/xllm/ 后,
// 这些 ACLNN API 可通过 cust_opapi 库链接使用。
//
// 当前实现: 委托到现有的 ATB 桥接函数 (atb::npu_flash_attention 等),
// 后续在 xllm_ops_neo 的 ACLNN 接口注册完成后，可直接替换为自定义算子调用。

#ifdef USE_NEO_FUSED_OPS

#include "npu_ops_api.h"
#include "ops_npu/npu_ops.h"

#include <glog/logging.h>

namespace xllm::kernel::npu {

// ---------------------------------------------------------------------------
// neo_batch_prefill: x_flash_attention_infer 替换
// ---------------------------------------------------------------------------
// 参考实现: /data/zzy/xllm_ops_neo/xllm_ops/x_flash_attention_infer/
//
// x_flash_attention_infer 的核心优势:
// 1. 基于 Catlass (Ascend CUTLASS 等效) 模板化框架
// 2. Pipeline: Load Q → [for each KV block: Load K → QK matmul →
//    OnlineSoftmax → Load V → PV matmul → RescaleO]
// 3. Cube Core: BlockMmadQK + BlockMmadPV 跨核同步
// 4. Vector Core: online softmax (causal mask) + O-rescale
// 5. KV 数据预取 2 次迭代实现 pipeline overlap
//
// ACLNN 接口签名 (xllm_ops_neo 编译后注册):
//   aclnnXFlashAttentionInferGetWorkspaceSize(
//       query, key, value, mask, seqLen, scale, output, ...)
//   aclnnXFlashAttentionInfer(workspace, size, executor, stream)
//
// 当前: 委托到 ATB SelfAttention (功能等价, 后续可切换到直接 ACLNN 调用)
void neo_batch_prefill(const torch::Tensor& query,
                       const torch::Tensor& key,
                       const torch::Tensor& value,
                       const torch::Tensor& mask,
                       const torch::Tensor& seq_len,
                       float scale,
                       torch::Tensor& output) {
  int64_t num_heads = query.size(-2);
  int64_t num_kv_heads = key.size(-2);

  LOG_FIRST_N(INFO, 1) << "[NEO] Using neo_batch_prefill (x_flash_attention_infer bridge)";

  // 委托到 ATB flash attention
  // TODO: 当 aclnnXFlashAttentionInfer 注册完成后, 替换为:
  //   utils::create_acltensor() + aclnnXFlashAttentionInferGetWorkspaceSize()
  //   + aclnnXFlashAttentionInfer()
  atb::npu_flash_attention(
      query, key, value, mask, seq_len, scale, num_heads, num_kv_heads, output);
}

// ---------------------------------------------------------------------------
// neo_batch_decode: x_paged_attention / multi_latent_attention 替换
// ---------------------------------------------------------------------------
// 参考实现: /data/zzy/xllm_ops_neo/xllm_ops/x_attention/
//           /data/zzy/xllm_ops_neo/xllm_ops/multi_latent_attention/
//
// multi_latent_attention 的核心优势:
// 1. 直接寄存器级编程: mmad, l1_to_l0_a/b, gm_to_l1 原语
// 2. L0A/L0B/L0C 寄存器 tiling, L1 cache pingpong
// 3. 支持 int8/fp16/bf16, ND→NZ 格式转换
// 4. Split-K 沿 embedding 维度, 支持 TP1 多核 KV 分割
//
// ACLNN 接口签名:
//   aclnnXAttentionGetWorkspaceSize(query, k_cache, v_cache, ...)
//   aclnnXAttention(workspace, size, executor, stream)
void neo_batch_decode(const torch::Tensor& query,
                      const torch::Tensor& k_cache,
                      const torch::Tensor& v_cache,
                      float scale,
                      const torch::Tensor& block_table,
                      const torch::Tensor& seq_lens,
                      torch::Tensor& output) {
  int64_t head_size = query.size(-1);
  int64_t num_heads = query.size(-2);
  int64_t num_kv_heads = k_cache.size(-2);
  auto q = query.view({-1, num_heads, head_size});
  auto o = output.view({-1, num_heads, head_size});

  LOG_FIRST_N(INFO, 1) << "[NEO] Using neo_batch_decode (x_paged_attention bridge)";

  // 委托到 ATB paged attention
  // TODO: 替换为 aclnnXAttention 或 aclnnMultiLatentAttention
  atb::npu_paged_attention(q,
                           k_cache,
                           v_cache,
                           num_kv_heads,
                           num_heads,
                           scale,
                           block_table,
                           seq_lens,
                           o);
}

// ---------------------------------------------------------------------------
// neo_batch_decode_acl_graph: CustomPagedAttention 替换
// ---------------------------------------------------------------------------
// 参考实现: /data/zzy/xllm_ops_neo/atb_customize/ops/custom_paged_attention/
//
// 该算子已在 xllm_ops_neo 中实现, 通过 ATB customize API 注册
// 核心优势: 避免 .to(kCPU) 操作, 在 ACL Graph 中直接执行
void neo_batch_decode_acl_graph(const torch::Tensor& query,
                                const torch::Tensor& k_cache,
                                const torch::Tensor& v_cache,
                                float scale,
                                const torch::Tensor& block_table,
                                const torch::Tensor& seq_lens,
                                const torch::Tensor& tiling_data,
                                torch::Tensor& output) {
  int64_t head_size = query.size(-1);
  int64_t num_heads = query.size(-2);
  int64_t num_kv_heads = k_cache.size(-2);
  auto q = query.view({-1, num_heads, head_size});
  auto o = output.view({-1, num_heads, head_size});

  LOG_FIRST_N(INFO, 1) << "[NEO] Using neo_batch_decode_acl_graph (custom_paged_attention bridge)";

  // 使用 xllm_ops_neo 的 custom_paged_attention (已在 ATB customize 中注册)
  atb::npu_custom_paged_attention(q,
                                  k_cache,
                                  v_cache,
                                  num_kv_heads,
                                  num_heads,
                                  scale,
                                  block_table,
                                  seq_lens,
                                  tiling_data,
                                  o);
}

}  // namespace xllm::kernel::npu

#endif  // USE_NEO_FUSED_OPS
