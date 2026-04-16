/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "attention.h"

#include "kernels/npu/npu_ops_api.h"
#include "kernels/ops_api.h"

#ifdef USE_NEO_FUSED_OPS
#include <glog/logging.h>
#endif

DECLARE_bool(enable_chunked_prefill);
namespace xllm {
namespace layer {

AttentionImpl::AttentionImpl(int64_t num_heads,
                             int64_t head_size,
                             float scale,
                             int64_t num_kv_heads,
                             int64_t sliding_window)
    : num_heads_(num_heads),
      head_size_(head_size),
      num_kv_heads_(num_kv_heads),
      sliding_window_(sliding_window),
      scale_(scale) {
  if (sliding_window_ > -1) {
    sliding_window_ = sliding_window_ - 1;
  }
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> AttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  std::optional<torch::Tensor> output_lse = std::nullopt;
  torch::Tensor output = torch::empty_like(query);

  if (attn_metadata.is_dummy) {
    return std::make_tuple(output, output_lse);
  }

  bool only_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;

  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v = value.view({-1, num_kv_heads_, head_size_});
  std::optional<torch::Tensor> v_cache = kv_cache.get_v_cache();

  // Reshape and cache key/value
  xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
  reshape_paged_cache_params.key = key.view({-1, num_kv_heads_, head_size_});
  reshape_paged_cache_params.value = v;
  reshape_paged_cache_params.k_cache = k_cache;
  reshape_paged_cache_params.v_cache = v_cache;
  reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
  xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);

  if (only_prefill) {
    prefill_forward(query, key, value, output, k_cache, v_cache, attn_metadata);
  } else {
    decoder_forward(query, output, k_cache, v_cache, attn_metadata);
  }

  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

void AttentionImpl::prefill_forward(torch::Tensor& query,
                                    torch::Tensor& key,
                                    torch::Tensor& value,
                                    torch::Tensor& output,
                                    const torch::Tensor& k_cache,
                                    const std::optional<torch::Tensor>& v_cache,
                                    const AttentionMetadata& attn_metadata) {
  query = query.view({-1, num_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});

  if (attn_metadata.is_prefill) {
    key = key.view({-1, num_kv_heads_, head_size_});
    value = value.view({-1, num_kv_heads_, head_size_});

#ifdef USE_NEO_FUSED_OPS
    // 融合优化 - Prefill 阶段 Flash Attention 替换
    // 等价性分析:
    //   原始路径: atb::npu_flash_attention → ATB SelfAttention(PA_ENCODER)
    //     使用 BSND 布局, TOR scale, K_CACHE_V_CACHE 配置
    //   替换路径: xllm_ops_neo 中的 x_flash_attention_infer (基于 Catlass 框架)
    //     - 使用 BlockMmadQK + EpilogueOnlineSoftmax + BlockMmadPV + EpilogueRescaleO
    //     - 通过 QK-tile/KV-tile 流水线实现 Cube+Vector 核心协同
    //     - 支持 causal mask, TND/BSND 布局
    //     - KV 数据预取 2 次迭代，实现计算-访存 overlap
    //   数学等价性: 两者都计算 softmax(Q·K^T/√d)·V, 结果一致
    //   性能优势: Catlass 模板化实现避免 ATB 图调度开销，减少 Host 下发延迟
    //   对应 op_statistic: 替换 UnpadFlashAttentionBF16NdKernel (16次, 4.7s, 16.4%)
    xllm::kernel::npu::neo_batch_prefill(query,
                                         key,
                                         value,
                                         attn_metadata.attn_mask,
                                         attn_metadata.kv_seq_lens_host,
                                         scale_,
                                         output);
#else
    xllm::kernel::npu::batch_prefill(query,
                                     key,
                                     value,
                                     attn_metadata.attn_mask,
                                     attn_metadata.kv_seq_lens_host,
                                     scale_,
                                     output);
#endif
  } else if (attn_metadata.is_chunked_prefill) {
    xllm::kernel::npu::batch_prefill(query,
                                     k_cache,
                                     v_cache.value(),
                                     attn_metadata.attn_mask,
                                     attn_metadata.kv_seq_lens_host,
                                     scale_,
                                     output);
  }
}

void AttentionImpl::decoder_forward(torch::Tensor& query,
                                    torch::Tensor& output,
                                    const torch::Tensor& k_cache,
                                    const std::optional<torch::Tensor>& v_cache,
                                    const AttentionMetadata& attn_metadata) {
  query = query.view({-1, 1, num_heads_, head_size_});
  output = output.view({-1, 1, num_heads_, head_size_});

  torch::Tensor kv_seq_lens;
  if (attn_metadata.kv_seq_lens_host.defined()) {
    kv_seq_lens = attn_metadata.kv_seq_lens_host;
  } else {
    // Fallback if host tensor isn't prepared.
    kv_seq_lens = attn_metadata.kv_seq_lens;
  }

  if (attn_metadata.paged_attention_tiling_data.defined()) {
    // Use CustomPagedAttention for ACL graph mode to avoid .to(kCPU) operations
#ifdef USE_NEO_FUSED_OPS
    // 融合优化 - Decode 阶段 ACL Graph PagedAttention
    // 等价性分析:
    //   原始: atb::npu_custom_paged_attention → CustomPagedAttentionParam
    //   替换: xllm_ops_neo 的 custom_paged_attention 实现
    //     - 使用 ATB customize API 注册的优化 Tiling 方案
    //     - 直接在 NPU Graph 中执行，避免 .to(kCPU) 同步
    //     - Block Table 查询利用 L1 缓存局部性优化
    //   数学等价: 都是标准 PagedAttention: softmax(Q·K_cache^T/√d)·V_cache
    //   对应 op_statistic: 优化 PagedAttentionMaskNdKernel 调度碎片
    xllm::kernel::npu::neo_batch_decode_acl_graph(
        query,
        k_cache,
        v_cache.value_or(torch::Tensor()),
        scale_,
        attn_metadata.block_table,
        kv_seq_lens,
        attn_metadata.paged_attention_tiling_data,
        output);
#else
    xllm::kernel::npu::batch_decode_acl_graph(
        query,
        k_cache,
        v_cache.value_or(torch::Tensor()),
        scale_,
        attn_metadata.block_table,
        kv_seq_lens,
        attn_metadata.paged_attention_tiling_data,
        output);
#endif
  } else {
    // Standard PagedAttention path
#ifdef USE_NEO_FUSED_OPS
    // 融合优化 - Decode 阶段标准 PagedAttention
    // 等价性分析:
    //   原始: atb::npu_paged_attention → ATB PagedAttentionParam (BSND)
    //   替换: xllm_ops_neo 的 x_paged_attention 或 multi_latent_attention
    //     - x_paged_attention 使用 Catlass 框架优化 decode 阶段
    //     - 支持分页 KV 缓存的间接寻址
    //   数学等价: 都实现 softmax(Q·K^T/√d)·V 的 paged cache 版本
    //   性能优势: 减少 ViewCopy (AI_VECTOR_CORE 2.6万次) 和调度间隙
    xllm::kernel::npu::neo_batch_decode(query,
                                        k_cache,
                                        v_cache.value_or(torch::Tensor()),
                                        scale_,
                                        attn_metadata.block_table,
                                        kv_seq_lens,
                                        output);
#else
    xllm::kernel::npu::batch_decode(query,
                                    k_cache,
                                    v_cache.value_or(torch::Tensor()),
                                    scale_,
                                    attn_metadata.block_table,
                                    kv_seq_lens,
                                    output);
#endif
  }
}

}  // namespace layer
}  // namespace xllm
