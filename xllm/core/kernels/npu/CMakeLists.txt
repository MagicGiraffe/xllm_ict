include(cc_library)

add_subdirectory(xllm_ops)
add_subdirectory(tilelang)

cc_library(
  NAME
    npu_kernels
  HDRS
    npu_ops_api.h
  SRCS
    active.cpp
    attention.cpp
    fused_layernorm.cpp
    matmul.cpp
    neo_attention_ops.cpp
    npu_grouped_matmul.cpp
    npu_moe_gating_topk_softmax.cpp
    npu_moe_init_routing_v2.cpp
    npu_moe_token_unpermute.cpp
    rope.cpp
  DEPS
    :torch_npu_kernels
    :tilelang_kernels
    ascendc_ops
)
