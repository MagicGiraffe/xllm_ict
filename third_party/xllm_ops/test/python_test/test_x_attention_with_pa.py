import torch
import torch_npu
import custom_ops
import numpy as np
import pytest
import random

torch.manual_seed(1)
random.seed(1)

def get_max_block_num(shared_kv_lens, unshared_kv_len, block_size):
    max_block_num = 0
    for shared_kv_len in shared_kv_lens:
        kv_len = shared_kv_len + unshared_kv_len
        block_num = (kv_len + block_size - 1) // block_size
        max_block_num = max(max_block_num, block_num)
    return max_block_num

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("q_head_num, kv_head_num", [(16, 8), (32, 8)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("batch", [1, 6, 8])
@pytest.mark.parametrize("beam_size", [128, 256, 512])
@pytest.mark.parametrize("max_decode_step", [2, 4, 8, 12, 16, 20, 50, 100])
@pytest.mark.parametrize("prompt_length", [128, 256, 512, 1024])
def test_x_attention_with_pa(dtype, q_head_num, kv_head_num, head_dim, batch, beam_size, max_decode_step, prompt_length):
    shared_block_tokens = batch * prompt_length + 8192
    unshared_block_num = batch + 209
    q_min_range = -50.0
    q_max_range = 50.0
    k_min_range = -500.0
    k_max_range = 500.0
    v_min_range = -3.0
    v_max_range = 3.0
    unshared_kv_len = random.randint(1, max_decode_step)

    # generate test data
    total_tokens = batch * beam_size
    query = (torch.empty((total_tokens, q_head_num, head_dim), dtype=dtype)
                    .uniform_(q_min_range, q_max_range))
    shared_key_block = (torch.empty((shared_block_tokens, kv_head_num, head_dim), dtype=dtype).uniform_(k_min_range, k_max_range))
    shared_value_block = (torch.empty((shared_block_tokens, kv_head_num, head_dim), dtype=dtype).uniform_(v_min_range, v_max_range))
    shared_kv_lens = torch.tensor([prompt_length] * batch, dtype=torch.int32)
    
    unshared_block_tables = torch.randperm(unshared_block_num)[:batch].to(torch.int32)
    unshared_key_block = torch.zeros((unshared_block_num, beam_size, kv_head_num, max_decode_step, head_dim), dtype=dtype)
    unshared_value_block = torch.zeros((unshared_block_num, beam_size, kv_head_num, max_decode_step, head_dim), dtype=dtype)
    
    for block_idx in unshared_block_tables:
        unshared_key_cache = (torch.empty(
                    (beam_size, kv_head_num, unshared_kv_len, head_dim),
                    dtype=dtype
                ).uniform_(k_min_range, k_max_range))
        unshared_value_cache = (torch.empty(
                    (beam_size, kv_head_num, unshared_kv_len, head_dim),
                    dtype=dtype
                ).uniform_(v_min_range, v_max_range))
        unshared_key_block[block_idx, :, :, :unshared_kv_len, :] = unshared_key_cache
        unshared_value_block[block_idx, :, :, :unshared_kv_len, :] = unshared_value_cache
    decode_step_tensor = torch.tensor([unshared_kv_len], dtype=torch.int32)
    x_attention_out = custom_ops.x_attention_npu(
            query.npu(), shared_key_block.npu(), shared_value_block.npu(), unshared_key_block.npu(), unshared_value_block.npu(), 
            shared_kv_lens.npu(), decode_step_tensor.npu(),
            None, unshared_block_tables.npu()
        )
    torch.npu.synchronize()
    block_size = 128
    max_block_num = get_max_block_num(shared_kv_lens.tolist(), unshared_kv_len, block_size)
    pa_block_num = total_tokens * max_block_num + 1024
    pa_key_block = torch.zeros((pa_block_num, block_size, kv_head_num, head_dim), dtype=dtype)
    pa_value_block = torch.zeros((pa_block_num, block_size, kv_head_num, head_dim), dtype=dtype)

    pa_block_table = torch.randperm(pa_block_num)[:(max_block_num * total_tokens)].reshape(total_tokens, max_block_num).to(torch.int32)
    shared_kv_offset = 0
    for batch_idx in range(batch):
        shared_kv_len = shared_kv_lens[batch_idx].item()
        unshared_block_index = unshared_block_tables[batch_idx]
        shared_kv_block_num = (shared_kv_len + block_size - 1) // block_size
        total_kv_len = shared_kv_len + unshared_kv_len
        total_kv_block_num = (total_kv_len + block_size - 1) // block_size
        batch_shared_key = shared_key_block[shared_kv_offset: (shared_kv_offset + shared_kv_len), ...]
        batch_shared_value = shared_value_block[shared_kv_offset: (shared_kv_offset + shared_kv_len), ...]
        
        base_token_idx = batch_idx * beam_size
        block_indices = pa_block_table[base_token_idx: base_token_idx + beam_size, :total_kv_block_num]
        
        for i in range(shared_kv_block_num):
            block_kv_len = block_size if i < shared_kv_block_num - 1 else shared_kv_len - i * block_size
            src_start = i * block_size
            src_end = src_start + block_kv_len
            pa_key_block[block_indices[:, i], :block_kv_len, ...] = batch_shared_key[src_start:src_end, ...].unsqueeze(0).repeat(beam_size, 1, 1, 1)
            pa_value_block[block_indices[:, i], :block_kv_len, ...] = batch_shared_value[src_start:src_end, ...].unsqueeze(0).repeat(beam_size, 1, 1, 1)
        
        unshared_keys = unshared_key_block[unshared_block_index.item(), :, :, :unshared_kv_len, :].permute(0, 2, 1, 3)
        unshared_values = unshared_value_block[unshared_block_index.item(), :, :, :unshared_kv_len, :].permute(0, 2, 1, 3)
        
        if shared_kv_block_num == total_kv_block_num:
            block_indices_last = pa_block_table[base_token_idx: base_token_idx + beam_size, -1]
            unshared_kv_start_idx = shared_kv_len % block_size
            pa_key_block[block_indices_last, unshared_kv_start_idx: unshared_kv_start_idx + unshared_kv_len, ...] = unshared_keys
            pa_value_block[block_indices_last, unshared_kv_start_idx: unshared_kv_start_idx + unshared_kv_len, ...] = unshared_values
        else:
            unshared_kv_left_len = unshared_kv_len
            if shared_kv_len % block_size == 0:
                unshared_start_block_idx = shared_kv_block_num
                start_idx = 0
            else:
                unshared_start_block_idx = shared_kv_block_num - 1
                start_idx = shared_kv_len % block_size
            
            for i in range(unshared_start_block_idx, total_kv_block_num):
                block_indices_i = pa_block_table[base_token_idx: base_token_idx + beam_size, i]
                curr_start_idx = 0 if i > unshared_start_block_idx else start_idx
                block_tail_len = block_size - curr_start_idx
                if block_tail_len == 0:
                    continue
                block_copy_len = min(block_tail_len, unshared_kv_left_len)
                pa_key_block[block_indices_i, curr_start_idx: curr_start_idx + block_copy_len, ...] = unshared_keys[:, :block_copy_len, ...]
                pa_value_block[block_indices_i, curr_start_idx: curr_start_idx + block_copy_len, ...] = unshared_values[:, :block_copy_len, ...]
                unshared_keys = unshared_keys[:, block_copy_len:, ...]
                unshared_values = unshared_values[:, block_copy_len:, ...]
                unshared_kv_left_len -= block_copy_len
        shared_kv_offset += shared_kv_len
    
    context_lens = []
    for batch_idx in range(batch):
        shared_kv_len = shared_kv_lens[batch_idx].item()
        context_lens.extend([shared_kv_len + unshared_kv_len] * beam_size)
    context_lens = torch.tensor(context_lens).to(torch.int32)
    atb_pa_out = torch.zeros_like(x_attention_out)
    torch_npu._npu_paged_attention(
            query.npu(), pa_key_block.npu(), pa_value_block.npu(),
            kv_head_num, q_head_num, 1.0 / np.sqrt(head_dim),
            pa_block_table.npu(), context_lens, atb_pa_out.npu()
        )
    torch.npu.synchronize()

    if dtype == torch.bfloat16:
        atol = 0.01
        rtol = 0.01
    else:
        atol = 0.001
        rtol = 0.001

    x_attention_out = x_attention_out.float().cpu()
    atb_pa_out = atb_pa_out.float().cpu()


    flag = torch.allclose(x_attention_out, atb_pa_out, atol=atol, rtol=rtol, equal_nan=True)
    assert flag
