import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from lean_vllm.layers.triton_decode_attn import decode_attention_fwd
from lean_vllm.layers.triton_unified_attn import KVQuantMode, unified_attention
from lean_vllm.utils.context import get_context

# split-K 每片的最少工作量，用来定 num_kv_splits
MIN_WORK_PER_SPLIT = 512


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    k_scale_ptr,
    v_scale_ptr,
    D: tl.constexpr,
    QUANT: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    if QUANT:
        # 写入时量化成 fp8（越界先 clamp）
        ks = tl.load(k_scale_ptr)
        vs = tl.load(v_scale_ptr)
        key = key.to(tl.float32) / ks
        value = value.to(tl.float32) / vs
        key = tl.minimum(tl.maximum(key, -448.0), 448.0).to(tl.float8e4nv)
        value = tl.minimum(tl.maximum(value, -448.0), 448.0).to(tl.float8e4nv)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor, k_scale: torch.Tensor, v_scale: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    quant = k_cache.dtype == torch.float8_e4m3fn
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache,
                               slot_mapping, k_scale, v_scale, D, quant)


class Attention(nn.Module):

    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.k_cache = self.v_cache = torch.tensor([])
        # 由 EngineRunner 注入；默认全关走 flash-attn
        self.fp8_kv = False
        self.page_size = 256
        self.num_kv_splits = 1
        self.k_scale = torch.ones((), dtype=torch.float32)
        self.v_scale = torch.ones((), dtype=torch.float32)
        # split-K 临时缓冲，由 runner 分配共享的一份
        self.decode_scratch = None

    def _decode_triton(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor):
        """纯 decode 走 Triton kernel（支持 fp8 池，也能进 CUDA graph）"""
        context = get_context()
        batch = q.shape[0]
        logits, lse = self.decode_scratch
        o = torch.empty_like(q)
        decode_attention_fwd(q, k_cache, v_cache, o, lse[:batch],
                             context.page_tables, context.context_lens, logits[:batch],
                             self.num_kv_splits, self.scale, self.page_size,
                             k_scale=self.k_scale, v_scale=self.v_scale)
        return o

    def _unified(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor):
        """含 prefill 的步走 unified kernel"""
        context = get_context()
        o = torch.empty_like(q)
        used_k = context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]
        unified_attention(q, k_cache, v_cache, o,
                          context.cu_seqlens_q, context.max_seqlen_q,
                          used_k, context.max_seqlen_k,
                          self.scale, True, (-1, -1), context.page_tables, 0.0,
                          self.k_scale, self.k_scale, self.v_scale,
                          kv_quant_mode=KVQuantMode.FP8_PER_TENSOR)
        return o

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping,
                          self.k_scale, self.v_scale)
        if context.num_prefill_tokens > 0:
            # 本步含 prefill：池子非空时读池子
            if context.page_tables is not None:
                if self.fp8_kv:
                    o = self._unified(q, k_cache, v_cache)
                else:
                    o = flash_attn_varlen_func(q, k_cache, v_cache,
                                               max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                               max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                               softmax_scale=self.scale, causal=True, block_table=context.page_tables)
            else:
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True)
        else:    # decode
            if self.fp8_kv:
                o = self._decode_triton(q, k_cache, v_cache)
            else:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.page_tables,
                                            softmax_scale=self.scale, causal=True)
        return o
