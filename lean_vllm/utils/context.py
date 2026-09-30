from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    # 本步要算的 prefill token 数，0 表示整步都是 decode
    num_prefill_tokens: int = 0
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    page_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(num_prefill_tokens=0, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, page_tables=None):
    global _CONTEXT
    _CONTEXT = Context(num_prefill_tokens, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, page_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
