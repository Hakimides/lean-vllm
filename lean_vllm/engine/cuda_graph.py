"""CUDA 图：纯 decode 步按批大小分档，含 prefill 的步按 token 桶分档。

对外只暴露三件事：选档（pick_*）、重放（replay_*）、以及捕图（构造时完成）。
"""
from __future__ import annotations

import torch

from lean_vllm.config import Config
from lean_vllm.utils.context import get_context, reset_context, set_context

# 含 prefill 的步进图用的 token 桶（只捕小桶）
PREFILL_BUCKETS = (128, 256, 512, 1024)
# 图的 batch 维上限
MAX_GRAPH_SEQS = 512


class GraphRunner:

    def __init__(self, model, config: Config, page_size: int):
        self.model = model
        self.config = config
        self.page_size = page_size
        self.max_seqs = min(config.max_num_seqs, MAX_GRAPH_SEQS)
        self.max_pages = (config.max_model_len + page_size - 1) // page_size
        self.pool = None
        self.decode: dict[int, torch.cuda.CUDAGraph] = {}
        self.decode_vars: dict = {}
        self.prefill: dict[int, dict] = {}
        self._capture_decode()
        self._capture_prefill()

    # ---------- 选档 ----------

    def pick_decode(self, num_seqs: int) -> int | None:
        """纯 decode 步：返回覆盖该批大小的档位"""
        if num_seqs > self.max_seqs:
            return None
        return next((b for b in self.decode_bs if b >= num_seqs), None)

    def pick_prefill(self, num_tokens: int, num_seqs: int) -> int | None:
        """含 prefill 的步：token 数落进桶、batch 不超上限、写满 KV、且读的是 KV 池"""
        context = get_context()
        if not self.prefill or num_seqs > self.max_seqs:
            return None
        if context.page_tables is None or context.slot_mapping is None:
            return None
        if context.slot_mapping.numel() != num_tokens:
            return None
        return next((b for b in sorted(self.prefill) if b >= num_tokens), None)

    # ---------- 重放 ----------

    def replay_decode(self, key: int, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """重放 decode 图，返回 batch 内的 hidden states"""
        context = get_context()
        bs = input_ids.size(0)
        gv = self.decode_vars
        gv["input_ids"][:bs] = input_ids
        gv["positions"][:bs] = positions
        gv["slot_mapping"].fill_(-1)
        gv["slot_mapping"][:bs] = context.slot_mapping
        gv["context_lens"].zero_()
        gv["context_lens"][:bs] = context.context_lens
        gv["page_tables"][:bs, :context.page_tables.size(1)] = context.page_tables
        self.decode[key].replay()
        return gv["outputs"][:bs]

    def replay_prefill(self, bucket: int, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """把形状对齐到桶再重放，返回本步 token 的 hidden states"""
        ctx = get_context()
        gv = self.prefill[bucket]
        num_tokens = input_ids.size(0)
        num_seqs = ctx.cu_seqlens_q.numel() - 1

        gv["input_ids"][:num_tokens] = input_ids
        gv["positions"][:num_tokens] = positions
        gv["slot_mapping"].fill_(-1)
        gv["slot_mapping"][:num_tokens] = ctx.slot_mapping
        gv["cu_seqlens_q"][:num_seqs + 1] = ctx.cu_seqlens_q
        gv["cu_seqlens_q"][num_seqs + 1:].fill_(int(ctx.cu_seqlens_q[-1]))
        gv["cu_seqlens_k"][:num_seqs + 1] = ctx.cu_seqlens_k
        gv["cu_seqlens_k"][num_seqs + 1:].fill_(int(ctx.cu_seqlens_k[-1]))
        gv["page_tables"][:num_seqs, :ctx.page_tables.size(1)] = ctx.page_tables

        # 图内用对齐后的形状与固定的 max_seqlen_q
        set_context(bucket, gv["cu_seqlens_q"], gv["cu_seqlens_k"], bucket, bucket,
                    gv["slot_mapping"], None, gv["page_tables"])
        gv["graph"].replay()
        # 交还真实 context
        set_context(ctx.num_prefill_tokens, ctx.cu_seqlens_q, ctx.cu_seqlens_k,
                    ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.slot_mapping,
                    ctx.context_lens, ctx.page_tables)
        return gv["outputs"][:num_tokens]

    # ---------- 捕图 ----------

    @torch.inference_mode()
    def _capture_decode(self):
        max_bs = self.max_seqs
        hf_config = self.config.hf_config
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        page_tables = torch.zeros(max_bs, self.max_pages, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.decode_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))

        for bs in reversed(self.decode_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(0, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs],
                        page_tables=page_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])            # warmup
            with torch.cuda.graph(graph, self.pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])        # capture
            if self.pool is None:
                self.pool = graph.pool()
            self.decode[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.decode_vars = dict(
            input_ids=input_ids, positions=positions, slot_mapping=slot_mapping,
            context_lens=context_lens, page_tables=page_tables, outputs=outputs,
        )

    @torch.inference_mode()
    def _capture_prefill(self):
        """每个 token 桶各捕一张；batch 维固定为 max_bs，空位用长度 0 的序列补"""
        max_bs = self.max_seqs
        hf_config = self.config.hf_config
        cu_seqlens_q = torch.zeros(max_bs + 1, dtype=torch.int32)
        cu_seqlens_k = torch.zeros(max_bs + 1, dtype=torch.int32)

        for bucket in PREFILL_BUCKETS if self.config.prefill_graph else ():
            tokens = torch.zeros(bucket, dtype=torch.int64)
            pos = torch.zeros(bucket, dtype=torch.int64)
            slots = torch.full((bucket,), -1, dtype=torch.int32)
            pages = torch.zeros(max_bs, self.max_pages, dtype=torch.int32)
            out = torch.zeros(bucket, hf_config.hidden_size)
            set_context(bucket, cu_seqlens_q, cu_seqlens_k, bucket, bucket, slots, None, pages)
            out[:] = self.model(tokens, pos)                                     # warmup
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, self.pool):
                out[:] = self.model(tokens, pos)                                 # capture
            self.prefill[bucket] = dict(
                graph=graph, input_ids=tokens, positions=pos, slot_mapping=slots,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                page_tables=pages, outputs=out,
            )
            torch.cuda.synchronize()
            reset_context()
