import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from lean_vllm.layers.linear import FP8_MAX, fp8_linear, make_act_scale
from lean_vllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader
        # 量化后的 fp8 词表矩阵；为 None 表示走 bf16
        self.register_buffer("weight_fp8", None, persistent=False)
        self.register_buffer("weight_scale", None, persistent=False)
        self.register_buffer("act_scale", None, persistent=False)
        self.act_inv = 1.0      # 只有 lm_head 用得上

    def quantize_weight(self, act_amax: float | None = None):
        """把词表矩阵量化成 fp8，存成未转置的 [V, H]；重复调用安全

        act_amax 在这里用不上（查表的"激活"是 token id），只为和 LinearBase 签名一致。
        """
        if self.weight is None:
            return
        w = self.weight.data
        scale = (w.abs().amax().clamp_min(1e-12) / FP8_MAX).float()
        wf = w.to(torch.float32)
        wf.div_(scale)
        self.weight_fp8 = wf.to(torch.float8_e4m3fn)
        self.weight_scale = scale.reshape(())
        self.register_parameter("weight", None)

    def set_act_scale(self, act_amax: float):
        """设激活 scale（只有 lm_head 需要）"""
        self.act_scale, self.act_inv = make_act_scale(act_amax, self.weight_fp8.device)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        if self.weight_fp8 is not None:
            # 查表只读命中的行，反量化回来即可
            y = F.embedding(x, self.weight_fp8).to(torch.bfloat16) * self.weight_scale
        else:
            y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.num_prefill_tokens > 0:
            # prefill 时每条序列只取最后一个 token 算 logits
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        if self.weight_fp8 is not None:
            # 每步都要读满整张词表矩阵，量化收益最直接
            logits = fp8_linear(x, self.weight_fp8.t(), self.weight_scale,
                                self.act_scale, self.act_inv, None)
        else:
            logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
