from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


FP8_MAX = 448.0          # e4m3 能表示的最大绝对值


def quantize_activation(x2: torch.Tensor, act_inv: float) -> torch.Tensor:
    """按静态 scale 裁剪并量化激活到 fp8"""
    xq = torch.mul(x2, act_inv)
    xq = torch.clamp(xq, -FP8_MAX, FP8_MAX)
    return xq.to(torch.float8_e4m3fn)


def fp8_linear(x, weight_fp8_t, weight_scale, act_scale, act_inv, bias):
    """W8A8 线性层；激活与权重都用 per-tensor 标量 scale"""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    xq = quantize_activation(x2, act_inv)
    y = torch._scaled_mm(xq, weight_fp8_t, scale_a=act_scale,
                         scale_b=weight_scale, out_dtype=torch.bfloat16)
    y = y.reshape(*shape[:-1], weight_fp8_t.shape[1])
    if bias is not None:
        y = y + bias
    return y


def make_act_scale(amax: float, device):
    """由激活 amax 算出 (scale, 它的倒数)"""
    scale = max(float(amax), 1e-6) / FP8_MAX
    return torch.tensor(scale, device=device, dtype=torch.float32).reshape(()), 1.0 / scale


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module, ABC):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
        # 量化后的 fp8 权重；为 None 表示这条线性层走 bf16
        self.register_buffer("weight_fp8_t", None, persistent=False)
        self.register_buffer("weight_scale", None, persistent=False)
        self.register_buffer("act_scale", None, persistent=False)
        self.act_inv = 1.0      # 激活的乘数（python float，省一次 tensor 派发）

    def quantize_weight(self, act_amax: float):
        """把权重转成 fp8（标量 scale），并记下该层的激活 scale；重复调用安全"""
        if self.weight is None:
            return
        if act_amax is None:
            raise ValueError("校准表里没有这一层的激活 amax；先跑 benchmarks/calibrate.py")
        w = self.weight.data
        wscale = (w.abs().amax().clamp_min(1e-12) / FP8_MAX).float()
        wf = w.to(torch.float32)
        wf.div_(wscale)
        self.weight_fp8_t = wf.to(torch.float8_e4m3fn).t()
        self.weight_scale = wscale.reshape(())
        self.act_scale, self.act_inv = make_act_scale(act_amax, w.device)
        self.register_parameter("weight", None)

    def _linear(self, x: torch.Tensor, bias) -> torch.Tensor:
        """量化过走 fp8，否则退回 bf16"""
        if self.weight_fp8_t is not None:
            return fp8_linear(x, self.weight_fp8_t, self.weight_scale,
                              self.act_scale, self.act_inv, bias)
        return F.linear(x, self.weight, bias)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear(x, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._linear(x, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
