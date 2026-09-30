import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    # 一步里最多给 prefill 多少 token。0 = 不限。
    # 调小它就把一条长 prompt 的 prefill 摊到多步 —— 那一步更短，decode 被冻住的时间更短。
    # 代价有两头：首 token 变慢，以及每步的 token 预算用不完（同样的活要分更多步做，
    # 每步重付一次固定的权重读）。
    # 默认 1024。实测（λ=4.0, seed 0，两轮一致）：ITL p99 137 → 81ms，吞吐 −1%、TTFT +1~2.6%。
    max_prefill_tokens_per_step: int = 1024
    # 准入余量：收新请求前，KV 空闲页里必须留出这么多比例不动用。
    # 默认 0.1（10% = 16 页）。留一点能避免把 KV 吃干 —— 一吃干就开始反复抢占，
    # 而抢占是 λ=4.0 上尾部炸掉的根因（vLLM 的 watermark 就是这个用途）。
    # 实测（seed 0）：10% 把 λ=4.0 的抢占从 76 次降到 1 次、ITL max 从 7.5s 降到 1.6s，
    # 吞吐不变、TTFT 只退 4%。
    kv_admission_watermark: float = 0.1
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_page_size: int = 256
    num_kvcache_pages: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_page_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
