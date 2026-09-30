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
    # 一步最多给 prefill 算多少 token，0 表示不限
    max_prefill_tokens_per_step: int = 1024
    # 收新请求前必须留空的 KV 页比例
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
