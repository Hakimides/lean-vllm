import json
import os
import pickle
import torch
import torch.distributed as dist
import triton
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from lean_vllm.config import Config
from lean_vllm.engine.cuda_graph import GraphRunner
from lean_vllm.engine.sequence import Request
from lean_vllm.models.qwen3 import Qwen3ForCausalLM
from lean_vllm.layers.attention import MIN_WORK_PER_SPLIT
from lean_vllm.layers.linear import FP8_MAX
from lean_vllm.layers.sampler import Sampler
from lean_vllm.utils.context import reset_context, set_context
from lean_vllm.utils.loader import load_model


def load_fp8_scales(path: str) -> dict:
    """读 fp8 校准表"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到 fp8 校准表 {path}；先跑 benchmarks/calibrate.py")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class EngineRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.page_size = config.kvcache_page_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.cuda_graphs: GraphRunner | None = None

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        # 量化须在 acquire_kv_cache 之前
        if config.fp8_linear:
            self.model.quantize_fp8(load_fp8_scales(config.fp8_scales_path)["activation_amax"])
        self.sampler = Sampler()
        self.warmup_model()
        self.acquire_kv_cache()
        if not self.enforce_eager:
            self.cuda_graphs = GraphRunner(self.model, config, self.page_size)
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="lean_vllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="lean_vllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        self.cuda_graphs = None
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Request([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.scheduled_len = seq_len
        self.run(seqs)
        torch.cuda.empty_cache()

    def acquire_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        kv_dtype = torch.float8_e4m3fn if config.kv_fp8 else hf_config.dtype
        page_bytes = 2 * hf_config.num_hidden_layers * self.page_size * num_kv_heads * head_dim * kv_dtype.itemsize
        config.num_kvcache_pages = int(total * config.gpu_memory_utilization - used - peak + current) // page_bytes
        assert config.num_kvcache_pages > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_pages, self.page_size, num_kv_heads, head_dim, dtype=kv_dtype)
        k_scale = v_scale = None
        if config.kv_fp8:
            kv_amax = load_fp8_scales(config.fp8_scales_path).get("kv_amax")
            if kv_amax is None:
                raise ValueError("校准表里没有 kv_amax；重跑 benchmarks/calibrate.py")
            k_scale = torch.tensor(kv_amax["k"] / FP8_MAX, device="cuda", dtype=torch.float32).reshape(())
            v_scale = torch.tensor(kv_amax["v"] / FP8_MAX, device="cuda", dtype=torch.float32).reshape(())
        num_kv_splits = min(
            triton.next_power_of_2(max(1, config.max_model_len // MIN_WORK_PER_SPLIT)),
            torch.cuda.get_device_properties(0).multi_processor_count * 2)
        scratch = None
        if config.kv_fp8:
            # 各层共用一份 split-K 临时缓冲
            num_heads = hf_config.num_attention_heads // self.world_size
            scratch = (
                torch.empty(config.max_num_seqs, num_heads, num_kv_splits, head_dim + 1,
                            dtype=torch.float32, device="cuda"),
                torch.empty(config.max_num_seqs, num_heads, dtype=torch.float32, device="cuda"),
            )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                module.fp8_kv = config.kv_fp8
                module.page_size = self.page_size
                module.num_kv_splits = num_kv_splits
                if scratch is not None:
                    module.k_scale, module.v_scale = k_scale, v_scale
                    module.decode_scratch = scratch
                layer_id += 1

    def prepare_page_tables(self, seqs: list[Request]):
        max_len = max(len(seq.page_table) for seq in seqs)
        page_tables = [seq.page_table + [-1] * (max_len - len(seq.page_table)) for seq in seqs]
        page_tables = torch.tensor(page_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return page_tables

    def prepare_batch(self, seqs: list[Request]):
        """摊平一批请求，返回 (input_ids, positions, prefill token 数)"""
        # decode 步按每条序列各自的 KV 长度去读
        is_decode_only = all(not seq.is_prefill for seq in seqs)

        input_ids = []
        positions = []
        slot_mapping = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        context_lens = []
        num_prefill_tokens = 0

        for seq in seqs:
            if seq.is_prefill:
                start = seq.cached_len
                seqlen_q = seq.scheduled_len
                num_prefill_tokens += seqlen_q
            else:
                start = len(seq) - 1
                seqlen_q = 1
                context_lens.append(len(seq))
            end = start + seqlen_q
            seqlen_k = end

            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.page_table:    # 预热阶段还没有 KV 页，不往 cache 里写
                continue
            start_page = start // self.page_size
            end_page = (end + self.page_size - 1) // self.page_size
            for i in range(start_page, end_page):
                slot_start = seq.page_table[i] * self.page_size
                if i == start_page:
                    slot_start += start % self.page_size
                if i != end_page - 1:
                    slot_end = seq.page_table[i] * self.page_size + self.page_size
                else:
                    slot_end = seq.page_table[i] * self.page_size + end - i * self.page_size
                slot_mapping.extend(range(slot_start, slot_end))

        # 用 Python 列表判断
        needs_page_tables = is_decode_only or cu_seqlens_k[-1] > cu_seqlens_q[-1]

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # 纯 decode 步不构造 cu_seqlens
        if num_prefill_tokens > 0:
            cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        else:
            cu_seqlens_q = cu_seqlens_k = None

        # decode 步必须有 block_table，prefill 步仅在命中前缀缓存时需要
        page_tables = self.prepare_page_tables(seqs) if needs_page_tables else None
        if is_decode_only:
            context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        else:
            context_lens = None

        set_context(num_prefill_tokens, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    slot_mapping, context_lens, page_tables)
        return input_ids, positions, num_prefill_tokens

    def prepare_sample(self, seqs: list[Request]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor,
                  decode_key: int | None = None, bucket: int | None = None):
        if bucket is not None:
            hidden = self.cuda_graphs.replay_prefill(bucket, input_ids, positions)
        elif decode_key is not None:
            hidden = self.cuda_graphs.replay_decode(decode_key, input_ids, positions)
        else:
            hidden = self.model(input_ids, positions)
        return self.model.compute_logits(hidden)

    def run(self, seqs: list[Request]) -> list[int]:
        input_ids, positions, num_prefill_tokens = self.prepare_batch(seqs)
        decode_key = bucket = None
        if self.cuda_graphs is not None:
            if num_prefill_tokens:
                bucket = self.cuda_graphs.pick_prefill(input_ids.size(0), len(seqs))
            else:
                decode_key = self.cuda_graphs.pick_decode(len(seqs))
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, decode_key, bucket)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

