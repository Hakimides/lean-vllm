import atexit
from dataclasses import fields, replace
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from lean_vllm.config import Config
from lean_vllm.sampling_params import SamplingParams
from lean_vllm.engine.sequence import Request
from lean_vllm.engine.planner import BatchScheduler
from lean_vllm.engine.runner import EngineRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Request.page_size = config.kvcache_page_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=EngineRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = EngineRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = BatchScheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        # 长度校验：一条序列的 prompt + 输出不能超过 max_model_len。
        #
        # 超了会崩，而且崩得很远：capture_cudagraph() 按 max_model_len 预留了
        # decode 用的 page table，列数写死为 ceil(max_model_len / page_size)。
        # 序列一旦长过 max_model_len，page table 就要更多列，往那张表里拷的时候
        # 形状对不上，直接抛异常。
        #
        # 这里的处理办法和别家引擎一样：把请求的输出长度截到剩余预算，而不是放它跑。
        max_model_len = self.model_runner.config.max_model_len
        budget = max_model_len - len(prompt)
        if budget <= 0:
            raise ValueError(
                f"prompt 已经占满 max_model_len={max_model_len}（实际 {len(prompt)}），没有位置生成输出"
            )
        if sampling_params.max_tokens > budget:
            print(f"[warn] max_tokens={sampling_params.max_tokens} 超出剩余预算 {budget}，已截断")
            sampling_params = replace(sampling_params, max_tokens=budget)
        seq = Request(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """排一步、跑一步。返回 (完成的请求, 本步 prefill token 数, 本步 decode 条数)。

        一步里可以同时有 prefill 和 decode，所以不再是一个正负号能表达的
        （原来用符号区分两类步，混合批之后这个约定失效）。
        """
        seqs, num_prefill_tokens, num_decode = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs)
        # postprocess 里那个判断自带逐条分流：decode 的 cached_len 已经等于 num_tokens，
        # 只有还没算完的分块 prefill 才会被它跳过。所以这里传"本步有没有 prefill"即可。
        self.scheduler.postprocess(seqs, token_ids, num_prefill_tokens > 0)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_prefill_tokens, num_decode

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_prefill_tokens, num_decode = self.step()
            elapsed = perf_counter() - t
            # 一步里两类可能同时有，所以两边分别更新
            if num_prefill_tokens:
                prefill_throughput = num_prefill_tokens / elapsed
            if num_decode:
                decode_throughput = num_decode / elapsed
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
