from collections import deque

from lean_vllm.config import Config
from lean_vllm.engine.sequence import Request, RequestState
from lean_vllm.engine.page_manager import PageManager


class BatchScheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_prefill_tokens_per_step = config.max_prefill_tokens_per_step
        # 准入时不许动用的页数。num_kvcache_pages 由 runner 在构造本对象之前算好
        self.reserve_pages = int(config.kv_admission_watermark * config.num_kvcache_pages)
        self.eos = config.eos
        self.page_size = config.kvcache_page_size
        self.page_manager = PageManager(config.num_kvcache_pages, config.kvcache_page_size)
        self.waiting: deque[Request] = deque()
        self.running: deque[Request] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Request):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Request], int, int]:
        """排一步，返回 (本步要跑的序列, 本步的 prefill token 数, 本步的 decode 条数)。

        顺序是刻意的：**先给已经在跑的序列排 decode**（各 1 个 token），
        剩下的预算才给等待队列做 prefill。原来反着来 —— prefill 独占一整步，
        一条长 prompt 进来就把所有在跑的序列整步冻住，那是 ITL 尖峰的来源。
        """
        scheduled_seqs: list[Request] = []
        num_batched_tokens = 0
        num_prefill_tokens = 0
        num_decode = 0
        evicted = False

        # ---- ① 先排 running：decode，各 1 个 token ----
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # 页不够就抢占，腾出地方
            while not self.page_manager.can_append(seq):
                if self.running:
                    self.evict_running(self.running.pop())
                    evicted = True
                else:
                    self.evict_running(seq)
                    evicted = True
                    break
            else:
                seq.scheduled_len = 1
                seq.is_prefill = False
                self.page_manager.may_append(seq)
                scheduled_seqs.append(seq)
                num_batched_tokens += 1
                num_decode += 1
        # 排上的放回队首，保持原来的先后
        self.running.extendleft(reversed(scheduled_seqs))

        # ---- ② 再用剩下的预算排 waiting：prefill ----
        # 本步发生过抢占就跳过 prefill —— 抢占说明页已经不够，这时再塞新请求
        # 只会把在跑的挤得更惨。但一条都没排上时例外，否则这一步就是空的。
        if not evicted or not scheduled_seqs:
            skipped: list[Request] = []
            # 这一步只算了一块、prompt 还没算完的序列。它们必须**从队首取出来**，
            # 否则本轮循环的下一轮又会取到同一条、把同一块重复塞进这一批。
            # 上限一开就会发生：prompt 永远算不完，队首永远不弹出。
            unfinished: list[Request] = []
            while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
                seq = self.waiting[0]
                remaining = self.max_num_batched_tokens - num_batched_tokens
                if remaining == 0:
                    break
                if not seq.page_table:
                    num_cached_pages = self.page_manager.can_acquire(seq)
                    if num_cached_pages == -1:
                        # 页凑不齐：跳过这条、看下一条，而不是 break ——
                        # 后面的短请求还塞得进来，队首的长 prompt 不该把它们全堵死。
                        self.waiting.popleft()
                        skipped.append(seq)
                        continue
                    # 准入余量：收下它之后，空闲页不能低于 reserve_pages。
                    # 余量不够就 break（别再收了）—— 把 KV 吃干只会逼出反复抢占，
                    # 而抢占才是 λ=4.0 上尾部炸掉的根因。
                    if (len(self.page_manager.free_page_ids) - (seq.num_pages - num_cached_pages)
                            < self.reserve_pages):
                        break
                    num_tokens = seq.num_tokens - num_cached_pages * self.page_size
                else:
                    num_tokens = seq.num_tokens - seq.cached_len
                # 每步 prefill 的 token 上限：调小它就把长 prompt 的 prefill 摊到多步，
                # decode 被冻住的时间跟着变短，代价是首 token 变慢
                if self.max_prefill_tokens_per_step:
                    num_tokens = min(num_tokens, self.max_prefill_tokens_per_step)
                # 分块只对本步第一条 prefill 开放
                if remaining < num_tokens and num_prefill_tokens:
                    break
                if not seq.page_table:
                    self.page_manager.acquire(seq, num_cached_pages)
                seq.scheduled_len = min(num_tokens, remaining)
                num_batched_tokens += seq.scheduled_len
                num_prefill_tokens += seq.scheduled_len
                if seq.cached_len + seq.scheduled_len == seq.num_tokens:
                    seq.status = RequestState.RUNNING
                    self.waiting.popleft()
                    self.running.append(seq)
                else:
                    # 只算了一块，先取出来，免得本轮又被取到
                    self.waiting.popleft()
                    unfinished.append(seq)
                scheduled_seqs.append(seq)
            # 本轮跳过的插回队首，保持先来后到
            if skipped:
                self.waiting.extendleft(reversed(skipped))
            # 没算完的那条本来就在队首（比"跳过的"更靠前），所以最后放回
            if unfinished:
                self.waiting.extendleft(reversed(unfinished))

        assert scheduled_seqs
        return scheduled_seqs, num_prefill_tokens, num_decode

    def evict_running(self, seq: Request):
        seq.status = RequestState.WAITING
        seq.is_prefill = True
        self.page_manager.release(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Request], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.page_manager.hash_pages(seq)
            seq.cached_len += seq.scheduled_len
            seq.scheduled_len = 0
            if is_prefill and seq.cached_len < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = RequestState.FINISHED
                self.page_manager.release(seq)
                self.running.remove(seq)
