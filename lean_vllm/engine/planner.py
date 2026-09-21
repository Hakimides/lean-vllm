from collections import deque

from lean_vllm.config import Config
from lean_vllm.engine.sequence import Request, RequestState
from lean_vllm.engine.page_manager import PageManager


class BatchScheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.page_size = config.kvcache_page_size
        self.page_manager = PageManager(config.num_kvcache_pages, config.kvcache_page_size)
        self.waiting: deque[Request] = deque()
        self.running: deque[Request] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Request):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Request], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.page_table:
                num_cached_pages = self.page_manager.can_acquire(seq)
                if num_cached_pages == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_pages * self.page_size
            else:
                num_tokens = seq.num_tokens - seq.cached_len
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.page_table:
                self.page_manager.acquire(seq, num_cached_pages)
            seq.scheduled_len = min(num_tokens, remaining)
            num_batched_tokens += seq.scheduled_len
            if seq.cached_len + seq.scheduled_len == seq.num_tokens:
                seq.status = RequestState.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.page_manager.can_append(seq):
                if self.running:
                    self.evict_running(self.running.pop())
                else:
                    self.evict_running(seq)
                    break
            else:
                seq.scheduled_len = 1
                seq.is_prefill = False
                self.page_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

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
