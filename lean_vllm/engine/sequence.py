from copy import copy
from enum import Enum, auto
from itertools import count

from lean_vllm.sampling_params import SamplingParams


class RequestState(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Request:
    page_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Request.counter)
        self.status = RequestState.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.cached_len = 0
        self.scheduled_len = 0
        self.is_prefill = True
        self.page_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == RequestState.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_pages(self):
        return (self.num_tokens + self.page_size - 1) // self.page_size

    @property
    def last_page_num_tokens(self):
        return self.num_tokens - (self.num_pages - 1) * self.page_size

    def page(self, i):
        assert 0 <= i < self.num_pages
        return self.token_ids[i*self.page_size: (i+1)*self.page_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.cached_len, self.scheduled_len, self.page_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.cached_len, self.scheduled_len, self.page_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
