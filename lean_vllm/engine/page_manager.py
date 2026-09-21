from collections import deque
import xxhash
import numpy as np

from lean_vllm.engine.sequence import Request


class Page:

    def __init__(self, page_id):
        self.page_id = page_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class PageManager:

    def __init__(self, num_pages: int, page_size: int):
        self.page_size = page_size
        self.pages: list[Page] = [Page(i) for i in range(num_pages)]
        self.hash_to_page_id: dict[int, int] = dict()
        self.free_page_ids: deque[int] = deque(range(num_pages))
        self.used_page_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _acquire_page(self) -> int:
        page_id = self.free_page_ids.popleft()
        page = self.pages[page_id]
        assert page.ref_count == 0
        if page.hash != -1 and self.hash_to_page_id.get(page.hash) == page_id:
            del self.hash_to_page_id[page.hash]
        page.reset()
        self.used_page_ids.add(page_id)
        return page_id

    def _release_page(self, page_id: int):
        assert self.pages[page_id].ref_count == 0
        self.used_page_ids.remove(page_id)
        self.free_page_ids.append(page_id)

    def can_acquire(self, seq: Request) -> int:
        h = -1
        num_cached_pages = 0
        num_new_pages = seq.num_pages
        for i in range(seq.num_pages - 1):
            token_ids = seq.page(i)
            h = self.compute_hash(token_ids, h)
            page_id = self.hash_to_page_id.get(h, -1)
            if page_id == -1 or self.pages[page_id].token_ids != token_ids:
                break
            num_cached_pages += 1
            if page_id in self.used_page_ids:
                num_new_pages -= 1
        if len(self.free_page_ids) < num_new_pages:
            return -1
        return num_cached_pages

    def acquire(self, seq: Request, num_cached_pages: int):
        assert not seq.page_table
        h = -1
        for i in range(num_cached_pages):
            token_ids = seq.page(i)
            h = self.compute_hash(token_ids, h)
            page_id = self.hash_to_page_id[h]
            page = self.pages[page_id]
            if page_id in self.used_page_ids:
                page.ref_count += 1
            else:
                page.ref_count = 1
                self.free_page_ids.remove(page_id)
                self.used_page_ids.add(page_id)
            seq.page_table.append(page_id)
        for i in range(num_cached_pages, seq.num_pages):
            seq.page_table.append(self._acquire_page())
        seq.cached_len = num_cached_pages * self.page_size

    def release(self, seq: Request):
        for page_id in reversed(seq.page_table):
            page = self.pages[page_id]
            page.ref_count -= 1
            if page.ref_count == 0:
                self._release_page(page_id)
        seq.cached_len = 0
        seq.page_table.clear()

    def can_append(self, seq: Request) -> bool:
        return len(self.free_page_ids) >= (len(seq) % self.page_size == 1)

    def may_append(self, seq: Request):
        if len(seq) % self.page_size == 1:
            seq.page_table.append(self._acquire_page())

    def hash_pages(self, seq: Request):
        start = seq.cached_len // self.page_size
        end = (seq.cached_len + seq.scheduled_len) // self.page_size
        if start == end: return
        h = self.pages[seq.page_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            page = self.pages[seq.page_table[i]]
            token_ids = seq.page(i)
            h = self.compute_hash(token_ids, h)
            page.update(h, token_ids)
            self.hash_to_page_id[h] = page.page_id
