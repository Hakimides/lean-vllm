from __future__ import annotations

import random
from dataclasses import dataclass
from typing import NamedTuple

from lean_vllm import SamplingParams


@dataclass(frozen=True)
class Shape:
    """一种负载形状：长度闭区间加配比权重"""

    name: str
    weight: int
    prompt_lo: int
    prompt_hi: int
    output_lo: int
    output_hi: int


SHAPES: tuple[Shape, ...] = (
    Shape("短问短答", 20, 16, 127, 16, 127),
    Shape("短问长答", 40, 16, 127, 513, 1536),
    Shape("长问短答", 25, 1536, 3840, 16, 127),
    Shape("长问长答", 15, 1536, 2560, 513, 1536),
)


class BenchRequest(NamedTuple):
    shape: str
    prompt_ids: list[int]
    sampling: SamplingParams
    arrival: float        # 相对起跑线的计划到达时刻，单位秒


@dataclass
class WorkloadConfig:
    n_requests: int = 256
    lam: float = 4.0            # 平均到达率，条每秒
    seed: int = 0
    temperature: float = 0.6
    max_model_len: int = 4096


def _allocate(n: int, weights: list[int]) -> list[int]:
    """按权重把 n 条分到各形状，用最大余数法保证总和正好是 n"""
    total = sum(weights)
    exact = [n * w / total for w in weights]
    counts = [int(x) for x in exact]
    rest = n - sum(counts)
    order = sorted(range(len(weights)), key=lambda i: exact[i] - counts[i], reverse=True)
    for i in order[:rest]:
        counts[i] += 1
    return counts


def build_requests(vocab_size: int, cfg: WorkloadConfig | None = None) -> list[BenchRequest]:
    """造一批请求，带计划到达时刻，按到达时刻升序返回"""
    cfg = cfg or WorkloadConfig()

    # 每种形状的输入上界加输出上界都不能超过 max_model_len
    for shape in SHAPES:
        assert shape.prompt_hi + shape.output_hi <= cfg.max_model_len, (
            f"{shape.name} 的长度区间上界 {shape.prompt_hi}+{shape.output_hi} "
            f"超过 max_model_len={cfg.max_model_len}"
        )

    rng = random.Random(cfg.seed)
    rng_arrival = random.Random(cfg.seed + 7919)     # 到达时刻用单独的发生器

    body: list[tuple[str, list[int], SamplingParams]] = []
    for shape, n in zip(SHAPES, _allocate(cfg.n_requests, [s.weight for s in SHAPES])):
        for _ in range(n):
            plen = rng.randint(shape.prompt_lo, shape.prompt_hi)
            olen = rng.randint(shape.output_lo, shape.output_hi)
            body.append((
                shape.name,
                [rng.randrange(vocab_size) for _ in range(plen)],
                # ignore_eos 强制跑满指定长度
                SamplingParams(temperature=cfg.temperature, ignore_eos=True, max_tokens=olen),
            ))

    # 打散，让每次到达落到哪种形状是随机的
    rng.shuffle(body)

    # 泊松到达：间隔服从指数分布，均值 1/lam
    intervals = [rng_arrival.expovariate(cfg.lam) for _ in range(len(body))]

    # 缩放一次，让样本平均到达率等于 lam
    scale = (1.0 / cfg.lam) / (sum(intervals) / len(intervals))
    t = 0.0
    requests: list[BenchRequest] = []
    for (shape_name, prompt_ids, sampling), gap in zip(body, intervals):
        t += gap * scale
        requests.append(BenchRequest(shape_name, prompt_ids, sampling, t))
    return requests


def describe(requests: list[BenchRequest]) -> str:
    """这批请求的规模，写进实验记录用"""
    prompt_lens = [len(r.prompt_ids) for r in requests]
    out_lens = [r.sampling.max_tokens for r in requests]
    span = max(r.arrival for r in requests)
    return (
        f"{len(requests)} requests | "
        f"prompt tokens: min={min(prompt_lens)} max={max(prompt_lens)} sum={sum(prompt_lens)} | "
        f"output tokens: min={min(out_lens)} max={max(out_lens)} sum={sum(out_lens)} | "
        f"到达跨度 {span:.1f}s（实际 {len(requests) / span:.2f} 条/s）"
    )


def shape_table(requests: list[BenchRequest]) -> str:
    """按形状列出条数和长度，核对配比有没有落对"""
    lines = []
    for shape in SHAPES:
        group = [r for r in requests if r.shape == shape.name]
        if not group:
            continue
        pl = [len(r.prompt_ids) for r in group]
        ol = [r.sampling.max_tokens for r in group]
        share = 100 * len(group) / len(requests)
        lines.append(
            f"  {shape.name}  {len(group):>4} 条 ({share:>4.1f}%) | "
            f"输入 {min(pl)}~{max(pl)} | 输出 {min(ol)}~{max(ol)} | "
            f"合计 {sum(pl) + sum(ol)} token"
        )
    return "\n".join(lines)
