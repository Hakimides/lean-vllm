"""逐请求记录的汇总，唯一实现。

measure.py 落盘时用它，sweep.py 跨种子汇总时也用它，避免同一批原始数据被两处各算一遍。
"""
from __future__ import annotations

import numpy as np

# 规范字段到汇总 json 标签的对应表（sweep 的 summary.json 沿用这套标签）
LABELS = {
    ("ttft_ms", "p50"): "TTFT p50",
    ("ttft_ms", "p99"): "TTFT p99",
    ("queue_ms", "p50"): "排队 p50",
    ("prefill_ms", "p50"): "prefill p50",
    ("e2e_ms", "p99"): "E2E p99",
    ("itl_ms", "p50"): "ITL p50",
    ("itl_ms", "p99"): "ITL p99",
    ("itl_ms", "max"): "ITL max",
    ("arrival_lag_ms", "p99"): "到达偏差 p99",
}

# 整轮取中位数的指标：标签到 overall 取数函数
RUN_ITEMS = {
    "输出吞吐": lambda o: o["throughput"]["output_tok_s"],
    "总吞吐": lambda o: o["throughput"]["total_tok_s"],
    "抢占次数": lambda o: float(o["n_preemptions"]),
    "重算token": lambda o: float(o["recompute_tokens"]),
    "最大并发": lambda o: float(o["concurrency"]["max_running"]),
    "队列最深": lambda o: float(o["concurrency"]["max_waiting"]),
    "实际耗时 s": lambda o: o["wall_s"],
    "decode占比%": lambda o: 100 * o["phase_ms"]["decode_total"] / 1000 / o["wall_s"],
    "空闲占比%": lambda o: 100 * o["idle_s"] / o["wall_s"],
}


def pct(values: list[float], q: float) -> float:
    """分位数，没数据返回 nan"""
    return float(np.percentile(values, q)) if values else float("nan")


def summarize_requests(reqs: list[dict]) -> dict:
    """逐请求记录汇总成分位数"""
    def col(key):
        return [r[key] for r in reqs if r[key] == r[key]]      # 顺手滤掉 nan

    ttft, queue, prefill = col("ttft_ms"), col("queue_ms"), col("prefill_ms")
    e2e, lag = col("e2e_ms"), col("arrival_lag_ms")
    itls = [x for r in reqs for x in r["itls"]]
    return {
        "n": len(reqs),
        "output_tokens": sum(r["n_out"] for r in reqs),
        "ttft_ms": {
            "p50": pct(ttft, 50), "p99": pct(ttft, 99),
            "mean": float(np.mean(ttft)) if ttft else float("nan"),
            "max": max(ttft) if ttft else float("nan"),
        },
        "queue_ms": {"p50": pct(queue, 50), "p99": pct(queue, 99)},
        "prefill_ms": {"p50": pct(prefill, 50), "p99": pct(prefill, 99)},
        "e2e_ms": {"p50": pct(e2e, 50), "p99": pct(e2e, 99)},
        "itl_ms": {
            "p50": pct(itls, 50), "p99": pct(itls, 99),
            "mean": float(np.mean(itls)) if itls else float("nan"),
            "max": max(itls) if itls else float("nan"),
        },
        "arrival_lag_ms": {
            "p50": pct(lag, 50), "p99": pct(lag, 99),
            "max": max(lag) if lag else float("nan"),
        },
    }


def flat(summary: dict) -> dict:
    """规范嵌套转成标签值表"""
    return {label: summary[sec][stat] for (sec, stat), label in LABELS.items()}


def summarize_runs(runs: list[dict]) -> dict:
    """只能整轮取中位数的指标（吞吐、抢占这类）"""
    def med(fn):
        return float(np.median([fn(r["overall"]) for r in runs]))

    return {label: med(fn) for label, fn in RUN_ITEMS.items()}


# 噪声底要比的项：标签到 overall 取数函数（集合固定）
SPREAD_ITEMS = {
    "输出吞吐": lambda o: o["throughput"]["output_tok_s"],
    "TTFT p50": lambda o: o["ttft_ms"]["p50"],
    "排队 p50": lambda o: o["queue_ms"]["p50"],
    "prefill p50": lambda o: o["prefill_ms"]["p50"],
    "E2E p99": lambda o: o["e2e_ms"]["p99"],
    "ITL p50": lambda o: o["itl_ms"]["p50"],
    "ITL p99": lambda o: o["itl_ms"]["p99"],
    "ITL max": lambda o: o["itl_ms"]["max"],
    "抢占次数": lambda o: o["n_preemptions"],
    "重算token": lambda o: o["recompute_tokens"],
    "最大并发": lambda o: o["concurrency"]["max_running"],
    "队列最深": lambda o: o["concurrency"]["max_waiting"],
    "实际耗时 s": lambda o: o["wall_s"],
}


def spread(runs: list[dict]) -> dict:
    """同一配置重跑之间的相对差，这就是噪声底"""
    out = {}
    for label, fn in SPREAD_ITEMS.items():
        v = [fn(r["overall"]) for r in runs]
        mid = float(np.median(v)) if v else float("nan")
        out[label] = ((max(v) - min(v)) / mid * 100) if mid else float("nan")
    return out
