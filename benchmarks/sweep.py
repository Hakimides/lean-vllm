from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np

from benchmarks import metrics
from benchmarks.common import REPO_ROOT, RESULTS_DIR, load_run


def _pad(text: str, width: int) -> str:
    shown = sum(2 if ord(c) > 0x2E80 else 1 for c in text)
    return text + " " * max(1, width - shown)


def _run_one(tag: str, seed: int, n: int, lam: float, fp8: bool = False,
             no_prefill_graph: bool = False, kv_fp8: bool = False,
             warmup_gpu_s: float | None = None) -> dict:
    """跑一次 measure.py，返回它写出的 json"""
    cmd = [
        sys.executable, "-m", "benchmarks.measure",
        "--tag", tag, "--seed", str(seed), "--n", str(n), "--lam", str(lam),
    ]
    if fp8:
        cmd.append("--fp8-linear")
    if kv_fp8:
        cmd.append("--kv-fp8")
    if no_prefill_graph:
        cmd.append("--no-prefill-graph")
    if warmup_gpu_s is not None:
        cmd += ["--warmup-gpu-s", str(warmup_gpu_s)]
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return load_run(tag)


def _summarize(prefix: str, per_seed: dict[int, dict], meta: dict) -> dict:
    """跨种子汇总：每个指标给出各种子的值 + 中位数 + 极差。"""
    names = list(next(iter(per_seed.values())).keys())
    table = {}
    for name in names:
        values = [per_seed[s][name] for s in sorted(per_seed)]
        table[name] = {
            "per_seed": values,
            "median": float(np.median(values)),
            "min": min(values),
            "max": max(values),
        }
    return {"prefix": prefix, "seeds": sorted(per_seed), **meta, "metrics": table}


def _warmup_cell(run: dict) -> str:
    """暖机判稳"""
    ramp = ((run.get("warmup") or {}).get("ramp")) or {}
    if not ramp:
        return "-"
    mark = "OK" if ramp.get("stable") else "!! 未稳"
    return f"{ramp['delta_mhz']:+.0f} MHz {mark}"


def _print_summary(summary: dict) -> None:
    m = summary["metrics"]
    print("\n" + "=" * 78)
    print(
        f"汇总 {summary['prefix']}：{len(summary['seeds'])} 个种子 x "
        f"{summary['repeats']} 次 | n={summary['n']} λ={summary['lam']}"
    )
    print("=" * 78)
    print(_pad("指标", 16) + _pad("中位数", 14) + _pad("最小值", 14) + "最大值")
    for name, v in m.items():
        print(
            _pad(name, 16)
            + _pad(f"{v['median']:.2f}", 14)
            + _pad(f"{v['min']:.2f}", 14)
            + f"{v['max']:.2f}"
        )


# 主流程

def cmd_run(args) -> None:
    per_seed: dict[int, dict] = {}
    spread_by_seed: dict[int, dict] = {}
    for seed in args.seeds:
        # 同一个种子连跑完再换下一个：外层种子、内层重复
        runs = [_run_one(f"{args.tag}_s{seed}_r{r}", seed, args.n, args.lam,
                         args.fp8_linear, args.no_prefill_graph, args.kv_fp8,
                         args.warmup_gpu_s)
                for r in range(args.repeats)]
        pooled = [row for run in runs for row in run["per_request"]]
        merged = {**metrics.flat(metrics.summarize_requests(pooled)),
                  **metrics.summarize_runs(runs)}
        per_seed[seed] = merged

        print(f"\n--- 种子 {seed}：逐次运行 ---")
        print(_pad("运行", 8) + _pad("输出吞吐", 12) + _pad("TTFT p50", 12)
              + _pad("ITL p99", 10) + _pad("ITL max", 11) + _pad("抢占", 8) + "暖机判稳")
        for i, run in enumerate(runs):
            o = run["overall"]
            print(
                _pad(f"r{i}", 8) + _pad(f"{o['throughput']['output_tok_s']:.1f}", 12)
                + _pad(f"{o['ttft_ms']['p50']:.0f}", 12) + _pad(f"{o['itl_ms']['p99']:.2f}", 10)
                + _pad(f"{o['itl_ms']['max']:.0f}", 11) + _pad(str(o["n_preemptions"]), 8)
                + _warmup_cell(run)
            )

        spread = metrics.spread(runs)
        spread_by_seed[seed] = spread
        print(f"\n--- 种子 {seed}：重跑差异（极差/中位），这就是噪声底 ---")
        for k, v in spread.items():
            print(f"    {_pad(k, 14)}{v:6.2f}%")

        print(f"\n--- 种子 {seed} 汇总 ---")
        for k, v in merged.items():
            print(f"    {_pad(k, 16)}{v:.2f}")

    summary = _summarize(
        args.tag, per_seed,
        {"n": args.n, "lam": args.lam, "repeats": args.repeats},
    )
    summary["run_spread_by_seed"] = spread_by_seed
    out = RESULTS_DIR / f"{args.tag}_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    _print_summary(summary)
    print(f"\n已写入 {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="批量跑基准并汇总")
    ap.add_argument("--tag", required=True, help="实验名前缀，决定输出文件名")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="种子列表，每个种子算一组")
    ap.add_argument("--repeats", type=int, default=2, help="连跑至少两次")
    ap.add_argument("--n", type=int, default=256, help="每轮请求数")
    ap.add_argument("--lam", type=float, default=1.0, help="轻载用 1.0，过载用 4.0")
    ap.add_argument("--fp8-linear", action="store_true",
                    help="线性层与词表矩阵用 fp8（W8A8）")
    ap.add_argument("--no-prefill-graph", action="store_true",
                    help="关掉含 prefill 的步进图（改前改后对照用）")
    ap.add_argument("--kv-fp8", action="store_true",
                    help="KV 池用 fp8（写入时量化）")
    ap.add_argument("--warmup-gpu-s", type=float, default=None,
                    help="GPU 暖机秒数；不给就沿用 measure.py 默认（45）")
    args = ap.parse_args()
    cmd_run(args)


if __name__ == "__main__":
    main()
