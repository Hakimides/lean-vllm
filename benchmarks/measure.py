from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from lean_vllm import LLM

from benchmarks import metrics, workloads
from benchmarks.common import RESULTS_DIR, resolve_model_path


def _git_provenance() -> dict:
    """这份数据对应的 git 提交；dirty 表示当时有未提交改动"""
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return {"commit": commit or None, "dirty": bool(dirty)}
    except Exception:
        return {"commit": None, "dirty": None}


def _gpu_state() -> dict | None:
    """取一次 GPU 状态，取不到返回 None"""
    query = ("clocks.sm,clocks.max.sm,temperature.gpu,power.draw,"
             "clocks_throttle_reasons.active")
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        text = out.stdout.strip()
        if out.returncode != 0 or not text:
            return None

        def num(field: str):
            try:
                return float(field.split()[0])
            except (ValueError, IndexError):
                return None

        parts = [p.strip() for p in text.split(",")]
        return {
            "raw": text,
            "sm_mhz": num(parts[0]) if len(parts) > 0 else None,
            "sm_max_mhz": num(parts[1]) if len(parts) > 1 else None,
            "temp_c": num(parts[2]) if len(parts) > 2 else None,
            "power_w": num(parts[3]) if len(parts) > 3 else None,
            "throttle": parts[4] if len(parts) > 4 else None,
        }
    except Exception:
        return None


@dataclass
class Rec:
    """一条请求的观测记录，时刻都是 perf_counter 的绝对时间"""

    shape: str
    n_out: int                          # 期望生成多少 token
    planned: float                      # 计划到达时刻
    t_add: float = 0.0                  # 实际进引擎的时刻
    t_accept: float | None = None       # 首次被受理拿到 KV page 的时刻
    token_times: list[float] = field(default_factory=list)
    had_pages: bool = False             # 用来识别被抢占
    n_preempt: int = 0                  # 这条被抢占了几次


def _drain(llm: LLM) -> None:
    """把引擎里剩下的请求跑完"""
    while not llm.is_finished():
        llm.step()


def _clock_ramp(samples: list[list], window_s: float = 30.0,
                tol_mhz: float = 60.0) -> dict | None:
    """频率还在不在爬升，比最后 window_s 秒前后两半的中位数"""
    if not samples:
        return None
    t_end = samples[-1][0]
    win = [s for s in samples if s[0] >= t_end - window_s and s[1] is not None]
    if len(win) < 6:
        return None
    half = len(win) // 2
    a = float(statistics.median([s[1] for s in win[:half]]))
    b = float(statistics.median([s[1] for s in win[half:]]))
    return {
        "first_half_median_mhz": round(a, 1),
        "last_half_median_mhz": round(b, 1),
        "delta_mhz": round(b - a, 1),
        "stable": abs(b - a) <= tol_mhz,
        "window_s": window_s,
        "n_samples": len(win),
    }


def _warmup_gpu(llm: LLM, vocab_size: int, seconds: float, target_inflight: int,
                seed: int, max_model_len: int, tol_mhz: float,
                period_s: float = 2.0) -> dict:
    """GPU 暖机：用饱和负载顶频率，期间采频率"""
    pool = workloads.build_requests(
        vocab_size,
        workloads.WorkloadConfig(n_requests=512, lam=4.0, seed=seed,
                                 max_model_len=max_model_len),
    )
    samples: list[list] = []              # [相对秒, SM频率MHz, 温度, 功耗]
    idx = 0
    t0 = time.perf_counter()
    next_sample = 0.0
    while True:
        elapsed = time.perf_counter() - t0
        inflight = len(llm.scheduler.running) + len(llm.scheduler.waiting)
        # 补足在跑条数
        while inflight < target_inflight and idx < len(pool):
            req = pool[idx]
            llm.add_request(req.prompt_ids, req.sampling)
            idx += 1
            inflight += 1
        # 引擎空着时不能 step()
        if inflight == 0 and idx >= len(pool):
            break
        if elapsed >= seconds:
            break
        if elapsed >= next_sample:
            st = _gpu_state()
            if st and st["sm_mhz"] is not None:
                samples.append([round(elapsed, 2), st["sm_mhz"], st["temp_c"], st["power_w"]])
            next_sample = elapsed + period_s
        llm.step()

    _drain(llm)
    ramp = _clock_ramp(samples, tol_mhz=tol_mhz)
    clocks = [s[1] for s in samples if s[1] is not None]
    return {
        "load_s": round(seconds, 1),
        "inflight": target_inflight,
        "seed": seed,
        "n_samples": len(samples),
        "clock_min_mhz": min(clocks) if clocks else None,
        "clock_max_mhz": max(clocks) if clocks else None,
        "ramp": ramp,
        "trace": samples,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="按泊松到达在引擎上跑一批请求并测延迟")
    ap.add_argument("--tag", default="run", help="实验名，决定输出文件名")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=256, help="请求总数")
    ap.add_argument("--lam", type=float, default=4.0, help="平均到达率，条每秒")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--warmup-gpu-s", type=float, default=45.0,
                    help="GPU 暖机用持续饱和负载顶频率的秒数，0 表示不做")
    ap.add_argument("--warmup-inflight", type=int, default=32,
                    help="预热期间保持多少条请求在跑")
    ap.add_argument("--warmup-clock-tol", type=float, default=60.0,
                    help="判频率不再爬升的阈值 MHz")
    ap.add_argument("--warmup-seed", type=int, default=-1,
                    help="预热负载的种子，-1 表示用 seed+1000")
    ap.add_argument("--gpu-sample-s", type=float, default=0.0,
                    help="运行中采 GPU 状态的间隔秒数，0 表示不采")
    ap.add_argument("--kv-watermark", type=float, default=None,
                    help="准入余量：留出的空闲 KV 页比例。不给就沿用引擎默认（0.1 = 10%）")
    ap.add_argument("--prefill-cap", type=int, default=None,
                    help="一步里最多给 prefill 多少 token。不给就沿用引擎默认（0 = 不限）")
    ap.add_argument("--fp8-linear", action="store_true",
                    help="线性层与词表矩阵用 fp8（W8A8）；不给就走 bf16")
    ap.add_argument("--fp8-scales", default=None,
                    help="fp8 校准表路径；不给就用引擎默认（仓库根目录 fp8_scales.json）")
    ap.add_argument("--kv-fp8", action="store_true",
                    help="KV 池用 fp8（写入时量化）；容量翻倍、attention 读字节减半")
    ap.add_argument("--no-prefill-graph", action="store_true",
                    help="关掉含 prefill 的步进图（改前改后对照用）")
    ap.add_argument("--model", default=resolve_model_path())
    ap.add_argument("--out", default=None, help="输出路径，默认写进 benchmarks/results/")
    args = ap.parse_args()

    if not args.model:
        ap.error("没有模型路径：请写 lean-vllm/local_settings.py 或设 MODEL_PATH 环境变量")

    # 只在显式指定时传给引擎
    engine_kwargs = {}
    if args.kv_watermark is not None:
        engine_kwargs["kv_admission_watermark"] = args.kv_watermark
    if args.prefill_cap is not None:
        engine_kwargs["max_prefill_tokens_per_step"] = args.prefill_cap
    if args.fp8_linear:
        engine_kwargs["fp8_linear"] = True
    if args.fp8_scales is not None:
        engine_kwargs["fp8_scales_path"] = args.fp8_scales
    if args.kv_fp8:
        engine_kwargs["kv_fp8"] = True
    if args.no_prefill_graph:
        engine_kwargs["prefill_graph"] = False
    llm = LLM(args.model, enforce_eager=False, max_model_len=args.max_model_len,
              **engine_kwargs)
    # 随机 token id 的上界与长度预算取自引擎配置
    cfg = workloads.WorkloadConfig(
        n_requests=args.n,
        lam=args.lam,
        seed=args.seed,
        max_model_len=llm.model_runner.config.max_model_len,
    )
    requests = workloads.build_requests(llm.model_runner.config.hf_config.vocab_size, cfg)
    print(f"负载：{workloads.describe(requests)}")
    print(workloads.shape_table(requests))

    warmup = None
    if args.warmup_gpu_s > 0:
        warmup_seed = args.seed + 1000 if args.warmup_seed < 0 else args.warmup_seed
        print(f"\nGPU 暖机：持续饱和负载 {args.warmup_gpu_s:.0f}s"
              f"（保持 {args.warmup_inflight} 条在跑）...", flush=True)
        warmup = _warmup_gpu(
            llm, llm.model_runner.config.hf_config.vocab_size,
            args.warmup_gpu_s, args.warmup_inflight, warmup_seed,
            llm.model_runner.config.max_model_len, args.warmup_clock_tol,
        )
        ramp = warmup["ramp"]
        if ramp is None:
            print(f"  预热期只采到 {warmup['n_samples']} 个频率样本，判不了稳不稳")
        else:
            print(f"  频率 前半 {ramp['first_half_median_mhz']:.0f} -> "
                  f"后半 {ramp['last_half_median_mhz']:.0f} MHz"
                  f"（{ramp['delta_mhz']:+.0f}）"
                  f" | 区间 {warmup['clock_min_mhz']:.0f}~{warmup['clock_max_mhz']:.0f}")
            if ramp["stable"]:
                print("  频率已不再爬升，可以开测")
            else:
                print(f"  频率还在爬升（|{ramp['delta_mhz']:+.0f}| > "
                      f"{args.warmup_clock_tol:.0f} MHz），把 --warmup-gpu-s 调大再跑")
    if torch.cuda.is_available():
        # 放在暖机之后
        torch.cuda.reset_peak_memory_stats()

    # 主循环：按到达时刻表提交，自己驱动 step()
    records: list[Rec] = []
    live: list[tuple[object, Rec]] = []      # (引擎里的 seq 对象, 记录)
    # 每步一行：[prefill token 数, decode 条数, 耗时, 在跑, 在等, 驻留 token]
    # 驻留 token = 在跑序列的 KV 长度之和
    step_log: list[list] = []
    n_preempt = 0
    n_prefill_steps = n_decode_steps = n_mixed_steps = 0
    prefill_tokens = 0
    phase_ms = [0.0, 0.0]                    # 纯 prefill / 纯 decode 步各自累计的毫秒
    mixed_ms = 0.0                           # 混合步累计的毫秒（不摊到上面两个里，免得重复计）
    idle_s = 0.0                             # 引擎空转等到达的累计秒数
    nxt = 0                                  # 下一条待提交的请求下标

    gpu_before = _gpu_state()
    gpu_trace: list[list] = []               # [相对秒, SM频率MHz, 温度, 功耗]
    next_gpu_sample = 0.0
    t0 = time.perf_counter()
    while nxt < len(requests) or not llm.is_finished():
        elapsed = time.perf_counter() - t0

        # 1 把已经到点的请求提交进引擎
        while nxt < len(requests) and requests[nxt].arrival <= elapsed:
            req = requests[nxt]
            rec = Rec(shape=req.shape, n_out=req.sampling.max_tokens, planned=req.arrival)
            rec.t_add = time.perf_counter()
            llm.add_request(req.prompt_ids, req.sampling)
            # 只能从 waiting 队尾取刚加进去的那条
            live.append((llm.scheduler.waiting[-1], rec))
            records.append(rec)
            nxt += 1

        # 2 引擎空着就等到下一条到达，不要空转 step
        if llm.is_finished():
            t_idle = time.perf_counter()
            time.sleep(max(0.0, (t0 + requests[nxt].arrival) - t_idle))
            idle_s += time.perf_counter() - t_idle
            continue

        # 3 采样 GPU 状态（在 step 计时之外）
        if args.gpu_sample_s > 0 and elapsed >= next_gpu_sample:
            st = _gpu_state()
            if st:
                gpu_trace.append([round(elapsed, 2), st["sm_mhz"], st["temp_c"], st["power_w"]])
            next_gpu_sample = elapsed + args.gpu_sample_s

        # 4 跑一步
        s = time.perf_counter()
        _, num_prefill_tokens, num_decode = llm.step()
        e = time.perf_counter()
        step_ms = (e - s) * 1000
        # 第 6 列：在跑序列的 KV 长度之和
        resident_tokens = sum(seq.cached_len for seq in llm.scheduler.running)
        step_log.append([
            num_prefill_tokens,
            num_decode,
            round(step_ms, 3),
            len(llm.scheduler.running),
            len(llm.scheduler.waiting),
            resident_tokens,
        ])
        prefill_tokens += num_prefill_tokens
        if num_prefill_tokens and num_decode:
            n_mixed_steps += 1
            mixed_ms += step_ms
        elif num_prefill_tokens:
            n_prefill_steps += 1
            phase_ms[0] += step_ms
        else:
            n_decode_steps += 1
            phase_ms[1] += step_ms

        # 5 扫还活着的请求：谁吐了 token、谁被受理、谁被抢占
        still_live = []
        for seq, rec in live:
            # 记步开始时刻（受理发生在 schedule() 里）
            if rec.t_accept is None and (seq.page_table or rec.token_times):
                rec.t_accept = s
            n = seq.num_completion_tokens
            if n > len(rec.token_times):
                rec.token_times.extend([e] * (n - len(rec.token_times)))
            if seq.is_finished:
                continue
            # page_table 被清空即为被抢占，只算先前拿到过 page 的
            if seq.page_table:
                rec.had_pages = True
            elif rec.had_pages:
                n_preempt += 1
                rec.n_preempt += 1
                rec.had_pages = False
            still_live.append((seq, rec))
        live = still_live
    t_end = time.perf_counter()
    gpu_after = _gpu_state()

    wall = t_end - t0

    # 汇总
    per_request: list[dict] = []
    for rec in records:
        if not rec.token_times:
            continue
        itls = [round((b - a) * 1000, 3) for a, b in zip(rec.token_times, rec.token_times[1:])]
        row = {
            "shape": rec.shape,
            "planned_s": round(rec.planned, 4),
            "arrival_lag_ms": round((rec.t_add - t0 - rec.planned) * 1000, 3),
            "n_out": rec.n_out,
            "n_preempt": rec.n_preempt,
            "ttft_ms": round((rec.token_times[0] - rec.t_add) * 1000, 3),
            "e2e_ms": round((rec.token_times[-1] - rec.t_add) * 1000, 3),
            "itls": itls,
        }
        row["queue_ms"] = (
            round((rec.t_accept - rec.t_add) * 1000, 3) if rec.t_accept is not None else float("nan")
        )
        row["prefill_ms"] = (
            round((rec.token_times[0] - rec.t_accept) * 1000, 3) if rec.t_accept is not None else float("nan")
        )
        per_request.append(row)

    total_out = sum(r["n_out"] for r in per_request)
    prompt_tokens = sum(len(req.prompt_ids) for req in requests)
    overall = metrics.summarize_requests(per_request)
    # 吞吐用整轮口径
    overall["throughput"] = {
        "output_tok_s": total_out / wall,
        "total_tok_s": (total_out + prompt_tokens) / wall,
        "prefill_tok_s": prefill_tokens / wall,
    }
    overall["wall_s"] = wall
    overall["arrival_span_s"] = max(req.arrival for req in requests)
    overall["achieved_arrival_rate"] = len(requests) / overall["arrival_span_s"]
    overall["n_steps"] = len(step_log)
    overall["n_prefill_steps"] = n_prefill_steps
    overall["n_decode_steps"] = n_decode_steps
    overall["n_mixed_steps"] = n_mixed_steps
    overall["n_preemptions"] = n_preempt

    # 四段加起来约等于实际耗时减去空闲等待
    overall["phase_ms"] = {
        "prefill_total": phase_ms[0],
        "decode_total": phase_ms[1],
        "mixed_total": mixed_ms,
    }
    overall["idle_s"] = idle_s

    # 批大小与队列深度
    runnings = [row[3] for row in step_log]
    waitings = [row[4] for row in step_log]
    overall["concurrency"] = {
        "max_running": max(runnings) if runnings else 0,
        "mean_running": round(float(np.mean(runnings)), 2) if runnings else 0.0,
        "max_waiting": max(waitings) if waitings else 0,
    }

    # 重算量 = prefill 处理的 token 减去原始 prompt 总量
    overall["recompute_tokens"] = prefill_tokens - prompt_tokens

    memory = None
    if torch.cuda.is_available():
        memory = {
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }
        overall["memory"] = memory

    # KV 容量与待处理 token 总量
    page_size = llm.scheduler.page_size
    num_pages = llm.model_runner.config.num_kvcache_pages
    kv_capacity = num_pages * page_size
    overall["engine"] = {
        "max_num_batched_tokens": llm.scheduler.max_num_batched_tokens,
        "max_num_seqs": llm.scheduler.max_num_seqs,
        "max_model_len": llm.model_runner.config.max_model_len,
        "page_size": page_size,
        "num_kvcache_pages": num_pages,
        "kv_capacity_tokens": kv_capacity,
    }

    # 打印
    o = overall
    print(
        f"\n实际耗时 {wall:.2f}s | 到达跨度 {o['arrival_span_s']:.1f}s "
        f"（{o['achieved_arrival_rate']:.2f} 条/s）| {len(step_log)} 步"
        f"（prefill {n_prefill_steps} / 混合 {n_mixed_steps} / decode {n_decode_steps}）"
        f"| 抢占 {o['n_preemptions']} 次"
    )
    print(
        f"吞吐：输出 {o['throughput']['output_tok_s']:.0f} tok/s"
        f" | 含 prompt {o['throughput']['total_tok_s']:.0f} tok/s"
    )
    print(
        f"耗时去向：prefill {phase_ms[0] / 1000:.1f}s + decode {phase_ms[1] / 1000:.1f}s"
        f" + 混合 {mixed_ms / 1000:.1f}s + 空闲 {idle_s:.1f}s = {wall:.1f}s"
    )
    print(
        f"并发度：max {o['concurrency']['max_running']}"
        f" / 均值 {o['concurrency']['mean_running']}"
        f" | 队列最深 {o['concurrency']['max_waiting']}"
        f" | 重算 {o['recompute_tokens']} token"
        f"（prefill 总量的 {100 * o['recompute_tokens'] / max(1, prompt_tokens):.1f}%）"
    )
    print(
        f"延迟 ms：TTFT p50 {o['ttft_ms']['p50']:.0f} / p99 {o['ttft_ms']['p99']:.0f}"
        f" | 排队 p50 {o['queue_ms']['p50']:.2f}"
        f" | prefill p50 {o['prefill_ms']['p50']:.0f}"
        f" | E2E p50 {o['e2e_ms']['p50']:.0f} / p99 {o['e2e_ms']['p99']:.0f}"
    )
    print(
        f"ITL ms：p50 {o['itl_ms']['p50']:.2f} / p99 {o['itl_ms']['p99']:.2f}"
        f" / max {o['itl_ms']['max']:.0f}"
    )
    lag = o["arrival_lag_ms"]
    print(f"到达偏差：p50 {lag['p50']:.2f} / p99 {lag['p99']:.2f} ms")
    if memory:
        print(f"显存峰值：已分配 {memory['peak_allocated_mib']:.0f} MiB")
    need = total_out + prompt_tokens
    print(f"KV 容量 {kv_capacity} tokens | 负载总量 {need} tokens（{need / kv_capacity:.1f} 倍）")
    if gpu_before or gpu_after:
        print(f"GPU 跑前 {gpu_before} | 跑后 {gpu_after}")

    # 落盘
    payload = {
        "tag": args.tag,
        "provenance": _git_provenance(),
        "model": args.model,
        "workload_config": asdict(cfg),
        "workload_desc": workloads.describe(requests),
        "gpu_sample": {"period_s": args.gpu_sample_s},
        # 记引擎里实际生效的值
        "sched_knobs": {
            "kv_admission_watermark": llm.model_runner.config.kv_admission_watermark,
            "reserve_pages": llm.scheduler.reserve_pages,
            "max_prefill_tokens_per_step": llm.scheduler.max_prefill_tokens_per_step,
        },
        # 这一轮开了哪些开关
        "engine_flags": {
            "fp8_linear": llm.model_runner.config.fp8_linear,
            "kv_fp8": llm.model_runner.config.kv_fp8,
            "prefill_graph": llm.model_runner.config.prefill_graph,
        },
        # 暖机报告含预热期的频率轨迹
        "warmup": warmup,
        "env": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
        },
        "overall": overall,
        "per_request": per_request,
        "step_log": step_log,
        "gpu_trace": gpu_trace,
    }
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"{args.tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入 {out_path}")


if __name__ == "__main__":
    main()
