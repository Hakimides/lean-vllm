"""生成 fp8 静态量化用的激活校准表，写成 fp8_scales.json"""
from __future__ import annotations

import argparse
import json
import os

from transformers import AutoTokenizer

from lean_vllm import LLM, SamplingParams
from lean_vllm.layers.attention import Attention

try:
    from local_settings import MODEL_PATH
except ImportError:
    MODEL_PATH = os.environ.get("MODEL_PATH", "")

# 校准文本（古文长诗）
CALIB_TEXT = """春江潮水连海平，海上明月共潮生。滟滟随波千万里，何处春江无月明。
江流宛转绕芳甸，月照花林皆似霰。空里流霜不觉飞，汀上白沙看不见。
江天一色无纤尘，皎皎空中孤月轮。江畔何人初见月，江月何年初照人。
人生代代无穷已，江月年年望相似。不知江月待何人，但见长江送流水。
白云一片去悠悠，青枫浦上不胜愁。谁家今夜扁舟子，何处相思明月楼。
可怜楼上月徘徊，应照离人妆镜台。玉户帘中卷不去，捣衣砧上拂还来。
此时相望不相闻，愿逐月华流照君。鸿雁长飞光不度，鱼龙潜跃水成文。"""


def collect_activation_amax(llm: LLM = None, model=None, text: str = None):
    """跑一遍文本，返回 ({模块名: 输入的最大绝对值}, {k/v 的最大绝对值})"""
    amax: dict[str, float] = {}
    kv_amax = {"k": 0.0, "v": 0.0}
    hooks = []
    for name, module in model.named_modules():
        if hasattr(module, "quantize_weight"):

            def make(nm):
                def hook(mod, inputs, output):
                    v = float(inputs[0].detach().abs().amax())
                    if v > amax.get(nm, 0.0):
                        amax[nm] = v
                return hook

            hooks.append(module.register_forward_hook(make(name)))
        elif isinstance(module, Attention):

            def kv_hook(mod, args):
                # Attention.forward(q, k, v)：这里的 k 已过旋转
                _, k, v = args
                kv_amax["k"] = max(kv_amax["k"], float(k.detach().abs().amax()))
                kv_amax["v"] = max(kv_amax["v"], float(v.detach().abs().amax()))

            hooks.append(module.register_forward_pre_hook(kv_hook))
    try:
        llm.generate([text], SamplingParams(temperature=0.6, max_tokens=1), use_tqdm=False)
    finally:
        for h in hooks:
            h.remove()
    return amax, kv_amax


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 fp8 静态量化的激活校准表")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--out", default=None, help="输出路径，默认写进仓库根目录 fp8_scales.json")
    ap.add_argument("--text", default=None, help="自定义校准文本文件（默认用内置的一段中文）")
    args = ap.parse_args()

    if not args.model:
        ap.error("没有模型路径：请写 lean-vllm/local_settings.py 或设 MODEL_PATH")

    text = CALIB_TEXT
    if args.text:
        with open(args.text, encoding="utf-8") as f:
            text = f.read()

    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fp8_scales.json")

    n_tok = len(AutoTokenizer.from_pretrained(args.model, use_fast=True).encode(text))
    print(f"校准文本 {n_tok} token")

    # 用引擎自己的模块名
    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1)
    amax, kv_amax = collect_activation_amax(llm=llm, model=llm.model_runner.model, text=text)

    if not amax:
        raise RuntimeError("一个激活 amax 都没收到，检查 hook 挂对没有")

    vals = sorted(amax.values())
    print(f"收到 {len(amax)} 层：最小 {vals[0]:.3g} / 中位 {vals[len(vals)//2]:.3g} / 最大 {vals[-1]:.3g}")
    print(f"KV amax：k {kv_amax['k']:.3g} / v {kv_amax['v']:.3g}")

    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "n_tokens": n_tok,
                   "activation_amax": amax, "kv_amax": kv_amax},
                  f, ensure_ascii=False, indent=1)
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
