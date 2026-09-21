import os

from lean_vllm import LLM, SamplingParams
from transformers import AutoTokenizer

try:
    from local_settings import MODEL_PATH
except ImportError:
    MODEL_PATH = os.environ.get("MODEL_PATH", "")


def main():
    path = MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "用一句话介绍杭州。",
        "把「今天天气很好」翻译成英文。",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
