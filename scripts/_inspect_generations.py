"""Look at raw base-model output to diagnose format and stopping behaviour."""

import os

import requests

from llm_rl.data import Example, load_examples

URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8001")
MODEL = "Qwen/Qwen3.5-0.8B-Base"

gsm = load_examples("openai/gsm8k", split="test", config="main", max_samples=3, deduplicate=False)
ex = gsm[0]
print("=== PROMPT ===")
print(ex.prompt)
print(f"\n=== GOLD: {ex.answer} ===\n")

for max_tokens in (2048,):
    payload = {
        "model": MODEL,
        "prompt": ex.prompt,
        "n": 2,
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "seed": 0,
    }
    body = requests.post(f"{URL}/v1/completions", json=payload, timeout=1200).json()
    for i, choice in enumerate(body["choices"]):
        text = choice["text"]
        print(f"--- sample {i} (max_tokens={max_tokens}) finish={choice['finish_reason']} chars={len(text)} ---")
        print(text[:1500])
        print("   [...]" if len(text) > 1500 else "")
        print(f"   contains \\boxed: {'\\boxed' in text}")
        print()

# Does the model ever emit EOS on its own, and does a stop string help?
print("=== with stop sequences ===")
payload = {
    "model": MODEL,
    "prompt": ex.prompt,
    "n": 4,
    "max_tokens": 2048,
    "temperature": 0.6,
    "top_p": 0.95,
    "seed": 0,
    "stop": ["\nUser:", "\nUser :", "\n\nUser"],
}
body = requests.post(f"{URL}/v1/completions", json=payload, timeout=1200).json()
for i, choice in enumerate(body["choices"]):
    text = choice["text"]
    boxed = "\\boxed" in text
    print(f"  sample {i}: finish={choice['finish_reason']:8s} chars={len(text):5d} boxed={boxed} stop={choice.get('stop_reason')!r}")
    if boxed:
        idx = text.rfind("\\boxed")
        print(f"     ...{text[max(0,idx-60):idx+40]!r}")
