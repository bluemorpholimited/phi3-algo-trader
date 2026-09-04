#!/usr/bin/env python3
"""
GGUF Conversion + Quantization + Verification Script.
Merges LoRA adapters into base model, converts to GGUF,
quantizes to Q4_K_M, and runs verification.

Writes progress to /marimo/outputs/gguf_progress.json
"""
import os, json, time, sys, subprocess

DPO_MODEL_PATH = "/marimo/outputs/dpo"
MERGED_MODEL_PATH = "/marimo/outputs/merged"
GGUF_DIR = "/marimo/outputs/gguf"
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
HF_TOKEN = "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"
HF_REPO = "bluemorpholimited/phi3-algo-trader-gguf"
QUANT_TYPE = "q4_k_m"

PROGRESS_FILE = "/marimo/outputs/gguf_progress.json"
LOG_FILE = "/marimo/outputs/gguf_conversion.log"

def write_progress(status, **kwargs):
    data = {"status": status, "timestamp": time.time(), **kwargs}
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f)

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    print(msg, flush=True)

log("=" * 60)
log("GGUF CONVERSION + QUANTIZATION")
log("=" * 60)

# Step 1: Merge LoRA adapters
log("[1/5] Merging LoRA adapters...")
write_progress("merging_lora")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="cpu",
    attn_implementation="eager",
)

# Load DPO model with adapters
model = PeftModel.from_pretrained(base_model, DPO_MODEL_PATH)
log("  Merging adapters...")
merged_model = model.merge_and_unload()

log(f"  Saving merged model to {MERGED_MODEL_PATH}...")
os.makedirs(MERGED_MODEL_PATH, exist_ok=True)
merged_model.save_pretrained(MERGED_MODEL_PATH, safe_serialization=False)
tokenizer.save_pretrained(MERGED_MODEL_PATH)
log("  Merge complete!")
write_progress("merge_complete")

# Free VRAM
del base_model, model, merged_model
torch.cuda.empty_cache()

# Step 2: Convert to GGUF format
log("[2/5] Converting to GGUF format...")
write_progress("converting_gguf")

# Install llama.cpp
log("  Installing llama.cpp...")
result = subprocess.run(
    ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", "/marimo/llama.cpp"],
    capture_output=True, text=True, timeout=120
)
if result.returncode != 0 and "already exists" not in result.stderr:
    log(f"  Git clone result: {result.stderr}")

# Install build deps and build
log("  Building llama.cpp...")
build_cmds = [
    "cd /marimo/llama.cpp && cmake -B build -DBLAS=ON",
    "cd /marimo/llama.cpp && cmake --build build --config Release -j$(nproc)"
]
for cmd in build_cmds:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        log(f"  Build step failed: {result.stderr[-500:]}")
    else:
        log(f"  Build step OK")

# Convert to GGUF
convert_script = "/marimo/llama.cpp/convert_hf_to_gguf.py"
gguf_f16_path = os.path.join(GGUF_DIR, "model-f16.gguf")
os.makedirs(GGUF_DIR, exist_ok=True)

log(f"  Converting HF model to GGUF (f16)...")
result = subprocess.run(
    [sys.executable, convert_script, MERGED_MODEL_PATH, "--outfile", gguf_f16_path, "--outtype", "f16"],
    capture_output=True, text=True, timeout=600
)
log(f"  Convert stdout: {result.stdout[-500:]}")
if result.returncode != 0:
    log(f"  Convert stderr: {result.stderr[-500:]}")
    write_progress("convert_failed", error=result.stderr[-500:])
    sys.exit(1)
else:
    log(f"  GGUF conversion complete: {gguf_f16_path}")
    write_progress("gguf_converted", path=gguf_f16_path)

# Step 3: Quantize to Q4_K_M
log(f"[3/5] Quantizing to {QUANT_TYPE}...")
write_progress("quantizing", quant_type=QUANT_TYPE)

quantize_binary = "/marimo/llama.cpp/build/bin/llama-quantize"
gguf_quant_path = os.path.join(GGUF_DIR, f"model-{QUANT_TYPE}.gguf")

result = subprocess.run(
    [quantize_binary, gguf_f16_path, gguf_quant_path, QUANT_TYPE.upper()],
    capture_output=True, text=True, timeout=600
)
log(f"  Quantize stdout: {result.stdout[-500:]}")
if result.returncode != 0:
    log(f"  Quantize failed: {result.stderr[-500:]}")
    write_progress("quantize_failed", error=result.stderr[-500:])
    sys.exit(1)
else:
    file_size = os.path.getsize(gguf_quant_path) / 1e9
    log(f"  Quantization complete! Size: {file_size:.2f} GB")
    write_progress("quantized", path=gguf_quant_path, size_gb=file_size)

# Step 4: Verification - test the quantized model
log("[4/5] Verifying quantized model...")
write_progress("verifying")

try:
    from llama_cpp import Llama
    llm = Llama(
        model_path=gguf_quant_path,
        n_ctx=2048,
        n_gpu_layers=99,
        verbose=False,
    )

    test_prompts = [
        {
            "system": "You are a financial analyst specializing in stock direction prediction. Analyze technical indicators and historical prices to predict stock direction, generating step-by-step reasoning followed by a final VERDICT: BUY, SELL, or HOLD.",
            "user": "Analyze AAPL as of 2024-01-15 and predict the stock direction.\n\nPrice history (last 20 days): $185.30 -> $186.20 -> $184.50 -> $183.10 -> $185.80 -> $187.40 -> $188.20 -> $186.90 -> $184.30 -> $182.50 -> $180.20 -> $181.10 -> $182.80 -> $183.50 -> $184.90 -> $185.30 -> $186.10 -> $187.20 -> $188.50 -> $189.10",
        },
        {
            "system": "You are a financial analyst. Provide a clear BUY, SELL, or HOLD verdict with reasoning.",
            "user": "MSFT RSI is 45.2, volume is 1.3x the 20-day average. Price has been declining for 5 days. What is your recommendation?",
        },
        {
            "system": "You are a quantitative breakout detector. Output a structured JSON block containing directional confidence, regime classification, and prediction horizon.",
            "user": "Asset: NVDA\nWavelet Price History: [425.08, 430.03, 428.47, 435.60, 442.42]\nDenoised Metrics: RSI_BIN_6 | VOL_BIN_4 | VOLATILITY_BIN_3",
        },
    ]

    verification_results = []
    for i, prompt in enumerate(test_prompts):
        log(f"  Test {i+1}: {prompt['user'][:80]}...")
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            max_tokens=500,
            temperature=0.7,
        )
        output = response["choices"][0]["message"]["content"]
        log(f"  Response: {output[:300]}")

        # Check for valid output (contains BUY/SELL/HOLD or JSON structure)
        has_signal = any(sig in output.upper() for sig in ["BUY", "SELL", "HOLD"])
        has_json = "{" in output and "}" in output
        is_valid = has_signal or has_json

        verification_results.append({
            "test": i + 1,
            "prompt": prompt["user"][:100],
            "response": output[:500],
            "has_signal": has_signal,
            "has_json": has_json,
            "valid": is_valid,
        })

    valid_count = sum(1 for r in verification_results if r["valid"])
    log(f"  Verification: {valid_count}/{len(verification_results)} tests passed")

    # Save verification results
    with open(os.path.join(GGUF_DIR, "verification_results.json"), "w") as f:
        json.dump(verification_results, f, indent=2)

    write_progress("verified", tests=len(verification_results), passed=valid_count)
    del llm

except ImportError:
    log("  llama-cpp-python not installed, installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python"], capture_output=True, text=True, timeout=120)
    log("  Installed. Re-run verification manually.")

# Step 5: Upload GGUF to HuggingFace
log("[5/5] Uploading GGUF to HuggingFace...")
write_progress("uploading_hf")

from huggingface_hub import HfApi, create_repo
api = HfApi(token=HF_TOKEN)

# Create repo
try:
    create_repo(HF_REPO, repo_type="model", token=HF_TOKEN, exist_ok=True, private=True)
    log(f"  Created repo: {HF_REPO}")
except Exception as e:
    log(f"  Repo creation: {e}")

# Upload quantized GGUF
log(f"  Uploading {gguf_quant_path}...")
api.upload_file(
    path_or_fileobj=gguf_quant_path,
    path_in_repo=f"model-{QUANT_TYPE}.gguf",
    repo_id=HF_REPO,
    token=HF_TOKEN,
)
log(f"  Uploaded Q4_K_M model!")

# Upload verification results
api.upload_file(
    path_or_fileobj=os.path.join(GGUF_DIR, "verification_results.json"),
    path_in_repo="verification_results.json",
    repo_id=HF_REPO,
    token=HF_TOKEN,
)

# Also upload f16 if space allows
if os.path.exists(gguf_f16_path):
    f16_size = os.path.getsize(gguf_f16_path) / 1e9
    if f16_size < 10:  # only upload if reasonable size
        log(f"  Uploading f16 model ({f16_size:.2f} GB)...")
        api.upload_file(
            path_or_fileobj=gguf_f16_path,
            path_in_repo="model-f16.gguf",
            repo_id=HF_REPO,
            token=HF_TOKEN,
        )

write_progress("uploaded", hf_repo=HF_REPO)
log(f"  GGUF model on HF: https://huggingface.co/{HF_REPO}")

stats = {
    "merged_model_path": MERGED_MODEL_PATH,
    "gguf_f16_path": gguf_f16_path,
    "gguf_quant_path": gguf_quant_path,
    "quant_type": QUANT_TYPE,
    "quant_size_gb": os.path.getsize(gguf_quant_path) / 1e9,
    "verification_passed": valid_count if 'valid_count' in dir() else 0,
    "hf_repo": HF_REPO,
}
with open(os.path.join(GGUF_DIR, "gguf_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

write_progress("done", **stats)
log("GGUF PHASE COMPLETE!")
log(f"Quantized model: {gguf_quant_path}")
log(f"HF: https://huggingface.co/{HF_REPO}")
