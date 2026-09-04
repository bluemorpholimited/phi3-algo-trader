#!/usr/bin/env python3
"""
SFT Training Script - runs as a standalone process on Molab.
Uses nohup to survive kernel disconnects.
Writes progress to /marimo/outputs/training_progress.json
"""
import torch, os, json, time, sys
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import TrainerCallback

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR = "/marimo/outputs/sft"
MAX_SEQ_LEN = 2048
BATCH_SIZE = 4
GRAD_ACCUM = 8
LR = 2e-4
EPOCHS = 2
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
SFT_EXAMPLES = 50000
HF_REPO = "bluemorpholimited/phi3-algo-trader-sft"
HF_TOKEN = "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"

PROGRESS_FILE = "/marimo/outputs/training_progress.json"
LOG_FILE = "/marimo/outputs/sft_training.log"

def write_progress(status, **kwargs):
    data = {"status": status, "timestamp": time.time(), **kwargs}
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f)
    print(f"[{time.strftime('%H:%M:%S')}] {status} {kwargs}")

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    print(msg, flush=True)

log("=" * 60)
log("SFT TRAINING PIPELINE (QLoRA 4-bit)")
log("=" * 60)

# Step 1: Load datasets
log("[1/5] Loading datasets...")
write_progress("loading_datasets")

ds_signals = load_dataset("ewin-reg/Stock-Market-Trading-Signals", split="train")
log(f"  Trading signals: {len(ds_signals)}")

ds_json = load_dataset("ewin-reg/Stock-Market-Trading-Signals-V11-JSON", split="train")
log(f"  JSON signals: {len(ds_json)}")

ds_finance = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k", split="train").shuffle(seed=42)
finance_budget = max(0, SFT_EXAMPLES - len(ds_signals) - len(ds_json))
ds_finance = ds_finance.select(range(min(finance_budget, len(ds_finance))))
log(f"  Finance instruct: {len(ds_finance)}")

write_progress("formatting_datasets", signals=len(ds_signals), json_signals=len(ds_json), finance=len(ds_finance))

# Step 2: Format
log("[2/5] Formatting datasets...")

def format_signals(ex):
    if "messages" in ex: return {"messages": ex["messages"]}
    return {"messages": [{"role": "system", "content": "You are a financial analyst."}, {"role": "user", "content": ex.get("inputs", "")}, {"role": "assistant", "content": ex.get("answer", "")}]}

def format_json_signals(ex):
    text = ex.get("text", "")
    if not text: return {"messages": [{"role": "user", "content": "N/A"}, {"role": "assistant", "content": "N/A"}]}
    messages = []
    for part in text.split("<|im_start|>")[1:]:
        cp = part.split("<|im_end|>")
        if len(cp) >= 2:
            rc = cp[0].strip()
            lines = rc.split("\n", 1)
            messages.append({"role": lines[0].strip(), "content": lines[1].strip() if len(lines) > 1 else ""})
    return {"messages": messages} if len(messages) >= 2 else {"messages": [{"role": "user", "content": "N/A"}, {"role": "assistant", "content": "N/A"}]}

def format_finance(ex):
    return {"messages": [{"role": "system", "content": ex.get("system_prompt", "You are a financial assistant.")}, {"role": "user", "content": ex.get("inputs", "")}, {"role": "assistant", "content": ex.get("answer", "")}]}

ds_signals_fmt = ds_signals.map(format_signals, remove_columns=ds_signals.column_names)
ds_json_fmt = ds_json.map(format_json_signals, remove_columns=ds_json.column_names)
ds_finance_fmt = ds_finance.map(format_finance, remove_columns=ds_finance.column_names)

combined = concatenate_datasets([ds_signals_fmt, ds_json_fmt, ds_finance_fmt]).shuffle(seed=42)
if len(combined) > SFT_EXAMPLES:
    combined = combined.select(range(SFT_EXAMPLES))
log(f"  Final dataset: {len(combined)}")

# Step 3: Load model
log("[3/5] Loading model (QLoRA 4-bit)...")
write_progress("loading_model")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager",
)
model = prepare_model_for_kbit_training(model)
log(f"  Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# Step 4: Configure trainer
log("[4/5] Setting up trainer...")
write_progress("setting_up_trainer")

peft_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# Check for resume
resume_from = None
if os.path.exists(OUTPUT_DIR):
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if checkpoints:
        latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        resume_from = os.path.join(OUTPUT_DIR, latest)
        log(f"  Resuming from: {resume_from}")

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=5,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_length=MAX_SEQ_LEN,
    packing=True,
    report_to="none",
    seed=42,
    optim="adamw_torch_fused",
    weight_decay=0.01,
    max_grad_norm=1.0,
    push_to_hub=True,
    hub_model_id=HF_REPO,
    hub_token=HF_TOKEN,
    hub_strategy="checkpoint",
    hub_private_repo=True,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=combined,
    peft_config=peft_config,
    processing_class=tokenizer,
)

# Progress callback
class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        write_progress("training",
            step=state.global_step,
            loss=logs.get("loss", 0) if logs else 0,
            epoch=logs.get("epoch", 0) if logs else 0,
            lr=logs.get("learning_rate", 0) if logs else 0,
            total_steps=EPOCHS * (len(combined) // (BATCH_SIZE * GRAD_ACCUM)))

trainer.add_callback(ProgressCallback())

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
log(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)")

# Step 5: Train
log("[5/5] Starting training...")
total_steps = EPOCHS * (len(combined) // (BATCH_SIZE * GRAD_ACCUM))
write_progress("training_started", total_steps=total_steps)
log(f"  Total steps: {total_steps}")

train_result = trainer.train(resume_from_checkpoint=resume_from)

log(f"  Training complete! Loss: {train_result.training_loss:.4f}, Steps: {train_result.global_step}")
write_progress("training_complete", train_loss=train_result.training_loss, global_step=train_result.global_step)

# Save
log("Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

log(f"Pushing to {HF_REPO}...")
trainer.push_to_hub()
write_progress("model_pushed", hf_repo=HF_REPO)

stats = {
    "base_model": BASE_MODEL,
    "sft_examples": len(combined),
    "train_loss": train_result.training_loss,
    "global_step": train_result.global_step,
    "output_dir": OUTPUT_DIR,
    "hf_repo": HF_REPO,
}
with open(os.path.join(OUTPUT_DIR, "training_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

write_progress("done", **stats)
log("SFT PHASE COMPLETE!")
log(f"Model at: {OUTPUT_DIR}")
log(f"Model on HF: https://huggingface.co/{HF_REPO}")
