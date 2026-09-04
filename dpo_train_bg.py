#!/usr/bin/env python3
"""
DPO Training Script - runs after SFT completes.
Loads SFT model, applies DPO on finance preference pairs.
Writes progress to /marimo/outputs/dpo_progress.json
"""
import torch, os, json, time, sys
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig
from transformers import TrainerCallback

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SFT_MODEL_PATH = "/marimo/outputs/sft"
OUTPUT_DIR = "/marimo/outputs/dpo"
MAX_SEQ_LEN = 2048
BATCH_SIZE = 2
GRAD_ACCUM = 16  # effective batch = 32
LR = 5e-5
EPOCHS = 1
BETA = 0.1
DPO_EXAMPLES = 5000
HF_REPO = "bluemorpholimited/phi3-algo-trader-dpo"
HF_TOKEN = "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"

PROGRESS_FILE = "/marimo/outputs/dpo_progress.json"
LOG_FILE = "/marimo/outputs/dpo_training.log"

def write_progress(status, **kwargs):
    data = {"status": status, "timestamp": time.time(), **kwargs}
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f)

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    print(msg, flush=True)

log("=" * 60)
log("DPO TRAINING PIPELINE")
log("=" * 60)

# Step 1: Load DPO datasets
log("[1/4] Loading DPO datasets...")
write_progress("loading_datasets")

# Primary DPO dataset: finance DPO pairs
ds_dpo = load_dataset("ZixuanKe/sujet_finance_instruct_sup_dpo_binarized", split="train")
log(f"  DPO pairs: {len(ds_dpo)}")

# Supplemental if needed
if len(ds_dpo) < DPO_EXAMPLES:
    ds_extra = load_dataset("gandhiraketla277/finance-dpo-dataset", split="train")
    log(f"  Extra DPO: {len(ds_extra)}")
    from datasets import concatenate_datasets
    ds_dpo = concatenate_datasets([ds_dpo, ds_extra]).shuffle(seed=42)

# Trim to target
if len(ds_dpo) > DPO_EXAMPLES:
    ds_dpo = ds_dpo.shuffle(seed=42).select(range(DPO_EXAMPLES))
log(f"  Final DPO dataset: {len(ds_dpo)}")

# Step 2: Format DPO data
log("[2/4] Formatting DPO data...")
write_progress("formatting")

# DPO expects: prompt, chosen, rejected
def format_dpo(example):
    prompt = example.get("prompt", "")
    chosen = example.get("chosen", "")
    rejected = example.get("rejected", "")

    # Handle if chosen/rejected are lists (conversation format)
    if isinstance(chosen, list):
        chosen = chosen[-1]["content"] if chosen else ""
    if isinstance(rejected, list):
        rejected = rejected[-1]["content"] if rejected else ""

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

ds_dpo_fmt = ds_dpo.map(format_dpo)
# Filter out empty examples
ds_dpo_fmt = ds_dpo_fmt.filter(lambda x: len(x["prompt"]) > 10 and len(x["chosen"]) > 10 and len(x["rejected"]) > 10)
log(f"  After filtering: {len(ds_dpo_fmt)}")

# Step 3: Load SFT model
log("[3/4] Loading SFT model...")
write_progress("loading_model")

tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    SFT_MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)

log(f"  Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# Step 4: Configure DPO trainer
log("[4/4] Setting up DPO trainer...")
write_progress("setting_up_trainer")

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_steps=25,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_length=MAX_SEQ_LEN,
    max_prompt_length=1024,
    beta=BETA,
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

trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=ds_dpo_fmt,
    processing_class=tokenizer,
    peft_config=peft_config,
)

class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        write_progress("training",
            step=state.global_step,
            loss=logs.get("loss", 0) if logs else 0,
            reward=logs.get("rewards/margins", 0) if logs else 0,
            epoch=logs.get("epoch", 0) if logs else 0)

trainer.add_callback(ProgressCallback())

# Train
log("Starting DPO training...")
total_steps = EPOCHS * (len(ds_dpo_fmt) // (BATCH_SIZE * GRAD_ACCUM))
write_progress("training_started", total_steps=total_steps)
log(f"  Total steps: {total_steps}")

train_result = trainer.train()

log(f"  DPO complete! Loss: {train_result.training_loss:.4f}")
write_progress("training_complete", train_loss=train_result.training_loss, global_step=train_result.global_step)

# Save
log("Saving DPO model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

log(f"Pushing to {HF_REPO}...")
trainer.push_to_hub()
write_progress("model_pushed", hf_repo=HF_REPO)

stats = {
    "dpo_examples": len(ds_dpo_fmt),
    "train_loss": train_result.training_loss,
    "global_step": train_result.global_step,
    "output_dir": OUTPUT_DIR,
    "hf_repo": HF_REPO,
}
with open(os.path.join(OUTPUT_DIR, "training_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

write_progress("done", **stats)
log("DPO PHASE COMPLETE!")
log(f"Model at: {OUTPUT_DIR}")
log(f"Model on HF: https://huggingface.co/{HF_REPO}")
