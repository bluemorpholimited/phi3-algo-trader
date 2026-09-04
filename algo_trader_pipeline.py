import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    mo.status.toast("Algo Trading LLM Pipeline — SFT → DPO → GGUF on Blackwell RTX PRO 6000")
    print("Pipeline initialized")
    return (mo,)


@app.cell
def _():
    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    return (torch,)


@app.cell
def _():
    # Install required packages
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
        "peft", "trl", "accelerate", "bitsandbytes", "huggingface_hub"],
        capture_output=True, text=True, timeout=300)
    import trl, peft, accelerate, bitsandbytes
    print(f"trl={trl.__version__}, peft={peft.__version__}")
    from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
    print("All ML imports OK!")
    return


@app.cell
def _():
    # ============================================================
    # CONFIG
    # ============================================================
    BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
    OUTPUT_DIR = "/marimo/outputs/sft"
    DPO_DIR = "/marimo/outputs/dpo"
    MERGED_DIR = "/marimo/outputs/merged"
    GGUF_DIR = "/marimo/outputs/gguf"
    MAX_SEQ_LEN = 2048
    BATCH_SIZE = 4
    GRAD_ACCUM = 8
    LR = 2e-4
    EPOCHS = 2
    LORA_R = 64
    LORA_ALPHA = 128
    LORA_DROPOUT = 0.05
    SFT_EXAMPLES = 50000
    DPO_EXAMPLES = 5000
    HF_REPO_SFT = "bluemorpholimited/phi3-algo-trader-sft"
    HF_REPO_DPO = "bluemorpholimited/phi3-algo-trader-dpo"
    HF_REPO_GGUF = "bluemorpholimited/phi3-algo-trader-gguf"
    HF_TOKEN = "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"

    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    for d in [OUTPUT_DIR, DPO_DIR, MERGED_DIR, GGUF_DIR]:
        os.makedirs(d, exist_ok=True)
    print("Config ready")
    return


@app.cell
def _():
    # ============================================================
    # STEP 1: LOAD & FORMAT SFT DATASETS
    # ============================================================
    from datasets import load_dataset, concatenate_datasets

    print("Loading datasets...")
    ds_signals = load_dataset("ewin-reg/Stock-Market-Trading-Signals", split="train")
    print(f"  Trading signals: {len(ds_signals)}")

    ds_json = load_dataset("ewin-reg/Stock-Market-Trading-Signals-V11-JSON", split="train")
    print(f"  JSON signals: {len(ds_json)}")

    ds_finance = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k", split="train").shuffle(seed=42)
    finance_budget = max(0, 50000 - len(ds_signals) - len(ds_json))
    ds_finance = ds_finance.select(range(min(finance_budget, len(ds_finance))))
    print(f"  Finance instruct: {len(ds_finance)}")

    # Format functions
    def format_signals(ex):
        if "messages" in ex:
            return {"messages": ex["messages"]}
        return {"messages": [{"role": "system", "content": "You are a financial analyst."}, {"role": "user", "content": ex.get("inputs", "")}, {"role": "assistant", "content": ex.get("answer", "")}]}

    def format_json_signals(ex):
        text = ex.get("text", "")
        if not text:
            return {"messages": [{"role": "user", "content": "N/A"}, {"role": "assistant", "content": "N/A"}]}
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

    ds_s = ds_signals.map(format_signals, remove_columns=ds_signals.column_names)
    ds_j = ds_json.map(format_json_signals, remove_columns=ds_json.column_names)
    ds_f = ds_finance.map(format_finance, remove_columns=ds_finance.column_names)

    combined = concatenate_datasets([ds_s, ds_j, ds_f]).shuffle(seed=42)
    if len(combined) > 50000:
        combined = combined.select(range(50000))
    print(f"Final SFT dataset: {len(combined)} examples")
    return (combined,)


@app.cell
def _(combined, torch):
    # ============================================================
    # STEP 2: LOAD MODEL (QLoRA 4-bit)
    # ============================================================
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)
    print(f"Model loaded (4-bit). VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return (model, tokenizer, bnb_config, LoraConfig, prepare_model_for_kbit_training)


@app.cell
def _(combined, model, tokenizer, LoraConfig):
    # ============================================================
    # STEP 3: SFT TRAINING
    # ============================================================
    from trl import SFTTrainer, SFTConfig

    peft_config = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    sft_config = SFTConfig(
        output_dir="/marimo/outputs/sft",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=2048,
        packing=True,
        report_to="none",
        seed=42,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        max_grad_norm=1.0,
        push_to_hub=True,
        hub_model_id="bluemorpholimited/phi3-algo-trader-sft",
        hub_token="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",
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

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)")
    print("Starting SFT training...")
    return (trainer,)


@app.cell
def _(trainer):
    # Run SFT training
    train_result = trainer.train()
    print(f"SFT complete! Loss: {train_result.training_loss:.4f}, Steps: {train_result.global_step}")
    trainer.save_model("/marimo/outputs/sft")
    trainer.push_to_hub()
    print("SFT model saved and pushed to HF!")
    return (train_result,)


@app.cell
def _():
    # ============================================================
    # STEP 4: DPO TRAINING
    # ============================================================
    from datasets import load_dataset as _ls
    from transformers import AutoModelForCausalLM as _AM, AutoTokenizer as _AT
    from trl import DPOTrainer, DPOConfig
    from peft import LoraConfig as _LC
    import torch as _t

    ds_dpo = _ls("ZixuanKe/sujet_finance_instruct_sup_dpo_binarized", split="train")
    if len(ds_dpo) > 5000:
        ds_dpo = ds_dpo.shuffle(seed=42).select(range(5000))
    print(f"DPO dataset: {len(ds_dpo)}")

    def format_dpo(ex):
        p, c, r = ex.get("prompt",""), ex.get("chosen",""), ex.get("rejected","")
        if isinstance(c, list): c = c[-1]["content"] if c else ""
        if isinstance(r, list): r = r[-1]["content"] if r else ""
        return {"prompt": p, "chosen": c, "rejected": r}

    ds_dpo = ds_dpo.map(format_dpo).filter(lambda x: len(x["prompt"])>10 and len(x["chosen"])>10 and len(x["rejected"])>10)
    print(f"After filtering: {len(ds_dpo)}")

    tok = _AT.from_pretrained("/marimo/outputs/sft")
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    dpo_model = _AM.from_pretrained("/marimo/outputs/sft", dtype=_t.bfloat16, device_map="auto", attn_implementation="eager")

    dpo_cfg = DPOConfig(
        output_dir="/marimo/outputs/dpo",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        warmup_steps=25,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=2048,
        max_prompt_length=1024,
        beta=0.1,
        report_to="none",
        seed=42,
        optim="adamw_torch_fused",
        push_to_hub=True,
        hub_model_id="bluemorpholimited/phi3-algo-trader-dpo",
        hub_token="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",
        hub_strategy="checkpoint",
        hub_private_repo=True,
    )

    dpo_trainer = DPOTrainer(
        model=dpo_model,
        args=dpo_cfg,
        train_dataset=ds_dpo,
        processing_class=tok,
        peft_config=_LC(r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]),
    )
    print("Starting DPO training...")
    dpo_result = dpo_trainer.train()
    print(f"DPO complete! Loss: {dpo_result.training_loss:.4f}")
    dpo_trainer.save_model("/marimo/outputs/dpo")
    dpo_trainer.push_to_hub()
    print("DPO model saved and pushed!")
    return (dpo_trainer,)


@app.cell
def _():
    # ============================================================
    # STEP 5: MERGE LoRA → GGUF → Q4_K_M → VERIFY → UPLOAD
    # ============================================================
    import os, sys, subprocess, json, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import HfApi, create_repo

    HF_TOKEN = "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"
    HF_REPO = "bluemorpholimited/phi3-algo-trader-gguf"

    # 5a: Merge LoRA
    print("Merging LoRA adapters...")
    base = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", dtype=torch.float16, device_map="cpu", attn_implementation="eager")
    peft_model = PeftModel.from_pretrained(base, "/marimo/outputs/dpo")
    merged = peft_model.merge_and_unload()
    os.makedirs("/marimo/outputs/merged", exist_ok=True)
    merged.save_pretrained("/marimo/outputs/merged", safe_serialization=False)
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct").save_pretrained("/marimo/outputs/merged")
    print("Merge complete!")
    del base, peft_model, merged
    torch.cuda.empty_cache()

    # 5b: Clone & build llama.cpp
    print("Building llama.cpp...")
    if not os.path.exists("/marimo/llama.cpp"):
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", "/marimo/llama.cpp"], check=True)
    subprocess.run("cd /marimo/llama.cpp && cmake -B build -DBLAS=ON && cmake --build build --config Release -j$(nproc)", shell=True, check=True)

    # 5c: Convert to GGUF
    print("Converting to GGUF (f16)...")
    gguf_f16 = "/marimo/outputs/gguf/model-f16.gguf"
    os.makedirs("/marimo/outputs/gguf", exist_ok=True)
    subprocess.run([sys.executable, "/marimo/llama.cpp/convert_hf_to_gguf.py", "/marimo/outputs/merged", "--outfile", gguf_f16, "--outtype", "f16"], check=True)
    print("GGUF conversion done!")

    # 5d: Quantize to Q4_K_M
    print("Quantizing to Q4_K_M...")
    gguf_q4 = "/marimo/outputs/gguf/model-q4_k_m.gguf"
    subprocess.run(["/marimo/llama.cpp/build/bin/llama-quantize", gguf_f16, gguf_q4, "Q4_K_M"], check=True)
    size_gb = os.path.getsize(gguf_q4) / 1e9
    print(f"Quantized! Size: {size_gb:.2f} GB")

    # 5e: Verify
    print("Verifying model...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "llama-cpp-python"], capture_output=True)
    from llama_cpp import Llama
    llm = Llama(model_path=gguf_q4, n_ctx=2048, n_gpu_layers=99, verbose=False)
    test = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a financial analyst. Predict stock direction with reasoning. End with VERDICT: BUY, SELL, or HOLD."},
            {"role": "user", "content": "Analyze AAPL. RSI: 58.4, Volume: 1.55x average. Price trending up 5 days. What is your recommendation?"},
        ],
        max_tokens=300,
    )
    response = test["choices"][0]["message"]["content"]
    print(f"Verification response:\n{response}")
    has_signal = any(sig in response.upper() for sig in ["BUY", "SELL", "HOLD"])
    print(f"Contains trading signal: {has_signal}")

    # 5f: Upload to HF
    print("Uploading GGUF to HuggingFace...")
    api = HfApi(token=HF_TOKEN)
    create_repo(HF_REPO, repo_type="model", token=HF_TOKEN, exist_ok=True, private=True)
    api.upload_file(path_or_fileobj=gguf_q4, path_in_repo="model-q4_k_m.gguf", repo_id=HF_REPO, token=HF_TOKEN)
    print(f"Uploaded! https://huggingface.co/{HF_REPO}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"SFT model:  https://huggingface.co/bluemorpholimited/phi3-algo-trader-sft")
    print(f"DPO model:  https://huggingface.co/bluemorpholimited/phi3-algo-trader-dpo")
    print(f"GGUF model: https://huggingface.co/bluemorpholimited/phi3-algo-trader-gguf")
    return


if __name__ == "__main__":
    app.run()
