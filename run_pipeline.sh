#!/usr/bin/env bash
# Master pipeline: SFT → DPO → GGUF → HF Upload
# Runs on Molab sandbox, survives kernel disconnects via nohup
set -e

echo "[$(date)] Starting full training pipeline..."
echo "[$(date)] Pipeline: SFT → DPO → GGUF → HF Upload"

PYTHON=/usr/local/bin/python3
LOG=/marimo/outputs/pipeline.log

echo "[$(date)] === STAGE 1: SFT TRAINING ===" | tee -a $LOG
$PYTHON /marimo/sft_train_bg.py 2>&1 | tee -a $LOG

echo "[$(date)] === STAGE 2: DPO TRAINING ===" | tee -a $LOG
$PYTHON /marimo/dpo_train_bg.py 2>&1 | tee -a $LOG

echo "[$(date)] === STAGE 3: GGUF + QUANTIZATION + VERIFICATION ===" | tee -a $LOG
$PYTHON /marimo/gguf_convert_bg.py 2>&1 | tee -a $LOG

echo "[$(date)] === PIPELINE COMPLETE ===" | tee -a $LOG
echo "[$(date)] Check outputs at:" | tee -a $LOG
echo "  SFT: https://huggingface.co/bluemorpholimited/phi3-algo-trader-sft" | tee -a $LOG
echo "  DPO: https://huggingface.co/bluemorpholimited/phi3-algo-trader-dpo" | tee -a $LOG
echo "  GGUF: https://huggingface.co/bluemorpholimited/phi3-algo-trader-gguf" | tee -a $LOG
