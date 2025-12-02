# CLEVR-X Visual Question Answering with Qwen2-VL

This repository contains the source code for the I491E competition. The goal is to solve a Visual Question Answering (VQA) task using a model with fewer than 4 billion parameters.

**Score achieved:** 0.876 (Rank 18)

## 🏗️ Architecture
- **Model:** Qwen2-VL-2B-Instruct
- **Method:** Supervised Fine-Tuning with LoRA (Low-Rank Adaptation)
- **Precision:** bfloat16 (Optimized for NVIDIA A100)

## 📂 Files Description
- `train_best.py`: The main training script. It handles data loading, LoRA configuration, training loop (5 epochs), and inference.
- `technical_report.pdf`: Detailed analysis of the experiments and results.

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
