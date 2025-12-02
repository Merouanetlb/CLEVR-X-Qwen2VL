# CLEVR-X Visual Question Answering with Qwen2-VL

This repository contains the source code for the I491E competition. The goal is to solve a Visual Question Answering (VQA) task using a model with fewer than 4 billion parameters.

**Score achieved:** 0.876 (Rank 18)

## 🏗️ Architecture
- **Model:** [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- **Method:** Supervised Fine-Tuning (SFT) with LoRA (Low-Rank Adaptation)
- **Precision:** bfloat16 (Optimized for NVIDIA A100)

## 📂 Files Description
- `train_best.py`: The main training script used to achieve the best score. It handles data loading, LoRA configuration, training loop, and inference.
- `technical_report.pdf`: Detailed analysis of the experiments, failure cases (Model Collapse), and final results.
- `requirements.txt`: List of dependencies.

## 🚀 How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the training script:**
   ```bash
   python train_best.py
   ```

## ⚙️ Hyperparameters (Best Run)
- **Epochs:** 5
- **Learning Rate:** 5e-5
- **LoRA Rank:** 128
- **Batch Size:** 4 (Gradient Accumulation: 4)
- **Context Length:** 512

## 👤 Author
Student ID: s2512017
