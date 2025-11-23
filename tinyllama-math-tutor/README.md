# TinyLlama Math Tutor – LoRA Fine-Tuning Example

This project demonstrates how to fine-tune the **TinyLlama-1.1B-Chat** model using **LoRA / QLoRA** on a small, custom dataset to create a simple **math-tutor assistant**.

The goal of this project is to learn the fundamentals of:
- loading LLMs with Hugging Face Transformers  
- quantizing models using bitsandbytes (4-bit loading)  
- applying LoRA for parameter-efficient fine-tuning  
- preparing instruction-tuning datasets (JSONL format)  
- training and testing a model inside Google Colab  

This project is a complete end-to-end LLM fine-tuning example.

## 📂 Project Structure

```
tinyllama-math-tutor/
│
├── notebook.ipynb                 # Full Google Colab training script
├── math_tutor_train_v2.jsonl      # Custom math-tutor dataset
└── README.md                      # This file
```

## 📘 What This Project Does

### 1. **Loads a pre-trained model**
We load `TinyLlama/TinyLlama-1.1B-Chat-v1.0` using:

- `AutoModelForCausalLM`
- 4-bit quantization (`load_in_4bit=True`)
- automatic GPU placement (`device_map="auto"`)

### 2. **Loads and prepares a custom instruction dataset**
A JSONL dataset with fields:
```json
{"instruction": "...", "output": "..."}
```

This dataset contains simple math problems and explanations.

### 3. **Applies LoRA for efficient fine-tuning**
Instead of updating the full 1.1B model, we:
- freeze the base model
- add small LoRA adapter layers
- train only those new layers

### 4. **Trains the model using Hugging Face Trainer**
Training runs quickly (a few minutes) and produces a LoRA adapter folder.

### 5. **Tests the fine-tuned model**
Inference is run on example questions to check improvement.

## 📊 Fine-Tuning Method

This project uses:
- **PEFT**  
- **LoRA / QLoRA**  
- **BitsAndBytes**  
- **Transformers Trainer**

## 🧠 Dataset Format

```
{"instruction": "What is 7 + 8?", "output": "7 + 8 = 15."}
{"instruction": "What is speed?", "output": "Speed is how fast something moves."}
```

## 🚀 How to Run This Project

1. Open `notebook.ipynb` in Google Colab  
2. Upload `math_tutor_train_v2.jsonl`  
3. Run all cells  
4. Use `ask()` to test the model  

## 📌 Requirements

- transformers  
- datasets  
- peft  
- accelerate  
- bitsandbytes  

## 🎯 Purpose

Part of a collection of LLM fine-tuning experiments.

## 📬 Contact

Open an issue or pull request for improvements.
