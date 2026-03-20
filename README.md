# Bert Online Project

A bert-based online prediction project using Flask/Python.

## Projects Structure:
- `app.py`: Project entry point.
- `predict.py`: Model prediction logic.
- `textCNN.py`: Model architecture definition.

## Setup Requirements:
- Python 3.8+
- PyTorch (Cuda enabled)
- Hugging Face Transformers

## Local Usage:
Ensure your model weights (`pytorch_model.bin`) are in the `data/bert_pretrain/` directory before running.
```bash
python app.py
```
