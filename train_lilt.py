
import json
import inspect
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    LayoutLMv3TokenizerFast,
    LiltForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import precision_recall_fscore_support


# =========================================================
# 1. CONFIGURATION
# =========================================================

PROJECT_DIR = Path(r"C:\Internship\LiLT")
DATA_DIR = PROJECT_DIR / "indian_synthetic_invoice_dataset_200"
OUTPUT_DIR = PROJECT_DIR / "lilt-indian-invoice-model"

MODEL_NAME = "SCUT-DLVCLab/lilt-roberta-en-base"

MAX_LENGTH = 512
BATCH_SIZE = 2
EPOCHS = 1             # Start with 1; increase after testing
LEARNING_RATE = 2e-5

# =========================================================
# 2. CHECK FILES
# =========================================================

required_files = [
    DATA_DIR / "train.jsonl",
    DATA_DIR / "validation.jsonl",
    DATA_DIR / "test.jsonl",
    DATA_DIR / "label_map.json",
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing file: {file_path}\n"
            "Check that the dataset ZIP is extracted correctly."
        )

print("Dataset folder:", DATA_DIR)

# =========================================================
# 3. DEVICE CHECK
# =========================================================

if hasattr(torch, "xpu") and torch.xpu.is_available():
    device_name = torch.xpu.get_device_name(0)
    print("Using Intel GPU:", device_name)
else:
    print("Intel XPU unavailable; training will use CPU.")

print("PyTorch version:", torch.__version__)

# Hugging Face Trainer automatically selects a supported
# accelerator when the installed PyTorch build exposes it.

# =========================================================
# 4. LOAD LABEL MAP
# =========================================================

with open(DATA_DIR / "label_map.json", encoding="utf-8") as f:
    label2id = json.load(f)

# JSON label-map values may be integers already.
label2id = {label: int(idx) for label, idx in label2id.items()}
id2label = {idx: label for label, idx in label2id.items()}

num_labels = len(label2id)

print("Number of labels:", num_labels)
print("Labels:", label2id)

# =========================================================
# 5. LOAD DATASET SPLITS
# =========================================================

raw_datasets = load_dataset(
    "json",
    data_files={
        "train": str(DATA_DIR / "train.jsonl"),
        "validation": str(DATA_DIR / "validation.jsonl"),
        "test": str(DATA_DIR / "test.jsonl"),
    },
)

for split in raw_datasets:
    print(f"{split}: {len(raw_datasets[split])} invoices")

# =========================================================
# 6. LOAD THE CORRECT TOKENIZER
# =========================================================

tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
    MODEL_NAME
)

# =========================================================
# 7. TOKENIZE AND ALIGN LABELS
# =========================================================


def prepare_example(example):
    words = example["words"]
    boxes = example["boxes_normalized_0_1000"]
    word_labels = example["labels"]

    encoding = tokenizer(
        words,
        boxes=boxes,
        word_labels=word_labels,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

    return dict(encoding)





tokenized_datasets = raw_datasets.map(
    prepare_example,
    remove_columns=raw_datasets["train"].column_names,
    desc="Tokenizing invoices",
)

# =========================================================
# 8. LOAD PRETRAINED LiLT FOR TOKEN CLASSIFICATION
# =========================================================

model = LiltForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)

# The pretrained model weights are retained where compatible.
# The classification head is initialized for your invoice labels.

# =========================================================
# 9. METRICS
# =========================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    correct = 0
    total = 0
    true_ids = []
    pred_ids = []

    for pred_row, label_row in zip(predictions, labels):
        for pred_id, label_id in zip(pred_row, label_row):
            if label_id == -100:
                continue

            true_ids.append(int(label_id))
            pred_ids.append(int(pred_id))

            total += 1
            if int(pred_id) == int(label_id):
                correct += 1

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            true_ids,
            pred_ids,
            average="macro",
            zero_division=0,
        )
    )

    return {
        "token_accuracy": correct / max(total, 1),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


# =========================================================
# 10. TRAINING CONFIGURATION
# =========================================================

args_params = inspect.signature(TrainingArguments.__init__).parameters

eval_key = (
    "eval_strategy"
    if "eval_strategy" in args_params
    else "evaluation_strategy"
)

training_kwargs = {
    "output_dir": str(OUTPUT_DIR),
    "learning_rate": LEARNING_RATE,
    "per_device_train_batch_size": BATCH_SIZE,
    "per_device_eval_batch_size": BATCH_SIZE,
    "num_train_epochs": EPOCHS,
    "weight_decay": 0.01,
    "save_strategy": "epoch",
    "logging_steps": 10,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_macro_f1",
    "greater_is_better": True,
    "save_total_limit": 2,
    "report_to": "none",
    "seed": 42,
}

training_kwargs[eval_key] = "epoch"

# Use CPU if the Intel GPU is not detected.
if "use_cpu" in args_params:
    training_kwargs["use_cpu"] = not (
        hasattr(torch, "xpu") and torch.xpu.is_available()
    )

training_args = TrainingArguments(**training_kwargs)

# =========================================================
# 11. TRAINER
# =========================================================

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# =========================================================
# 12. TRAIN
# =========================================================

print("\nStarting supervised LiLT fine-tuning...\n")

trainer.train()

# =========================================================
# 13. SAVE MODEL AND TOKENIZER
# =========================================================

trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print("\nTraining completed.")
print("Model saved at:", OUTPUT_DIR)

# =========================================================
# 14. EVALUATE ON HELD-OUT TEST SET
# =========================================================

print("\nEvaluating on test invoices...")

test_results = trainer.evaluate(
    eval_dataset=tokenized_datasets["test"],
    metric_key_prefix="test",
)

print("\nTest results:")
for metric, value in test_results.items():
    print(f"{metric}: {value}")

with open(
    OUTPUT_DIR / "test_results.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(test_results, f, indent=2)

print("\nAll done.")
