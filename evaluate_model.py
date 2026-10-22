
import json
from pathlib import Path

import torch
from datasets import Dataset
from seqeval.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from transformers import (
    LayoutLMv3TokenizerFast,
    LiltForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

PROJECT_DIR = Path(r"C:\Internship\LiLT")
MODEL_PATH = PROJECT_DIR / "lilt_invoice_model" / "final_model"

# Automatically find the folder containing test.jsonl.
matches = list(PROJECT_DIR.glob("**/test.jsonl"))
matches = [
    p for p in matches
    if "lilt_invoice_model" not in str(p)
    and "env" not in p.parts
]

if not matches:
    raise FileNotFoundError(
        "Cannot find test.jsonl. Set DATASET_DIR manually."
    )

# If multiple matches appear, select your actual dataset folder.
DATASET_DIR = matches[0].parent

print("Model:", MODEL_PATH)
print("Dataset:", DATASET_DIR)

if not (MODEL_PATH / "config.json").exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")


# --------------------------------------------------
# 2. LOAD LABEL MAP AND MODEL
# --------------------------------------------------

with open(
    DATASET_DIR / "label_map.json",
    "r",
    encoding="utf-8",
) as f:
    label_map = json.load(f)

label_map = {
    str(label): int(label_id)
    for label, label_id in label_map.items()
}

id2label = {
    label_id: label
    for label, label_id in label_map.items()
}

tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
    str(MODEL_PATH)
)

model = LiltForTokenClassification.from_pretrained(
    str(MODEL_PATH)
)

model.eval()

print("Loaded saved model successfully.")
print("Labels:", id2label)


# --------------------------------------------------
# 3. LOAD TEST DATA AND CONVERT LABELS
# --------------------------------------------------

test_records = []

with open(
    DATASET_DIR / "test.jsonl",
    "r",
    encoding="utf-8",
) as f:
    for line_number, line in enumerate(f, start=1):
        if not line.strip():
            continue

        example = json.loads(line)

        words = example["words"]

        if "boxes_normalized_0_1000" in example:
            boxes = example["boxes_normalized_0_1000"]
        elif "boxes" in example:
            boxes = example["boxes"]
        elif "bboxes" in example:
            boxes = example["bboxes"]
        else:
            raise KeyError(
                f"Missing boxes field on test line {line_number}"
            )

        labels = example["labels"]

        if not (len(words) == len(boxes) == len(labels)):
            raise ValueError(
                f"Words, boxes and labels differ on line {line_number}"
            )

        integer_labels = []

        for label in labels:
            if isinstance(label, str):
                if label not in label_map:
                    raise ValueError(f"Unknown label: {label}")
                integer_labels.append(label_map[label])
            else:
                integer_labels.append(int(label))

        test_records.append({
            "words": words,
            "boxes_normalized_0_1000": boxes,
            "labels": integer_labels,
        })

print("Test examples:", len(test_records))

if not test_records:
    raise ValueError("The test dataset is empty.")


# --------------------------------------------------
# 4. TOKENIZE TEST DATA
# --------------------------------------------------

def prepare_example(example):
    encoding = tokenizer(
        example["words"],
        boxes=example["boxes_normalized_0_1000"],
        word_labels=example["labels"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

    return dict(encoding)


test_dataset = Dataset.from_list(test_records)

test_dataset = test_dataset.map(
    prepare_example,
    remove_columns=test_dataset.column_names,
    desc="Preparing test data",
)


# --------------------------------------------------
# 5. EVALUATE SAVED MODEL
# --------------------------------------------------

args = TrainingArguments(
    output_dir=str(PROJECT_DIR / "evaluation_output"),
    per_device_eval_batch_size=2,
    report_to="none",
    dataloader_pin_memory=False,
)

collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding="max_length",
    max_length=512,
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=collator,
)

print("\nCalculating test loss and predictions...")

results = trainer.evaluate(
    eval_dataset=test_dataset,
    metric_key_prefix="test",
)

print("\nTest loss and runtime metrics:")
for key, value in results.items():
    print(f"{key}: {value}")


# --------------------------------------------------
# 6. CALCULATE ENTITY-LEVEL METRICS
# --------------------------------------------------

predictions, labels, _ = trainer.predict(test_dataset)

predicted_ids = predictions.argmax(axis=-1)

true_sequences = []
predicted_sequences = []

for pred_row, label_row in zip(predicted_ids, labels):
    true_tags = []
    pred_tags = []

    for pred_id, true_id in zip(pred_row, label_row):
        # Ignore special tokens, padding and ignored subwords.
        if true_id == -100:
            continue

        true_tags.append(id2label[int(true_id)])
        pred_tags.append(id2label[int(pred_id)])

    true_sequences.append(true_tags)
    predicted_sequences.append(pred_tags)

precision = precision_score(
    true_sequences, predicted_sequences, zero_division=0
)
recall = recall_score(
    true_sequences, predicted_sequences, zero_division=0
)
f1 = f1_score(
    true_sequences, predicted_sequences, zero_division=0
)

print("\n" + "=" * 55)
print("OVERALL ENTITY-LEVEL METRICS")
print("=" * 55)
print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1-score:  {f1:.4f} ({f1 * 100:.2f}%)")

print("\nPER-ENTITY CLASSIFICATION REPORT")
print(
    classification_report(
        true_sequences,
        predicted_sequences,
        zero_division=0,
    )
)


# --------------------------------------------------
# 7. SAVE RESULTS
# --------------------------------------------------

report = classification_report(
    true_sequences,
    predicted_sequences,
    zero_division=0,
    output_dict=True,
)

output = {
    "test_loss_and_runtime": results,
    "entity_precision": precision,
    "entity_recall": recall,
    "entity_f1": f1,
    "classification_report": report,
    "test_examples": len(test_records),
}

output_path = PROJECT_DIR / "actual_test_metrics.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"\nMetrics saved to: {output_path}")
