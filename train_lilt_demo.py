
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    LayoutLMv3TokenizerFast,
    LiltForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"C:\Internship\LiLT")

MODEL_NAME = "SCUT-DLVCLab/lilt-roberta-en-base"

MAX_LENGTH = 512
NUM_EPOCHS = 5
BATCH_SIZE = 2
LEARNING_RATE = 2e-5

DATASET_DIR = None
OUTPUT_DIR = PROJECT_DIR / "lilt_invoice_model"


# ============================================================
# 2. FIND THE DATASET DIRECTORY
# ============================================================

def find_dataset_dir():
    """
    Finds the directory containing train.jsonl,
    validation.jsonl, test.jsonl and label_map.json.
    """

    candidates = [PROJECT_DIR]

    candidates.extend(
        p for p in PROJECT_DIR.iterdir() if p.is_dir()
    )

    for parent in list(candidates):
        if parent.is_dir():
            candidates.extend(
                p for p in parent.iterdir() if p.is_dir()
            )

    for folder in candidates:
        required = [
            "train.jsonl",
            "validation.jsonl",
            "test.jsonl",
            "label_map.json",
        ]

        if all((folder / name).is_file() for name in required):
            return folder

    raise FileNotFoundError(
        "Could not find the dataset files.\n"
        "Expected train.jsonl, validation.jsonl, "
        "test.jsonl and label_map.json in the same folder.\n"
        "Set DATASET_DIR manually near the top of this script."
    )


if DATASET_DIR is None:
    DATASET_DIR = find_dataset_dir()
else:
    DATASET_DIR = Path(DATASET_DIR)

print(f"\nDataset directory: {DATASET_DIR}")


# ============================================================
# 3. LOAD THE LABEL MAP
# ============================================================

with open(
    DATASET_DIR / "label_map.json",
    "r",
    encoding="utf-8",
) as file:
    label_map = json.load(file)


# Expected format:
# {
#     "O": 0,
#     "B-INVOICE_NO": 1,
#     "I-INVOICE_NO": 2
# }

if not isinstance(label_map, dict) or not label_map:
    raise ValueError("label_map.json must contain a non-empty JSON object.")

# Ensure that labels map to integer IDs.
label_map = {
    str(label): int(label_id)
    for label, label_id in label_map.items()
}

label_ids = sorted(label_map.values())

if label_ids != list(range(len(label_ids))):
    raise ValueError(
        "Label IDs must be consecutive integers starting at 0. "
        "Check label_map.json."
    )

id2label = {
    label_id: label
    for label, label_id in label_map.items()
}

num_labels = len(label_map)

print(f"Number of labels: {num_labels}")
print(f"Labels: {label_map}")


# ============================================================
# 4. LOAD JSONL AND CONVERT LABELS BEFORE ARROW CREATION
# ============================================================

def load_jsonl(file_path):
    """
    Reads raw JSONL records before constructing a Hugging Face
    Dataset. Converts string labels into integer IDs first,
    avoiding PyArrow's string-versus-integer inference error.
    """

    records = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            example = json.loads(line)

            # Support common field-name variants.
            if "words" not in example:
                raise KeyError(
                    f"'words' is missing in {file_path}, "
                    f"line {line_number}. Available fields: "
                    f"{list(example.keys())}"
                )

            if "labels" not in example:
                raise KeyError(
                    f"'labels' is missing in {file_path}, "
                    f"line {line_number}."
                )

            if "boxes_normalized_0_1000" in example:
                boxes = example["boxes_normalized_0_1000"]
            elif "boxes" in example:
                boxes = example["boxes"]
            elif "bboxes" in example:
                boxes = example["bboxes"]
            else:
                raise KeyError(
                    f"No bounding-box field found in {file_path}, "
                    f"line {line_number}. Available fields: "
                    f"{list(example.keys())}"
                )

            words = example["words"]
            labels = example["labels"]

            if not (
                len(words) == len(boxes) == len(labels)
            ):
                raise ValueError(
                    f"Word, box and label counts differ in "
                    f"{file_path}, line {line_number}: "
                    f"{len(words)} words, {len(boxes)} boxes, "
                    f"{len(labels)} labels."
                )

            integer_labels = []

            for label in labels:
                if isinstance(label, str):
                    if label not in label_map:
                        raise ValueError(
                            f"Unknown label {label!r} in "
                            f"{file_path}, line {line_number}. "
                            f"Check label_map.json."
                        )
                    integer_labels.append(label_map[label])
                else:
                    label_id = int(label)

                    if label_id not in id2label:
                        raise ValueError(
                            f"Unknown label ID {label_id} in "
                            f"{file_path}, line {line_number}."
                        )

                    integer_labels.append(label_id)

            # Normalize bounding-box field name.
            example["words"] = words
            example["boxes_normalized_0_1000"] = [
                [int(coord) for coord in box]
                for box in boxes
            ]
            example["labels"] = integer_labels

            # Validate each box.
            for box in example["boxes_normalized_0_1000"]:
                if len(box) != 4:
                    raise ValueError(
                        f"Every bounding box must have four "
                        f"coordinates. Found: {box}"
                    )

                if any(coord < 0 or coord > 1000 for coord in box):
                    raise ValueError(
                        f"Bounding-box coordinates must be "
                        f"normalized to 0-1000. Found: {box}"
                    )

                if box[2] < box[0] or box[3] < box[1]:
                    raise ValueError(
                        f"Invalid bounding box: {box}"
                    )

            records.append(example)

    if not records:
        raise ValueError(f"No examples found in {file_path}")

    print(f"Loaded {len(records)} examples from {file_path.name}")

    return records


train_records = load_jsonl(DATASET_DIR / "train.jsonl")
validation_records = load_jsonl(DATASET_DIR / "validation.jsonl")
test_records = load_jsonl(DATASET_DIR / "test.jsonl")


# IMPORTANT:
# Labels are already integers BEFORE Dataset.from_list().
# This is the fix for the PyArrow ArrowInvalid error.

train_dataset = Dataset.from_list(train_records)
validation_dataset = Dataset.from_list(validation_records)
test_dataset = Dataset.from_list(test_records)


# ============================================================
# 5. LOAD THE TOKENIZER
# ============================================================

tokenizer = LayoutLMv3TokenizerFast.from_pretrained(MODEL_NAME)

print(f"\nTokenizer class: {type(tokenizer)}")
print(f"Tokenizer is fast: {tokenizer.is_fast}")


# ============================================================
# 6. TOKENIZE WORDS, BOXES AND LABELS
# ============================================================

def prepare_example(example):
    words = example["words"]
    boxes = example["boxes_normalized_0_1000"]
    word_labels = example["labels"]

    # Do not pass is_split_into_words=True here.
    # The layout-aware tokenizer receives the boxes alongside
    # the words and labels.
    encoding = tokenizer(
        words,
        boxes=boxes,
        word_labels=word_labels,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

    return dict(encoding)


print("\nTokenizing training data...")

train_dataset = train_dataset.map(
    prepare_example,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing training data",
)

validation_dataset = validation_dataset.map(
    prepare_example,
    remove_columns=validation_dataset.column_names,
    desc="Tokenizing validation data",
)

test_dataset = test_dataset.map(
    prepare_example,
    remove_columns=test_dataset.column_names,
    desc="Tokenizing test data",
)


# ============================================================
# 7. LOAD THE LILT MODEL
# ============================================================

model = LiltForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label_map,
)

# Intel GPU support through PyTorch XPU, if available.
if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
    print("\nIntel GPU detected: using XPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("\nCUDA GPU detected: using CUDA")
else:
    device = torch.device("cpu")
    print("\nNo supported GPU detected: using CPU")

model.to(device)


# ============================================================
# 8. TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    dataloader_pin_memory=False,
)


# ============================================================
# 9. DATA COLLATOR
# ============================================================

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding="max_length",
    max_length=MAX_LENGTH,
)


# ============================================================
# 10. TRAIN
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
)

print("\nStarting LiLT training...\n")

trainer.train()


# ============================================================
# 11. SAVE MODEL AND TOKENIZER
# ============================================================

final_model_dir = OUTPUT_DIR / "final_model"
final_model_dir.mkdir(parents=True, exist_ok=True)

trainer.save_model(str(final_model_dir))
tokenizer.save_pretrained(str(final_model_dir))

print(f"\nModel saved to: {final_model_dir}")


# ============================================================
# 12. EVALUATE
# ============================================================

print("\nEvaluating on the validation dataset...")

validation_metrics = trainer.evaluate(
    eval_dataset=validation_dataset
)

print("Validation metrics:")
print(validation_metrics)

print("\nEvaluating on the test dataset...")

test_metrics = trainer.evaluate(
    eval_dataset=test_dataset,
    metric_key_prefix="test",
)

print("Test metrics:")
print(test_metrics)

with open(
    OUTPUT_DIR / "evaluation_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            "validation": validation_metrics,
            "test": test_metrics,
        },
        file,
        indent=2,
    )

print("\nTraining and evaluation completed.")
