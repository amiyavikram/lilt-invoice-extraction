from pathlib import Path

PROJECT_DIR = Path(r"C:\Internship\LiLT")
DATA_DIR = PROJECT_DIR / "indian_synthetic_invoice_dataset_200"

print("Dataset exists:", DATA_DIR.exists())
print("Training file exists:", (DATA_DIR / "train.jsonl").exists())