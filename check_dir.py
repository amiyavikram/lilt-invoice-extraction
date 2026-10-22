from pathlib import Path

PROJECT_DIR = Path(r"C:\Internship\LiLT")
DATA_DIR = PROJECT_DIR / "indian_synthetic_invoice_dataset_200"

print("Searching for training files...\n")

for file in DATA_DIR.rglob("train.jsonl"):
    print("Found training file:", file)

print("\nFiles in dataset folder:")
for item in DATA_DIR.iterdir():
    print(item.name)