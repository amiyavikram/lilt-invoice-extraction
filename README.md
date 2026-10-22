# LiLT invoice pipeline: image → OCR → model → extracted fields

This starter pipeline uses PaddleOCR to read a real invoice image, converts OCR boxes to LiLT's 0–1000 coordinates, runs your saved token-classification model, and writes JSON/debug outputs.

## Paths for your setup

- Model: `C:\Internship\LiLT\lilt_invoice_model\final_model`
- Example invoice path: `C:\Internship\LiLT\sample_invoices\example_invoice.png`
- Output folder: `C:\Internship\LiLT\pipeline_output`

Put a real invoice image at the example path, or pass another image using `--invoice`.

## Install

In VS Code PowerShell:

```powershell
cd C:\Internship\LiLT
.\env\Scripts\python.exe -m pip install paddleocr paddlepaddle opencv-python pillow
```

If PaddlePaddle installation fails for your Python version, follow the official PaddleOCR installation instructions. The rest of the script can also run with `--ocr-json` using precomputed OCR words/boxes.

## Run

```powershell
.\env\Scripts\python.exe .\pipeline.py
```

Or:

```powershell
.\env\Scripts\python.exe .\pipeline.py --invoice "C:\Internship\LiLT\invoices\invoice_001.jpg"
```

Outputs:
- `ocr_words.json`: OCR words and boxes
- `prediction.json`: word labels and merged fields
- `annotated_invoice.jpg`: image annotated with predicted labels

## Important

The script assumes the tokenizer saved with your model supports `word_ids()` and that your model labels use BIO names like `B-BUYER`, `I-BUYER`, `O`. Adjust `FIELD_ALIASES` in the script if your actual labels differ. Your synthetic test scores do not guarantee performance on real invoices; test with invoices from unseen vendors and verify extracted values manually.

For PDFs, convert each page to PNG first (for example with `pdf2image` and Poppler), then process each page image separately.
