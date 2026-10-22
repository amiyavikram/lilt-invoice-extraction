# Synthetic Indian Invoice Dataset — 200 documents

Contains 200 synthetic Indian B2B/GST-style invoice PNGs across eight business archetypes: FMCG/wholesale, manufacturing/electrical, pharma/medical, construction/materials, IT/professional services, logistics/transport, textiles/apparel, and agriculture.

Files:
- images/: rendered PNG invoices
- annotations/: word boxes, labels, structured fields, line items
- ocr_json/: compact OCR-style word/box/label JSON
- dataset.jsonl: all records
- train.jsonl, validation.jsonl, test.jsonl: 140/30/30 split
- splits.json, label_map.json, manifest.csv

Important:
- All names, transactions, values and GSTIN-like strings are fictional. GSTIN-like strings are placeholders and are not guaranteed valid.
- No real company invoice templates are copied.
- Word boxes and labels are generated from known renderer positions; no OCR engine was run. This is clean synthetic ground truth, not noisy OCR predictions.
- Before LiLT training, verify checkpoint/tokenizer requirements and align word-level boxes/labels to tokenizer subwords.
- The provided random document split can share layout families. For stronger generalization evaluation, reserve a whole layout family or template for testing.
- This is a research starter dataset, not a GST compliance dataset.
