from pathlib import Path
import argparse, json
import cv2
import numpy as np
import torch
from PIL import Image
from paddleocr import PaddleOCR
from transformers import LayoutLMv3TokenizerFast, LiltForTokenClassification

MODEL_PATH = Path(r"C:\Internship\LiLT\lilt_invoice_model\final_model")
INVOICE_PATH = Path(r"C:\Internship\LiLT\sample_invoices\example_invoice.png")
OUTPUT_DIR = Path(r"C:\Internship\LiLT\pipeline_output")
MAX_LENGTH = 512

# Edit these names to match the labels in your model.config.id2label.
FIELD_ALIASES = {
    "INVOICE_NO": {"INVOICE_NO", "INVOICE_NUMBER", "INV_NO", "INVOICE"},
    "INVOICE_DATE": {"INVOICE_DATE", "DATE"},
    "BUYER": {"BUYER", "BUYER_NAME", "BILL_TO", "CUSTOMER"},
    "SELLER": {"SELLER", "SELLER_NAME", "VENDOR", "SUPPLIER"},
    "TOTAL": {"TOTAL", "GRAND_TOTAL", "AMOUNT_DUE", "INVOICE_TOTAL"},
    "SUBTOTAL": {"SUBTOTAL", "SUB_TOTAL"},
    "TAX": {"TAX", "GST", "GST_AMOUNT", "TAX_AMOUNT"},
    "GSTIN": {"GSTIN", "GST_NUMBER", "GSTIN_NO"},
    "PO_NUMBER": {"PO_NUMBER", "PURCHASE_ORDER", "PO_NO"},
}

def pixel_box(poly):
    arr = np.asarray(poly, dtype=float).reshape(-1, 2)
    return [int(arr[:, 0].min()), int(arr[:, 1].min()),
            int(arr[:, 0].max()), int(arr[:, 1].max())]

def norm_box(box, width, height):
    x0, y0, x1, y1 = box
    return [max(0, min(1000, int(1000*x0/max(width, 1)))),
            max(0, min(1000, int(1000*y0/max(height, 1)))),
            max(0, min(1000, int(1000*x1/max(width, 1)))),
            max(0, min(1000, int(1000*y1/max(height, 1))))]

def run_ocr(image_path):
    # PaddleOCR 3.x API. If your installed version is 2.x, see README troubleshooting.
    ocr = PaddleOCR(lang="en", use_textline_orientation=True,enable_mkldnn=False)
    results = ocr.predict(str(image_path))
    words, boxes, details = [], [], []
    for page in results or []:
        data = getattr(page, "json", page)
        if callable(data):
            data = data()
        if isinstance(data, dict) and "res" in data:
            data = data["res"]
        if isinstance(data, dict) and "rec_texts" in data:
            texts = data.get("rec_texts") or []
            polys = data.get("rec_polys", data.get("dt_polys", [])) or []
            scores = data.get("rec_scores") or []
            for i, (txt, poly) in enumerate(zip(texts, polys)):
                txt = str(txt).strip()
                if not txt:
                    continue
                box = pixel_box(poly)
                words.append(txt); boxes.append(box)
                details.append({"text": txt, "box_pixels": box,
                                "ocr_confidence": float(scores[i]) if i < len(scores) else None})
        elif isinstance(page, list):
            # PaddleOCR 2.x-style output fallback
            for line in page:
                try:
                    poly, rec = line[0], line[1]
                    txt, score = str(rec[0]).strip(), float(rec[1])
                    if txt:
                        box = pixel_box(poly)
                        words.append(txt); boxes.append(box)
                        details.append({"text": txt, "box_pixels": box, "ocr_confidence": score})
                except (TypeError, IndexError, ValueError):
                    continue
    if not words:
        raise RuntimeError("OCR returned no words. Check image/PaddleOCR version, or use --ocr-json.")
    return words, boxes, details

def split_label(label):
    label = str(label)
    if label == "O":
        return "O", "O"
    if "-" in label:
        prefix, entity = label.split("-", 1)
        if prefix in {"B", "I", "E", "S"}:
            return prefix, entity.upper()
    return "B", label.upper()

def output_field(entity):
    for field, aliases in FIELD_ALIASES.items():
        if entity.upper() in aliases:
            return field
    return None

def predict_words(words, boxes, tokenizer, model, device):
    # Match the same input style as the training pipeline: words + normalized boxes.
    enc = tokenizer(words, boxes=boxes, truncation=True, max_length=MAX_LENGTH,
                    padding="max_length", return_tensors="pt")
    try:
        word_ids = enc.word_ids(batch_index=0)
    except Exception as e:
        raise RuntimeError(
            "Tokenizer has no word_ids() alignment. Use the same compatible tokenizer "
            "that was used during training and saved alongside the model."
        ) from e
    inputs = {k: v.to(device) for k, v in enc.items()
              if k in {"input_ids", "attention_mask", "token_type_ids", "bbox"}}
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(-1).tolist()
        confs = probabilities.max(-1).values.tolist()
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    output, seen = [], set()
    for token_i, word_i in enumerate(word_ids):
        if word_i is None or word_i >= len(words) or word_i in seen:
            continue
        seen.add(word_i)
        label_id = int(pred_ids[token_i])
        output.append({
            "word_index": int(word_i), "text": words[word_i], "box_pixels": boxes[word_i],
            "label": id2label.get(label_id, str(label_id)),
            "confidence": round(float(confs[token_i]), 4)
        })
    return output

def merge_entities(predictions):
    candidates = {}
    entity, span = None, []
    def flush():
        nonlocal entity, span
        if entity and span:
            key = output_field(entity)
            if key:
                candidates.setdefault(key, []).append({
                    "value": " ".join(x["text"] for x in span),
                    "confidence": round(sum(x["confidence"] for x in span)/len(span), 4)
                })
        entity, span = None, []
    for item in predictions:
        prefix, ent = split_label(item["label"])
        if prefix == "O":
            flush(); continue
        if prefix in {"B", "S"} or (entity is not None and ent != entity):
            flush(); entity, span = ent, [item]
        elif prefix in {"I", "E"} and entity == ent:
            span.append(item)
            if prefix == "E":
                flush()
        else:
            flush(); entity, span = ent, [item]
    flush()
    fields = {k: (v[0]["value"] if len(v) == 1 else [x["value"] for x in v])
              for k, v in candidates.items()}
    return fields, candidates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--invoice", type=Path, default=INVOICE_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--ocr-json", type=Path, help="JSON with words, boxes_pixels, and image_size")
    args = parser.parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model folder not found: {args.model}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.ocr_json:
        raw = json.loads(args.ocr_json.read_text(encoding="utf-8"))
        words, pixel_boxes = raw["words"], raw["boxes_pixels"]
        width, height = int(raw["image_size"]["width"]), int(raw["image_size"]["height"])
        ocr_details = raw.get("ocr_details", [])
        image_path = None
    else:
        if not args.invoice.exists():
            raise FileNotFoundError(
                f"Invoice not found: {args.invoice}. Put an image at this path or use --invoice."
            )
        image_path = args.invoice
        with Image.open(image_path) as im:
            width, height = im.size
        words, pixel_boxes, ocr_details = run_ocr(image_path)

    boxes = [norm_box(b, width, height) for b in pixel_boxes]
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(str(args.model))
    model = LiltForTokenClassification.from_pretrained(str(args.model))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    predictions = predict_words(words, boxes, tokenizer, model, device)
    fields, candidates = merge_entities(predictions)

    (args.output_dir / "ocr_words.json").write_text(json.dumps({
        "image_size": {"width": width, "height": height},
        "words": words, "boxes_pixels": pixel_boxes,
        "boxes_normalized_0_1000": boxes, "ocr_details": ocr_details
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {
        "invoice": str(image_path) if image_path else None,
        "model": str(args.model), "device": str(device),
        "fields": fields, "field_candidates": candidates,
        "word_predictions": predictions,
        "warning": "Review values against the original invoice. Token confidence is not calibrated field accuracy."
    }
    (args.output_dir / "prediction.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if image_path:
        image = cv2.imread(str(image_path))
        if image is not None:
            for p in predictions:
                x0, y0, x1, y1 = p["box_pixels"]
                cv2.rectangle(image, (x0, y0), (x1, y1), (0, 180, 0), 1)
                cv2.putText(image, f'{p["label"]}: {p["text"]}'[:80],
                            (x0, max(15, y0-3)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(args.output_dir / "annotated_invoice.jpg"), image)

    print("Extracted fields:")
    print(json.dumps(fields, ensure_ascii=False, indent=2))
    print(f"Saved results in: {args.output_dir}")

if __name__ == "__main__":
    main()
