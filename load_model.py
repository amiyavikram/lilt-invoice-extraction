
  from transformers import (
      LayoutLMv3TokenizerFast,
      LiltForTokenClassification,
  )
  
  MODEL_PATH = r"C:\Internship\LiLT\lilt_invoice_model\final_model"
  
  tokenizer = LayoutLMv3TokenizerFast.from_pretrained(MODEL_PATH)
  
  model = LiltForTokenClassification.from_pretrained(MODEL_PATH)
  
  model.eval()
  
  print("Trained LiLT model loaded successfully!")
  print("Labels:", model.config.id2label)
