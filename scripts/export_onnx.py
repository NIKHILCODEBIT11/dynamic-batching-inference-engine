import os
import torch
from transformers import AutoTokenizer, AutoModel
from onnxruntime.quantization import quantize_dynamic, QuantType

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ONNX_FP32_PATH = os.path.join(OUTPUT_DIR, "model_fp32.onnx")
ONNX_INT8_PATH = os.path.join(OUTPUT_DIR, "model_int8.onnx")

print("1. HuggingFace se base model download ho raha hai...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)
model.eval()

# Dummy input taaki ONNX graph samajh sake
dummy_text = ["This is a sample sentence for dynamic batching tracing."]
inputs = tokenizer(
    dummy_text,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

print("2. ONNX format mein export ho raha hai (FP32)...")
torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
    ONNX_FP32_PATH,
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "token_type_ids": {0: "batch_size", 1: "sequence_length"},
        "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
    },
    opset_version=14
)

print("3. Model ko INT8 format mein compress (quantize) kar rahe hain...")
quantize_dynamic(
    model_input=ONNX_FP32_PATH,
    model_output=ONNX_INT8_PATH,
    weight_type=QuantType.QInt8
)

tokenizer.save_pretrained(OUTPUT_DIR)
print(f"SUCCESS: Quantized model ready at {ONNX_INT8_PATH}")