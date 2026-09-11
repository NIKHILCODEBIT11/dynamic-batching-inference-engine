import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

class ONNXEmbeddingEngine:
    def __init__(self, model_dir: str = "models"):
        model_path = os.path.join(model_dir, "model_int8.onnx")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # ONNX Runtime CPU multi-threading configuration
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path, 
            sess_options, 
            providers=["CPUExecutionProvider"]
        )

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
            
        # 1. Tokenize batched input texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="np"
        )
        
        ort_inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
            "token_type_ids": encoded.get("token_type_ids", np.zeros_like(encoded["input_ids"])).astype(np.int64)
        }
        
        # 2. Run batched inference via ONNX Runtime
        outputs = self.session.run(["last_hidden_state"], ort_inputs)
        last_hidden_state = outputs[0]  # Shape: (batch_size, seq_len, 384)
        
        # 3. Mean Pooling: Average token embeddings to get 1 vector per sentence
        input_mask_expanded = np.expand_dims(encoded["attention_mask"], -1).astype(float)
        sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        mean_pooled = sum_embeddings / sum_mask
        
        # 4. Normalize embeddings (Cosine distance ready)
        norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        normalized_embeddings = mean_pooled / np.clip(norms, a_min=1e-9, a_max=None)
        
        return normalized_embeddings.tolist()