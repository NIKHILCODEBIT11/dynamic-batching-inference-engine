from locust import HttpUser, task, between
import random

SAMPLE_TEXTS = [
    "Machine learning dynamic batching inference optimization.",
    "FastAPI non-blocking async architecture benchmark.",
    "ONNX Runtime INT8 quantization enables faster token processing.",
    "Evaluating p95 and p99 latency under heavy concurrent loads.",
    "High throughput vector embedding serving for enterprise search."
]

class EmbeddingUser(HttpUser):
    # Har virtual user request ke beech 10ms se 50ms ka gap rakhega
    wait_time = between(0.01, 0.05)

    @task
    def test_embed_endpoint(self):
        payload = {"text": random.choice(SAMPLE_TEXTS)}
        headers = {"Content-Type": "application/json"}
        with self.client.post("/embed", json=payload, headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status: {response.status_code}")