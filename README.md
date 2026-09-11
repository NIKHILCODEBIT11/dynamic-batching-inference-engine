# High-Throughput Dynamic Batching ML Inference Engine

A high-performance, asynchronous embedding inference service built with **FastAPI**, **ONNX Runtime (INT8)**, and an in-memory **Dynamic Batching Queue**. Designed to maximize CPU hardware utilization and prevent thread blocking under heavy concurrent request loads.

## Architecture

```text
[Concurrent HTTP Requests] 
            │
            ▼
┌─────────────────────────┐
│     FastAPI Gateway     │ (Async/Non-blocking endpoint)
└───────────┬─────────────┘
            │ Enqueue (Request + Future ticket)
            ▼
┌─────────────────────────┐
│ Dynamic Batching Worker │ ── Wait max 10ms OR Batch Size == 16
└───────────┬─────────────┘
            │ Stack into batched 2D tensor
            ▼
┌─────────────────────────┐
│   ONNX Runtime (INT8)   │ (Quantized all-MiniLM-L6-v2)
└───────────┬─────────────┘
            │ Resolve individual futures
            ▼
[Return 384-d normalized vectors to respective clients]
```

## Performance & Load Benchmark (Locust)

The system was benchmarked using Locust with **50 concurrent virtual users** executing continuous embedding queries:

| Metric | Measured Value |
| :--- | :--- |
| **Total Requests Processed** | 2,144 |
| **Failure Rate** | 0.0% |
| **Peak Throughput** | ~25 RPS (CPU Execution Provider) |
| **Median Latency (p50)** | 1.30 s |
| **95th Percentile Latency (p95)** | 4.10 s |
| **Model Footprint** | Quantized INT8 (~60% reduction vs FP32) |

## Key Technical Highlights

- **Dynamic Batch Aggregation:** Batches incoming single requests up to `max_batch_size=16` within a `10ms` micro-window, minimizing redundant forward passes.
- **Asynchronous Decoupling:** Employs `asyncio.Future` pattern to keep the FastAPI web loop unblocked while computing embeddings on a worker threadpool via `run_in_executor`.
- **Inference Optimization:** Converted Hugging Face transformer model to ONNX with INT8 dynamic quantization and dynamic axes support.

## Setup & Run

1. Clone repository and set up virtual environment:
   python -m venv venv
   source venv/bin/activate # On Windows: venv\Scripts\activate
   pip install -r requirements.txt

2. Export and quantize ONNX model:
   python scripts/export_onnx.py

3. Launch the API server:
   uvicorn app.main:app --host 0.0.0.0 --port 8000

4. Run the Locust load benchmark:
   locust -f scripts/benchmark_locust.py --host http://127.0.0.1:8000