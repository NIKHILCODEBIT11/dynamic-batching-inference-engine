import asyncio
import time
from dataclasses import dataclass
from app.model_engine import ONNXEmbeddingEngine

@dataclass
class QueueItem:
    text: str
    future: asyncio.Future

class DynamicBatcher:
    def __init__(
        self,
        engine: ONNXEmbeddingEngine,
        max_batch_size: int = 16,
        max_wait_time_ms: float = 10.0
    ):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time_ms / 1000.0  # Seconds
        self.queue: asyncio.Queue[QueueItem] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._running = False

    def start(self):
        self._running = True
        self._worker_task = asyncio.create_task(self._process_queue())

    async def stop(self):
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def enqueue(self, text: str) -> list[float]:
        # Har request ko ek Future (ticket) assign karo
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self.queue.put(QueueItem(text=text, future=future))
        return await future

    async def _process_queue(self):
        while self._running:
            # 1. First request aane tak wait karo
            item = await self.queue.get()
            batch: list[QueueItem] = [item]
            start_time = time.monotonic()

            # 2. Max 10ms wait karo ya jab tak 16 requests na aa jayein
            while len(batch) < self.max_batch_size:
                elapsed = time.monotonic() - start_time
                remaining_time = self.max_wait_time - elapsed
                if remaining_time <= 0:
                    break
                try:
                    next_item = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=remaining_time
                    )
                    batch.append(next_item)
                except asyncio.TimeoutError:
                    break

            # 3. Model ko batch bhejo alag thread pool mein (non-blocking)
            texts = [b.text for b in batch]
            try:
                loop = asyncio.get_running_loop()
                embeddings = await loop.run_in_executor(None, self.engine.encode_batch, texts)
                
                # 4. Har request ko uska result de do
                for b, emb in zip(batch, embeddings):
                    if not b.future.done():
                        b.future.set_result(emb)
            except Exception as exc:
                for b in batch:
                    if not b.future.done():
                        b.future.set_exception(exc)