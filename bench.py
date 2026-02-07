#!/usr/bin/env python3
"""
vLLM Benchmark CLI — stress-test an OpenAI-compatible server with parallel requests.
"""

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import dataclass, field

try:
    import aiohttp
except ImportError:
    print("Requires aiohttp: pip install aiohttp")
    sys.exit(1)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ---------------------------------------------------------------------------
# Prompt pools (grouped by approximate output size)
# ---------------------------------------------------------------------------

SMALL_PROMPTS = [
    "Qual è la capitale della Francia?",
    "What is 2 + 2?",
    "Name three primary colors.",
    "Who wrote Romeo and Juliet?",
    "What is the boiling point of water in Celsius?",
    "Translate 'hello' to Spanish.",
    "What planet is closest to the Sun?",
    "How many continents are there?",
    "What is the chemical symbol for gold?",
    "Name the largest ocean on Earth.",
    "What year did the Titanic sink?",
    "What is the square root of 144?",
    "Who painted the Mona Lisa?",
    "What is the currency of Japan?",
    "How many legs does a spider have?",
]

MEDIUM_PROMPTS = [
    "Explain the difference between TCP and UDP in simple terms.",
    "Write a short paragraph about the history of the Internet.",
    "Describe how a neural network learns, step by step.",
    "What are the main differences between Python and JavaScript?",
    "Summarize the plot of '1984' by George Orwell in a few sentences.",
    "Explain what a hash table is and why it is useful.",
    "Describe the water cycle in detail.",
    "What are the pros and cons of remote work?",
    "Explain the concept of recursion with a simple example.",
    "Describe the main causes of climate change.",
]

LARGE_PROMPTS = [
    "Write a detailed tutorial on how to build a REST API with Python and Flask, including code examples.",
    "Explain the theory of relativity in depth, covering both special and general relativity with examples.",
    "Write a comprehensive comparison of SQL and NoSQL databases, including use cases, advantages, and disadvantages.",
    "Describe the complete lifecycle of a machine learning project from data collection to deployment.",
    "Write a detailed essay about the history and evolution of programming languages from the 1950s to today.",
    "Explain distributed systems concepts: CAP theorem, consensus algorithms, and partition tolerance with examples.",
    "Write a thorough guide to container orchestration with Kubernetes, covering pods, services, and deployments.",
    "Describe the architecture of a modern web application, from frontend to backend to infrastructure.",
]


def pick_prompts(n: int, size: str) -> list[str]:
    pool = {"small": SMALL_PROMPTS, "medium": MEDIUM_PROMPTS, "large": LARGE_PROMPTS}[size]
    return [random.choice(pool) for _ in range(n)]


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    prompt: str = ""
    status: str = "ok"
    ttft_ms: float = 0.0          # time-to-first-token
    total_time_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_sec: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> RequestResult:
    result = RequestResult(prompt=prompt)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }

    t0 = time.perf_counter()
    first_token_time = None

    try:
        async with session.post(url, json=body) as resp:
            if resp.status != 200:
                result.status = "error"
                result.error = f"HTTP {resp.status}: {await resp.text()}"
                result.total_time_s = time.perf_counter() - t0
                return result

            if stream:
                completion_text = ""
                last_chunk = None
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        last_chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    delta = last_chunk.get("choices", [{}])[0].get("delta", {})
                    token_text = delta.get("content") or ""
                    if token_text and first_token_time is None:
                        first_token_time = time.perf_counter()
                    completion_text += token_text

                result.total_time_s = time.perf_counter() - t0
                # rough token count (chars / 4 fallback)
                result.completion_tokens = max(1, len(completion_text) // 4)
                if first_token_time is not None:
                    result.ttft_ms = (first_token_time - t0) * 1000

                # try to get usage from last chunk
                if last_chunk:
                    usage = last_chunk.get("usage") or {}
                    if usage.get("completion_tokens"):
                        result.completion_tokens = usage["completion_tokens"]
                    if usage.get("prompt_tokens"):
                        result.prompt_tokens = usage["prompt_tokens"]
            else:
                data = await resp.json()
                result.total_time_s = time.perf_counter() - t0
                usage = data.get("usage", {})
                result.prompt_tokens = usage.get("prompt_tokens", 0)
                result.completion_tokens = usage.get("completion_tokens", 0)

            if result.completion_tokens and result.total_time_s > 0:
                result.tokens_per_sec = result.completion_tokens / result.total_time_s

    except Exception as e:
        result.status = "error"
        result.error = str(e)
        result.total_time_s = time.perf_counter() - t0

    return result


# ---------------------------------------------------------------------------
# RAM monitor
# ---------------------------------------------------------------------------

class RamMonitor:
    """Samples system RAM usage in a background thread."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.samples: list[float] = []  # used GB
        self._stop = False
        self._total_gb: float = 0.0

    def start(self):
        if not HAS_PSUTIL:
            return
        import threading
        self._stop = False
        self._total_gb = psutil.virtual_memory().total / (1024 ** 3)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True

    def _run(self):
        while not self._stop:
            mem = psutil.virtual_memory()
            self.samples.append(mem.used / (1024 ** 3))
            time.sleep(self.interval)

    def summary(self) -> dict:
        if not self.samples:
            return {}
        return {
            "total_gb": self._total_gb,
            "start_gb": self.samples[0],
            "peak_gb": max(self.samples),
            "avg_gb": sum(self.samples) / len(self.samples),
            "end_gb": self.samples[-1],
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_bench(args: argparse.Namespace):
    prompts = pick_prompts(args.requests, args.prompt_size)
    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    sem = asyncio.Semaphore(args.parallel)

    async def bounded(p: str) -> RequestResult:
        async with sem:
            return await send_request(
                session, url, args.model, p,
                args.max_tokens, args.temperature, args.stream,
            )

    print(f"\n{'='*60}")
    print(f" vLLM Bench — {args.requests} requests, {args.parallel} parallel")
    print(f" Server : {url}")
    print(f" Model  : {args.model}")
    print(f" Max tok: {args.max_tokens}  |  Prompt size: {args.prompt_size}")
    print(f" Stream : {args.stream}")
    print(f"{'='*60}\n")

    ram = RamMonitor()
    ram.start()

    results: list[RequestResult] = []
    t_start = time.perf_counter()

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        tasks = [asyncio.create_task(bounded(p)) for p in prompts]
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            r = await coro
            results.append(r)
            mark = "OK" if r.status == "ok" else "ERR"
            tps = f"{r.tokens_per_sec:6.1f} t/s" if r.tokens_per_sec else "  n/a  "
            ttft = f"{r.ttft_ms:6.0f}ms" if r.ttft_ms else "   n/a "
            print(f"  [{i:>{len(str(args.requests))}}/{args.requests}] "
                  f"{mark}  {tps}  TTFT {ttft}  "
                  f"({r.completion_tokens:>4} tok in {r.total_time_s:.2f}s)  "
                  f"{'| ' + r.error if r.error else ''}")

    wall = time.perf_counter() - t_start
    ram.stop()
    ram_info = ram.summary()

    ok = [r for r in results if r.status == "ok"]
    errs = len(results) - len(ok)

    if not ok:
        print("\nAll requests failed.")
        return

    total_comp = sum(r.completion_tokens for r in ok)
    total_prompt = sum(r.prompt_tokens for r in ok)
    avg_tps = sum(r.tokens_per_sec for r in ok) / len(ok)
    aggregate_tps = total_comp / wall if wall else 0
    avg_ttft = sum(r.ttft_ms for r in ok if r.ttft_ms) / max(1, sum(1 for r in ok if r.ttft_ms))
    p50 = sorted(r.tokens_per_sec for r in ok)[len(ok) // 2]
    fastest = max(r.tokens_per_sec for r in ok)
    slowest = min(r.tokens_per_sec for r in ok)

    print(f"\n{'='*60}")
    print(f" RESULTS")
    print(f"{'='*60}")
    print(f"  Wall time         : {wall:.2f}s")
    print(f"  Requests          : {len(ok)} ok / {errs} errors")
    print(f"  Prompt tokens     : {total_prompt}")
    print(f"  Completion tokens : {total_comp}")
    print(f"  ---")
    print(f"  Aggregate tok/s   : {aggregate_tps:.1f}  (total tokens / wall time)")
    print(f"  Avg tok/s per req : {avg_tps:.1f}")
    print(f"  Median tok/s      : {p50:.1f}")
    print(f"  Fastest           : {fastest:.1f} t/s")
    print(f"  Slowest           : {slowest:.1f} t/s")
    if avg_ttft:
        print(f"  Avg TTFT          : {avg_ttft:.0f}ms")
    if ram_info:
        print(f"  ---")
        print(f"  RAM total         : {ram_info['total_gb']:.1f} GB")
        print(f"  RAM start         : {ram_info['start_gb']:.1f} GB")
        print(f"  RAM peak          : {ram_info['peak_gb']:.1f} GB")
        print(f"  RAM end           : {ram_info['end_gb']:.1f} GB")
        print(f"  RAM avg           : {ram_info['avg_gb']:.1f} GB")
    elif not HAS_PSUTIL:
        print(f"  ---")
        print(f"  RAM               : install psutil for RAM monitoring")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Benchmark an OpenAI-compatible vLLM server under parallel load.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s                                  # 4 parallel, 8 requests, small prompts
  %(prog)s -p 8 -n 20 --prompt-size medium  # 8 parallel, 20 requests, medium prompts
  %(prog)s -p 16 -n 50 --max-tokens 512 --prompt-size large
  %(prog)s --base-url http://remote:8000 --model my-model
""",
    )
    p.add_argument("-b", "--base-url", default="http://localhost:8000",
                   help="Server base URL (default: http://localhost:8000)")
    p.add_argument("-m", "--model", default="default",
                   help="Model name to request (default: 'default')")
    p.add_argument("-k", "--api-key", default="",
                   help="API key / bearer token (default: none)")
    p.add_argument("-p", "--parallel", type=int, default=4,
                   help="Number of concurrent requests (default: 4)")
    p.add_argument("-n", "--requests", type=int, default=8,
                   help="Total number of requests to send (default: 8)")
    p.add_argument("--max-tokens", type=int, default=64,
                   help="Max tokens per completion (default: 64)")
    p.add_argument("--prompt-size", choices=["small", "medium", "large"], default="small",
                   help="Prompt complexity: small (default), medium, large")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature (default: 0.7)")
    p.add_argument("--no-stream", dest="stream", action="store_false",
                   help="Disable streaming (streaming is on by default)")
    p.add_argument("--timeout", type=int, default=120,
                   help="Per-request timeout in seconds (default: 120)")
    args = p.parse_args()
    asyncio.run(run_bench(args))


if __name__ == "__main__":
    main()
