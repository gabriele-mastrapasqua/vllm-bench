# vLLM Bench

A lightweight CLI to stress-test any OpenAI-compatible server (vLLM, Ollama, llama.cpp, etc.) with configurable parallel requests and measure throughput under load.

## Table of Contents

- [Requirements](#requirements)
- [Quick Install (one command)](#quick-install-one-command)
- [Alternative Installation](#alternative-installation)
- [Usage](#usage)
- [Test Scenarios](#test-scenarios)
  - [1. Smoke Test](#1-smoke-test--single-request)
  - [2. Baseline](#2-baseline--low-concurrency-small-prompts)
  - [3. Light Load](#3-light-load--few-parallel-users)
  - [4. Medium Load](#4-medium-load--moderate-concurrency-longer-prompts)
  - [5. Heavy Load](#5-heavy-load--high-concurrency-large-prompts)
  - [6. Saturation Test](#6-saturation-test--maximum-parallel-requests)
  - [7. Long Generation Stress](#7-long-generation-stress)
  - [8. Streaming vs Non-Streaming](#8-streaming-vs-non-streaming-comparison)
  - [9. Scaling Ladder](#9-scaling-ladder--incremental-concurrency)
  - [10. Mixed Prompt Sizes](#10-mixed-prompt-sizes-manual)
- [Metrics Explained](#metrics-explained)
- [What to Look For](#what-to-look-for)

## Requirements

- Python **>= 3.10**
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Quick Install (one command)

```bash
uv tool install git+https://github.com/gabriele-mastrapasqua/vllm-bench.git
```

This installs `vllm-bench` globally — no clone, no venv, just run:

```bash
vllm-bench --help
```

To upgrade later:

```bash
uv tool upgrade vllm-bench
```

To uninstall:

```bash
uv tool uninstall vllm-bench
```

## Alternative Installation

### From source with uv

```bash
git clone https://github.com/gabriele-mastrapasqua/vllm-bench.git && cd vllm-bench
uv sync
uv run vllm-bench --help
```

### From source with pip

```bash
git clone https://github.com/gabriele-mastrapasqua/vllm-bench.git && cd vllm-bench
python3 -m venv .venv && source .venv/bin/activate
pip install .
vllm-bench --help
```

## Usage

If installed globally via `uv tool install`:

```bash
vllm-bench [options]
```

If running from the cloned repo folder:

```bash
uv run vllm-bench [options]
```

Run `vllm-bench --help` (or `uv run vllm-bench --help`) for all available flags.

By default the tool targets `http://localhost:8000` (standard vLLM port). Use `-b` to override.

> **Note:** All examples below use `uv run vllm-bench` (works from the repo folder).
> If you installed globally with `uv tool install`, drop the `uv run` prefix.

## Test Scenarios

### 1. Smoke Test — Single Request

Verify the server is reachable and responding correctly.

```bash
uv run vllm-bench -b http://localhost:8000 -p 1 -n 1 -m my-model
```

### 2. Baseline — Low Concurrency, Small Prompts

Establish a single-user throughput baseline with short Q&A prompts and minimal output.

```bash
uv run vllm-bench -b http://localhost:8000 -p 1 -n 10 --prompt-size small --max-tokens 64 -m my-model
```

### 3. Light Load — Few Parallel Users

Simulate a handful of concurrent users with small prompts.

```bash
uv run vllm-bench -b http://localhost:8000 -p 4 -n 16 --prompt-size small --max-tokens 64 -m my-model
```

### 4. Medium Load — Moderate Concurrency, Longer Prompts

Increase both concurrency and prompt complexity to see how the server handles heavier KV-cache usage.

```bash
uv run vllm-bench -b http://localhost:8000 -p 8 -n 32 --prompt-size medium --max-tokens 256 -m my-model
```

### 5. Heavy Load — High Concurrency, Large Prompts

Push the server towards its limits with many parallel requests and long-form generation.

```bash
uv run vllm-bench -b http://localhost:8000 -p 16 -n 50 --prompt-size large --max-tokens 512 -m my-model
```

### 6. Saturation Test — Maximum Parallel Requests

Find the breaking point: flood the server with concurrent requests.

```bash
uv run vllm-bench -b http://localhost:8000 -p 32 -n 64 --prompt-size medium --max-tokens 256 -m my-model
```

### 7. Long Generation Stress

Test sustained generation with high token counts to stress memory and throughput over time.

```bash
uv run vllm-bench -b http://localhost:8000 -p 4 -n 8 --prompt-size large --max-tokens 1024 -m my-model
```

### 8. Streaming vs Non-Streaming Comparison

Run the same workload with and without streaming to compare TTFT and throughput.

```bash
# Streaming (default)
uv run vllm-bench -b http://localhost:8000 -p 8 -n 20 --prompt-size medium --max-tokens 256 -m my-model

# Non-streaming
uv run vllm-bench -b http://localhost:8000 -p 8 -n 20 --prompt-size medium --max-tokens 256 --no-stream -m my-model
```

### 9. Scaling Ladder — Incremental Concurrency

Gradually increase parallelism to find the optimal concurrency level for your hardware.

```bash
for p in 1 2 4 8 16 32; do
  echo "=== Parallel: $p ==="
  uv run vllm-bench -b http://localhost:8000 -p $p -n 32 --prompt-size small --max-tokens 128 -m my-model
done
```

### 10. Mixed Prompt Sizes (Manual)

Run different prompt sizes back-to-back to simulate a realistic mix of user queries.

```bash
uv run vllm-bench -b http://localhost:8000 -p 8 -n 16 --prompt-size small  --max-tokens 64  -m my-model
uv run vllm-bench -b http://localhost:8000 -p 8 -n 16 --prompt-size medium --max-tokens 256 -m my-model
uv run vllm-bench -b http://localhost:8000 -p 8 -n 16 --prompt-size large  --max-tokens 512 -m my-model
```

## Metrics Explained

| Metric | Description |
|---|---|
| **Aggregate tok/s** | Total completion tokens / wall-clock time. Best measure of overall server throughput under load. |
| **Avg tok/s per req** | Mean per-request generation speed. Reflects individual user experience. |
| **Median tok/s** | 50th percentile per-request speed. Less sensitive to outliers than the average. |
| **TTFT** | Time-to-first-token (streaming only). How fast the server starts responding. |
| **Fastest / Slowest** | Best and worst per-request throughput. A large gap suggests queuing or resource contention. |

## What to Look For

- **Aggregate tok/s should increase** as you add concurrency — up to a point. When it plateaus or drops, you've hit the server's throughput ceiling.
- **Per-request tok/s will decrease** under heavy load due to batching and resource sharing. This is expected.
- **TTFT increasing sharply** means the server is queuing requests and users will notice latency.
- **Errors appearing** under load may indicate OOM, timeout, or the server dropping requests.
