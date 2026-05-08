# vLLM Expert-Level Cheat Sheet

`vLLM` is a high-performance serving engine for LLMs, optimized for throughput and latency through efficient memory management (PagedAttention) and request scheduling (Continuous Batching). This guide is designed for senior developers (7+ years) focusing on production-grade deployments and performance tuning.

---

## 0. The Basics

### 0.1. Installation & Environment
```bash
pip install vllm
# Recommended for optimized kernels (For NVIDIA GPUs only) 
pip install flash-attn --no-build-isolation
```

### 0.2. Quick Start: Offline Inference
The `LLM` class is the main entry point for local/offline usage.
```python
from vllm import LLM, SamplingParams

# SamplingParams defines generation configuration (similar to OpenAI's params)
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)

# Initializing LLM engine (loads weights and profiles GPU memory)
llm = LLM(model="facebook/opt-125m")

outputs = llm.generate(["Explain quantum computing in one sentence."], sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated: {output.outputs[0].text!r}")
```

### 0.3. OpenAI-Compatible API Server
vLLM provides a production-ready FastAPI-based server.
```bash
# Using the CLI (v0.5.0+)
vllm serve facebook/opt-125m --port 8000

# Legacy entrypoint
python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m
```

### 0.4. Core CLI Arguments
*   `--model`: HuggingFace ID or local filesystem path.
*   `--dtype`: Compute data type (`auto`, `half`, `float16`, `bfloat16`, `float`).
*   `--max-model-len`: Override model's max sequence length (useful for OOM management).
*   `--trust-remote-code`: Must be set for models with custom model code (e.g., DBRX, Falcon).

---

## 1. Intermediate Configuration & Usage

### 1.1. Advanced Sampling Parameters
Beyond temperature, vLLM supports a wide range of sampling knobs:
*   **`top_k`**: Samples from the top K tokens. Reduces the tail of the distribution.
*   **`min_p`**: Scale-invariant alternative to `top_p`. Filter tokens with probability less than `min_p * prob(top_token)`.
*   **`presence_penalty` / `frequency_penalty`**: Penalize tokens based on whether they appeared or how often.
*   **`stop`**: List of strings or token IDs to terminate generation.
*   **`best_of` & `n`**: Generate multiple candidate sequences.

### 1.2. Chat Templates & Tokenization
vLLM integrates with HuggingFace's chat templates.
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct")
messages = [{"role": "user", "content": "How does PagedAttention work?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# outputs = llm.generate([prompt], sampling_params)
```

### 1.3. AsyncLLMEngine
Use `AsyncLLMEngine` for non-blocking applications or custom API servers.
```python
from vllm import AsyncLLMEngine, AsyncEngineArgs
engine_args = AsyncEngineArgs(model="facebook/opt-125m")
engine = AsyncLLMEngine.from_engine_args(engine_args)

async for request_output in engine.generate(prompt, sampling_params, request_id):
    print(request_output.outputs[0].text)
```

### 1.4. Practical Quantization
Loading models with reduced precision to save VRAM.
*   **AWQ (`--quantization awq`)**: High performance 4-bit quantization.
*   **GPTQ (`--quantization gptq`)**: Broadly available 4-bit quantization.
*   **GGUF**: Native support for loading GGUF files directly.
*   **Installation**: Usually requires `autoawq` or `auto-gptq` libraries in the environment.

### 1.5. Basic Parallelism (Multi-GPU)
*   **Tensor Parallelism (TP)**: Set `--tensor-parallel-size N`. Shards weights across N GPUs.
*   **When to use**: If a model exceeds the VRAM of a single GPU (e.g., a 70B model requires ~140GB in FP16, fitting on 2-4 A100s).

---

## 2. Core Architecture & Memory Management

### 2.1. PagedAttention Internals
PagedAttention solves the KV cache fragmentation problem by treating memory like an OS treats virtual memory.
*   **Logical vs. Physical Blocks:** KV cache is divided into fixed-size blocks (default: 16 tokens). 
*   **Block Manager:** Maps logical blocks of a request to physical blocks on the GPU.
*   **Prefix Caching:** Uses a Radix Tree to store and reuse KV blocks for common prefixes (e.g., system prompts). Enable with `--enable-prefix-caching`.
*   **Performance Knob:** `--block-size {8,16,32}`. Smaller blocks reduce internal fragmentation but increase management overhead. 16 is usually optimal for A100/H100.

### 2.2. Continuous Batching
Traditional batching (static) waits for all sequences to finish. vLLM uses iteration-level scheduling.
*   **Preemption:** If GPU memory is exhausted, the scheduler preempts lower-priority requests (recompute or swap to CPU).
*   **Chunked Prefill:** Prevents long prompts from "starving" the generation of existing requests by breaking prefills into chunks. Enable with `--enable-chunked-prefill`.

---

## 3. Performance Tuning & Scaling

### 3.1. Memory Configuration
*   **`--gpu-memory-utilization`**: Default is 0.90. In multi-model or multi-tenant setups, tune this to leave headroom for other processes or NCCL buffers.
*   **`--max-num-batched-tokens`**: Limits the number of tokens processed in one iteration. Higher values increase throughput but can spike latency.
*   **`--cache-dtype {auto,fp8}`**: Use `fp8` (on H100/L40) to halve KV cache memory usage, allowing for 2x larger batches or context lengths.

### 3.2. Distributed Serving (Parallelism)
*   **Tensor Parallelism (TP):** Shards weights across GPUs. Best for latency. Use `--tensor-parallel-size (TP)`.
*   **Pipeline Parallelism (PP):** Shards layers across GPUs. Use when model doesn't fit in one GPU even with TP, or to increase throughput. Use `--pipeline-parallel-size (PP)`.
*   **Expert Parallelism (EP):** Specifically for MoE models (e.g., Mixtral, DeepSeek). Shards experts across ranks.
*   **Context Parallelism (CP/DCP):** Shards the sequence length dimension. Useful for extremely long contexts where KV cache for a single sequence exceeds one GPU's memory.
*   **MLA (Multi-Head Latent Attention):** Optimized attention for DeepSeek-V3 models, significantly reducing KV cache size. vLLM has native kernels for this.
*   **Data Parallelism (DP):** Multiple independent replicas for high-throughput scenarios.

### 3.3. NCCL & Communication Tuning
For multi-GPU/multi-node, NCCL is the bottleneck.
*   **`VLLM_NCCL_SKIP_P2P_CHECK=1`**: Skip P2P checks if using specific virtualization (e.g., some K8s setups).
*   **`NCCL_DEBUG=INFO`**: Essential for debugging hangs during initialization.
*   **Custom All-Reduce:** vLLM uses custom kernels for TP. If they fail, fallback with `--disable-custom-all-reduce`.

---

## 4. Advanced Features

### 4.1. Multi-LoRA Serving
vLLM supports serving multiple LoRA adapters concurrently with the base model.
*   **Setup:** `--enable-lora --max-loras 4 --max-lora-rank 16`.
*   **Request:** Specify `lora_request` in the API call.
*   **Overhead:** Memory for LoRA weights is allocated on the fly or pre-cached on CPU.

### 4.2. Structured Outputs (Json/Regex)
Native support for constrained decoding via multiple backends.
*   **Backends:** `xgrammar` (fastest), `outlines`, `guidance`, `lm-format-enforcer`.
*   **Usage:** Pass `response_format={"type": "json_schema", "json_schema": ...}` in API.
*   **Reasoning Models:** Support for `reasoning_parser` (e.g., DeepSeek-R1) to separate <think> tokens from final output.

### 4.3. Multi-modal Support
vLLM handles vision (VLM), audio, and video models.
*   **Config:** `--limit-per-prompt image=2` (allow 2 images per prompt).
*   **Chunked Prefill:** Essential for VLMs due to large image embeddings (tokens).

### 4.4. Speculative Decoding
Speeds up generation by using a smaller "draft" model to predict tokens, which the "target" model verifies in parallel.
*   **Configuration:** `--speculative-model {draft_model_name} --num-speculative-tokens 5`.
*   **Efficiency:** Can achieve 1.5x - 2.5x speedup depending on draft model alignment.

### 4.5. Quantization Support
| Type | Engine | Best For |
| :--- | :--- | :--- |
| **AWQ** | `vllm` | General purpose 4-bit, fast. |
| **GPTQ** | `vllm` | Broad compatibility. |
| **FP8** | `vllm` | Native H100 support, minimal accuracy loss. |
| **GGUF** | `vllm` | CPU/Llama.cpp ecosystem. |

---

## 5. The vLLM V2 Architecture (Experimental)
vLLM is transitioning to a more decoupled "V2" architecture.
*   **Decoupled Frontend:** API server and Engine can run in separate processes more efficiently.
*   **Async Post-processing:** Tokenization and detokenization are moved out of the critical path.
*   **Improved Scheduling:** Finer-grained control over preemption and swap policies.
*   **Enable:** Currently via internal flags or specific builds (check `VLLM_USE_V1=0` or similar env vars in newer versions).

---

## 6. Production Observability

### 6.1. Metrics (Prometheus)
vLLM exposes a `/metrics` endpoint. Key metrics to monitor:
*   `vllm:num_requests_running`: Current active requests.
*   `vllm:num_requests_swapped`: Requests swapped to CPU (sign of memory pressure).
*   `vllm:gpu_cache_usage_percent`: KV cache utilization.
*   `vllm:time_to_first_token_seconds`: Critical for UX (TTFT).
*   `vllm:time_per_output_token_seconds`: Generation speed (TPOT).

### 6.2. Logging & Debugging
*   `VLLM_LOGGING_LEVEL=DEBUG`
*   Use `vllm.entrypoints.openai.api_server` for standard OpenAI-compatible serving.

---

## 7. Troubleshooting (Senior Dev Edition)

### 7.1. Common Issues
*   **OOM during Profiling:** vLLM profiles memory by running a dummy pass. If it fails, reduce `--gpu-memory-utilization` or `--max-model-len`.
*   **NCCL Timeouts:** Often caused by mismatched TP/PP sizes or network issues in multi-node. Check `export NCCL_IB_DISABLE=1` if not using InfiniBand.
*   **Slow TTFT:** Often due to long prompts in a busy system. Enable Chunked Prefill or increase TP size.
*   **High Latency Spikes:** Check for "Requests Swapped". If high, decrease `--max-num-seqs` or increase GPU count.

### 7.2. Kernel Optimizations
*   vLLM uses **FlashInfer** and **XFormers** under the hood. 
*   Check `vllm.platforms.current_platform` to ensure you are utilizing the correct kernels for your hardware (CUDA vs ROCm).

---

*Cheat Sheet compiled for vLLM v0.4.0+. Architecture and flags may evolve.*
