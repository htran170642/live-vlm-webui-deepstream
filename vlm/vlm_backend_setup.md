# Setting Up VLM Backends

This guide covers different VLM backend options to power your Live VLM WebUI.

## Option A: Ollama (Easiest)

**Best for:** Quick start, easy model management, beginners

```bash
# Install ollama from https://ollama.ai/download
# Pull a vision model
ollama pull llama3.2-vision:11b

# Start ollama server
ollama serve
```

**Recommended Models:**
- `llama3.2-vision:11b` - Good balance of quality and speed
- `llava:7b` - Faster, lighter model
- `llava:13b` - Higher quality

---

## Option B: vLLM (Recommended for Performance)

**Best for:** Production deployments, high throughput, GPU optimization

```bash
# Install vLLM
pip install vllm

# Start vLLM server with a vision model
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-11B-Vision-Instruct \
  --port 8000
```
---

## Port Reference

| Backend | Default Port | API Base URL |
|---------|-------------|--------------|
| **Ollama** | 11434 | `http://localhost:11434/v1` |
| **vLLM** | 8000 | `http://localhost:8000/v1` |
| **SGLang** | 30000 | `http://localhost:30000/v1` |
| **NVIDIA API** | - | `https://ai.api.nvidia.com/v1/gr` |
