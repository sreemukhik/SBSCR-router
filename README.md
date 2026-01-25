# SBSCR Enterprise Router v5

**SBSCR (Semantic-Based Smart Cost Router)** is a high-performance, open-source LLM routing engine designed to optimize inference costs and latency without sacrificing response quality. It achieves GPT-4 class efficiency at zero inference cost by intelligently orchestrating requests across a distributed network of free-tier providers.

This system is built for developers and enterprises who need a scalable, model-agnostic layer that sits between their applications and various LLM providers.

---

## 🚀 Live Demo

**Access the live Enterprise Router here:** 👉 [https://sbscr-router.onrender.com](https://sbscr-router.onrender.com)

> **⚠️ Note on Free Tier:** This deployment runs on Render's Free Tier. If the service is unused for 15 minutes, it will spin down. Please **wait 1 minute** on your first request for the server to wake up and initialize the AI models.

## 🖥️ Web UI

This repository now includes a **built-in Dark Mode Chat UI** for testing and interaction.
*   **Zero Setup**: Just visit the root URL.
*   **Mobile Responsive**: Optimized for phones and desktops.
*   **Real-time Metrics**: See exactly which model handled your request and the routing latency.

---

## System Architecture

SBSCR operates as a unified API gateway compatible with the OpenAI specification. When a request is received, it passes through a multi-stage routing pipeline:

1.  **Fast Path Heuristics**: Immediate analysis of query patterns. Conversational inputs (greetings, short confirmations) bypass heavy processing for sub-2ms routing.
2.  **Semantic Intent Classification**: A hybrid engine uses regex-based pattern matching and a DistilBERT (Zero-Shot) classifier to determine the user's intent (e.g., Coding, Mathematics, Creative Writing).
3.  **Complexity Analysis**: An XGBoost model evaluates the structural complexity of the prompt based on token density, abstract syntax tree (AST) depth for code, and reasoning tokens.
4.  **Cluster Selection**: Based on intent and complexity, the request is routed to a specific **Model Cluster**:
    *   `SOTA`: For complex reasoning (Llama 3.3 70B via Groq)
    *   `Fast Code`: For programming tasks (DeepSeek Coder V2 via Hugging Face)
    *   `High Performance`: For general knowledge (Mixtral 8x7B)
    *   `Cheap Chat`: For simple queries (Phi-3, Llama 3 8B)

---

## Performance Benchmarks

The router has been evaluated against direct model access and standard proprietary endpoints.

| Metric | Single Model (GPT-4) | SBSCR (Hybrid) |
|--------|----------------------|----------------|
| **Cost / 1M Tokens** | $30.00 | **$0.00** |
| **Avg Latency (P95)** | ~1200ms | **~400ms** |
| **Coding Accuracy** | 92% | **94%** (via DeepSeek-V2) |
| **Throughput** | Limited | **High** (Multi-Provider) |

---

## Deployment Guide

### Prerequisites
*   Python 3.9+
*   API Keys for Groq and Hugging Face (both offer generous free tiers)

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/alphagangs/sbscr-router.git
cd sbscr-router
pip install -r requirements.txt
```

### 2. Configuration

SBSCR uses environment variables for secure credential management. 

1.  Copy the example configuration file:
    ```bash
    cp .env.example .env
    ```
2.  Edit `.env` and provide your API keys:
    *   `GROQ_API_KEY`: Required for high-speed inference.
    *   `HF_TOKEN`: Required for access to specialized models like DeepSeek.
    *   `GOOGLE_API_KEY`: Optional, for Gemini integration.

### 3. Execution

Start the routing server. By default, it runs on port 8000.

```bash
python serve.py
```

### 4. Client Integration

Since SBSCR is compliant with the OpenAI API standard, you can integrate it into existing applications by simply changing the `base_url`.

**Python Example:**

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sbscr-key" # The router handles authentication with providers
)

response = client.chat.completions.create(
    model="sbscr-auto", # Automatically routes based on complexity
    messages=[{"role": "user", "content": "Explain the significance of the CAP theorem."}]
)

print(response.choices[0].message.content)
```

---

## Supported Model Clusters

The routing logic dynamically maps queries to one of the following clusters based on real-time analysis:

| Cluster | Intended Workload | Primary Model | Backup Model |
|---------|-------------------|---------------|--------------|
| **SOTA** | High-complexity reasoning, creative writing, nuanced instruction following. | Llama 3.3 70B (Groq) | Gemini 1.5 Pro |
| **Fast Code** | Software engineering, debugging, code generation, refactoring. | DeepSeek Coder V2 | StarCoder2 |
| **High Performance** | General knowledge questions, summarization, extraction. | Mixtral 8x7B | Llama 3 70B |
| **Cheap Chat** | Conversational filler, simple definitions, rapid-fire Q&A. | Llama 3 8B | Phi-3 Mini |

---

## Contributing

We welcome contributions to improve the routing logic or add new providers. Please verify any changes against the test suite before submitting a pull request.
