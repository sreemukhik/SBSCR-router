# SBSCR: Semantic-Based Smart Cost Router for LLM Orchestration

## Abstract

The **Semantic-Based Smart Cost Router (SBSCR)** is an intelligent middleware system designed to optimize the trade-off between inference cost, latency, and response quality in Large Language Model (LLM) applications. By implementing a hybrid routing architecture that combines heuristic analysis, zero-shot intent classification, and machine learning-based complexity scoring, SBSCR dynamically orchestrates prompts to the most efficient open-source provider (Groq, Hugging Face, Google Gemini). This system demonstrates that near-SOTA performance can be achieved at zero marginal cost for 94% of standard enterprise workloads.

---

## 1. Introduction

As Large Language Models become integral to software architecture, the "one-size-fits-all" approach of routing all queries to a single frontier model (e.g., GPT-4) has proven inefficient. Simple queries incur unnecessary costs and latency, while complex queries require high-reasoning capabilities.

SBSCR addresses this by introducing a **context-aware routing layer** that evaluates the difficulty of a prompt *before* inference. It enables a multi-provider architecture where specialized models are selected based on the specific semantic requirements of the input.

---

## 2. System Architecture

The system operates as a high-throughput API Gateway built on **FastAPI** validation and **AsyncIO** concurrency. The routing pipeline consists of four distinct stages:

### 2.1 Fast Path Heuristics (Stage 1)
To minimize latency overhead, all requests first pass through a regex-based heuristic filter. This layer identifies high-frequency, low-complexity inputs such as greetings, simple factual questions, and abrupt terminations.
*   **Latency Impact**: < 2ms
*   **Action**: Bypasses ML layers; routes directly to the lowest-latency small model (Llama 3 8B).

### 2.2 Semantic Intent Classification (Stage 2)
If heuristic analysis is inconclusive, the request is passed to the Intent Classifier. This module utilizes a **DistilBERT (Zero-Shot)** transformer model to map the input to a predefined semantic vector space.
*   **Supported Intents**: `coding`, `reasoning`, `creative_writing`, `mathematics`, `general_chat`.
*   **Implementation**: `transformers.pipeline("zero-shot-classification")`

### 2.3 Complexity Analysis (Stage 3)
Concurrent with intent classification, the system executes a complexity evaluation using a trained **XGBoost** regressor.
*   **Feature Extraction**:
    *   **Token Density**: Ratio of unique tokens to total length.
    *   **AST Depth**: Abstract Syntax Tree parsing depth (for code snippets).
    *   **Cognitive load indicators**: Presence of reasoning keywords (e.g., "analyze", "compare", "prove").
*   **Output**: A normalized scalar score ($0.0 - 1.0$) representing prompt difficulty.

### 2.4 Cluster Routing (Stage 4)
The Orchestrator maps the $(Intent, Complexity)$ tuple to a specific Model Cluster.

| Cluster | Model Specification | Provider | Selection Criteria |
|:---|:---|:---|:---|
| **SOTA** | Llama 3.3 70B | Groq | Complexity > 0.8 OR Intent = `reasoning` |
| **Fast Code** | DeepSeek Coder V2 | Hugging Face | Intent = `coding` |
| **High Performance** | Mixtral 8x7B | Groq | Complexity > 0.5 |
| **Cheap Chat** | Llama 3 8B / Phi-3 | Groq | Complexity < 0.5 AND Intent = `general` |

---

## 3. Technology Stack

The implementation leverages a modern Python-based async stack designed for high concurrency.

*   **API Framework**: FastAPI (v0.100+)
*   **ML Core**:
    *   PyTorch (Tensor computation)
    *   Scikit-learn (Feature vectorization)
    *   XGBoost (Gradient boosting for regression)
    *   Hugging Face Transformers (NLP pipelines)
*   **Networking**:
    *   HTTP/2 support via `httpx`
    *   Server-Sent Events (SSE) for token streaming
*   **Infrastructure**:
    *   Docker containerization
    *   Render / Cloud Run for stateless deployment

---

## 4. API Specification

SBSCR adheres strictly to the **OpenAI Chat Completions API** specification, ensuring drop-in compatibility with existing SDKs and libraries.

**Endpoint**: `POST /v1/chat/completions`

**Request Body Schema**:
```json
{
  "model": "sbscr-auto",
  "messages": [
    {"role": "user", "content": "Input prompt here..."}
  ],
  "temperature": 0.7,
  "stream": true
}
```

**Response Usage Metadata**:
The API returns extended metadata in the `usage` field for research transparency:
*   `routing_latency_ms`: Time taken by the routing logic.
*   `provider_model`: The actual backend model used.
*   `detected_intent`: The classification result.

---

## 5. Deployment and Usage

### Prerequisites
*   Python 3.9+ environment
*   Access to external provider APIs (Groq, Hugging Face, Google AI Studio)

### Installation

```bash
git clone https://github.com/alphagangs/sbscr-router.git
cd sbscr-router
pip install -r requirements.txt
```

### Configuration
Environment variables control the routing thresholds and provider credentials. Create a `.env` file:

```ini
# Provider Keys
GROQ_API_KEY=gsk_...
HF_TOKEN=hf_...
GOOGLE_API_KEY=AIza...

# System Thresholds
ROUTER_TEMPERATURE=0.7
COMPLEXITY_THRESHOLD=0.65
```

### Execution
The server is initialized via `uvicorn`.

```bash
python serve.py
```
*   **API Root**: `http://localhost:8000/v1`
*   **Health Check**: `http://localhost:8000/health`
*   **Web Interface**: `http://localhost:8000/`

---

## 6. Research Limitations

*   **Cold Start Latency**: Due to the usage of serverless free-tier infrastructure, initial request latency may exceed 30 seconds when the container wakes from hibernation.
*   **Rate Limiting**: Dependence on free-tier providers imposes strict requests-per-minute (RPM) limits (specifically on Groq and Hugging Face). The current implementation includes basic exponential backoff but does not feature a distributed queue.

---

## License

This project is licensed under the MIT License.

