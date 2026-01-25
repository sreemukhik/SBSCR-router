# SBSCR: Semantic-Based Smart Cost Router

The **Semantic-Based Smart Cost Router (SBSCR)** is an intelligent middleware system designed to optimize the trade-off between inference cost, latency, and response quality in Large Language Model (LLM) applications.

It implements a hybrid routing architecture that combines heuristic analysis, zero-shot intent classification, and machine learning-based complexity scoring to dynamically orchestrate prompts to the most efficient open-source provider (Groq, Hugging Face, Google Gemini).

## 🔗 Live Demo

**Access the Live Router:** [https://sbscr-router.onrender.com](https://sbscr-router.onrender.com)

> **Note:** The deployment runs on a free-tier instance. The first request may take up to 60 seconds to wake the server. Subsequent requests will be processed with standard latency.

---

## System Architecture

The system operates as a high-throughput API Gateway built on **FastAPI** and **AsyncIO**. The routing pipeline consists of four distinct stages:

### 1. Fast Path Heuristics
To minimize latency overhead, all requests first pass through a regex-based heuristic filter. This layer identifies high-frequency, low-complexity inputs such as greetings and simple factual questions.
*   **Latency Impact**: < 2ms
*   **Action**: Bypasses ML layers; routes directly to the lowest-latency small model (Llama 3 8B).

### 2. Semantic Intent Classification
If heuristic analysis is inconclusive, the request is passed to the Intent Classifier. This module filters input through a **DistilBERT (Zero-Shot)** transformer model to map it to a predefined semantic vector space.
*   **Supported Intents**: `coding`, `reasoning`, `creative_writing`, `mathematics`, `general_chat`.

### 3. Complexity Analysis
Concurrent with intent classification, the system executes a complexity evaluation using a trained **XGBoost** regressor. Features extracted include token density, AST depth (for code), and cognitive load indicators.
*   **Output**: A normalized scalar score ($0.0 - 1.0$) representing prompt difficulty.

### 4. Cluster Routing
The Orchestrator maps the $(Intent, Complexity)$ tuple to a specific Model Cluster.

| Cluster | Model Specification | Provider | Selection Criteria |
|:---|:---|:---|:---|
| **SOTA** | Llama 3.3 70B | Groq | Complexity > 0.8 OR Intent = `reasoning` |
| **Fast Code** | DeepSeek Coder V2 | Hugging Face | Intent = `coding` |
| **High Performance** | Mixtral 8x7B | Groq | Complexity > 0.5 |
| **Cheap Chat** | Llama 3 8B / Phi-3 | Groq | Complexity < 0.5 AND Intent = `general` |

---

## Technology Stack

The implementation leverages a modern Python-based async stack designed for high concurrency.

*   **API Framework**: FastAPI (v0.100+)
*   **ML Core**: PyTorch, Scikit-learn, XGBoost, Hugging Face Transformers
*   **Networking**: HTTP/2 support via `httpx`, Server-Sent Events (SSE)
*   **Infrastructure**: Docker, Render (Stateless deployment)

---

## API Specification

SBSCR adheres to the **OpenAI Chat Completions API** specification, ensuring drop-in compatibility with existing SDKs.

**Endpoint**: `POST /v1/chat/completions`

**Request Body Schema**:
```json
{
  "model": "sbscr-auto",
  "messages": [
    {"role": "user", "content": "Input prompt here..."}
  ],
  "temperature": 0.7
}
```

The API returns extended metadata in the `usage` field for transparency, including `routing_latency_ms` and `detected_intent`.

---

## Installation and Usage

### Prerequisites
*   Python 3.9+ environment
*   Access to external provider APIs (Groq, Hugging Face, Google AI Studio)

### Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/alphagangs/sbscr-router.git
    cd sbscr-router
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configuration**:
    Create a `.env` file with your credentials:
    ```ini
    GROQ_API_KEY=gsk_...
    HF_TOKEN=hf_...
    GOOGLE_API_KEY=AIza...
    ```

4.  **Run the Server**:
    ```bash
    python serve.py
    ```
    The API will be available at `http://localhost:8000/v1`.

---

## License

This project is licensed under the MIT License.

