# ⚡ SBSCR Enterprise Router v5
### Intelligent Semantic Routing for LLMs. Zero Cost. Sub-Second Latency.

**SBSCR (Semantic-Based Smart Cost Router)** is a next-generation AI gateway that gives you **GPT-4 class performance for free** by intelligently routing your prompts to the best available open-source model.

Instead of sending every request to an expensive model, SBSCR analyzes the **complexity** and **intent** of your prompt (e.g., Coding, Creative Writing, Math) and routes it to the optimal free provider (Groq, Hugging Face, Gemini).

---

## 🚀 Live Demo

**Access the live Enterprise Router here:** 👉 [**https://sbscr-router.onrender.com**](https://sbscr-router.onrender.com)

> **⚠️ Note on Free Tier:** This deployment runs on Render's Free Tier. If the service is unused for 15 minutes, it will spin down. Please **wait 1 minute** on your first request for the server to wake up and initialize the AI models.

## 🌟 Key Features

*   **💸 100% Free Inference**: Exploits the generous free tiers of high-performance providers like **Groq**, **Google Gemini**, and **Hugging Face**.
*   **🧠 Intelligent Routing Layers**:
    *   **Fast Path**: Regex-based analysis for instant conversational replies (< 2ms).
    *   **Intent Classification**: Zero-shot DistilBERT model to detect if you need Coding, Math, or Creative writing.
    *   **Complexity Scoring**: XGBoost model that predicts how hard a prompt is to answer.
*   **🏎️ Sub-Second Latency**: Optimized for speed. Leverages Groq's LPU inference for instant results.
*   **🛡️ Auto-Fallback**: If one provider is down, it automatically retries with the next best model.
*   **💻 Built-in UI**: A beautiful, dark-mode chat interface included out-of-the-box.

---

## 🏗️ How It Works

When you send a prompt, SBSCR doesn't just forward it. It *understands* it.

1.  **Input Analysis**: The router extracts semantic features (token density, code blocks, AST depth).
2.  **Intent Detection**: "Is this a Python script?" "Is this a poem?" "Is this a simple greeting?"
3.  **Cluster Assignment**:
    *   **SOTA Cluster** (Llama 3.3 70B via Groq): For complex reasoning and hard logic.
    *   **Fast Code Cluster** (DeepSeek Coder V2 via HF): For software engineering.
    *   **High-Perf Cluster** (Mixtral 8x7B): For general knowledge.
    *   **Cheap Chat Cluster** (Llama 3 8B): For simple chat and filler.
4.  **Execution**: The prompt is sent to the winning model, and the response is streamed back.

---

## 🛠️ Technology Stack

*   **Framework**: FastAPI (High-performance Async Python)
*   **ML Core**: XGBoost, Scikit-Learn, PyTorch
*   **Providers**: Groq (Llama 3), Google AI (Gemini), Hugging Face (DeepSeek/Qwen)
*   **Frontend**: HTML5, Vanilla JS, CSS3 (No build steps required)

---

## ⚡ Quick Start

### Prerequisites
*   Python 3.9+
*   API Keys: [Groq](https://console.groq.com), [Hugging Face](https://huggingface.co), [Google AI](https://aistudio.google.com) (All Free)

### 1. Clone & Install
```bash
git clone https://github.com/alphagangs/sbscr-router.git
cd sbscr-router
pip install -r requirements.txt
```

### 2. Configure Keys
Create a `.env` file:
```env
GROQ_API_KEY=gsk_...
HF_TOKEN=hf_...
GOOGLE_API_KEY=AIza...
```

### 3. Run Server
```bash
python serve.py
```
Visit `http://localhost:8000` to see the UI!

---

## 🔌 API Integration

SBSCR is **OpenAI Compatible**. Use it as a drop-in replacement in your existing apps.

```python
import openai

client = openai.OpenAI(
    base_url="https://sbscr-router.onrender.com/v1",
    # No payment method required
    api_key="sbscr-free-tier"
)

response = client.chat.completions.create(
    model="sbscr-auto", # Let the router decide!
    messages=[{"role": "user", "content": "Write a snake game in Python."}]
)

print(response.choices[0].message.content)
```

## 📊 Benchmarks vs GPT-4

| Metric | GPT-4 Turbo | SBSCR v5 |
|:---|:---:|:---:|
| **Cost** | $30.00 / 1M tokens | **$0.00** |
| **Speed** | ~40 t/s | **~300 t/s** (via Groq) |
| **Quality** | State of the Art | **Near-SOTA** (Hybrid Llama 3/DeepSeek) |

---

## 🤝 Contributing

We love open source! Fork the repo, make your changes to the logic in `sbscr/routers/sbscr.py`, and submit a PR.

**License**: MIT
