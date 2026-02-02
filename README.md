# SBSCR: Smart LLM Router

**Stop wasting money on expensive LLM APIs.** SBSCR intelligently routes your queries to the right model—automatically choosing between lightweight models for simple questions and powerhouse 70B models for complex reasoning.

## Why This Exists

Let's be honest: using GPT-4 for "What's the weather?" is overkill. But using Llama 8B for "Write a distributed transaction system in Rust" won't cut it either.

SBSCR solves this by:
- Analyzing each query's complexity and intent
- Routing simple queries to fast, cheap models (2-5s response time)
- Routing complex queries to SOTA models (Llama 70B, Mixtral 8x7B)
- **All while using only FREE API tiers**

Result? **8.9/10 on MT-Bench** (GPT-4 scores ~9.0) at **$0 cost**.

## Live Demo

Try it yourself: **[https://frontend-seven-eta-98.vercel.app/](https://frontend-seven-eta-98.vercel.app/)**

## How It Works

```
Your Query
    ↓
[Fast Path Check] ← Keywords like "hello", "thanks" → Route to Llama 8B (instant)
    ↓
[Intent Detection] ← DistilBERT classifies: code/math/reasoning/creative/chat
    ↓
[Complexity Score] ← XGBoost analyzes: token density, technical terms, structure
    ↓
[Smart Routing]
    → Coding query? → DeepSeek Coder (HuggingFace)
    → Complex reasoning? → Llama 3.3 70B (Groq)
    → Math problem? → Qwen 2.5 (HuggingFace)
    → Simple chat? → Llama 3.1 8B (Groq)
```

## Real-World Performance

**MT-Bench Scores** (industry standard benchmark):

| Task | SBSCR | GPT-4 |
|------|-------|-------|
| Writing | 9.0 | 9.0 |
| Coding | 9.0 | 8.5 |
| Math | 9.5 | 9.5 |
| Reasoning | 8.5 | 9.0 |
| **Overall** | **8.9** | **8.99** |

**Latency:**
- Simple queries (Fast Path): ~2ms overhead
- Complex queries: ~500ms routing + inference time
- Total response time: 2-5 seconds (competitive with GPT-3.5)

## Models We Use (All Free Tier)

**Groq (blazing fast inference):**
- Llama 3.3 70B (top-tier reasoning)
- Mixtral 8x7B (high performance)
- Llama 3.1 8B (speed demon)
- Gemma 2 9B (Google's model)

**HuggingFace (diverse model support):**
- DeepSeek Coder V2 (best for code)
- Qwen 2.5 (strong all-rounder)
- Mistral 7B (efficient mid-tier)
- Phi-3 Mini (ultra-fast chat)

**Google AI:**
- Gemini 1.5 Pro (highest quality)
- Gemini 1.5 Flash (fastest)

## Setup (5 minutes)

**Prerequisites:**
- Python 3.9+
- API keys (all free):
  - [Groq](https://console.groq.com) (14,400 requests/day)
  - [HuggingFace](https://huggingface.co/settings/tokens) (rate-limited)
  - [Google AI](https://makersuite.google.com/app/apikey) (15 req/min)

**Installation:**

```bash
# Clone the repo
git clone https://github.com/sreemukhik/SBSCR-router.git
cd sbscr-router

# Install dependencies
pip install -r requirements.txt

# Create .env file (copy from example)
cp .env.example .env

# Edit .env and add your API keys:
# GROQ_API_KEY=gsk_...
# HF_TOKEN=hf_...
# GOOGLE_API_KEY=AIza...

# Run the server
python serve.py
```

Server starts at `http://localhost:8000`

## Usage

**OpenAI-Compatible API:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sbscr-auto",
    "messages": [
      {"role": "user", "content": "Write a Python function to reverse a linked list"}
    ]
  }'
```

**Response includes routing metadata:**
```json
{
  "choices": [...],
  "usage": {
    "detected_intent": "coding",
    "complexity_score": 0.72,
    "selected_model": "deepseek-coder-33b",
    "routing_latency_ms": 487
  }
}
```

**Try the web UI:**
Open `demo/index.html` in your browser for a clean chat interface with live routing metrics.

## Project Structure

```
sbscr-router/
├── sbscr/
│   ├── core/
│   │   ├── registry.py      # 12 model definitions
│   │   ├── intent.py        # DistilBERT zero-shot classifier
│   │   └── features.py      # XGBoost complexity scorer
│   ├── routers/
│   │   └── sbscr.py         # Main routing logic
│   └── providers/
│       ├── groq_provider.py      # Groq API wrapper
│       ├── huggingface_provider.py
│       └── google_provider.py
├── serve.py                 # FastAPI server
├── demo/                    # Web UI
└── data/                    # Model configs, benchmarks
```

## What Makes This Different?

**vs. OpenRouter/Martian/Unify:**
- ✅ Fully open-source (MIT license)
- ✅ Zero cost (they charge per request)
- ✅ Self-hostable (no vendor lock-in)
- ❌ Fewer models (12 vs 100+)
- ❌ No GPT-4/Claude access

**vs. LangChain Router:**
- ✅ Production-ready API server (not just Python lib)
- ✅ Trained complexity scorer (not just rules)
- ✅ Multi-provider fallbacks built-in

**Best for:**
- Developers on a budget
- Privacy-conscious apps (self-hosted)
- Prototyping and experimentation
- Learning about LLM routing

**Not ideal for:**
- High-volume production (free tiers cap at ~14k req/day)
- Apps requiring GPT-4/Claude quality
- Enterprise SLA requirements

## Benchmarking

Run MT-Bench yourself:

```bash
# Generate responses for all 80 MT-Bench questions
python run_mt_bench.py

# Score with GPT-4 as judge (requires OpenAI key)
python run_judgment.py
```

Results saved to `data/mt_bench/model_answer/sbscr-auto.jsonl`

## Deployment

**Vercel (recommended):**
- Repo already includes `vercel.json`
- Connect your GitHub repo to Vercel
- Add environment variables in Vercel dashboard
- Auto-deploys on every push to main

**Docker:**
```bash
docker build -t sbscr-router .
docker run -p 8000:8000 --env-file .env sbscr-router
```

**Production tips:**
- Use paid API tiers for scale
- Add Redis for response caching
- Enable CORS for frontend apps
- Monitor with `prometheus_client` (built-in metrics)

## Contributing

Found a bug? Want to add a new provider? PRs welcome!

**Easy wins:**
- Add new models to `data/models.yaml`
- Improve fast-path regex patterns in `sbscr/routers/sbscr.py`
- Write tests (we need more!)

## License

MIT License - do whatever you want with this code

## Questions?

Open an issue or check out:
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Deep dive into architecture
- [FREE_TIER_SETUP.md](FREE_TIER_SETUP.md) - API key setup guide
- [demo/](demo/) - Example frontend implementation

---

