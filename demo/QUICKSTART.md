# 🚀 Quick Start Guide - SBSCR Demo

## Step 1: Start the SBSCR Server

Open a terminal and run:

```bash
cd c:\Users\KUNCHE SREEMUKHI\PycharmProjects\SBSCR-router
python serve.py
```

**Expected output:**
```
🚀 Initializing Enterprise Router v5...
🔌 Initializing 3 FREE Providers (Groq, HF, Google)...
✅ Available models: [...]
🌐 Server: http://localhost:8000
```

Keep this terminal open!

## Step 2: Start the Demo Server

Open a **NEW** terminal and run:

```bash
cd c:\Users\KUNCHE SREEMUKHI\PycharmProjects\SBSCR-router\demo
python run_demo.py
```

**Expected output:**
```
🎨 SBSCR DEMO SERVER
✅ Server running at: http://localhost:8080
🌐 Opening browser automatically...
```

The browser should open automatically. If not, manually open: http://localhost:8080

## Step 3: Test the Demo

Try these queries to impress reviewers:

### 💻 Code Query (Routes to DeepSeek Coder)
```
Write a Python function to scrape product prices from an e-commerce website
```

### 🧮 Math Query (Routes to Llama 3.3 70B)
```
Solve the differential equation: dy/dx = x^2 + 3x + 2
```

### 🎨 Creative Query (Routes to Mixtral)
```
Write a creative short story about an AI that learns to dream
```

### ⚡ Simple Query (Routes to Gemini Flash)
```
What is the capital of France?
```

## What to Show Reviewers

### 1. **Beautiful UI**
   - Glassmorphic design with animated gradients
   - Professional, modern interface
   - Smooth animations and transitions

### 2. **Intelligent Routing**
   - Watch the "Routing Intelligence" panel
   - See which model is selected for each query type
   - Notice the routing happens in milliseconds

### 3. **Performance Metrics**
   - Routing Latency: ~2-500ms
   - Inference Time: ~1-3s
   - Cost Saved: Accumulates with each query

### 4. **Model Tiers**
   - **SOTA** (Purple): Llama 3.3 70B - Complex reasoning
   - **HIGH** (Pink): Mixtral 8x7B - General knowledge
   - **CODE** (Yellow): DeepSeek V2 - Programming tasks
   - **FAST** (Blue): Gemini Flash - Simple queries

## Troubleshooting

### "Could not connect to server"
- Make sure SBSCR server is running on port 8000
- Check that both terminals are open
- Verify API keys are set in `.env`

### Demo won't open
- Manually navigate to http://localhost:8080
- Try a different port: edit `run_demo.py` and change `PORT = 8080` to `PORT = 8081`

### Styling looks broken
- Make sure all files are in the `demo/` folder:
  - index.html
  - styles.css
  - script.js
- Clear browser cache (Ctrl+Shift+R)

## Tips for Reviewers

1. **Compare Quality**: Try the same query on ChatGPT vs SBSCR
2. **Show Cost Savings**: Point out the "Cost Saved" metric
3. **Highlight Speed**: Routing happens in milliseconds
4. **Demonstrate Intelligence**: Show how different queries route to different models
5. **Emphasize Zero Cost**: All models are free tier!

## Stopping the Demo

1. Close the browser
2. In the demo terminal: Press `Ctrl+C`
3. In the SBSCR server terminal: Press `Ctrl+C`

---

**Need help?** Check the full documentation in `demo/README.md`
