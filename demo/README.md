# SBSCR Router - Interactive Demo

A stunning, glassmorphic chatbot interface to showcase the SBSCR Enterprise LLM Router to reviewers.

## 🎨 Features

- **Glassmorphic Design**: Modern, transparent UI with blur effects
- **Real-time Routing Visualization**: See which model is selected for each query
- **Performance Metrics**: Live display of routing latency, inference time, and cost savings
- **Animated Interactions**: Smooth transitions and micro-animations
- **Responsive Layout**: Works on desktop, tablet, and mobile

## 🚀 Quick Start

### 1. Start the SBSCR Server

First, make sure the SBSCR server is running:

```bash
# From the project root
python serve.py
```

The server should be running at `http://localhost:8000`

### 2. Open the Demo

Simply open `index.html` in your browser:

```bash
# Option 1: Double-click index.html in File Explorer

# Option 2: Use Python's built-in server
cd demo
python -m http.server 8080
# Then open http://localhost:8080 in your browser

# Option 3: Use Live Server (VS Code extension)
# Right-click index.html -> "Open with Live Server"
```

### 3. Start Chatting!

Try these example queries to see the router in action:

- **Code**: "Write a Python web scraper for product prices"
- **Math**: "Solve the integral of x^2 * sin(x)"
- **Creative**: "Write a short story about AI and humanity"
- **Reasoning**: "Explain the trolley problem and its implications"

## 📊 What You'll See

### Routing Intelligence Panel
Watch in real-time as SBSCR:
- Analyzes your query semantically
- Determines complexity
- Selects the optimal model
- Shows routing latency (typically 2-500ms)

### Performance Metrics
- **Routing Latency**: Time to select the model
- **Inference Time**: Time for the model to generate response
- **Total Time**: End-to-end latency
- **Cost Saved**: Cumulative savings vs GPT-4 pricing

### Model Tiers
The demo shows all 4 model tiers:
- 🟣 **SOTA**: Llama 3.3 70B (for complex reasoning)
- 🔴 **HIGH**: Mixtral 8x7B (for general knowledge)
- 🟡 **CODE**: DeepSeek V2 (for programming)
- 🔵 **FAST**: Gemini Flash (for simple queries)

## 🎯 Perfect for Reviewers

This demo is designed to impress:

1. **Visual Appeal**: Modern glassmorphic design with animated gradients
2. **Transparency**: See exactly which model handles each query
3. **Performance**: Real metrics showing sub-second routing
4. **Cost Savings**: Live counter showing $0 API costs
5. **Quality**: Compare responses to commercial alternatives

## 🛠️ Customization

### Change API Endpoint

Edit `script.js` line 2:

```javascript
const API_BASE_URL = 'http://localhost:8000';
```

### Modify Colors

Edit `styles.css` CSS variables:

```css
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    /* ... more colors ... */
}
```

### Add Example Queries

Edit `script.js` around line 300:

```javascript
const exampleQueries = [
    "Your custom query here",
    // ... more queries
];
```

## 📱 Browser Compatibility

- ✅ Chrome/Edge (Recommended)
- ✅ Firefox
- ✅ Safari
- ⚠️ IE11 (Limited support)

## 🐛 Troubleshooting

### "Could not connect to server"

Make sure:
1. SBSCR server is running (`python serve.py`)
2. Server is at `http://localhost:8000`
3. No CORS issues (use same origin or configure CORS)

### Styling looks broken

Make sure:
1. `styles.css` is in the same directory as `index.html`
2. Browser supports backdrop-filter (enable in Firefox about:config)

### Animations are laggy

Try:
1. Closing other browser tabs
2. Disabling browser extensions
3. Using Chrome/Edge for better performance

## 📄 License

MIT License - Same as SBSCR Router
