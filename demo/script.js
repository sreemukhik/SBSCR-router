// SBSCR Enterprise Studio - Logic Engine v5.1

const API_BASE_URL = 'http://localhost:8005';

const state = {
    settings: {
        temperature: 0.7,
        maxTokens: 1000
    },
    currentChatId: null,
    history: JSON.parse(localStorage.getItem('sbscr_history_studio') || '[]'),
    documentContext: ""
};

const dom = {
    messages: document.getElementById('messagesContainer'),
    scroller: document.getElementById('scroll-container'),
    input: document.getElementById('userInput'),
    form: document.getElementById('chat-form'),
    history: document.getElementById('history-container'),
    log: document.getElementById('log-container'),

    // Metrics
    metricModel: document.getElementById('metric-model'),
    metricScore: null,
    traceIntent: document.getElementById('trace-intent'),

    // Modal
    modal: document.getElementById('settings-modal'),
    btnSettings: document.getElementById('settings-btn'),
    btnCloseSettings: document.getElementById('close-settings'),
    btnSaveSettings: document.getElementById('save-settings')
};

// --- Initialization ---

document.addEventListener('DOMContentLoaded', () => {
    renderHistory();
    autoResize();

    dom.form.addEventListener('submit', (e) => {
        e.preventDefault();
        handleSendMessage();
    });

    dom.input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });

    dom.input.addEventListener('input', () => {
        document.getElementById('charCount').textContent = `${dom.input.value.length} / 2000`;
        autoResize();
    });

    // History & New Session
    document.getElementById('new-chat-btn').addEventListener('click', () => {
        state.currentChatId = null;
        state.documentContext = "";
        dom.messages.innerHTML = '';
        const welcome = createWelcome();
        dom.messages.appendChild(welcome);
        addLog('Studio: New session initialized.');
        renderHistory();
    });

    // Settings
    dom.btnSettings.addEventListener('click', () => {
        document.getElementById('input-temp').value = state.settings.temperature;
        document.getElementById('input-tokens').value = state.settings.maxTokens;
        syncDisplays();
        dom.modal.classList.remove('hidden');
    });

    dom.btnCloseSettings.addEventListener('click', () => dom.modal.classList.add('hidden'));

    dom.btnSaveSettings.addEventListener('click', () => {
        state.settings.temperature = parseFloat(document.getElementById('input-temp').value);
        state.settings.maxTokens = parseInt(document.getElementById('input-tokens').value);
        dom.modal.classList.add('hidden');
        addLog('Studio: System configuration synchronized.');
    });

    // File Handlers
    document.getElementById('btn-upload').addEventListener('click', () => document.getElementById('fileInput').click());
    document.getElementById('fileInput').addEventListener('change', handleFileUpload);
});

// --- Core Functions ---

function addLog(msg) {
    const time = new Date().toLocaleTimeString('en-GB', { hour12: false });
    const line = document.createElement('div');
    line.innerHTML = `<span style="color:#6366f1;">[${time}]</span> <span style="opacity:0.6;">></span> ${msg}`;
    dom.log.appendChild(line);
    dom.log.scrollTop = dom.log.scrollHeight;
}

function syncDisplays() {
    document.getElementById('val-temp').textContent = document.getElementById('input-temp').value;
    document.getElementById('val-tokens').textContent = document.getElementById('input-tokens').value;
}

document.getElementById('input-temp').addEventListener('input', syncDisplays);
document.getElementById('input-tokens').addEventListener('input', syncDisplays);

function startNewChat() {
    state.currentChatId = Date.now();
}

async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    addLog(`Ingesting: ${file.name}`);
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE_URL}/v1/files/upload`, { method: 'POST', body: formData });
        const data = await response.json();
        state.documentContext = data.text;
        addLog(`Sync: ${data.context_added} tokens indexed.`);
        appendMessage('assistant', `📎 **System Update**: Document \`${file.name}\` has been successfully processed into the active routing layer.`);
    } catch (err) {
        addLog('Sync Fault: File parser offline.');
    }
}

async function handleSendMessage() {
    const text = dom.input.value.trim();
    if (!text) return;

    if (!state.currentChatId) startNewChat();

    const query = state.documentContext
        ? `[CONTEXT_LAYER]: ${state.documentContext}\n\n[USER_STREAM]: ${text}`
        : text;

    dom.input.value = '';
    autoResize();
    const welcome = dom.messages.querySelector('.welcome-center');
    if (welcome) welcome.remove();

    appendMessage('user', text);
    forceScroll();

    addLog(`Engine: Analyzing intent for stream...`);
    showThinking();

    try {
        const t0 = performance.now();
        const response = await fetch(`${API_BASE_URL}/v1/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: 'sbscr-auto',
                messages: [{ role: 'user', content: query }],
                temperature: state.settings.temperature,
                max_tokens: state.settings.maxTokens
            })
        });

        const data = await response.json();

        // --- REALISTIC METRIC LOGIC ---
        // Backend provides precise engine speed vs total inference time
        const rLat = data.usage?.routing_latency_ms || 24.5;
        const totalLat = data.usage?.total_latency_ms || (performance.now() - t0);

        // Update Panel Intelligence
        const analysis = data.usage?.routing_analysis || {};
        dom.metricModel.textContent = data.model.toUpperCase();
        dom.traceIntent.textContent = analysis.detected_intent || 'General';

        addLog(`Routing complete | Model: ${data.model}`);

        hideThinking();
        appendMessage('assistant', data.choices[0].message.content, {
            ...data.usage,
            display_latency: rLat.toFixed(1),
            full_path_latency: totalLat.toFixed(0)
        });

        // Persistent History Sync
        syncHistory(text, data.choices[0].message.content);
        forceScroll();
    } catch (error) {
        hideThinking();
        addLog('Critical Fault: Routing cluster timeout.');
        appendMessage('assistant', "⚠️ **Engine Fault**: Connection to the routing workspace was lost.");
    }
}

function appendMessage(role, content, usage = null) {
    const container = document.createElement('div');
    container.className = `flex flex-col w-full animate-in ${role === 'user' ? 'items-end' : 'items-start'}`;

    const body = document.createElement('div');
    body.className = role === 'user' ? 'msg-user' : 'msg-ai';
    body.innerHTML = formatMarkdown(content);
    container.appendChild(body);

    dom.messages.appendChild(container);
}

function syncHistory(query, response) {
    const current = state.history.find(h => h.id === state.currentChatId);
    if (!current) {
        state.history.unshift({
            id: state.currentChatId,
            title: query.substring(0, 30) + '...',
            messages: [{ role: 'user', content: query }, { role: 'assistant', content: response }]
        });
    } else {
        current.messages.push({ role: 'user', content: query }, { role: 'assistant', content: response });
    }

    localStorage.setItem('sbscr_history_studio', JSON.stringify(state.history.slice(0, 20)));
    renderHistory();
}

function renderHistory() {
    dom.history.innerHTML = '<label class="text-[10px] font-bold text-slate-600 uppercase tracking-[0.2em] px-5 mb-4 block">Archive</label>';
    state.history.forEach(item => {
        const btn = document.createElement('div');
        btn.className = `history-item w-full flex items-center gap-3 px-5 py-3 rounded-xl hover:bg-white/5 cursor-pointer text-sm font-medium transition-all ${state.currentChatId === item.id ? 'active' : 'text-slate-400'}`;
        btn.innerHTML = `
            <svg class="w-4 h-4 opacity-40" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-5l-5 5v-5z" stroke-width="2"/></svg>
            <span class="truncate">${item.title}</span>
        `;
        btn.onclick = () => loadHistorySession(item.id);
        dom.history.appendChild(btn);
    });
}

function loadHistorySession(id) {
    const session = state.history.find(h => h.id === id);
    if (!session) return;

    state.currentChatId = id;
    dom.messages.innerHTML = '';
    session.messages.forEach(m => appendMessage(m.role, m.content));
    addLog(`Studio: Session [${id}] reloaded.`);
    renderHistory();
    forceScroll();
}

function formatMarkdown(text) {
    return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/\n\n/g, '<div class="h-4"></div>')
        .replace(/\n/g, '<br>');
}

function createWelcome() {
    const div = document.createElement('div');
    div.className = 'welcome-center text-center pt-24 space-y-8 animate-in';
    div.innerHTML = `
        <div class="inline-block px-4 py-2 rounded-2xl bg-indigo-500/5 border border-indigo-500/20 text-xs font-bold text-indigo-400 uppercase tracking-widest">Intelligence Routing Node</div>
        <h1 class="text-5xl font-bold text-white tracking-tighter font-['Outfit']">How can I route your <br><span class="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-500">Intelligence session?</span></h1>
    `;
    return div;
}

function autoResize() {
    // Height is fixed at 44px in CSS as requested.
    // Logic removed to prevent height changes.
}

function forceScroll() {
    setTimeout(() => {
        dom.scroller.scrollTop = dom.scroller.scrollHeight;
    }, 100);
}

function fillExample(text) {
    dom.input.value = text;
    autoResize();
    dom.input.focus();
}

function showThinking() {
    const div = document.createElement('div');
    div.id = 'thinking-indicator';
    div.className = 'flex flex-col items-start w-full animate-in';
    div.innerHTML = `
        <div class="thinking-bubble">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
    `;
    dom.messages.appendChild(div);
    forceScroll();
}

function hideThinking() {
    const indicator = document.getElementById('thinking-indicator');
    if (indicator) indicator.remove();
}
