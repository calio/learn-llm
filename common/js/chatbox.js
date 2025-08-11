(function (global) {
  const DEFAULTS = {
    endpoint: 'http://localhost:5057/api/chat',
    title: 'Ask ChatGPT',
    intro: 'Ask questions about this page.',
    position: 'bottom-right',
    welcome: 'Hi! How can I help?',
    system: 'You are a helpful tutor for an LLM learning repo. Answer concisely and, when relevant, point to files/sections by path.',
    pageContext: true,
    provider: 'openai', // 'proxy' | 'openai'
    // Sidebar layout options
    layout: 'bubble', // 'bubble' | 'sidebar'
    sidebarWidth: 380,
    // Page content inclusion
    pageContent: false,
    contextSelector: null,
    pageContentLimit: 8000,
    getPageContent: null,
  };

  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    if (attrs) {
      Object.entries(attrs).forEach(([k, v]) => {
        if (k === 'class') el.className = v;
        else if (k === 'html') el.innerHTML = v;
        else el.setAttribute(k, v);
      });
    }
    for (const c of children) {
      if (c == null) continue;
      if (typeof c === 'string') el.appendChild(document.createTextNode(c));
      else el.appendChild(c);
    }
    return el;
  }

  function formatMessage(role, content) {
    return { role, content };
  }

  function createUI(options) {
    const root = h('div', { class: 'cbx-root' });

    const toggle = h('button', { class: 'cbx-toggle', title: options.title }, 'Chat');

    const panel = h('div', { class: 'cbx-panel' },
      h('div', { class: 'cbx-header' },
        h('div', { class: 'cbx-title' }, options.title),
        h('div', { class: 'cbx-actions' },
          h('button', { class: 'cbx-key', title: 'Set API key' }, '🔑'),
          h('button', { class: 'cbx-collapse', title: 'Collapse sidebar' }, '⟨'),
          h('button', { class: 'cbx-close', title: 'Close' }, '✕')
        )
      ),
      h('div', { class: 'cbx-body' }),
      h('form', { class: 'cbx-input' },
        h('textarea', { class: 'cbx-text', rows: '1', placeholder: 'Ask a question...' }),
        h('button', { class: 'cbx-send', type: 'submit' }, 'Send')
      )
    );

    root.appendChild(toggle);
    root.appendChild(panel);

    return { root, toggle, panel };
  }

  function mount(opts) {
    const options = Object.assign({}, DEFAULTS, opts || {});

    const container = document.getElementById('chatbox-root') || document.body;
    const { root, toggle, panel } = createUI(options);
    container.appendChild(root);

    const bodyEl = panel.querySelector('.cbx-body');
    const closeEl = panel.querySelector('.cbx-close');
    const keyEl = panel.querySelector('.cbx-key');
    const collapseEl = panel.querySelector('.cbx-collapse');
    const formEl = panel.querySelector('form');
    const textEl = panel.querySelector('textarea');

    const history = [];

    // Layout setup
    let baseBodyPaddingRight = 0;
    try {
      baseBodyPaddingRight = parseFloat(getComputedStyle(document.body).paddingRight || '0') || 0;
    } catch (_) {}

    function applySidebarPadding(expanded) {
      try {
        document.body.style.paddingRight = expanded ? `${baseBodyPaddingRight + options.sidebarWidth}px` : `${baseBodyPaddingRight}px`;
      } catch (_) {}
    }

    if (options.layout === 'sidebar') {
      root.classList.add('cbx-layout-sidebar', 'cbx-open');
      root.style.setProperty('--cbx-sidebar-width', `${options.sidebarWidth}px`);
      applySidebarPadding(true);
      // Start with expanded sidebar by default; toggle hidden until collapsed
      toggle.classList.add('cbx-sidetab');
      toggle.style.display = 'none';
      closeEl.style.display = 'none';
    } else if (options.position === 'bottom-right') {
      root.classList.add('cbx-pos-br');
    }

    function addMessage(role, text) {
      const msg = h('div', { class: `cbx-msg cbx-${role}` },
        h('div', { class: 'cbx-bubble' }, text)
      );
      bodyEl.appendChild(msg);
      bodyEl.scrollTop = bodyEl.scrollHeight;
      history.push(formatMessage(role, text));
    }

    function setLoading(on) {
      formEl.querySelector('button').disabled = on;
      textEl.disabled = on;
    }

    function addTyping() {
      const el = h('div', { class: 'cbx-msg cbx-assistant cbx-typing' }, h('div', { class: 'cbx-bubble' }, 'Typing...'));
      bodyEl.appendChild(el);
      bodyEl.scrollTop = bodyEl.scrollHeight;
      return el;
    }

    function currentContext() {
      if (!options.pageContext) return null;
      return {
        url: window.location.href,
        path: window.location.pathname,
        title: document.title,
      };
    }

    function getApiKey() {
      const k = localStorage.getItem('OPENAI_API_KEY') || '';
      return k;
    }

    function promptApiKey() {
      const current = getApiKey();
      const v = window.prompt('Enter OpenAI API Key (stored in this browser). Use only for local/personal use.', current || '');
      if (v && v.trim()) {
        localStorage.setItem('OPENAI_API_KEY', v.trim());
        return v.trim();
      }
      return null;
    }

    function truncate(text, maxLen) {
      if (!text) return '';
      if (text.length <= maxLen) return text;
      return text.slice(0, maxLen) + `\n...[truncated ${text.length - maxLen} chars]`;
    }

    function collectPageContent() {
      try {
        if (typeof options.getPageContent === 'function') {
          const v = options.getPageContent();
          if (v) return v;
        }
        if (window.__MD_RAW) return String(window.__MD_RAW);
        if (options.contextSelector) {
          const node = document.querySelector(options.contextSelector);
          if (node) return node.innerText || node.textContent || '';
        }
        return document.body ? (document.body.innerText || document.body.textContent || '') : '';
      } catch (_) {
        return '';
      }
    }

    async function sendViaProxy(payload) {
      const res = await fetch(options.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      return data.reply || '(no reply)';
    }

    async function sendViaOpenAI(extraSystemMessages) {
      let apiKey = getApiKey();
      if (!apiKey) {
        apiKey = promptApiKey();
        if (!apiKey) throw new Error('API key required');
      }

      const sys = options.system;
      const msgs = [{ role: 'system', content: sys }];
      if (Array.isArray(extraSystemMessages)) msgs.push(...extraSystemMessages);
      msgs.push(...history.slice(-10));

      const res = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: 'gpt-4o-mini',
          messages: msgs,
          temperature: 0.3,
        })
      });
      if (!res.ok) {
        if (res.status === 401) localStorage.removeItem('OPENAI_API_KEY');
        throw new Error(`OpenAI HTTP ${res.status}`);
      }
      const data = await res.json();
      return (data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content) || '(no reply)';
    }

    async function send(text) {
      if (!text.trim()) return;
      addMessage('user', text);
      textEl.value = '';
      const typingEl = addTyping();
      setLoading(true);

      const ctx = currentContext();
      const extraSystems = [];
      if (ctx) extraSystems.push({ role: 'system', content: `CONTEXT: URL=${ctx.url} PATH=${ctx.path} TITLE=${ctx.title}` });
      if (options.pageContent) {
        const content = truncate(collectPageContent(), options.pageContentLimit);
        if (content) extraSystems.push({ role: 'system', content: `PAGE_CONTENT:\n${content}` });
      }

      const payload = {
        system: options.system,
        messages: history.slice(-10),
        context: ctx,
        page_content: options.pageContent ? collectPageContent().slice(0, options.pageContentLimit) : null,
      };

      try {
        const reply = options.provider === 'openai' ? await sendViaOpenAI(extraSystems) : await sendViaProxy(payload);
        typingEl.remove();
        addMessage('assistant', reply);
      } catch (e) {
        typingEl.remove();
        addMessage('assistant', 'Error contacting assistant. If using OpenAI mode, set a valid API key.');
        console.error('[Chatbox] request failed', e);
      } finally {
        setLoading(false);
      }
    }

    // Collapsing behavior for sidebar
    function setCollapsed(collapsed) {
      if (collapsed) {
        root.classList.add('cbx-collapsed');
        toggle.style.display = 'block';
        applySidebarPadding(false);
      } else {
        root.classList.remove('cbx-collapsed');
        toggle.style.display = 'none';
        applySidebarPadding(true);
      }
    }

    // Seed with welcome
    const modeNote = options.provider === 'openai' ? '\nNote: Using browser OpenAI mode. Your API key stays in this browser.' : '';
    addMessage('assistant', options.welcome + modeNote);

    // Events
    toggle.addEventListener('click', () => {
      if (root.classList.contains('cbx-layout-sidebar')) {
        setCollapsed(false);
      } else {
        root.classList.add('cbx-open');
        textEl.focus();
      }
    });
    if (collapseEl) {
      collapseEl.addEventListener('click', () => setCollapsed(true));
    }
    closeEl.addEventListener('click', () => root.classList.remove('cbx-open'));
    keyEl.addEventListener('click', () => promptApiKey());

    // In sidebar layout, always open and hide close button, use collapse instead
    if (options.layout === 'sidebar') {
      root.classList.add('cbx-open');
      closeEl.style.display = 'none';
    }

    formEl.addEventListener('submit', (e) => {
      e.preventDefault();
      send(textEl.value);
    });

    textEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        formEl.dispatchEvent(new Event('submit'));
      }
    });

    return { send, root };
  }

  global.Chatbox = { mount };
})(window);
