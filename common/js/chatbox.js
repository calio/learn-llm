(function (global) {
  const DEFAULTS = {
    endpoint: 'http://localhost:5057/api/chat',
    title: 'Ask ChatGPT',
    intro: 'Ask questions about this page.',
    position: 'bottom-right',
    welcome: 'Hi! How can I help?',
    system: 'You are a helpful tutor for an LLM learning repo. Answer concisely and, when relevant, point to files/sections by path.',
    pageContext: true,
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

    const toggle = h('button', { class: 'cbx-toggle', title: options.title }, '💬');

    const panel = h('div', { class: 'cbx-panel' },
      h('div', { class: 'cbx-header' },
        h('div', { class: 'cbx-title' }, options.title),
        h('button', { class: 'cbx-close', title: 'Close' }, '✕')
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

    if (!options.endpoint) {
      console.error('[Chatbox] Missing endpoint');
      return;
    }

    const container = document.getElementById('chatbox-root') || document.body;
    const { root, toggle, panel } = createUI(options);
    container.appendChild(root);

    if (options.position === 'bottom-right') root.classList.add('cbx-pos-br');

    const bodyEl = panel.querySelector('.cbx-body');
    const closeEl = panel.querySelector('.cbx-close');
    const formEl = panel.querySelector('form');
    const textEl = panel.querySelector('textarea');

    const history = [];

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

    async function send(text) {
      if (!text.trim()) return;
      addMessage('user', text);
      textEl.value = '';
      const typingEl = addTyping();
      setLoading(true);

      const payload = {
        system: options.system,
        messages: history.slice(-10),
        context: currentContext(),
      };

      try {
        const res = await fetch(options.endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        typingEl.remove();
        addMessage('assistant', data.reply || '(no reply)');
      } catch (e) {
        typingEl.remove();
        addMessage('assistant', 'Error contacting assistant. Is the server running?');
        console.error('[Chatbox] request failed', e);
      } finally {
        setLoading(false);
      }
    }

    // Seed with welcome
    addMessage('assistant', options.welcome);

    // Events
    toggle.addEventListener('click', () => {
      root.classList.add('cbx-open');
      textEl.focus();
    });
    closeEl.addEventListener('click', () => root.classList.remove('cbx-open'));

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
