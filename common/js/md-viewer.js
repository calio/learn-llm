(function () {
  async function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = src; s.onload = resolve; s.onerror = reject;
      document.head.appendChild(s);
    });
  }

  async function loadStyles(href) {
    return new Promise((resolve, reject) => {
      const l = document.createElement('link');
      l.rel = 'stylesheet'; l.href = href; l.onload = resolve; l.onerror = reject;
      document.head.appendChild(l);
    });
  }

  function getParam(name) {
    const u = new URL(window.location.href);
    return u.searchParams.get(name);
  }

  function showError(msg) {
    const container = document.getElementById('md-root');
    container.innerHTML = `<div style="color:#f88">${msg}</div>`;
  }

  async function fetchText(path) {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${path}`);
    return await res.text();
  }

  async function main() {
    const container = document.getElementById('md-root');
    const path = getParam('path');
    if (!path) {
      container.textContent = 'Missing ?path=... parameter';
      return;
    }

    try {
      // Load libs
      await loadScript('https://cdn.jsdelivr.net/npm/marked/marked.min.js');
      await loadScript('https://cdn.jsdelivr.net/npm/highlight.js/lib/common.min.js');
      await loadStyles('https://cdn.jsdelivr.net/npm/highlight.js/styles/github-dark.min.css');

      const raw = await fetchText(path);
      window.__MD_RAW = raw; // expose raw for chat context

      // Configure marked
      marked.setOptions({
        highlight: function (code, lang) {
          try {
            if (lang && hljs.getLanguage(lang)) {
              return hljs.highlight(code, { language: lang }).value;
            }
            return hljs.highlightAuto(code).value;
          } catch (e) {
            return code;
          }
        },
        gfm: true,
        breaks: false,
      });

      // Resolve relative links/images in markdown to be relative to the md file path
      const base = new URL(path, window.location.origin + window.location.pathname).toString();
      const html = marked.parse(raw);
      container.innerHTML = html;

      container.querySelectorAll('a[href]').forEach((a) => {
        const href = a.getAttribute('href');
        if (!href) return;
        if (href.endsWith('.md')) {
          a.addEventListener('click', (e) => {
            e.preventDefault();
            // Resolve against the md base
            const next = new URL(href, base).pathname.replace(/^\//, '');
            window.location.href = `docs.html?path=${encodeURIComponent(next)}`;
          });
        }
      });

      // Update page title
      document.title = `${path} – Viewer`;
    } catch (e) {
      console.error(e);
      showError('Failed to load document. If you opened this file using file://, please run the dev server: npm run dev');
    }
  }

  window.addEventListener('DOMContentLoaded', main);
})();
