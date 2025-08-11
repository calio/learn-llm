(function () {
  const axEl = document.getElementById('ax');
  const ayEl = document.getElementById('ay');
  const bxEl = document.getElementById('bx');
  const byEl = document.getElementById('by');
  const metricEl = document.getElementById('metric');
  const valuesEl = document.getElementById('values');
  const canvas = document.getElementById('plane');
  const ctx = canvas.getContext('2d');
  const randomizeBtn = document.getElementById('randomize');
  const resetBtn = document.getElementById('reset');

  const EPS = 1e-12;

  function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1];
  }

  function norm(a) {
    return Math.sqrt(a[0] * a[0] + a[1] * a[1]);
  }

  function cosineSimilarity(a, b) {
    const na = norm(a);
    const nb = norm(b);
    if (na < EPS || nb < EPS) return 0;
    return dot(a, b) / (na * nb);
  }

  function cosineDistance(a, b) {
    return 1 - cosineSimilarity(a, b);
  }

  function euclidean(a, b) {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    return Math.sqrt(dx * dx + dy * dy);
  }

  function manhattan(a, b) {
    return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);
  }

  function chebyshev(a, b) {
    return Math.max(Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1]));
  }

  function angular(a, b) {
    const c = Math.max(-1, Math.min(1, cosineSimilarity(a, b)));
    return Math.acos(c);
  }

  function readVectors() {
    return [
      [parseFloat(axEl.value), parseFloat(ayEl.value)],
      [parseFloat(bxEl.value), parseFloat(byEl.value)],
    ];
  }

  function draw() {
    const [a, b] = readVectors();

    // Clear
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Setup coordinates
    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const scale = 60; // pixels per unit

    // Grid
    ctx.strokeStyle = '#eee';
    ctx.lineWidth = 1;
    for (let x = 0; x <= w; x += 30) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y <= h; y += 30) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = '#999';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, cy);
    ctx.lineTo(w, cy);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, h);
    ctx.stroke();

    // Draw vector helper
    const drawVector = (v, color, label) => {
      const x = cx + v[0] * scale;
      const y = cy - v[1] * scale;

      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 3;

      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(x, y);
      ctx.stroke();

      // Arrowhead
      const angle = Math.atan2(cy - y, x - cx);
      const ah = 10;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x - ah * Math.cos(angle - Math.PI / 6), y + ah * Math.sin(angle - Math.PI / 6));
      ctx.lineTo(x - ah * Math.cos(angle + Math.PI / 6), y + ah * Math.sin(angle + Math.PI / 6));
      ctx.closePath();
      ctx.fill();

      // Label
      ctx.font = '12px system-ui, -apple-system, Segoe UI, Roboto, sans-serif';
      ctx.fillText(label, x + 6, y - 6);
    };

    drawVector(a, '#0070f3', 'A');
    drawVector(b, '#e91e63', 'B');

    // Angle arc between A and B
    const c = Math.max(-1, Math.min(1, cosineSimilarity(a, b)));
    const theta = Math.acos(c);
    const angA = Math.atan2(a[1], a[0]);
    const angB = Math.atan2(b[1], b[0]);
    const start = angA;
    // normalize sweep direction to smaller angle
    let delta = angB - angA;
    while (delta > Math.PI) delta -= 2 * Math.PI;
    while (delta < -Math.PI) delta += 2 * Math.PI;

    ctx.strokeStyle = '#444';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, 40, start, start + delta, delta < 0);
    ctx.stroke();

    // Results list
    const results = [
      ['Cosine similarity', cosineSimilarity(a, b)],
      ['Cosine distance', cosineDistance(a, b)],
      ['Euclidean (L2)', euclidean(a, b)],
      ['Manhattan (L1)', manhattan(a, b)],
      ['Chebyshev (L∞)', chebyshev(a, b)],
      ['Angular distance (rad)', theta],
    ];

    const selected = metricEl.value;
    valuesEl.innerHTML = '';
    for (const [name, value] of results) {
      const li = document.createElement('li');
      const isPrimary =
        (selected === 'cosine' && name.startsWith('Cosine similarity')) ||
        (selected === 'cosine_distance' && name.startsWith('Cosine distance')) ||
        (selected === 'euclidean' && name.startsWith('Euclidean')) ||
        (selected === 'manhattan' && name.startsWith('Manhattan')) ||
        (selected === 'chebyshev' && name.startsWith('Chebyshev')) ||
        (selected === 'angular' && name.startsWith('Angular'));
      li.textContent = `${name}: ${value.toFixed(6)}`;
      if (isPrimary) li.classList.add('primary');
      valuesEl.appendChild(li);
    }
  }

  function randomize() {
    function rnd() {
      return parseFloat((Math.random() * 2 - 1).toFixed(2));
    }
    axEl.value = rnd();
    ayEl.value = rnd();
    bxEl.value = rnd();
    byEl.value = rnd();
    draw();
  }

  function reset() {
    axEl.value = 1;
    ayEl.value = 0;
    bxEl.value = 0.7;
    byEl.value = 0.7;
    metricEl.value = 'cosine';
    draw();
  }

  [axEl, ayEl, bxEl, byEl, metricEl].forEach((el) => el.addEventListener('input', draw));
  randomizeBtn.addEventListener('click', randomize);
  resetBtn.addEventListener('click', reset);

  draw();
})();
