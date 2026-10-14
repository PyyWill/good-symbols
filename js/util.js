/* Shared helpers: colormap, canvas setup, easing, data loading, reveal-on-scroll. */
(function () {
  const COOLWARM = [[59,76,192],[68,90,204],[78,104,216],[88,117,225],[98,130,234],[108,143,241],[119,154,247],[130,166,251],[141,176,254],[152,185,255],[163,194,254],[174,201,252],[185,208,249],[195,213,244],[204,217,237],[213,219,229],[221,220,220],[229,216,209],[236,211,197],[241,204,184],[245,196,172],[247,186,159],[247,176,147],[246,165,134],[244,152,122],[240,139,110],[235,125,98],[228,110,86],[221,95,75],[212,78,65],[202,59,55],[190,36,46],[180,4,38]];
  const LUT = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255 * 32, k = Math.min(31, Math.floor(t)), f = t - k;
    for (let c = 0; c < 3; c++) LUT[i * 3 + c] = Math.round(COOLWARM[k][c] * (1 - f) + COOLWARM[k + 1][c] * f);
  }
  const reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function setupCanvas(canvas) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: r.width, h: r.height, dpr };
  }

  const ease = {
    inOut: t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2,
    out: t => 1 - Math.pow(1 - t, 3),
    outQuint: t => 1 - Math.pow(1 - t, 5),
  };

  const cache = {};
  function getJSON(url) {
    if (!cache[url]) cache[url] = fetch(url).then(r => { if (!r.ok) throw new Error(url + ' ' + r.status); return r.json(); });
    return cache[url];
  }

  /* Run fn once when el first becomes visible (or immediately when observers are unavailable). */
  function onVisible(el, fn, opts) {
    if (!el) return;
    if (!('IntersectionObserver' in window)) { fn(); return; }
    const io = new IntersectionObserver((ents) => {
      ents.forEach(e => { if (e.isIntersecting) { io.disconnect(); fn(); } });
    }, Object.assign({ threshold: 0.25 }, opts || {}));
    io.observe(el);
  }

  /* Entrance for elements below the fold only: content above the fold is never hidden. */
  function initRise() {
    document.documentElement.classList.add('js');
    const els = Array.from(document.querySelectorAll('.rise'));
    const vh = window.innerHeight;
    els.forEach((el, i) => {
      const r = el.getBoundingClientRect();
      if (reduced || r.top < vh * 0.95) return;
      el.classList.add('wait');
      onVisible(el, () => setTimeout(() => el.classList.remove('wait'), (i % 3) * 90), { threshold: 0.1 });
    });
  }

  function countUp(el, to, digits, ms) {
    if (reduced) { el.textContent = to.toFixed(digits); return; }
    const t0 = performance.now(), from = 0;
    function tick(now) {
      const t = Math.max(0, Math.min(1, (now - t0) / (ms || 900)));
      el.textContent = (from + (to - from) * ease.outQuint(t)).toFixed(digits);
      if (t < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

  /* Run fn once when el comes within about one screen of the viewport (above or below), so heavy parts load just before they are seen
     (and never compete with the hero on first load). */
  function whenNear(el, fn) {
    if (!el) return;
    if (!('IntersectionObserver' in window)) { fn(); return; }
    const io = new IntersectionObserver(ents => { if (ents.some(e => e.isIntersecting)) { io.disconnect(); fn(); } }, { rootMargin: '100% 0px 120% 0px' });
    io.observe(el);
  }

  /* three.js and its orbit controls, loaded on demand (only the 3-D panels need them) */
  const THREE_SRC = ['https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js', 'https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js'];
  const loadScript = src => new Promise((res, rej) => { const s = document.createElement('script'); s.src = src; s.onload = res; s.onerror = rej; document.head.appendChild(s); });
  let threeP = null;
  function loadThree() { return threeP || (threeP = window.THREE ? Promise.resolve() : loadScript(THREE_SRC[0]).then(() => loadScript(THREE_SRC[1]))); }

  window.SS = { LUT, COOLWARM, setupCanvas, ease, getJSON, onVisible, whenNear, loadThree, initRise, countUp, reduced, cssVar };
})();
