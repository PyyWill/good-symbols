/* Paper figures exported from PowerPoint as SVG: inlined so the web fonts apply, assembled left to right when
   they first scroll into view, and (for the CPR round) one panel highlighted at a time. */
(function () {
  const SS = window.SS, { onVisible, reduced } = SS;
  let uid = 0;

  async function inline(el) {
    const url = el.dataset.svg;
    let txt = await (await fetch(url)).text();
    // several PowerPoint SVGs share ids (clip0, img1, ...): prefix them per figure so references stay local
    const pfx = 'f' + (++uid) + '-';
    txt = txt.replace(/<\?xml[^>]*>/, '')
      .replace(/\bid="([^"]+)"/g, (m, id) => `id="${pfx}${id}"`)
      .replace(/url\(#([^)]+)\)/g, (m, id) => `url(#${pfx}${id})`)
      .replace(/(xlink:href|href)="#([^"]+)"/g, (m, attr, id) => `${attr}="#${pfx}${id}"`);
    el.innerHTML = txt;
    const svg = el.querySelector('svg');
    const w = parseFloat(svg.getAttribute('width')), h = parseFloat(svg.getAttribute('height'));
    if (!svg.getAttribute('viewBox') && w && h) svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    svg.removeAttribute('width'); svg.removeAttribute('height');
    svg.setAttribute('role', 'img');
    if (el.getAttribute('aria-label')) svg.setAttribute('aria-label', el.getAttribute('aria-label'));
    svg.style.width = '100%'; svg.style.height = 'auto';
    return { svg, w, h };
  }

  function figExtent(n, svg, w) {
    const R = svg.getBoundingClientRect(), r = n.getBoundingClientRect();
    if (!R.width || (!r.width && !r.height)) return null;
    return { x: (r.left - R.left) / R.width * w, w: r.width / R.width * w, yf: (r.top + r.height / 2 - R.top) / R.height };
  }

  function parts(svg) {
    const g = svg.querySelector(':scope > g');
    return g ? Array.from(g.children) : [];
  }

  function assemble(el, svg, w) {
    const items = parts(svg).map(n => { const e = figExtent(n, svg, w); return { n, x: e ? e.x + e.w * 0.15 : 0 }; });
    const r = el.getBoundingClientRect();
    if (reduced || r.top < window.innerHeight * 0.9) return;     // already on screen: leave it complete
    items.forEach(({ n }) => { n.style.opacity = 0; n.style.transition = 'opacity .5s ease'; });
    onVisible(el, () => {
      items.forEach(({ n, x }) => { n.style.transitionDelay = (Math.min(w, Math.max(0, x)) / w * 1.6).toFixed(2) + 's'; requestAnimationFrame(() => { n.style.opacity = 1; }); });
    }, { threshold: 0.3 });
  }

  const CPR_CAPS = [
    '<b>Sample graph.</b> Random-start descents on <i>h<sub>s</sub></i> give the graph nodes; edges join consecutive steps and nearest neighbours, scored by the highest energy along them, which is the barrier a descent must climb.',
    '<b>Verify.</b> Flooding the graph from its minima finds basins Φ<sub>1</sub>, Φ<sub>2</sub>, Φ<sub>3</sub>. Leaving Φ<sub>2</sub> or Φ<sub>3</sub> costs more than τ: the region is not faithful.',
    '<b>Sparsify.</b> New descents seeded at the flagged saddle reveal a low path between Φ<sub>1</sub> and Φ<sub>2</sub>: there exists a τ-descent faithful path between them, so Φ<sub>1</sub> ← Φ<sub>1</sub> ∪ Φ<sub>2</sub>, at no symbolic cost.',
    '<b>Split.</b> The remaining barrier is real. A cut through its lowest pass, along the gradient flip, invents one symbol, <i>ψ</i> ∈ {0, 1}.',
  ];

  async function initCPR() {
    const el = document.getElementById('cpr-fig');
    if (!el) return;
    const { svg, w } = await inline(el);
    // drop the panels' white backgrounds so the four panels sit directly on the page background
    svg.querySelectorAll('path[fill="#FFFFFF"]').forEach(p => { try { const b = p.getBBox(); if (b.width > 800 && b.height > 600) p.setAttribute('fill', 'none'); } catch (e) { /* not rendered */ } });
    const bounds0 = [0, w * 0.25, w * 0.5, w * 0.745, w + 1];
    // the bottom-row conclusion ("there exists a tau-descent faithful path ... : phi1 <- phi1 U phi2") belongs to panel (c)
    const items = parts(svg).map(n => { const e = figExtent(n, svg, w); let cx = e ? e.x + e.w / 2 : -1;
      if (e && e.yf > 0.8 && e.x >= bounds0[2] - 1) cx = (bounds0[2] + bounds0[3]) / 2;
      return { n, cx }; });
    items.forEach(({ n }) => { n.style.transition = 'opacity .45s ease'; });
    const bounds = bounds0;
    const btns = Array.from(document.querySelectorAll('.cpr-steps button')), cap = document.getElementById('cpr-cap');
    let cur = 0, timer = null, userTook = false;
    function show(k) {
      cur = k;
      items.forEach(({ n, cx }) => { const inPanel = cx >= bounds[k] && cx < bounds[k + 1]; n.style.opacity = inPanel ? 1 : 0.16; });
      btns.forEach((b, i) => b.setAttribute('aria-pressed', i === k ? 'true' : 'false'));
      cap.innerHTML = CPR_CAPS[k];
    }
    btns.forEach((b, i) => b.addEventListener('click', () => { userTook = true; clearInterval(timer); show(i); }));
    show(0);
    if (!reduced) onVisible(el, () => { timer = setInterval(() => { if (!userTook) show((cur + 1) % 4); }, 4200); }, { threshold: 0.4 });
  }

  async function initFigures() {
    // the two method figures (one sits in a hidden tab) load together when the Idea section is near; CPR when it is near
    const figs = Array.from(document.querySelectorAll('.pptfig[data-svg]')).filter(e => e.id !== 'cpr-fig');
    SS.whenNear(document.getElementById('idea'), async () => {
      for (const el of figs) {
        try { const { svg, w } = await inline(el); if (!el.closest('[hidden]')) assemble(el, svg, w); }
        catch (e) { console.warn('figure', el.dataset.svg, e); }
      }
    });
    SS.whenNear(document.getElementById('cpr-fig'), () => initCPR().catch(e => console.warn('cpr', e)));
    // instantiation tabs
    const tabs = [['tab-nn', 'inst-nn'], ['tab-pwa', 'inst-pwa']];
    tabs.forEach(([t, p]) => document.getElementById(t).addEventListener('click', () => {
      tabs.forEach(([t2, p2]) => { document.getElementById(t2).setAttribute('aria-selected', t2 === t); document.getElementById(p2).hidden = t2 !== t; });
      const svg = document.querySelector('#' + p + ' svg');
      if (svg && !reduced) {
        const items = parts(svg), w = svg.viewBox.baseVal.width;
        items.forEach(n => { n.style.transition = 'none'; n.style.opacity = 0; });
        requestAnimationFrame(() => items.forEach(n => { const e = figExtent(n, svg, w) || { x: 0, w: 0 };
          n.style.transition = 'opacity .45s ease'; n.style.transitionDelay = (Math.min(w, Math.max(0, e.x + e.w * 0.15)) / w * 1.2).toFixed(2) + 's'; n.style.opacity = 1; }));
      }
    }));
  }

  window.SS.initFigures = initFigures;
})();
