/* Hero: an illustrative rugged energy landscape drawn as a 2-D map and solved live in the page.
   Phase A: 24 random restarts settle in whatever basin they reach. Phase B: two cuts through the passes invent symbols
   (regions R1..R3). Phase C: one confined descent per region; the lowest floor is the global optimum.
   The solver is the paper's annealed normalized-gradient descent (600 steps + 15 polish); the landscape parameters
   come from data/hero_landscape.json (generator: tools/hero/landP.py, which mirrors h below). */
(function () {
  const N0 = 0.2, N1 = 0.005, POLISH = 15, REC = 10, R = 24;

  function makeH(P) {
    // cells are numbered by how many cuts lie on their left: 0, 1, 2 from left to right
    const cellOf = (x, y) => P.cuts.reduce((c, [ux, uy, b]) => c + (ux * x + uy * y >= b ? 1 : 0), 0);
    const nw = P.wmodes.length / 2;
    function h(x, y) {
      // base: soft minimum of one bowl per region, centred on the region's lowest point (boundaries = the straight cuts)
      const [ma, mb, md] = P.metric || [P.kappa, 0, P.kappa];   // shared (tilted, elliptic) bowl metric
      const E = P.centres.map(([cx, cy], k) => ma * (x - cx) ** 2 + 2 * mb * (x - cx) * (y - cy) + md * (y - cy) ** 2 + (P.offs ? P.offs[k] : 0));
      const m = Math.min(...E);
      let v = m - P.T * Math.log(E.reduce((a, e) => a + Math.exp(-(e - m) / P.T), 0));
      const r2 = (x / P.AX) ** 2 + (y / P.AY) ** 2;
      v += P.bowl * r2 + P.edge * Math.max(0, r2 - P.r0) ** 2;
      // pits at warped coordinates, so their shapes are irregular
      let wx = 0, wy = 0;
      P.wmodes.forEach(([k, t, p, a], i) => { const s = a * Math.sin(k * (Math.cos(t) * x + Math.sin(t) * y) + p); if (i < nw) wx += s; else wy += s; });
      const xw = x + P.warp * wx, yw = y + P.warp * wy;
      for (const [cx, cy, D, sx, sy, ang] of P.cores) {
        const c = Math.cos(ang), s = Math.sin(ang), dx = xw - cx, dy = yw - cy, a = c * dx + s * dy, b = -s * dx + c * dy;
        v -= D * Math.exp(-(a * a) / (2 * sx * sx) - (b * b) / (2 * sy * sy));
      }
      // pseudo-pits and bumps (skipped beyond 6 sigma: exp(-18) is invisible to the finite-difference gradient)
      for (const [cx, cy, D, s] of P.pits) { const d2 = (x - cx) ** 2 + (y - cy) ** 2; if (d2 < 36 * s * s) v -= D * Math.exp(-d2 / (2 * s * s)); }
      for (const [cx, cy, D, s] of P.spikes) { const d2 = (x - cx) ** 2 + (y - cy) ** 2; if (d2 < 36 * s * s) v += D * Math.exp(-d2 / (2 * s * s)); }
      // rugged multi-scale field, damped near the pits' floors and along the passes
      let nz = 0, calm = 1;
      for (const [k, th, ph, a] of P.modes) nz += a * Math.sin(k * (Math.cos(th) * x + Math.sin(th) * y) + ph);
      for (const [cx, cy] of P.calms) calm *= 1 - Math.exp(-((xw - cx) ** 2 + (yw - cy) ** 2) / (2 * P.calm * P.calm));
      for (const [ux, uy, b] of P.cuts) calm *= 1 - P.pass_ * Math.exp(-((ux * x + uy * y - b) ** 2) / (2 * P.passw * P.passw));
      return v + P.noise * (P.cmin + (1 - P.cmin) * calm) * nz;
    }
    return { h, cellOf };
  }

  function rng(seed) {               // mulberry32 + Box-Muller: the same animation on every load
    let a = seed >>> 0;
    const u = () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
    return { u, n: () => Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u()) };
  }

  // display only: a centred moving average whose window shrinks as the descent anneals (ends kept), so the early
  // high-noise exploration reads as a path rather than a scribble while the settling part stays exact
  function smooth(pts) {
    const n = pts.length / 2, out = [];
    for (let i = 0; i < n; i++) {
      const k = Math.min(Math.round(9 * Math.pow(1 - i / n, 2)) + 1, i, n - 1 - i); let sx = 0, sy = 0;
      for (let j = i - k; j <= i + k; j++) { sx += pts[2 * j]; sy += pts[2 * j + 1]; }
      out.push(sx / (2 * k + 1), sy / (2 * k + 1));
    }
    return out;
  }

  function solver(P, H) {
    const { h, cellOf } = H, S = 0.5 * (P.AX + P.AY), STEPS = P.steps || 600;
    const clampX = x => Math.max(-P.AX, Math.min(P.AX, x)), clampY = y => Math.max(-P.AY, Math.min(P.AY, y));
    function project(x, y, c) {
      for (let it = 0; it < 2; it++) P.cuts.forEach(([ux, uy, b], i) => {
        const s = ux * x + uy * y, t = c > i ? Math.max(s, b + 1e-3) : Math.min(s, b - 1e-3);
        x += (t - s) * ux; y += (t - s) * uy;
      });
      return [clampX(x), clampY(y)];
    }
    return function descend(x, y, r, conf) {
      const pts = [], e = 1e-4;
      const grad = (x, y) => { const gx = (h(x + e, y) - h(x - e, y)) / (2 * e), gy = (h(x, y + e) - h(x, y - e)) / (2 * e), n = Math.hypot(gx, gy) + 1e-12; return [gx / n, gy / n]; };
      for (let t = 0; t < STEPS; t++) {
        if (t % REC === 0) pts.push(x, y);
        const [gx, gy] = grad(x, y), nz = N0 * Math.pow(N1 / N0, t / (STEPS - 1));
        x = clampX(x - 0.02 * S * gx + nz * S * 0.5 * r.n()); y = clampY(y - 0.02 * S * gy + nz * S * 0.5 * r.n());
        if (conf != null) [x, y] = project(x, y, conf);
      }
      for (let t = 0; t < POLISH; t++) {
        const [gx, gy] = grad(x, y);
        x = clampX(x - 0.004 * S * gx); y = clampY(y - 0.004 * S * gy);
        if (conf != null) [x, y] = project(x, y, conf);
      }
      pts.push(x, y);
      return { pts: smooth(pts), v: h(x, y), cell: cellOf(x, y) };
    };
  }

  // value grid over the map (also gives each region's floor and the global minimum); row 0 is the top edge
  function grid(P, H, NX, NY) {
    const vals = new Float32Array(NX * NY), cells = new Uint8Array(NX * NY);
    for (let iy = 0; iy < NY; iy++) for (let ix = 0; ix < NX; ix++) {
      const x = -P.AX + 2 * P.AX * (ix + 0.5) / NX, y = P.AY - 2 * P.AY * (iy + 0.5) / NY, k = iy * NX + ix;
      vals[k] = H.h(x, y); cells[k] = H.cellOf(x, y);
    }
    return { NX, NY, vals, cells };
  }

  function build(P) {
    const H = makeH(P), descend = solver(P, H), G = grid(P, H, 400, 200);
    const regions = [];
    for (let c = 0; c <= P.cuts.length; c++) regions.push({ c, name: 'R' + (c + 1), floor: Infinity });
    let hmin = Infinity;
    G.vals.forEach((v, k) => { const g = regions[G.cells[k]]; g.floor = Math.min(g.floor, v); hmin = Math.min(hmin, v); });
    const r1 = rng(10), rest = [];
    for (let k = 0; k < R; k++) rest.push(descend(-P.AX + 2 * P.AX * r1.u(), -P.AY + 2 * P.AY * r1.u(), r1));
    const r2 = rng(P.seedPer || 5), per = regions.map(reg => {
      let x, y; do { x = -P.AX + 2 * P.AX * r2.u(); y = -P.AY + 2 * P.AY * r2.u(); } while (H.cellOf(x, y) !== reg.c);
      return Object.assign(descend(x, y, r2, reg.c), { reg });
    });
    // tag anchors: as high as possible in each region, clear of the cuts, the map border and the region's floor
    regions.forEach((g, k) => {
      const e = per[k].pts, fx = e[e.length - 2], fy = e[e.length - 1];
      let bestS = -Infinity;
      for (let iy = 0; iy <= 40; iy++) for (let ix = 0; ix <= 80; ix++) {
        const x = -P.AX + 2 * P.AX * ix / 80, y = -P.AY + 2 * P.AY * iy / 40;
        if (H.cellOf(x, y) !== g.c) continue;
        const dc = Math.min(...P.cuts.map(([ux, uy, b]) => Math.abs(ux * x + uy * y - b))), de = Math.min(P.AX - Math.abs(x), P.AY - Math.abs(y));
        if (dc < 0.2 || de < 0.12 || Math.hypot(x - fx, y - fy) < 0.4) continue;
        const sc = y - 0.25 * Math.abs(x - (fx + 0.0));   // prefer the top, near the floor's column
        if (sc > bestS) { bestS = sc; g.lx = x; g.ly = y; }
      }
      if (g.lx == null) { g.lx = fx; g.ly = fy + 0.2; }
    });
    return { H, G, hmin, regions, rest, per };
  }

  /* ---------------- 2-D rendering ---------------- */
  /* map pixels: coolwarm by height rank (balanced) blended with absolute height (deeper floors read darker), with a light
     hillshade so the relief reads at a glance. Returns RGBA bytes for the NX x NY value grid (row 0 = top). */
  function mapPixels(P, G, LUT) {
    const { NX, NY, vals } = G;
    const order = Array.from(vals.keys()).sort((i, j) => vals[i] - vals[j]), rank = new Float32Array(vals.length);
    order.forEach((k, r) => { rank[k] = r / (vals.length - 1); });
    const lo = vals[order[0]], hi = vals[order[Math.floor(order.length * 0.98)]];
    const px = new Uint8ClampedArray(NX * NY * 4), dX = 2 * P.AX / NX, dY = 2 * P.AY / NY;
    for (let iy = 0; iy < NY; iy++) for (let ix = 0; ix < NX; ix++) {
      const k = iy * NX + ix, t = Math.min(1, 0.72 * rank[k] + 0.28 * Math.min(1, (vals[k] - lo) / (hi - lo)));
      const j = Math.round(t * 255) * 3;
      const gx = (vals[iy * NX + Math.min(NX - 1, ix + 1)] - vals[iy * NX + Math.max(0, ix - 1)]) / (2 * dX);
      const gy = (vals[Math.max(0, iy - 1) * NX + ix] - vals[Math.min(NY - 1, iy + 1) * NX + ix]) / (2 * dY);
      const shade = 1 + 0.11 * Math.max(-1.5, Math.min(1.5, (-gx * 0.6 + gy * 0.8) * 0.45));   // light from the upper left
      px[4 * k] = LUT[j] * shade; px[4 * k + 1] = LUT[j + 1] * shade; px[4 * k + 2] = LUT[j + 2] * shade; px[4 * k + 3] = 255;
    }
    return { NX, NY, px };
  }

  /* the model the animation needs, rounded for shipping as JSON (tools/hero/export_model.js writes it with the map PNG) */
  function compact(P, M) {
    const r4 = a => Array.from(a, v => Math.round(v * 1e5) / 1e5);
    return {
      P: { AX: P.AX, AY: P.AY, cuts: P.cuts, tau: P.tau },
      M: { hmin: M.hmin, regions: M.regions.map(g => ({ c: g.c, name: g.name, floor: g.floor, lx: g.lx, ly: g.ly })),
        rest: M.rest.map(s => ({ pts: r4(s.pts), v: s.v })), per: M.per.map(s => ({ pts: r4(s.pts), v: s.v, c: s.reg.c })) },
    };
  }

  function startHero2D(P, M, mapImg) {
    const canvas = document.getElementById('hero-canvas'), ctx = canvas.getContext('2d');
    const ease = window.SS.ease;
    // base map: the precomputed image when available, else computed here from the value grid
    const baseC = document.createElement('canvas');
    if (mapImg) { baseC.width = mapImg.naturalWidth; baseC.height = mapImg.naturalHeight; baseC.getContext('2d').drawImage(mapImg, 0, 0); }
    else {
      const { NX, NY, px } = mapPixels(P, M.G, window.SS.LUT);
      baseC.width = NX; baseC.height = NY; baseC.getContext('2d').putImageData(new ImageData(px, NX, NY), 0, 0);
    }

    let W = 0, Hh = 0;
    const X = x => (x + P.AX) / (2 * P.AX) * W, Y = y => (P.AY - y) / (2 * P.AY) * Hh;
    function resize() {
      const r = canvas.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = r.width; Hh = r.height; canvas.width = Math.max(1, Math.round(W * dpr)); canvas.height = Math.max(1, Math.round(Hh * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function path(pts, p, style) {
      const n = pts.length / 2, q = (n - 1) * Math.min(1, p), m = Math.max(1, Math.floor(q)), f = q - m;
      ctx.beginPath(); ctx.moveTo(X(pts[0]), Y(pts[1]));
      for (let i = 1; i <= m; i++) ctx.lineTo(X(pts[2 * i]), Y(pts[2 * i + 1]));
      let hx = pts[2 * m], hy = pts[2 * m + 1];
      if (m < n - 1 && f > 0) { hx += (pts[2 * m + 2] - hx) * f; hy += (pts[2 * m + 3] - hy) * f; ctx.lineTo(X(hx), Y(hy)); }
      ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      if (style.halo) { ctx.strokeStyle = style.halo; ctx.lineWidth = style.w + 3; ctx.stroke(); }
      ctx.strokeStyle = style.c; ctx.lineWidth = style.w; ctx.stroke();
      return [X(hx), Y(hy)];
    }
    function dot(x, y, r, fill, stroke) {
      ctx.beginPath(); ctx.arc(x, y, r, 0, 7);
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.4; ctx.stroke(); }
    }
    function tag(text, x, y, a) {
      const fs = Math.max(11, Math.min(15, W / 64));   // smaller tags on narrow screens
      ctx.save(); ctx.globalAlpha = a; ctx.font = `italic 700 ${fs}px Caladea, Cambria, Georgia, serif`;
      const w = ctx.measureText(text).width + fs, h = fs * 1.6;
      ctx.fillStyle = 'rgba(63, 86, 201, 0.95)'; ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(x - w / 2, y - h / 2, w, h, 6); else ctx.rect(x - w / 2, y - h / 2, w, h);
      ctx.fill(); ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(text, x, y + 1); ctx.restore();
    }
    function cutLine([ux, uy, b], p) {
      if (p <= 0) return;
      const y0 = P.AY, y1 = P.AY - 2 * P.AY * p, x0 = (b - uy * y0) / ux, x1 = (b - uy * y1) / ux;
      ctx.save(); ctx.lineCap = 'round'; ctx.beginPath(); ctx.moveTo(X(x0), Y(y0)); ctx.lineTo(X(x1), Y(y1));
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.75)'; ctx.lineWidth = 5; ctx.stroke();
      ctx.setLineDash([9, 6]); ctx.strokeStyle = '#3f56c9'; ctx.lineWidth = 2.2; ctx.stroke(); ctx.restore();
    }

    const TAU = P.tau || 0.3, best = M.hmin, hits = M.rest.filter(s => s.v <= best + TAU).length;
    const win = M.per.reduce((b, s, i) => (s.v < M.per[b].v ? i : b), 0);
    const phaseEl = document.getElementById('hero-phase'), hitsEl = document.getElementById('hero-hits');
    if (hitsEl) hitsEl.textContent = String(hits);
    const setPhase = html => { if (phaseEl.innerHTML !== html) phaseEl.innerHTML = html; };
    const T = { b0: 5.2, c0: 8.0, c1: 11.2, end: 15.0 };
    let t0 = performance.now(), visible = true, raf = null, lastT = 0;

    function draw(t) {
      lastT = t;
      ctx.clearRect(0, 0, W, Hh);
      ctx.imageSmoothingEnabled = true; ctx.drawImage(baseC, 0, 0, W, Hh);
      // phase A: random restarts (end dot: dark = reaches the optimum, white = trapped elsewhere)
      const fadeA = t < T.b0 ? 1 : Math.max(0, 1 - (t - T.b0) / 0.8);
      if (fadeA > 0) {
        ctx.save(); ctx.globalAlpha = fadeA;
        M.rest.forEach((s, i) => {
          const p = Math.max(0, Math.min(1, (t - i * 0.07) / 2.9)); if (p <= 0) return;
          const hit = s.v <= best + TAU;
          dot(X(s.pts[0]), Y(s.pts[1]), 2.2, 'rgba(255, 255, 255, 0.9)', null);
          const [hx, hy] = path(s.pts, p, { c: 'rgba(20, 23, 29, 0.42)', w: 1.1 });   // all restarts drawn alike; the end dot tells hits
          if (p >= 1) { if (hit) dot(hx, hy, 3.6, '#14171d', '#fff'); else dot(hx, hy, 3.2, '#fff', 'rgba(20, 23, 29, 0.6)'); }
        });
        ctx.restore();
      }
      // phase B: two cuts through the passes; the regions are the symbols
      if (t >= T.b0) {
        P.cuts.forEach((c, i) => cutLine(c, ease.inOut(Math.max(0, Math.min(1, (t - T.b0 - i * 0.35) / 1.1)))));
        const a = Math.max(0, Math.min(1, (t - T.b0 - 1.1) / 0.6));
        if (a > 0) M.regions.forEach(g => tag(g.name, X(g.lx), Y(g.ly), a));
      }
      // phase C: one descent per region; the ring marks the lowest floor
      if (t >= T.c0) M.per.forEach((s, i) => {
        const p = Math.max(0, Math.min(1, (t - T.c0 - i * 0.25) / 2.4)); if (p <= 0) return;
        dot(X(s.pts[0]), Y(s.pts[1]), 3.4, '#fff', '#14171d');
        const [hx, hy] = path(s.pts, p, { c: '#14171d', w: 2.4, halo: 'rgba(255, 255, 255, 0.8)' });
        dot(hx, hy, 4.2, '#14171d', '#fff');
      });
      if (t >= T.c1) {
        const s = M.per[win], n = s.pts.length, a = Math.min(1, (t - T.c1) / 0.5), r = 11 + 5 * (1 - ease.out(a)), x = X(s.pts[n - 2]), y = Y(s.pts[n - 1]);
        ctx.save(); ctx.globalAlpha = a; ctx.beginPath(); ctx.arc(x, y, r, 0, 7);
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 4; ctx.stroke(); ctx.strokeStyle = '#14171d'; ctx.lineWidth = 1.8; ctx.stroke(); ctx.restore();
      }
      // status line under the map
      if (t < T.b0) { setPhase(`<b>Random restarts</b><span>${hits} of ${R} reach the optimum</span>`); }
      else if (t < T.c0) setPhase(`<b>Invent symbols</b><span>two cuts through the passes make ${M.regions.length} regions</span>`);
      else {
        setPhase(`<b>One descent per region</b><span>the lowest floor, in ${M.per[win].reg.name}, is the optimum</span>`);
      }
    }
    resize(); window.addEventListener('resize', () => { resize(); if (!raf) draw(lastT); });
    function frame(now) {
      let t = window.SS.reduced ? T.c1 + 1 : (now - t0) / 1000;
      if (t > T.end) { t0 = now; t = 0; }
      draw(t);
      raf = visible && !window.SS.reduced ? requestAnimationFrame(frame) : null;
    }
    if ('IntersectionObserver' in window) new IntersectionObserver(es => es.forEach(e => {
      visible = e.isIntersecting; if (visible && !raf) raf = requestAnimationFrame(frame);
    })).observe(canvas);
    raf = requestAnimationFrame(frame);
    document.getElementById('hero-replay').addEventListener('click', () => { t0 = performance.now(); if (!raf) raf = requestAnimationFrame(frame); });
    window.SS._heroSeek = t => { t0 = performance.now() - t * 1000; };   // for screenshots
  }

  const loadImage = src => new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = src; });

  async function startHero() {
    const canvas = document.getElementById('hero-canvas'); if (!canvas) return;
    try {   // precomputed model + map (identical to computing them here, without the ~1 s of work on the main thread)
      const [D, img] = await Promise.all([window.SS.getJSON('data/hero_model.json'), loadImage('data/hero_map.png')]);
      D.M.per.forEach(s => { s.reg = D.M.regions.find(g => g.c === s.c); });
      window.SS._heroModel = D.M;
      startHero2D(D.P, D.M, img);
    } catch (e) {
      const P = await window.SS.getJSON('data/hero_landscape.json');
      const M = build(P);
      window.SS._heroModel = M;
      startHero2D(P, M);
    }
  }

  window.SS.startHero = () => startHero().catch(e => console.warn('hero', e));
  window.SS._heroBuild = build;
  window.SS._heroMapPixels = mapPixels;
  window.SS._heroCompact = compact;
})();
