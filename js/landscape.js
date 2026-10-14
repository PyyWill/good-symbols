/* Energy-landscape canvas: a coolwarm heatmap of h on a 2-D slice, cross-fading between decisions,
   with the symbol cut, region labels, and descent paths drawn on top. Shared by the hero and the lab. */
(function () {
  const { LUT, setupCanvas } = window.SS;

  function b64ToBytes(b64) {
    const s = atob(b64), n = s.length, out = new Uint8Array(n);
    for (let i = 0; i < n; i++) out[i] = s.charCodeAt(i);
    return out;
  }
  function decodeU16(b64) { const b = b64ToBytes(b64); return new Uint16Array(b.buffer, b.byteOffset, b.byteLength >> 1); }
  function decodeI16(b64) { const b = b64ToBytes(b64); return new Int16Array(b.buffer, b.byteOffset, b.byteLength >> 1); }

  /* Decode one decision record (see build_data.py) into typed arrays and helpers. */
  function decodeDecision(d) {
    if (d._decoded) return d;
    d.grid = decodeU16(d.Hq);
    const P = d.P;
    const unpack = (b64, n) => {
      if (!b64) return null;
      const raw = decodeI16(b64), paths = [];
      for (let i = 0; i < n; i++) {
        const pts = new Float32Array(P * 2);
        for (let j = 0; j < P; j++) {
          pts[2 * j] = d.a[0] + (raw[(i * P + j) * 2] / 30000) * (d.a[1] - d.a[0]);
          pts[2 * j + 1] = d.b[0] + (raw[(i * P + j) * 2 + 1] / 30000) * (d.b[1] - d.b[0]);
        }
        paths.push(pts);
      }
      return paths;
    };
    d.conf = unpack(d.conf_q, d.G * d.G);
    d.free = unpack(d.free_q, d.G * d.G);
    d.rst = unpack(d.rst_q, d.n_rst);
    d.ctrl = unpack(d.ctrl_q, d.ctrl_cells.length);
    d._decoded = true;
    return d;
  }

  /* h value at slice coordinate (bilinear on the stored display transform, inverted). */
  function hAt(d, a, b) {
    const NX = d.NX || d.NG, NY = d.NY || d.NG;
    const fx = (a - d.a[0]) / (d.a[1] - d.a[0]) * (NX - 1), fy = (b - d.b[0]) / (d.b[1] - d.b[0]) * (NY - 1);
    const ix = Math.max(0, Math.min(NX - 1, Math.round(fx))), iy = Math.max(0, Math.min(NY - 1, Math.round(fy)));
    const q = d.grid[iy * NX + ix];
    if (q === 65535) return null;
    const L = d.lmin + (q / 65534) * (d.lmax - d.lmin);
    return Math.exp(L) - d.c + d.hmin;
  }

  function gridImage(d) {
    const NX = d.NX || d.NG, NY = d.NY || d.NG, cv = document.createElement('canvas');
    cv.width = NX; cv.height = NY;
    const ctx = cv.getContext('2d'), img = ctx.createImageData(NX, NY);
    for (let iy = 0; iy < NY; iy++) {
      for (let ix = 0; ix < NX; ix++) {
        const q = d.grid[iy * NX + ix], o = ((NY - 1 - iy) * NX + ix) * 4;     // y up
        if (q === 65535) { img.data[o] = 214; img.data[o + 1] = 216; img.data[o + 2] = 222; img.data[o + 3] = 255; continue; }
        const k = Math.min(255, q >> 8) * 3;
        img.data[o] = LUT[k]; img.data[o + 1] = LUT[k + 1]; img.data[o + 2] = LUT[k + 2]; img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    return cv;
  }

  class Landscape {
    constructor(canvas, opts) {
      this.canvas = canvas;
      this.opts = Object.assign({ pad: 0, labels: true, labelText: null }, opts || {});
      this.cur = null; this.prev = null; this.fade = 1;
      this.overlays = [];
      this.resize();
      window.addEventListener('resize', () => { this.resize(); this.draw(); });
    }
    resize() { const s = setupCanvas(this.canvas); this.ctx = s.ctx; this.w = s.w; this.h = s.h; }
    set(d) { decodeDecision(d); if (!d._img) d._img = gridImage(d); this.prev = null; this.cur = d; this.fade = 1; }
    fadeTo(d, ms) {
      decodeDecision(d); if (!d._img) d._img = gridImage(d);
      if (!this.cur || window.SS.reduced) { this.set(d); this.draw(); return Promise.resolve(); }
      this.prev = this.cur; this.cur = d; this.fade = 0;
      const t0 = performance.now(), dur = ms || 900;
      return new Promise(res => {
        const tick = (now) => {
          this.fade = Math.min(1, (now - t0) / dur);
          this.draw();
          if (this.fade < 1) requestAnimationFrame(tick); else { this.prev = null; res(); }
        };
        requestAnimationFrame(tick);
      });
    }
    /* fit rect: map the slice window to the canvas, keeping it square-ish and filling the pane */
    view(d) {
      d = d || this.cur;
      if (this.opts.fit !== 'crop') return { a0: d.a[0], a1: d.a[1], b0: d.b[0], b1: d.b[1] };
      // crop the data window to the canvas aspect (data units kept square), centred on the symbol cut
      const A = d.a[1] - d.a[0], B = d.b[1] - d.b[0], r = this.w / this.h;
      let va = A, vb = B;
      if (A / B > r) va = B * r; else vb = A / r;
      const ac = Math.min(d.a[1] - va / 2, Math.max(d.a[0] + va / 2, d.thr != null ? d.thr : (d.a[0] + d.a[1]) / 2));
      const bc = (d.b[0] + d.b[1]) / 2;
      return { a0: ac - va / 2, a1: ac + va / 2, b0: bc - vb / 2, b1: bc + vb / 2 };
    }
    toPx(a, b, d) {
      const v = this.view(d);
      return [(a - v.a0) / (v.a1 - v.a0) * this.w, this.h - (b - v.b0) / (v.b1 - v.b0) * this.h];
    }
    toAB(x, y, d) {
      const v = this.view(d);
      return [v.a0 + x / this.w * (v.a1 - v.a0), v.b0 + (this.h - y) / this.h * (v.b1 - v.b0)];
    }
    drawImageFor(d, alpha) {
      const ctx = this.ctx;
      ctx.globalAlpha = alpha;
      ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
      const v = this.view(d), NX = d._img.width, NY = d._img.height;
      const sx = (v.a0 - d.a[0]) / (d.a[1] - d.a[0]) * NX, sw = (v.a1 - v.a0) / (d.a[1] - d.a[0]) * NX;
      const sy = (d.b[1] - v.b1) / (d.b[1] - d.b[0]) * NY, sh = (v.b1 - v.b0) / (d.b[1] - d.b[0]) * NY;
      ctx.drawImage(d._img, sx, sy, sw, sh, 0, 0, this.w, this.h);
      ctx.globalAlpha = 1;
    }
    drawCut(d, alpha) {
      const ctx = this.ctx, [x] = this.toPx(d.thr, 0, d);
      ctx.save(); ctx.globalAlpha = alpha == null ? 1 : alpha;
      ctx.strokeStyle = 'rgba(255,255,255,.95)'; ctx.lineWidth = 2.2; ctx.setLineDash([8, 6]);
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, this.h); ctx.stroke();
      ctx.restore();
    }
    drawRegionLabels(d, alpha, texts) {
      const ctx = this.ctx, [x] = this.toPx(d.thr, 0, d);
      texts = texts || ['ψ = 0', 'ψ = 1'];
      ctx.save(); ctx.globalAlpha = alpha == null ? 1 : alpha;
      const tile = this.opts.compact ? (cx, label, sub) => {      // one line at the top edge: "ψ = 0 · pass on the left"
        ctx.font = 'italic 600 14px Caladea, Cambria, serif'; const w1 = ctx.measureText(label).width;
        ctx.font = '12.5px Carlito, Calibri, sans-serif'; const w2 = sub ? ctx.measureText(' · ' + sub).width : 0;
        const W = w1 + w2 + 18, y = 10, x0 = Math.max(6, Math.min(this.w - W - 6, cx - W / 2));
        ctx.fillStyle = 'rgba(63,86,201,.94)'; roundRect(ctx, x0, y, W, 25, 6); ctx.fill();
        ctx.fillStyle = '#fff'; ctx.textAlign = 'left';
        ctx.font = 'italic 600 14px Caladea, Cambria, serif'; ctx.fillText(label, x0 + 9, y + 17);
        if (sub) { ctx.font = '12.5px Carlito, Calibri, sans-serif'; ctx.fillStyle = 'rgba(255,255,255,.9)'; ctx.fillText(' · ' + sub, x0 + 9 + w1, y + 17); }
      } : (cx, label, sub) => {
        ctx.font = 'italic 600 15px Caladea, Cambria, serif';
        const w1 = ctx.measureText(label).width;
        ctx.font = '13px Carlito, Calibri, sans-serif';
        const w2 = sub ? ctx.measureText(sub).width : 0;
        const W = Math.max(w1, w2) + 20, Hh = sub ? 44 : 28, y = this.opts.labelsAt === 'bottom' ? this.h - Hh - 52 : 46;
        ctx.fillStyle = 'rgba(63,86,201,.94)';
        roundRect(ctx, cx - W / 2, y, W, Hh, 7); ctx.fill();
        ctx.fillStyle = '#fff'; ctx.textAlign = 'center';
        ctx.font = 'italic 600 15px Caladea, Cambria, serif'; ctx.fillText(label, cx, y + 19);
        if (sub) { ctx.font = '13px Carlito, Calibri, sans-serif'; ctx.fillStyle = 'rgba(255,255,255,.88)'; ctx.fillText(sub, cx, y + 36); }
      };
      const subs = this.opts.labelText || [null, null];
      tile(Math.max(60, x / 2), texts[0], subs[0]);
      tile(Math.min(this.w - 60, x + (this.w - x) / 2), texts[1], subs[1]);
      ctx.restore();
    }
    draw() {
      if (!this.cur) return;
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.w, this.h);
      if (this.prev && this.fade < 1) { this.drawImageFor(this.prev, 1); this.drawImageFor(this.cur, window.SS.ease.inOut(this.fade)); }
      else this.drawImageFor(this.cur, 1);
      for (const f of this.overlays) f(ctx, this);
    }
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }

  /* Draw a (partial) descent path: progress in [0,1]; style: 'ink' | 'light' | 'ghost' */
  function drawPath(ctx, L, pts, progress, style, d) {
    // progress 0 = only the start point; the path then grows from it (no segment is drawn before the descent moves)
    const n = pts.length / 2, upto = Math.max(0, Math.min(n - 1, progress * (n - 1)));
    const k = Math.floor(upto), f = upto - k;
    const P = i => L.toPx(pts[2 * i], pts[2 * i + 1], d);
    const thin = style === 'thin' || style === 'thinlight';
    if (style === 'thin') style = 'ink';
    if (style === 'thinlight') style = 'light';
    const col = style === 'light' ? (thin ? 'rgba(255,255,255,.8)' : 'rgba(255,255,255,.95)') : style === 'ghost' ? 'rgba(27,30,36,.35)' : (thin ? 'rgba(27,30,36,.55)' : 'rgba(27,30,36,.92)');
    ctx.save();
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    ctx.strokeStyle = col; ctx.lineWidth = style === 'ghost' ? 1.2 : thin ? 1.1 : 1.8;
    ctx.beginPath();
    let [x0, y0] = P(0); ctx.moveTo(x0, y0);
    let [px, py] = [x0, y0];
    for (let i = 1; i <= k; i++) { const [x, y] = P(i); ctx.quadraticCurveTo(px, py, (px + x) / 2, (py + y) / 2); px = x; py = y; }
    if (k >= 1) ctx.lineTo(px, py);
    let hx, hy;
    if (k < n - 1) { const [xa, ya] = P(k), [xb, yb] = P(k + 1); hx = xa + (xb - xa) * f; hy = ya + (yb - ya) * f; ctx.lineTo(hx, hy); }
    else { [hx, hy] = P(n - 1); }
    ctx.stroke();
    // start: open circle
    ctx.fillStyle = '#fff'; ctx.strokeStyle = style === 'light' ? 'rgba(27,30,36,.6)' : col; ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.arc(x0, y0, style === 'ghost' ? 2.6 : 3.6, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    if (progress >= 1) {           // floor: x mark
      ctx.strokeStyle = style === 'light' ? '#fff' : '#1b1e24'; ctx.lineWidth = style === 'ghost' ? 1.4 : 2.2;
      const s = style === 'ghost' ? 3.2 : 5;
      ctx.beginPath(); ctx.moveTo(hx - s, hy - s); ctx.lineTo(hx + s, hy + s); ctx.moveTo(hx + s, hy - s); ctx.lineTo(hx - s, hy + s); ctx.stroke();
    } else if (upto > 0.02) {     // moving head (none before the descent has moved)
      ctx.fillStyle = style === 'light' ? '#fff' : '#1b1e24';
      ctx.beginPath(); ctx.arc(hx, hy, style === 'ghost' ? 2.2 : 3.4, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
  }

  function drawRing(ctx, x, y, r, alpha) {
    ctx.save(); ctx.globalAlpha = alpha == null ? 1 : alpha;
    ctx.strokeStyle = '#1b1e24'; ctx.lineWidth = 2.2;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = 'rgba(255,255,255,.9)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(x, y, r + 2, 0, Math.PI * 2); ctx.stroke();
    ctx.restore();
  }

  window.SS.Landscape = Landscape;
  window.SS.decodeDecision = decodeDecision;
  window.SS.drawPath = drawPath;
  window.SS.drawRing = drawRing;
  window.SS.roundRect = roundRect;
  window.SS.hAt = hAt;
  window.SS.decodeU16 = decodeU16;
  window.SS.b64ToBytes = b64ToBytes;
})();
