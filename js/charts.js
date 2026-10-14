/* Results charts (SVG, drawn to scale), tables, task strip, steering and human-symbol panels. */
(function () {
  const { onVisible, reduced, ease } = window.SS;
  const NS = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  const fmt = (v, d) => v.toFixed(d == null ? 2 : d);
  const tween = (ms, fn) => {
    if (reduced || !(ms > 0)) { fn(1); return; }
    const t0 = performance.now();
    const tick = now => { const p = Math.max(0, Math.min(1, (now - t0) / ms)); fn(ease.out(p)); if (p < 1) requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  };

  function segButtons(container, labels, initial, onPick) {
    container.innerHTML = '';
    labels.forEach((l, i) => {
      const b = document.createElement('button'); b.type = 'button'; b.textContent = l; b.setAttribute('aria-pressed', i === initial);
      b.addEventListener('click', () => { container.querySelectorAll('button').forEach((x, j) => x.setAttribute('aria-pressed', j === i)); onPick(i); });
      container.appendChild(b);
    });
  }

  /* ---------------- PWA: success vs speed-up ---------------- */
  function pwaChart(R) {
    const T = R.table1, host = document.getElementById('pwa-chart');
    const W = 520, H = 330, m = { l: 46, r: 16, t: 14, b: 40 };
    const xs = v => m.l + (Math.log10(v) + 2) / 3 * (W - m.l - m.r);            // 0.01 .. 10
    const ys = v => m.t + (1 - (v - 0.55) / 0.47) * (H - m.t - m.b);              // 0.55 .. 1.02
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Success rate against speed-up per method' }, null);
    host.innerHTML = ''; host.appendChild(svg);
    [0.01, 0.1, 1, 10].forEach(v => {
      el('line', { x1: xs(v), x2: xs(v), y1: m.t, y2: H - m.b, class: 'ax' }, svg);
      const t = el('text', { x: xs(v), y: H - m.b + 18, 'text-anchor': 'middle', class: 'tick' }, svg); t.textContent = v >= 1 ? v + '×' : v + '×';
    });
    [0.6, 0.7, 0.8, 0.9, 1.0].forEach(v => {
      el('line', { x1: m.l, x2: W - m.r, y1: ys(v), y2: ys(v), class: 'ax' }, svg);
      const t = el('text', { x: m.l - 8, y: ys(v) + 4, 'text-anchor': 'end' }, svg); t.textContent = v.toFixed(1);
    });
    el('line', { x1: xs(1), x2: xs(1), y1: m.t, y2: H - m.b, stroke: 'var(--ink-3)', 'stroke-width': 1.2, 'stroke-dasharray': '4 3' }, svg);
    const e2e = el('text', { x: xs(1) + 6, y: m.t + 12 }, svg); e2e.textContent = 'full MILP = 1×';
    const xl = el('text', { x: (m.l + W - m.r) / 2, y: H - 4, 'text-anchor': 'middle' }, svg); xl.textContent = 'inference speed-up over the full MILP (log)';
    const yl = el('text', { x: 12, y: (m.t + H - m.b) / 2, 'text-anchor': 'middle', transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})` }, svg); yl.textContent = 'success rate';
    const pts = T.methods.map(meth => {
      const g = el('g', {}, svg);
      const ours = meth.family === 'ours';
      const c = el('circle', { r: ours ? 8 : 5.5, fill: ours ? 'var(--sym)' : meth.family === 'solver' ? 'var(--bg)' : 'var(--ink-3)', stroke: ours ? 'var(--sym)' : 'var(--ink-2)', 'stroke-width': ours ? 0 : 1.4 }, g);
      const lab = el('text', { 'font-size': ours ? 13 : 11.5, 'font-weight': ours ? 700 : 400, fill: ours ? 'var(--sym)' : 'var(--ink-2)' }, g); lab.textContent = meth.name;
      return { meth, g, c, lab, x: xs(1), y: ys(0.8) };
    });
    let cur = 3;
    function place(task, animate) {
      const targets = pts.map(p => ({ x: xs(p.meth.is[task]), y: ys(p.meth.sr[task]) }));
      // label placement: right of the point unless near the right edge; nudge collisions vertically
      const order = targets.map((t, i) => i).sort((a, b) => targets[a].y - targets[b].y);
      const used = [];
      const lpos = targets.map(() => null);
      order.forEach(i => {
        const t = targets[i], right = t.x < W - 110;
        let ly = t.y + 4;
        for (const u of used) if (Math.abs(u.y - ly) < 13 && Math.abs(u.x - t.x) < 90) ly = u.y + 13;
        used.push({ x: t.x, y: ly });
        lpos[i] = { x: right ? t.x + 11 : t.x - 11, y: ly, anchor: right ? 'start' : 'end' };
      });
      const from = pts.map(p => ({ x: p.x, y: p.y }));
      tween(animate ? 900 : 0, k => {
        pts.forEach((p, i) => {
          p.x = from[i].x + (targets[i].x - from[i].x) * k; p.y = from[i].y + (targets[i].y - from[i].y) * k;
          p.c.setAttribute('cx', p.x); p.c.setAttribute('cy', p.y);
          p.lab.setAttribute('x', lpos[i].x + (p.x - targets[i].x)); p.lab.setAttribute('y', lpos[i].y + (p.y - targets[i].y)); p.lab.setAttribute('text-anchor', lpos[i].anchor);
        });
      });
    }
    segButtons(document.getElementById('pwa-task'), T.tasks, cur, i => { cur = i; place(i, true); });
    place(cur, false);
    onVisible(host, () => { pts.forEach(p => { p.x = xs(1); p.y = ys(0.8); }); place(cur, true); });
    document.getElementById('pwa-legend').innerHTML =
      '<span><i style="background:var(--sym);border-radius:50%"></i>Ours</span><span><i style="background:var(--bg);border:1.4px solid var(--ink-2);border-radius:50%"></i>Solver-oriented (CoCo, GBD-MPC)</span><span><i style="background:var(--ink-3);border-radius:50%"></i>Symbol-oriented (PDDLStream, PGS, IVNTR, pix2pred)</span>';
  }

  /* ---------------- NN: descents to coverage ---------------- */
  function nnChart(R) {
    const T = R.table2, host = document.getElementById('nn-chart');
    const W = 520, H = 330, m = { l: 46, r: 12, t: 22, b: 58 };
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Descents to coverage per method' }, null);
    host.innerHTML = ''; host.appendChild(svg);
    const ymax = Math.log10(2000);
    const ys = v => H - m.b - Math.log10(Math.max(1, v)) / ymax * (H - m.t - m.b);
    [1, 10, 100, 1000].forEach(v => {
      el('line', { x1: m.l, x2: W - m.r, y1: ys(v), y2: ys(v), class: 'ax' }, svg);
      const t = el('text', { x: m.l - 8, y: ys(v) + 4, 'text-anchor': 'end' }, svg); t.textContent = v;
    });
    const yl = el('text', { x: 12, y: (m.t + H - m.b) / 2, 'text-anchor': 'middle', transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})` }, svg); yl.textContent = 'descents for 99.9% coverage (log)';
    const order = [5, 0, 1, 2, 3, 4];
    const bw = (W - m.l - m.r) / order.length;
    const bars = order.map((mi, j) => {
      const meth = T.methods[mi], x = m.l + j * bw + bw * 0.18, w = bw * 0.64;
      const fill = meth.ours ? 'var(--sym)' : mi === 0 ? 'var(--ink-2)' : 'var(--mid)';
      const r = el('rect', { x, width: w, y: ys(1), height: 0, fill, rx: 3 }, svg);
      const v = el('text', { x: x + w / 2, 'text-anchor': 'middle', 'font-weight': meth.ours ? 700 : 400, fill: meth.ours ? 'var(--sym)' : 'var(--ink)' }, svg);
      const n = el('text', { x: x + w / 2, y: H - m.b + 17, 'text-anchor': 'middle', 'font-weight': meth.ours ? 700 : 400, fill: meth.ours ? 'var(--sym)' : 'var(--ink-2)' }, svg);
      n.textContent = meth.ours ? 'Ours' : meth.name;
      const s = el('text', { x: x + w / 2, y: H - m.b + 34, 'text-anchor': 'middle', 'font-size': 11, class: 'tick' }, svg);
      return { meth, r, v, s };
    });
    const srl = el('text', { x: m.l - 8, y: H - m.b + 34, 'text-anchor': 'end', 'font-size': 11, class: 'tick' }, svg); srl.textContent = 'SR';
    const real = el('text', { x: W - m.r, y: m.t - 8, 'text-anchor': 'end', 'font-size': 11, fill: 'var(--ink-3)' }, svg);
    let cur = 0;
    function show(task, animate) {
      real.textContent = T.real[task] ? 'real robot' : 'simulation (D3IL)';
      tween(animate ? 1000 : 0, k => bars.forEach(b => {
        const dc = b.meth.dc[task], top = ys(1) - (ys(1) - ys(dc)) * k;
        b.r.setAttribute('y', top); b.r.setAttribute('height', Math.max(0, ys(1) - top));
        b.v.setAttribute('y', top - 6); b.v.textContent = (dc * k < 10 ? (dc * k).toFixed(2) : Math.round(dc * k));
        b.s.textContent = b.meth.sr[task].toFixed(2);
      }));
    }
    segButtons(document.getElementById('nn-task'), T.tasks, cur, i => { cur = i; show(i, true); });
    show(cur, false);
    onVisible(host, () => show(cur, true));
    document.getElementById('nn-legend').innerHTML =
      '<span><i style="background:var(--sym)"></i>DDPM + symbols: one descent per region</span><span><i style="background:var(--ink-2)"></i>DDPM: random restarts</span><span><i style="background:var(--mid)"></i>energy variants of ACT, BESO, IBC, BeT</span>';
  }

  /* ---------------- full tables ---------------- */
  function tables(R) {
    const t1 = R.table1, t2 = R.table2;
    let h = '<table class="data"><caption style="text-align:left;font-family:var(--font-ui);color:var(--ink-2);padding:6px 0">Piecewise-affine (MILP) tasks: success rate / speed-up over re-solving the full MILP at every step (full MILP = 1.00×)</caption><thead><tr><th>Method</th>' +
      t1.tasks.map(t => `<th>${t}</th>`).join('') + '</tr></thead><tbody>';
    t1.methods.forEach(mm => { h += `<tr class="${mm.family === 'ours' ? 'ours' : ''}"><td>${mm.name}</td>` + mm.sr.map((s, i) => `<td>${s.toFixed(2)} / ${mm.is[i].toFixed(2)}×</td>`).join('') + '</tr>'; });
    document.getElementById('table1').innerHTML = h + '</tbody></table>';
    h = `<table class="data"><caption style="text-align:left;font-family:var(--font-ui);color:var(--ink-2);padding:6px 0">Neural-policy tasks: success rate / descents needed to reach a near-optimal plan with 99.9% probability</caption><thead><tr><th>Method</th>` +
      t2.tasks.map((t, i) => `<th>${t}${t2.real[i] ? ' (real)' : ''}</th>`).join('') + '</tr></thead><tbody>';
    t2.methods.forEach(mm => { h += `<tr class="${mm.ours ? 'ours' : ''}"><td>${mm.name}</td>` + mm.sr.map((s, i) => `<td>${s.toFixed(2)} / ${mm.dc[i].toFixed(2)}</td>`).join('') + '</tr>'; });
    document.getElementById('table2').innerHTML = h + '</tbody></table>';
  }

  /* ---------------- task suites: two rows, as in the paper figure ---------------- */
  function suite() {
    const left = [['mpusht', 'MPushT'], ['hook', 'Hook'], ['packing', 'Packing'], ['block', 'Block'], ['cuboid', 'Cuboid'], ['pen', 'Pen']];
    const right = [['avoiding', 'Avoiding'], ['pushing', 'Pushing'], ['aligning', 'Aligning'], ['sorting', 'Sorting'], ['matching', 'Matching']];
    const fig = ([k, n, wide]) => `<figure${wide ? ' class="wide"' : ''}><img src="media/task_${k}${wide ? '_wide' : ''}.jpg" alt="${n} task" loading="lazy" width="${wide ? 750 : 400}" height="${wide ? 375 : 300}"><figcaption>${n}</figcaption></figure>`;
    right[3].push(true); right[4].push(true);   // the two real-robot tasks fill the second row, as in the paper figure
    const host = document.getElementById('suite'); if (!host) return;   // not on the page for now
    host.innerHTML =
      `<div class="grp"><h4>Piecewise-affine systems (MILP)</h4>${left.map(fig).join('')}</div>` +
      `<div class="grp six"><h4>Neural policies · simulation and real robot</h4>${right.map(fig).join('')}</div>`;
  }

  /* ---------------- human symbols ---------------- */
  function tnsn(R) {
    const F = R.fig7, host = document.getElementById('tnsn-chart');
    if (!host) return;
    const W = 520, H = 330, m = { l: 46, r: 50, t: 22, b: 56 }, TNMAX = 12, SHMAX = 0.75;
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Task necessity and share of factors by solver necessity, with the two Push-T pushes z1 and z2 marked in their bins' }, null);
    host.innerHTML = ''; host.appendChild(svg);
    const yL = v => H - m.b - v / TNMAX * (H - m.t - m.b), yR = v => H - m.b - v / SHMAX * (H - m.t - m.b);
    [0, 4, 8, 12].forEach(v => { el('line', { x1: m.l, x2: W - m.r, y1: yL(v), y2: yL(v), class: 'ax' }, svg); const t = el('text', { x: m.l - 8, y: yL(v) + 4, 'text-anchor': 'end', fill: 'var(--warm)' }, svg); t.textContent = v; });
    [0, 0.25, 0.5, 0.75].forEach(v => { const t = el('text', { x: W - m.r + 8, y: yR(v) + 4, fill: 'var(--cool)' }, svg); t.textContent = Math.round(v * 100) + '%'; });
    const gw = (W - m.l - m.r) / 3;
    const items = F.bins.map((b, i) => {
      const x0 = m.l + i * gw + gw * 0.2, w = gw * 0.28;
      const r1 = el('rect', { x: x0, width: w, y: yL(0), height: 0, fill: 'var(--warm)', rx: 3 }, svg);
      const r2 = el('rect', { x: x0 + w + 6, width: w, y: yL(0), height: 0, fill: 'var(--cool)', rx: 3 }, svg);
      const t1 = el('text', { x: x0 + w / 2, 'text-anchor': 'middle', fill: 'var(--warm)', 'font-weight': 700 }, svg);
      const t2 = el('text', { x: x0 + w * 1.5 + 6, 'text-anchor': 'middle', fill: 'var(--cool)', 'font-weight': 700 }, svg);
      const lab = el('text', { x: x0 + w + 3, y: H - m.b + 18, 'text-anchor': 'middle', fill: 'var(--ink)' }, svg); lab.textContent = b;
      return { i, x0, w, r1, r2, t1, t2 };
    });
    const xl = el('text', { x: (m.l + W - m.r) / 2, y: H - 12, 'text-anchor': 'middle' }, svg); xl.textContent = 'solver necessity (SN) of a factor';
    const l1 = el('text', { x: m.l, y: 12, fill: 'var(--warm)' }, svg); l1.textContent = '■ task necessity (TN)';
    const l2 = el('text', { x: W - m.r, y: 12, fill: 'var(--cool)', 'text-anchor': 'end' }, svg); l2.textContent = 'share of factors ■';

    // Push-T inset (as in the paper's figure): z1, a push at a corner of the bar, and z2, at the bottom of the stem
    const box = { x: m.l + 12, y: m.t + 8, w: 166, h: 128 }, s = 0.235, ox = box.x + 83 - 290 * s, oy = box.y + 6 - 190 * s;
    const P = (x, y) => [ox + x * s, oy + y * s];
    const g = el('g', {}, svg);
    el('rect', { x: box.x, y: box.y, width: box.w, height: box.h, rx: 8, fill: 'var(--paper)', stroke: 'var(--ink-3)', 'stroke-dasharray': '4 3' }, g);
    const tg = el('g', { transform: `translate(${ox} ${oy}) scale(${s})` }, g);
    tg.innerHTML = `<g fill="#8ee98e"><rect x="130" y="258" width="333" height="80"/><rect x="257" y="338" width="81" height="228"/></g>
      <g fill="#8da1b8" stroke="#6d7f91" stroke-width="6" stroke-linejoin="round"><polygon points="78,466 260,197 328,240 146,508"/><polygon points="230,350 262,305 445,470 405,528"/></g>
      <g fill="#111"><path d="M200 538 L160 516" stroke="#111" stroke-width="10" stroke-linecap="round"/><polygon points="148,509 182,512 168,534"/><circle cx="222" cy="551" r="24"/>
      <path d="M476 536 L438 514" stroke="#111" stroke-width="10" stroke-linecap="round"/><polygon points="426,507 460,510 446,532"/><circle cx="498" cy="549" r="24"/></g>`;
    const z1 = P(222, 551), z2 = P(498, 549);
    const zl = (p, sub, dx) => { const t = el('text', { x: p[0] + dx, y: p[1] + 5, style: 'font: italic 18px "STIX Two Math", "Cambria Math", serif; fill: #1b1e24' }, g);   // on the light inset in both themes
      t.innerHTML = `z<tspan style="font-size: 12px; font-style: normal" dy="4">${sub}</tspan>`; };
    zl(z1, 1, -27); zl(z2, 2, 9);
    // dashed connectors from each push to the bin of its solver necessity (z1: SN < 0.5, z2: SN = 1)
    const target = (i, v) => [items[i].x0 + items[i].w / 2, yL(v)];
    const conn = [[z1, target(0, 0.55)], [z2, target(2, 3.2)]].map(([a, b]) => {
      const midY = Math.max(a[1], b[1]) - 10;
      const d = a[1] < b[1] - 20 ? `M${a[0]} ${a[1] + 9} C${a[0]} ${b[1] - 40} ${b[0]} ${a[1] + 60} ${b[0]} ${b[1]}` : `M${a[0] + 9} ${a[1]} C${a[0] + 80} ${a[1]} ${b[0]} ${midY - 60} ${b[0]} ${b[1]}`;
      const p = el('path', { d, fill: 'none', stroke: 'var(--ink)', 'stroke-width': 1.4, 'stroke-dasharray': '5 4', opacity: 0 }, svg);
      const dot = el('circle', { cx: b[0], cy: b[1], r: 4, fill: 'var(--ink)', opacity: 0 }, svg);
      return { p, dot };
    });
    items.forEach(it => [it.t1, it.t2].forEach(t => { t.setAttribute('stroke', 'var(--bg)'); t.setAttribute('stroke-width', 4); t.setAttribute('paint-order', 'stroke'); svg.appendChild(t); }));

    function draw(k) {
      items.forEach(it => {
        const tn = F.tn[it.i] * k, sh = F.share[it.i] * k;
        it.r1.setAttribute('y', yL(tn)); it.r1.setAttribute('height', yL(0) - yL(tn));
        it.r2.setAttribute('y', yR(sh)); it.r2.setAttribute('height', yR(0) - yR(sh));
        it.t1.setAttribute('y', yL(tn) - 6); it.t1.textContent = tn.toFixed(1);
        it.t2.setAttribute('y', yR(sh) - 6); it.t2.textContent = Math.round(sh * 100) + '%';
      });
    }
    const showConn = () => conn.forEach(({ p, dot }, i) => {
      p.style.transition = dot.style.transition = 'opacity .5s ease ' + (0.15 + 0.35 * i) + 's';
      p.setAttribute('opacity', 1); dot.setAttribute('opacity', 1);
    });
    if (reduced) { draw(1); showConn(); return; }
    draw(0);                                     // bars grow from the axis when the chart scrolls into view
    onVisible(host, () => { tween(1200, draw); setTimeout(showConn, 1100); }, { threshold: 0.45 });
  }

  function semantics(R) {
    const host = document.getElementById('semantics');
    let h = '<div class="sem-row h"><span></span><span>Ours (CPR)</span><span>PGS</span><span>pix2pred</span></div>';
    R.semantics.forEach(row => {
      const cell = (pair, cls) => `<div class="sem ${cls}">${pair.map(([f, s]) => `<div><span class="f">${f}</span>${s ? `<br><span>${s}</span>` : ''}</div>`).join('')}</div>`;
      h += `<div class="sem-row"><span class="task">${row.task}</span>${cell(row.ours, 'ours')}${cell(row.pgs, '')}${cell(row.pix2pred, '')}</div>`;
    });
    host.innerHTML = h;
  }

  function horizon(R) {
    const T = R.table5, host = document.getElementById('horizon-chart');
    const W = 520, H = 210, m = { l: 40, r: 90, t: 14, b: 38 };
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Number of symbols against planning horizon' }, null);
    host.innerHTML = ''; host.appendChild(svg);
    const xs = i => m.l + i / (T.H.length - 1) * (W - m.l - m.r), ys = v => H - m.b - v / 8 * (H - m.t - m.b);
    [0, 2, 4, 6, 8].forEach(v => { el('line', { x1: m.l, x2: W - m.r, y1: ys(v), y2: ys(v), class: 'ax' }, svg); const t = el('text', { x: m.l - 8, y: ys(v) + 4, 'text-anchor': 'end' }, svg); t.textContent = v; });
    T.H.forEach((h, i) => { const t = el('text', { x: xs(i), y: H - m.b + 18, 'text-anchor': 'middle' }, svg); t.textContent = 'H = ' + h; });
    const yl = el('text', { x: 10, y: (m.t + H - m.b) / 2, 'text-anchor': 'middle', transform: `rotate(-90 10 ${(m.t + H - m.b) / 2})` }, svg); yl.textContent = 'symbols k';
    const cols = ['var(--cool)', 'var(--warm)'];
    const lines = T.rows.map((r, j) => {
      const d = r.k.map((k, i) => `${i ? 'L' : 'M'}${xs(i)},${ys(k)}`).join(' ');
      const p = el('path', { d, fill: 'none', stroke: cols[j], 'stroke-width': 2.4, 'stroke-linejoin': 'round' }, svg);
      r.k.forEach((k, i) => el('circle', { cx: xs(i), cy: ys(k), r: 4.2, fill: cols[j] }, svg));
      const t = el('text', { x: xs(T.H.length - 1) + 10, y: ys(r.k[r.k.length - 1]) + 4, fill: cols[j], 'font-weight': 700 }, svg); t.textContent = r.task;
      return p;
    });
    onVisible(host, () => lines.forEach(p => {
      const L = p.getTotalLength(); p.style.strokeDasharray = L; p.style.strokeDashoffset = L;
      tween(1200, k => { p.style.strokeDashoffset = L * (1 - k); });
    }));
  }





  /* ---------------- real robot ---------------- */
  function autoplayVideos() {
    document.querySelectorAll('video[data-autoplay]').forEach(v => {   // only opted-in videos; the dex and steering players drive their own
      if (!('IntersectionObserver' in window)) return;
      new IntersectionObserver(es => es.forEach(e => {
        if (e.isIntersecting) { if (!reduced) v.play().catch(() => {}); } else v.pause();
      }), { threshold: 0.35 }).observe(v);
    });
  }

  window.SS.initCharts = function (R) {
    try { pwaChart(R); } catch (e) { console.warn(e); }
    try { nnChart(R); } catch (e) { console.warn(e); }
    try { tables(R); } catch (e) { console.warn(e); }
    try { suite(); } catch (e) { console.warn(e); }
    try { tnsn(R); } catch (e) { console.warn(e); }
    autoplayVideos();
  };
})();
