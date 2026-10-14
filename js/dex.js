/* Three complete Block episodes (Shadow Hand, settings 1-3), each shown twice: the full MILP (top row) and the
   symbol-guided solver (bottom row). Both solvers follow the same trajectory (same contact-mode sequence, states
   within 0.002), so a row plays the same rendered episode. The replay clock is solve time: a row moves through
   control step i while its solver works on decision i, i.e. during K x the measured solve time of that decision
   (median of 5 runs), so playback time is exactly proportional to solve time and the counters of both rows agree at
   any moment until one finishes. All six panels share one clock. */
(function () {
  const K = 24, PRE = 0.8, HOLD = 2.6;               // solve-time stretch, still start, pause at the end (seconds)
  const fmt = ms => Math.round(ms) + ' ms';
  const METHODS = [{ key: 'e2e', name: 'Full MILP' }, { key: 'ours', name: 'Ours' }];

  async function initDex() {
    const row = document.getElementById('dex-row'); if (!row) return;
    const three = await fetch('data/dex/block_three.json').then(r => r.ok ? r.json() : null).catch(() => null);
    if (!three) return;
    const eps = await Promise.all(three.settings.map(async (S) => {
      const s = S.setting;
      const [ep, tm] = await Promise.all([fetch(`data/dex/block_s${s}_episode.json`).then(r => r.json()), fetch(`data/dex/block_s${s}_timing.json`).then(r => r.json())]);
      return { s, S, ep, ms: tm.per_step_ms_median_over_runs };
    }));

    const panel = (e, m, mi) => `
      <div class="dex-p" style="--e:${e.s};--m:${mi}" data-e="${e.s}" data-m="${m.key}">
        <div class="vid"><video muted playsinline preload="auto" poster="data/dex/block_s${e.s}_poster.jpg">
          <source src="data/dex/block_s${e.s}.webm" type="video/webm"><source src="data/dex/block_s${e.s}.mp4" type="video/mp4"></video>
          <span class="mtag${m.key === 'ours' ? ' us' : ''}">${m.name}</span></div>
        <div class="dex-read"><span class="st">step 0 / ${e.ep.n_steps}</span><span class="ms">0 ms</span></div>
      </div>`;
    row.innerHTML = eps.map(e => `<div class="dex-head" style="--e:${e.s}">Episode ${e.s}<span>${e.ep.n_steps} decisions</span></div>`).join('') +
      METHODS.map((m, mi) => eps.map(e => panel(e, m, mi)).join('')).join('');

    // per-panel schedule on the shared clock (seconds): step i spans [PRE + K * solved_before_i, PRE + K * solved_through_i]
    const panels = Array.from(row.querySelectorAll('.dex-p')).map(el => {
      const e = eps.find(x => x.s === +el.dataset.e), key = el.dataset.m, ms = e.ms[key], n = e.ep.n_steps;
      const cum = []; ms.reduce((a, v, i) => (cum[i] = a + v), 0);
      const seg = ms.map((v, i) => ({ i, t0: PRE + K * (i ? cum[i - 1] : 0) / 1000, t1: PRE + K * cum[i] / 1000 }));
      return {
        el, e, key, ms, cum, n, seg, end: seg[n - 1].t1, video: el.querySelector('video'),
        st: el.querySelector('.st'), out: el.querySelector('.ms'),
        f0: e.ep.first_state_frame || 30, fps: e.ep.fps || 30, fps_step: e.ep.frames_per_step || 20,
      };
    });
    const cycle = Math.max(...panels.map(p => p.end)) + HOLD;

    function frameTime(p, step) { return (p.f0 + p.fps_step * step) / p.fps; }
    function hold(v, t) { if (!v.paused) v.pause(); if (Math.abs(v.currentTime - t) > 0.05 && !v.seeking) v.currentTime = t; }
    function show(p, clock) {
      const v = p.video;
      if (clock >= p.end) {                                      // done: hold the goal state
        hold(v, frameTime(p, p.n));
        p.el.classList.add('done');
        p.st.innerHTML = `<b>goal reached</b><span class="n"> · ${p.n} steps</span>`; p.out.textContent = fmt(p.cum[p.n - 1]);
        return;
      }
      p.el.classList.remove('done');
      p.out.textContent = fmt(Math.max(0, clock - PRE) * 1000 / K);   // elapsed solve time: the same in both rows
      if (clock < PRE) { hold(v, frameTime(p, 0)); p.st.textContent = `step 0 / ${p.n}`; return; }
      const g = p.seg.find(s => clock < s.t1) || p.seg[p.n - 1];
      const dur = g.t1 - g.t0, target = frameTime(p, g.i) + (clock - g.t0) / dur * (p.fps_step / p.fps);
      v.playbackRate = Math.max(0.0625, Math.min(16, (p.fps_step / p.fps) / dur));
      if (v.paused) v.play().catch(() => {});
      if (Math.abs(v.currentTime - target) > 0.15 && !v.seeking) v.currentTime = target;
      p.st.textContent = `step ${g.i + 1} / ${p.n}`;
    }

    let t0 = null, raf = null, visible = false;
    function tick(now) {
      if (t0 == null) t0 = now;
      let clock = (now - t0) / 1000;
      if (clock > cycle) { t0 = now; clock = 0; }
      panels.forEach(p => show(p, clock));
      raf = visible ? requestAnimationFrame(tick) : null;
    }
    if (window.SS.reduced) panels.forEach(p => show(p, 1e9));
    else if ('IntersectionObserver' in window) new IntersectionObserver(es => es.forEach(en => {
      visible = en.isIntersecting;
      if (visible && !raf) { t0 = null; raf = requestAnimationFrame(tick); }
      if (!visible) panels.forEach(p => p.video.pause());
    }), { threshold: 0.25 }).observe(row);

  }

  window.SS.initDex = initDex;
})();
