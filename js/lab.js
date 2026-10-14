/* "Try it": the t = 0 decision only. Click a point -> a descent confined to that region slides downhill -> the robot
   runs the chunk that descent found and the controller keeps replanning until the episode ends. Every click start has
   its own recorded closed-loop simulation (data/avoiding/clicks.json: joint angles per control step; link poses are
   rebuilt here by forward kinematics of the Panda chain). */
(function () {
  const { Landscape, drawPath, drawRing, ease } = window.SS;

  /* joint angles (int16, per step) -> body poses [x y z w qx qy qz] of the 16 moving bodies, MuJoCo frame */
  function episodeFromClick(C, rec) {
    const raw = window.SS.b64ToBytes(rec.q), q16 = new Int16Array(raw.buffer, raw.byteOffset, raw.byteLength >> 1);
    const nq = C.qscale.length, T = q16.length / nq, chain = C.kin.chain, nb = chain.length;
    const poses = new Float32Array(T * nb * 7), ee = [];
    const rootP = new THREE.Vector3(...C.kin.root.pos), rootR = new THREE.Quaternion(C.kin.root.quat[1], C.kin.root.quat[2], C.kin.root.quat[3], C.kin.root.quat[0]);
    const bq = chain.map(c => new THREE.Quaternion(c.quat[1], c.quat[2], c.quat[3], c.quat[0]));
    const tmpQ = new THREE.Quaternion(), tmpV = new THREE.Vector3();
    for (let t = 0; t < T; t++) {
      const W = { [C.kin.root.body]: [rootP, rootR] };
      let k = 0;
      chain.forEach((c, b) => {
        const [pp, Rp] = W[c.parent];
        const p = tmpV.set(...c.pos).applyQuaternion(Rp).add(pp).clone(), R = Rp.clone().multiply(bq[b]);
        c.joints.forEach(j => {
          const v = q16[t * nq + k] * C.qscale[k]; k++;
          if (j.type === 3) R.multiply(tmpQ.setFromAxisAngle(new THREE.Vector3(...j.axis), v));
          else p.add(new THREE.Vector3(...j.axis).multiplyScalar(v).applyQuaternion(R));
        });
        W[c.body] = [p, R];
        poses.set([p.x, p.y, p.z, R.w, R.x, R.y, R.z], (t * nb + b) * 7);
        if (b === C.tcp) ee.push([p.x, p.y, p.z]);
      });
    }
    return { ee, _poses: poses, n_pose_bodies: nb, success: rec.success, steps: rec.steps, mode: rec.mode };
  }

  function startLab(data, scene) {
    const landCanvas = document.getElementById('land-canvas'), simCanvas = document.getElementById('sim-canvas');
    if (!landCanvas || !simCanvas) return;
    const eps = data.episodes, d = eps.right.decisions[0];      // same state and slice in both episodes at t = 0
    const L = new Landscape(landCanvas, { labelText: ['pass on the left', 'pass on the right'], compact: true });
    L.set(d);
    let sim = null;
    try { sim = new window.SS.Sim(simCanvas, scene, eps.right); } catch (e) { console.warn('WebGL unavailable', e); }
    const reset = document.getElementById('lab-reset'), landRead = document.getElementById('land-readout'), simRead = document.getElementById('sim-readout');
    const simHint = document.getElementById('sim-hint');
    const st = { anim: null, running: false, token: 0, ring: 0 };
    const clicks = window.SS.getJSON('data/avoiding/clicks.json').catch(e => { console.warn('clicks', e); return null; });

    L.overlays = [
      () => { L.drawCut(d); L.drawRegionLabels(d, 1); },
      (ctx) => {
        const now = performance.now();
        if (st.anim) {
          const p = Math.max(0, Math.min(1, (now - st.anim.t0) / st.anim.dur));
          drawPath(ctx, L, st.anim.pts, ease.inOut(p), 'ink', d);
          if (p < 1) requestAnimationFrame(() => L.draw());
          else if (st.ring > 0) {
            const pts = st.anim.pts, n = pts.length / 2, [x, y] = L.toPx(pts[2 * n - 2], pts[2 * n - 1], d);
            drawRing(ctx, x, y, 13, st.ring);
          }
        }
        if (st.hover && !st.anim) {
          ctx.save(); ctx.strokeStyle = 'rgba(255,255,255,.75)'; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.arc(st.hover[0], st.hover[1], 7, 0, Math.PI * 2); ctx.stroke(); ctx.restore();
        }
      },
    ];
    L.draw();

    const chip = (el, html, win) => { el.innerHTML = html ? `<span class="chip${win ? ' win' : ''}">${html}</span>` : ''; };
    const wait = ms => new Promise(r => setTimeout(r, window.SS.reduced ? 0 : ms));
    function animate(ms, fn) {
      if (window.SS.reduced) { fn(1); return Promise.resolve(); }
      const t0 = performance.now();
      return new Promise(res => { const tick = now => { const p = Math.max(0, Math.min(1, (now - t0) / ms)); fn(p); if (p < 1) requestAnimationFrame(tick); else res(); }; requestAnimationFrame(tick); });
    }
    function nearestStart(a, b) {
      const G = d.G;
      const ix = Math.round((a - d.ga[0]) / (d.ga[1] - d.ga[0]) * (G - 1)), iy = Math.round((b - d.gb[0]) / (d.gb[1] - d.gb[0]) * (G - 1));
      return Math.max(0, Math.min(G - 1, iy)) * G + Math.max(0, Math.min(G - 1, ix));
    }
    function doReset() {
      st.token++; st.running = false; st.anim = null; st.ring = 0;
      if (sim) { sim.setEpisode(eps.right); sim.setStep(0); sim.setOverlays({}); }
      L.draw();
      chip(landRead, 'Click a point to drop a descent'); chip(simRead, '');
    }

    landCanvas.addEventListener('click', async (ev) => {
      const r = landCanvas.getBoundingClientRect(), [a, b] = L.toAB(ev.clientX - r.left, ev.clientY - r.top, d);
      const i = nearestStart(a, b), tok = ++st.token;
      st.running = false; st.ring = 0;
      if (sim) { sim.setEpisode(eps.right); sim.setStep(0); sim.setOverlays({}); }
      chip(landRead, ''); chip(simRead, '');
      st.anim = { pts: d.conf[i], t0: performance.now(), dur: 1500 }; L.draw();
      const C = await clicks;
      await wait(1550);
      if (tok !== st.token) return;
      await animate(300, p => { st.ring = p; L.draw(); });
      chip(landRead, `your descent · <i class="m">h</i> = <b>${d.conf_v[i].toFixed(2)}</b>`, true);
      if (!sim) return;
      // the chunk this descent found runs at once; the controller then replans every 16 steps on its own
      const rec = C && C.episodes[i];
      const ep = rec ? episodeFromClick(C, rec) : (d.click_cell[i] === '0' ? eps.left : eps.right);
      const T = ep.ee.length - 1;
      st.running = true; sim.setEpisode(ep); sim.setStep(0); sim.setOverlays({});
      await animate(T * 130, p => { if (tok !== st.token) return; sim.setStep(T * p); });
      if (tok !== st.token) return;
      st.running = false;
      const ok = ep.success !== false;
      chip(simRead, ok ? 'goal reached' : (ep.steps >= 250 ? 'out of time' : 'hit an obstacle'), ok);
    });
    landCanvas.addEventListener('mousemove', (ev) => {
      const r = landCanvas.getBoundingClientRect(); st.hover = [ev.clientX - r.left, ev.clientY - r.top];
      if (!st.anim) L.draw();
    });
    landCanvas.addEventListener('mouseleave', () => { st.hover = null; L.draw(); });
    reset.addEventListener('click', doReset);
    if (sim) sim.onInteract = () => { simHint.style.opacity = 0; };
    doReset();
  }

  window.SS.startLab = startLab;
  window.SS._clickEpisode = episodeFromClick;   // for checks
})();
