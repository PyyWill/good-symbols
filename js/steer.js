/* Steering: one tab per task, two runs per task (one symbol value each). The real-robot task shows its two recorded
   videos; the simulated tasks show 3D players rendered like "Try it" in the same 16:9 frame as the videos,
   auto-playing the recorded preference-steered episodes while visible. */
(function () {
  const TASKS = {
    sorting: {
      kind: 'video', title: 'Sorting',
      runs: [['red_first', 'ψ = 0', 'red block first'], ['blue_first', 'ψ = 1', 'blue block first']],
    },
    avoiding: {
      title: 'Avoiding', data: 'data/avoiding/episodes.json', scene: 'data/avoiding/scene.json',
      cam: { pos: [1.5, 1.62, -0.05], target: [0.42, -0.1, -0.05], fov: 21 },   // framed for the 16:9 pane
      runs: [['left', 'ψ = 0', 'pass on the left'], ['right', 'ψ = 1', 'pass on the right']],
      speed: 130,
    },
    aligning: {
      title: 'Aligning', data: 'data/aligning/episodes.json', scene: 'data/aligning/scene.json',
      cam: { pos: [1.3, 1.42, -0.03], target: [0.37, -0.06, -0.03], fov: 23 },  // framed for the 16:9 pane
      runs: [['outside', 'ψ = 0', 'descend early, push from outside'], ['inside', 'ψ = 1', 'keep height, push from inside']],
      speed: 55,
    },
  };
  const cap = (sym, label) => `<div class="cap"><span class="symtile">${sym}</span><b>${label}</b></div>`;

  function addTarget(sim, target) {
    const box = sim.slots[0];                      // the pushed box is the first moving body
    if (!box) return;
    const g = new THREE.Group();
    box.children.forEach(m => {
      const c = m.clone(); c.material = new THREE.MeshBasicMaterial({ color: '#1b1e24', transparent: true, opacity: 0.18, depthWrite: false });
      c.castShadow = false; g.add(c);
    });
    g.position.set(target.pos[0], target.pos[1], box.position.z);
    g.quaternion.set(target.quat[1], target.quat[2], target.quat[3], target.quat[0]);
    sim.root.add(g);
  }

  function simPlayer(host, cfg, data, scene, run) {
    const [key, sym, label] = run, ep = data.episodes[key];
    const wrap = document.createElement('div'); wrap.className = 'player';
    wrap.innerHTML = `<div class="pane wide"><canvas aria-label="${cfg.title}: ${label}"></canvas><span class="tag">${cfg.title} · ${label}</span></div>${cap(sym, label)}`;
    host.appendChild(wrap);
    const canvas = wrap.querySelector('canvas');
    let sim;
    try { sim = new window.SS.Sim(canvas, scene, ep, { cam: cfg.cam }); } catch (e) { console.warn(e); return null; }
    if (data.target) addTarget(sim, data.target);
    sim.setEpisode(ep); sim.setStep(0); sim.setOverlays({});
    const T = ep.ee.length - 1;
    return {
      T, dispose() { sim.dispose(); }, setT(s) { sim.setStep(Math.min(T, s)); },
    };
  }

  function videoPlayer(host, cfg, run) {
    const [key, sym, label] = run;
    const wrap = document.createElement('div'); wrap.className = 'player';
    wrap.innerHTML = `<div class="pane video"><video muted playsinline loop preload="metadata" poster="media/sorting_${key}_poster.jpg" aria-label="${cfg.title}: ${label}">
      <source src="media/sorting_${key}.mp4" type="video/mp4"></video><span class="tag">${cfg.title} · ${label}</span><span class="vspeed">4×</span></div>${cap(sym, label)}`;
    host.appendChild(wrap);
    const v = wrap.querySelector('video');
    return { video: v, play() { if (!window.SS.reduced) v.play().catch(() => {}); }, pause() { v.pause(); }, dispose() { v.pause(); v.removeAttribute('src'); } };
  }

  let current = null, visible = false;
  async function show(name, host) {
    if (current) { current.stop(); current = null; }
    host.innerHTML = '';
    const cfg = TASKS[name];
    if (cfg.kind === 'video') {
      const vs = cfg.runs.map(run => videoPlayer(host, cfg, run));
      current = { wake() { vs.forEach(p => p.play()); }, sleep() { vs.forEach(p => p.pause()); }, stop() { vs.forEach(p => p.dispose()); } };
      if (visible) current.wake();
      return;
    }
    const [data, scene] = await Promise.all([window.SS.getJSON(cfg.data), window.SS.getJSON(cfg.scene), window.SS.loadThree()]);
    const ps = cfg.runs.map(run => simPlayer(host, cfg, data, scene, run)).filter(Boolean);
    const Tmax = Math.max(...ps.map(p => p.T)), dur = Tmax * cfg.speed + 1600;
    let t0 = performance.now(), raf = null, stopped = false;
    const loop = () => {
      if (stopped) return;
      const t = (performance.now() - t0) % dur, s = Math.max(0, (t - 600) / cfg.speed);
      ps.forEach(p => p.setT(s));
      raf = visible && !window.SS.reduced ? requestAnimationFrame(loop) : null;
    };
    ps.forEach(p => p.setT(window.SS.reduced ? p.T : 0));
    current = {
      wake() { if (!raf && !stopped && !window.SS.reduced) { t0 = performance.now(); raf = requestAnimationFrame(loop); } },
      sleep() {},
      stop() { stopped = true; ps.forEach(p => p.dispose()); },
    };
    if (visible) current.wake();
  }

  window.SS.initSteer = function () {
    const host = document.getElementById('players');
    if (!host) return;
    if ('IntersectionObserver' in window) new IntersectionObserver(es => es.forEach(e => {
      visible = e.isIntersecting;
      if (current) { if (visible) current.wake(); else current.sleep(); }
    }), { threshold: 0.25 }).observe(host);
    else visible = true;
    const tabs = Object.keys(TASKS).map(name => ['st-' + name, name]);
    tabs.forEach(([id, name]) => document.getElementById(id).addEventListener('click', () => {
      tabs.forEach(([id2]) => document.getElementById(id2).setAttribute('aria-selected', id2 === id));
      show(name, host).catch(e => console.warn('steer', e));
    }));
    show('sorting', host).catch(e => console.warn('steer', e));
  };
})();
