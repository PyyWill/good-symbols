/* Page bootstrap. Every module is optional: a failure in one leaves the rest of the page working. */
(function () {
  const SS = window.SS;
  SS.initRise();

  // BibTeX copy
  const copyBtn = document.getElementById('copy-bib');
  if (copyBtn) copyBtn.hidden = false;
  copyBtn && copyBtn.addEventListener('click', () => {
    const txt = document.getElementById('bibtex').textContent;
    const done = () => { copyBtn.textContent = 'Copied'; setTimeout(() => copyBtn.textContent = 'Copy', 1600); };
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(txt).then(done, () => selectBib());
    else selectBib();
  });
  function selectBib() { const r = document.createRange(); r.selectNodeContents(document.getElementById('bibtex')); const s = window.getSelection(); s.removeAllRanges(); s.addRange(r); copyBtn.textContent = 'Selected'; }

  // in-page links jump straight to the target (a smooth scroll would wake every lazy section on the way and land short
  // of the target as they grow), then keep re-aiming while sections near the target finish loading or the reader scrolls
  document.addEventListener('click', e => {
    const a = e.target.closest && e.target.closest('a[href^="#"]'), id = a && a.getAttribute('href').slice(1);
    const el = id && document.getElementById(id);
    if (!el) return;
    e.preventDefault();
    try { history.pushState(null, '', '#' + id); } catch (_) {}
    const aim = () => {
      const h = document.documentElement, b = h.style.scrollBehavior;
      h.style.scrollBehavior = 'auto'; window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY); h.style.scrollBehavior = b;
    };
    aim();
    if (!('ResizeObserver' in window)) return;
    const ro = new ResizeObserver(aim), stop = () => ro.disconnect();
    ro.observe(document.body); setTimeout(stop, 5000);
    window.addEventListener('wheel', stop, { once: true, passive: true }); window.addEventListener('touchstart', stop, { once: true, passive: true });
  });

  // the hero first; every other part loads when it comes within about a screen of the viewport
  try { SS.startHero(); } catch (e) { console.warn('hero', e); }
  SS.whenNear(document.getElementById('try'), () => {
    Promise.all([SS.getJSON('data/avoiding/episodes.json'), SS.getJSON('data/avoiding/scene.json'), SS.loadThree()]).then(([eps, scene]) => {
      try { SS.startLab(eps, scene); } catch (e) { console.warn('lab', e); }
    }).catch(e => console.warn('avoiding data', e));
  });

  // figures (inlined when near), results charts, steering
  SS.initFigures().catch(e => console.warn('figures', e));
  SS.getJSON('data/results.json').then(R => { SS.initCharts(R); }).catch(e => console.warn('charts', e));
  if (SS.initSteer) SS.whenNear(document.getElementById('steering'), () => SS.initSteer());

  // dexterous hands
  if (SS.initDex) SS.whenNear(document.getElementById('dex-row'), () => SS.initDex().catch(e => console.warn('dex', e)));

})();
