/* Page bootstrap. Every module is optional: a failure in one leaves the rest of the page working. */
(function () {
  const SS = window.SS;
  SS.initRise();

  // BibTeX copy
  const copyBtn = document.getElementById('copy-bib');
  copyBtn && copyBtn.addEventListener('click', () => {
    const txt = document.getElementById('bibtex').textContent;
    const done = () => { copyBtn.textContent = 'Copied'; setTimeout(() => copyBtn.textContent = 'Copy', 1600); };
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(txt).then(done, () => selectBib());
    else selectBib();
  });
  function selectBib() { const r = document.createRange(); r.selectNodeContents(document.getElementById('bibtex')); const s = window.getSelection(); s.removeAllRanges(); s.addRange(r); copyBtn.textContent = 'Selected'; }

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
