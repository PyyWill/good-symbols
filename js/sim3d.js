/* Avoiding scene in three.js, rebuilt from the MuJoCo model (meshes + primitive geoms) and driven by the
   recorded body poses of real closed-loop episodes. MuJoCo is z-up; the root group rotates it to three's y-up. */
(function () {
  function b64ToBytes(b64) { const s = atob(b64), n = s.length, o = new Uint8Array(n); for (let i = 0; i < n; i++) o[i] = s.charCodeAt(i); return o; }
  const f32 = b64 => { const b = b64ToBytes(b64); return new Float32Array(b.buffer, b.byteOffset, b.byteLength >> 2); };
  const u16 = b64 => { const b = b64ToBytes(b64); return new Uint16Array(b.buffer, b.byteOffset, b.byteLength >> 1); };
  const quat = q => new THREE.Quaternion(q[1], q[2], q[3], q[0]);

  class Sim {
    constructor(canvas, scene, data, opts) {
      this.canvas = canvas; this.data = data; this.opts = opts || {};
      const O = Object.assign({ pos: [1.72, 1.92, -0.05], target: [0.47, -0.04, -0.05], fov: 25, z: -0.0165 }, this.opts.cam || {});
      this.Z = O.z;
      const R = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, preserveDrawingBuffer: false });
      R.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      R.shadowMap.enabled = true; R.shadowMap.type = THREE.PCFSoftShadowMap;
      this.R = R;
      const S = new THREE.Scene();
      S.background = new THREE.Color('#e3e6ec');
      this.S = S;
      const cam = new THREE.PerspectiveCamera(O.fov, 1, 0.02, 20);
      cam.position.set(...O.pos);
      this.cam = cam;
      const ctl = new THREE.OrbitControls(cam, canvas);
      ctl.target.set(...O.target); ctl.enableDamping = true; ctl.dampingFactor = 0.08;
      ctl.minDistance = 0.7; ctl.maxDistance = 4.0; ctl.maxPolarAngle = Math.PI * 0.49; ctl.enablePan = false;
      ctl.addEventListener('start', () => this.onInteract && this.onInteract());
      this.ctl = ctl;
      S.add(new THREE.HemisphereLight(0xffffff, 0x9aa1ad, 0.62));
      const sun = new THREE.DirectionalLight(0xffffff, 0.62);
      sun.position.set(1.4, 2.2, 0.8); sun.castShadow = true;
      sun.shadow.mapSize.set(1024, 1024); sun.shadow.camera.left = -1; sun.shadow.camera.right = 1; sun.shadow.camera.top = 1; sun.shadow.camera.bottom = -1;
      sun.shadow.radius = 4; sun.shadow.bias = -0.0008;
      S.add(sun);
      const root = new THREE.Group(); root.rotation.x = -Math.PI / 2; S.add(root); this.root = root;
      // floor under the table
      const floor = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), new THREE.MeshStandardMaterial({ color: '#d9dde4', roughness: 1 }));
      floor.position.z = -0.94; floor.receiveShadow = true; root.add(floor);
      this.build(scene);
      this.trails = new THREE.Group(); root.add(this.trails);
      this.resize();
      window.addEventListener('resize', () => this.resize());
      this.dirty = true;
      const loop = () => { if (this.disposed) return; this.ctl.update(); if (this.visible) R.render(S, cam); requestAnimationFrame(loop); };
      this.visible = true;
      if ('IntersectionObserver' in window) new IntersectionObserver(es => es.forEach(e => this.visible = e.isIntersecting)).observe(canvas);
      requestAnimationFrame(loop);
    }
    dispose() { this.disposed = true; try { this.ctl.dispose(); this.R.dispose(); this.R.forceContextLoss(); } catch (e) { /* ignore */ } }
    resize() {
      const r = this.canvas.getBoundingClientRect();
      this.R.setSize(r.width, r.height, false);
      this.cam.aspect = r.width / Math.max(1, r.height); this.cam.updateProjectionMatrix();
    }
    material(g) {
      const n = (g.name || g.body_name || '');
      if (n.includes('obs')) return new THREE.MeshStandardMaterial({ color: '#b8323f', roughness: 0.6 });
      if (n.includes('finish')) return new THREE.MeshStandardMaterial({ color: '#3aa877', roughness: 0.9, transparent: true, opacity: 0.55 });
      if (g.body_name === 'table_plane') return new THREE.MeshStandardMaterial({ color: '#f7f5f0', roughness: 0.95 });
      if (n.startsWith('foot')) return new THREE.MeshStandardMaterial({ color: '#30343c', roughness: 0.8 });
      if (g.type === 7 && g.rgba[3] < 0.5) return new THREE.MeshStandardMaterial({ color: '#fbfbfc', roughness: 0.45, metalness: 0.0 });
      if (g.type === 7 || n.startsWith('rod')) return new THREE.MeshStandardMaterial({ color: '#7d828d', roughness: 0.5, metalness: 0.2 });
      if (g.rgba && g.rgba[3] > 0.5 && !/^(front_|back_|vert_|side_)/.test(g.body_name || '')) return new THREE.MeshStandardMaterial({ color: new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]), roughness: 0.6 });
      return new THREE.MeshStandardMaterial({ color: '#d7dbe2', roughness: 0.8 });
    }
    build(scene) {
      const meshGeo = {};
      for (const m of scene.meshes) {
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(f32(m.V), 3));
        g.setIndex(new THREE.BufferAttribute(u16(m.F), 1));
        g.computeVertexNormals();
        meshGeo[m.id] = g;
      }
      this.slots = [];
      for (let i = 0; i < this.data.n_pose_bodies; i++) { const b = new THREE.Group(); this.root.add(b); this.slots.push(b); }
      for (const g of scene.geoms) {
        if (/^(front_|back_|vert_|side_|foot_)/.test(g.body_name || '')) continue;   // table cage: clutter
        let geo;
        if (g.type === 7) geo = meshGeo[g.mesh];
        else if (g.type === 6) geo = new THREE.BoxGeometry(2 * g.size[0], 2 * g.size[1], 2 * g.size[2]);
        else if (g.type === 5) { geo = new THREE.CylinderGeometry(g.size[0], g.size[0], 2 * g.size[1], 40); geo.rotateX(Math.PI / 2); }
        else if (g.type === 2) geo = new THREE.SphereGeometry(g.size[0], 20, 14);
        else continue;
        const mesh = new THREE.Mesh(geo, this.material(g));
        mesh.castShadow = !(g.body_name === 'table_plane'); mesh.receiveShadow = true;
        mesh.position.set(...g.pos); mesh.quaternion.copy(quat(g.quat));
        if (g.slot >= 0) this.slots[g.slot].add(mesh);
        else {
          const holder = new THREE.Group(); holder.position.set(...g.body_pos); holder.quaternion.copy(quat(g.body_quat));
          holder.add(mesh); this.root.add(holder);
        }
      }
    }
    setEpisode(ep) {
      this.ep = ep;
      if (!ep._poses) ep._poses = f32(ep.poses);
      this.T = ep.ee.length - 1;
    }
    /* s: continuous control step in [0, T] */
    setStep(s) {
      const ep = this.ep, P = ep._poses, nb = ep.n_pose_bodies;
      s = Math.max(0, Math.min(this.T, s));
      const i = Math.floor(s), j = Math.min(this.T, i + 1), f = s - i;
      for (let b = 0; b < nb; b++) {
        const o1 = (i * nb + b) * 7, o2 = (j * nb + b) * 7;
        this.slots[b].position.set(P[o1] + (P[o2] - P[o1]) * f, P[o1 + 1] + (P[o2 + 1] - P[o1 + 1]) * f, P[o1 + 2] + (P[o2 + 2] - P[o1 + 2]) * f);
        const qa = new THREE.Quaternion(P[o1 + 4], P[o1 + 5], P[o1 + 6], P[o1 + 3]), qb = new THREE.Quaternion(P[o2 + 4], P[o2 + 5], P[o2 + 6], P[o2 + 3]);
        this.slots[b].quaternion.copy(qa.slerp(qb, f));
      }
      this.step = s;
      this.updateTrail();
    }
    line(points, color, opts) {
      opts = opts || {};
      const pts = points.map(p => new THREE.Vector3(p[0], p[1], opts.z != null ? opts.z : p[2]));
      if (pts.length < 2) return null;
      const curve = new THREE.CatmullRomCurve3(pts);
      const geo = new THREE.TubeGeometry(curve, Math.max(8, pts.length * 3), opts.r || 0.0035, 8, false);
      const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: opts.opacity == null ? 1 : opts.opacity, depthWrite: !(opts.opacity < 1) });
      return new THREE.Mesh(geo, mat);
    }
    clearTrails() { while (this.trails.children.length) { const c = this.trails.children.pop(); c.geometry && c.geometry.dispose(); } }
    /* ghost: the other episode's whole path; plans: [{pts, chosen}] for the current decision */
    setOverlays(o) { this.overlay = o; this.updateTrail(true); }
    updateTrail(force) {
      const s = Math.floor(this.step || 0);
      if (!force && s === this._lastTrail) return;
      this._lastTrail = s;
      this.clearTrails();
      const o = this.overlay || {};
      const Z = this.Z;   // drawn on the table top
      if (o.ghost) { const g = this.line(o.ghost.map(p => [p[0], p[1], Z]), '#8a90a0', { opacity: 0.55, r: 0.0022 }); g && this.trails.add(g); }
      const ee = this.ep.ee.slice(0, s + 1).map(p => [p[0], p[1], Z]);
      const t = this.line(ee, '#1b1e24', { r: 0.0042 }); t && this.trails.add(t);
      (o.plans || []).forEach(pl => {
        const geo = new THREE.SphereGeometry(pl.chosen ? 0.0062 : 0.0048, 12, 10);
        const mat = new THREE.MeshBasicMaterial({ color: pl.chosen ? '#3f56c9' : '#9aa3c4', transparent: !pl.chosen, opacity: pl.chosen ? 1 : pl.faint ? 0.3 : 0.85 });
        pl.pts.forEach((p, i) => { if (i % 2) return; const m = new THREE.Mesh(geo, mat); m.position.set(p[0], p[1], Z + 0.004); this.trails.add(m); });
      });
    }
  }

  window.SS.Sim = Sim;
})();
