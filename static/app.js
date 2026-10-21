// NIST Sim Lab front end: draws the real simulator's geoms (MuJoCo geom_xpos/geom_xmat, z-up metres) and drives it via /api.
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const $ = (id) => document.getElementById(id);
window.__simlabOK = true;
function setStatus(ok, msg) {
  $('status').innerHTML = ok ? '<span class="ok">● simulator connected</span>' : `<span class="bad">● simulator error: ${msg || 'no reply'}</span>`;
}
setStatus(false, 'connecting…');
const PEG_COLORS = ['#ff7a59', '#ffd166', '#06d6a0', '#4cc9f0', '#b388ff', '#f472b6', '#a3e635', '#f59e0b'];
const HAND_BODIES = new Set(['hand', 'fl', 'fr']);

// ---------- API ----------
async function api(cmd, args = {}) {
  let r;
  try { r = await fetch('/api', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cmd, ...args }) }); }
  catch (e) { setStatus(false, 'cannot reach the lab server (is server.py running?)'); throw e; }
  const j = await r.json();
  if (j.error) { const e = $('err'); e.textContent = j.error + '\n' + (j.trace || ''); e.style.display = 'block'; setStatus(false, j.error); throw new Error(j.error); }
  $('err').style.display = 'none';
  setStatus(true);
  return j;
}

// ---------- three.js scene ----------
const view = $('view');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
view.appendChild(renderer.domElement);
renderer.domElement.tabIndex = 0;
const scene = new THREE.Scene();
scene.background = new THREE.Color('#15171c');
const camera = new THREE.PerspectiveCamera(38, 1, 0.002, 20);
camera.up.set(0, 0, 1);
camera.position.set(0.0, -0.42, 0.32);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x30343c, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(0.3, -0.5, 1.0);
scene.add(sun);
const grid = new THREE.GridHelper(0.6, 60, 0x3a3f4a, 0x262a32);
grid.rotateX(Math.PI / 2);
grid.position.z = -0.0005;
scene.add(grid);
const world = new THREE.Group();
scene.add(world);
const overlay = new THREE.Group();
scene.add(overlay);

function resize() {
  const w = view.clientWidth, h = view.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);

let meshes = [], meta = null, last = null, follow = false, ghost = true, xray = false, cases = [];

function geomMesh(g) {
  const [a, b, c] = g.size;
  let geo = null;
  if (g.type === 'box') geo = new THREE.BoxGeometry(2 * a, 2 * b, 2 * c);
  else if (g.type === 'cylinder') geo = new THREE.CylinderGeometry(a, a, 2 * b, 40).rotateX(Math.PI / 2);
  else if (g.type === 'capsule') geo = new THREE.CapsuleGeometry(a, 2 * b, 8, 20).rotateX(Math.PI / 2);
  else if (g.type === 'sphere') geo = new THREE.SphereGeometry(a, 24, 16);
  else if (g.type === 'ellipsoid') geo = new THREE.SphereGeometry(1, 24, 16).scale(a, b, c);
  else if (g.type === 'plane') geo = new THREE.PlaneGeometry(a > 0 ? 2 * a : 0.8, b > 0 ? 2 * b : 0.8);
  if (!geo) return null;
  let color = new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]), opacity = g.rgba[3];
  const pm = /^peg(\d+)$/.exec(g.body);
  if (pm) color = new THREE.Color(PEG_COLORS[+pm[1] % 8]);
  else if (HAND_BODIES.has(g.body)) color = new THREE.Color(g.name.startsWith('pad') ? '#e0e0e0' : '#6b7280');
  else if (/^clutter/.test(g.body)) color = new THREE.Color('#5b7db1');
  else if (g.type === 'plane') color = new THREE.Color('#1c1f26');
  const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.05, transparent: opacity < 1, opacity });
  const m = new THREE.Mesh(geo, mat);
  m.matrixAutoUpdate = false;
  m.userData = { body: g.body, name: g.name, hand: HAND_BODIES.has(g.body), peg: pm ? +pm[1] : -1 };
  if (opacity === 0) m.visible = false;
  return m;
}

function buildScene(s) {
  world.clear();
  overlay.clear();
  meshes = s.geoms.map((g) => { const m = geomMesh(g); if (m) world.add(m); return m; });
  const first = !meta || meta.seed !== s.seed || meta.episode !== s.episode;
  controls.target.set(s.board_xy[0], s.board_xy[1], 0.03);
  if (first) camera.position.set(s.board_xy[0] + 0.05, s.board_xy[1] - 0.24, 0.2);
  const c = s.conditions;
  $('cond').textContent = `k ${c.k} pegs · horizon ${s.horizon} steps · friction ${c.friction} · mass ${c.peg_mass_g} g · ` +
    `clearance ${c.clearance_mm} mm · obs noise ${c.obs_noise_mm} mm · action delay ${c.act_delay} · clutter ${c.clutter}`;
  meta = s;
  applyXray();
}

const sphereGeo = new THREE.SphereGeometry(0.0025, 12, 8);
const ringGeo = new THREE.TorusGeometry(1, 0.06, 6, 40);

function drawOverlay(st) {
  overlay.clear();
  if (!ghost || !meta) return;
  st.pegs.forEach((p) => {
    const col = new THREE.Color(PEG_COLORS[p.i % 8]);
    const r = (meta.pegs_meta[p.i].dia_mm / 2000) + 0.002;
    const est = new THREE.Mesh(sphereGeo, new THREE.MeshBasicMaterial({ color: col, wireframe: true }));
    est.position.set(...p.est);
    overlay.add(est);
    const hole = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.5 }));
    hole.scale.setScalar(r); hole.position.set(p.hole[0], p.hole[1], 0.0205); overlay.add(hole);
    const he = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ color: col }));
    he.scale.setScalar(r * 0.8); he.position.set(p.hole_est[0], p.hole_est[1], 0.0207); overlay.add(he);
  });
  const t = st.hand.target;
  const cross = new THREE.Mesh(new THREE.OctahedronGeometry(0.003), new THREE.MeshBasicMaterial({ color: 0x5aa7ff, wireframe: true }));
  cross.position.set(t[0], t[1], t[2]); overlay.add(cross);
}

function applyXray() {
  meshes.forEach((m) => { if (m && m.userData.hand) { m.material.transparent = xray; m.material.opacity = xray ? 0.25 : 1; m.material.needsUpdate = true; } });
}

function applyFrames(b64) {
  const bin = atob(b64), buf = new ArrayBuffer(bin.length), u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  const f = new Float32Array(buf);
  meshes.forEach((m, g) => {
    if (!m) return;
    const o = g * 12;
    m.matrix.set(f[o + 3], f[o + 4], f[o + 5], f[o], f[o + 6], f[o + 7], f[o + 8], f[o + 1], f[o + 9], f[o + 10], f[o + 11], f[o + 2], 0, 0, 0, 1);
  });
}

// ---------- state display ----------
const fmt = (x, d = 1) => (x === null || x === undefined ? '—' : Number(x).toFixed(d));
function show(st) {
  last = st;
  applyFrames(st.frames);
  drawOverlay(st);
  const h = st.hand, pol = st.policy;
  const tgt = ['NEXT', 'IDLE'].includes(pol.phase) ? -1 : pol.target;
  meshes.forEach((m) => { if (m && m.userData.peg >= 0) m.material.emissive.set(m.userData.peg === tgt ? '#3a3a3a' : '#000'); });
  if (follow) controls.target.set(h.xyz[0], h.xyz[1], Math.max(0.02, h.xyz[2]));
  const kv = [
    ['step', `${st.step} / ${meta ? meta.horizon : '?'}`], ['episode end', st.done ? `${st.terminal} · score ${st.score}` : 'running'],
    ['policy phase', `${pol.phase} → peg ${pol.target}`], ['policy action', pol.action.map((x) => fmt(x, 2)).join('  ')],
    ['hand (mm)', h.xyz.map((x) => fmt(x * 1000)).join(', ') + `  yaw ${fmt(h.yaw * 57.2958)}°`],
    ['jaw gap', `${fmt(h.gap_mm, 2)} mm (cmd ${fmt(h.grip_cmd_mm * 2, 1)} mm)`],
    ['wrist F', h.force.map((x) => fmt(x, 2)).join(', ')], ['wrist τ / pad', h.torque.map((x) => fmt(x, 2)).join(', ') + ` / ${fmt(h.pad, 2)}`],
    ['mean |a|²', `${st.msa}`], ['my actions', `${st.n_actions}`],
  ];
  $('kv').innerHTML = kv.map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join('');
  const rows = st.pegs.map((p) => {
    const m = meta.pegs_meta[p.i];
    const e = Math.hypot(p.est[0] - p.pos[0], p.est[1] - p.pos[1]) * 1000;
    const az = p.az > 0.985 ? `<span class="ok">${fmt(p.az, 2)}</span>` : p.az < 0.5 ? `<span class="bad">${fmt(p.az, 2)}</span>` : `<span class="warn">${fmt(p.az, 2)}</span>`;
    const seat = p.seated_now ? '<span class="ok">yes</span>' : (p.first_seat !== null ? `<span class="warn">@${p.first_seat}</span>` : '—');
    return `<tr><td><span class="sw" style="background:${PEG_COLORS[p.i % 8]}"></span>${p.i} ${m.shape[0]}${m.dia_mm}</td><td>${m.start}</td>` +
      `<td>${az}</td><td>${seat}</td><td>${p.touch ? '✋' : ''}</td><td>${fmt(e, 1)}</td></tr>`;
  });
  $('pegs').innerHTML = '<tr><th>peg</th><th>start</th><th>upright</th><th>seated</th><th>grip</th><th>est err mm</th></tr>' + rows.join('');
  $('hud').textContent = `step ${st.step}/${meta ? meta.horizon : '?'} · ${mode === 'policy' ? 'POLICY' : 'MANUAL'} · ${pol.phase} → peg ${pol.target}` +
    (st.done ? ` · END ${st.terminal} score ${st.score}` : '');
}

// ---------- controls ----------
let mode = 'policy', playing = false, busy = false;
const keys = new Set();
function manualAction() {
  const m = +$('mag').value;
  const k = (c) => (keys.has(c) ? 1 : 0);
  return [(k('d') - k('a')) * m, (k('w') - k('s')) * m, (k('r') - k('f')) * m, (k('q') - k('e')) * m, +$('grip').value / 12 - 1];
}
function showAct() { const a = manualAction(); $('act').textContent = 'manual action: ' + a.map((x) => fmt(x, 2)).join('  '); }
document.addEventListener('keydown', (ev) => {
  if (ev.target.tagName === 'INPUT' || ev.target.tagName === 'SELECT') return;
  const c = ev.key.toLowerCase();
  if (c === 'g') { const g = $('grip'); g.value = +g.value > 12 ? 0 : 24; g.dispatchEvent(new Event('input')); }
  else if ('wasdrfqe'.includes(c)) keys.add(c);
  showAct();
});
document.addEventListener('keyup', (ev) => { keys.delete(ev.key.toLowerCase()); showAct(); });
$('mag').addEventListener('input', () => { $('magv').textContent = (+$('mag').value).toFixed(2); showAct(); });
$('grip').addEventListener('input', () => { $('gripv').textContent = `${(+$('grip').value).toFixed(1)} mm`; showAct(); });

async function stepOnce(n) {
  if (busy) return; busy = true;
  try { show(await api('step', { n, mode, action: manualAction() })); } finally { busy = false; }
}
function setMode(m) { mode = m; $('mPolicy').classList.toggle('on', m === 'policy'); $('mManual').classList.toggle('on', m === 'manual'); if (last) show(last); }
$('mPolicy').onclick = () => setMode('policy');
$('mManual').onclick = () => setMode('manual');
$('step1').onclick = () => stepOnce(1);
$('step25').onclick = () => stepOnce(25);
$('play').onclick = async () => {
  playing = !playing; $('play').textContent = playing ? '❚❚ Pause' : '▶ Play'; $('play').classList.toggle('on', playing);
  while (playing) {
    const t0 = performance.now();
    await stepOnce(+$('speed').value);
    if (last && last.done) { playing = false; $('play').textContent = '▶ Play'; $('play').classList.remove('on'); }
    const wait = 40 - (performance.now() - t0);   // 25 Hz = the controller's real rate
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
  }
};
$('reset').onclick = async () => { playing = false; const s = await api('reset', { seed: +$('seed').value, episode: +$('epi').value }); buildScene(s); show(s); };
$('loadcases').onclick = async () => {
  const r = await api('cases', { cause: $('cause').value });
  cases = r.cases;
  $('cases').innerHTML = r.cases.map((c, i) => `<option value="${i}">${c.seed}/${c.episode} peg ${c.peg} ${c.shape[0]}${c.dia_mm} ${c.start} · ${c.cause} · @${c.goto}</option>`).join('');
};
$('gocase').onclick = async () => {
  const o = $('cases').selectedOptions[0]; if (!o) return;
  const c = cases[+o.value];
  playing = false; $('seed').value = c.seed; $('epi').value = c.episode;
  const s = await api('goto', { seed: c.seed, episode: c.episode, step: c.goto });
  buildScene(s); show(s);
};
$('save').onclick = async () => {
  const r = await api('snapshot', { name: $('snapname').value || 's' });
  $('snaps').innerHTML = r.snapshots.map((n) => `<option>${n}</option>`).join('');
  $('snaps').value = r.saved;
};
$('restore').onclick = async () => { const n = $('snaps').value; if (!n) return; playing = false; const s = await api('restore', { name: n }); buildScene(s); show(s); };
$('export').onclick = async () => {
  const r = await api('export');
  $('status').innerHTML = `<span class="ok">● exported ${r.actions.length} actions → saved on the lab machine: ${r.saved_to || '(not saved)'}</span>`;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(r, null, 1)], { type: 'application/json' }));
  a.download = `simlab_${r.seed}_${r.episode}_step${r.step}.json`; a.click();
};
$('vTop').onclick = () => { follow = false; const t = controls.target; camera.position.set(t.x, t.y - 0.001, t.z + 0.45); };
$('vSide').onclick = () => { follow = false; const t = controls.target; camera.position.set(t.x, t.y - 0.32, t.z + 0.08); };
$('vFollow').onclick = () => { follow = !follow; $('vFollow').classList.toggle('on', follow); };
$('ghost').onclick = () => { ghost = !ghost; $('ghost').classList.toggle('on', ghost); if (last) drawOverlay(last); };
$('xray').onclick = () => { xray = !xray; $('xray').classList.toggle('on', xray); applyXray(); };

function loop() { controls.update(); renderer.render(scene, camera); requestAnimationFrame(loop); }
resize(); loop(); showAct();
$('reset').click();
