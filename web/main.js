// Neural Snake main page. This file contains NO game rules: every next frame comes from the model (web/model/snake.onnx).
// Every function carries a "// purpose:" line directly above it; scripts/audit_web.py prints them as the "what the JS still does" list.
import { installControls } from './controls.js'; // on-screen buttons and swipe: they only call setPending(action), exactly like an arrow key
import { chooseBackend, formatReports, errorText } from './selftest.js'; // runtime self-test: golden cases = the model's own recorded outputs, no game rules
import { fitCanvas, drawBoard, placeOrbs } from './render.js'; // drawing only: paints one frame from the cell classes it is given

const N = 12;
const TICK_MS = 150;
const NOOP = 4;
const VERBOSE = new URLSearchParams(location.search).get('verbose') === '1';
const KEY_TO_ACTION = { ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3 };
const BASE = new URL('./', import.meta.url);
const VENDOR = new URL('vendor/', BASE).href;
const MODEL_URL = new URL('model/snake.onnx', BASE).href;
const START_URL = new URL('model/start_state.json', BASE).href;
const SELFTEST_URL = new URL('model/selftest.json', BASE).href;
const SELFTEST_PANEL = new URLSearchParams(location.search).get('selftest') === '1'; // ?selftest=1: show every golden case's result on the page

// purpose: shorthand for document.getElementById.
const el = (id) => document.getElementById(id);
const canvas = el('board');
const ctx = canvas.getContext('2d');
const boardPx = fitCanvas(canvas); // canvas backing store: 480 px, or 960 px on high-density screens
const state = { ort: null, session: null, backend: null, note: '', reports: [], grid: null, start: null, over: false, busy: false, pendingAction: NOOP, steps: 0, times: [] };

// purpose: fetch a JSON file that ships with the page (the engine-generated start state).
async function loadJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + ' -> HTTP ' + r.status);
  return r.json();
}

// purpose: load one vendored onnxruntime-web bundle and create an inference session with exactly one execution provider.
async function openSession(bundle, provider) {
  const ort = await import(new URL(bundle, VENDOR).href);
  ort.env.wasm.wasmPaths = VENDOR;
  ort.env.wasm.numThreads = 1; // page is not cross-origin isolated
  // ?verbose=1 sets only the session option logSeverityLevel=0 (VERBOSE): ORT then logs "Node placements" (which EP each node runs on) to the console.
  const options = { executionProviders: [provider], ...(VERBOSE ? { logSeverityLevel: 0 } : {}) };
  const session = await ort.InferenceSession.create(MODEL_URL, options);
  return { ort, session };
}

// purpose: open a session for one named backend (the webgpu bundle for 'webgpu', the wasm bundle for 'wasm'); the self-tested backend chooser calls this.
function openBackend(name) {
  return name === 'webgpu' ? openSession('ort.webgpu.min.mjs', 'webgpu') : openSession('ort.wasm.min.mjs', 'wasm');
}

// purpose: load the golden cases and choose the backend: WebGPU first, kept only if it passes the self-test, else WASM (also self-tested; if it fails too we throw and the page does not play). Records which backend runs and why.
async function createSession() {
  const selftest = await loadJSON(SELFTEST_URL);
  try {
    const r = await chooseBackend({ open: openBackend, selftest, forceWasm: new URLSearchParams(location.search).get('backend') === 'wasm', testBoth: SELFTEST_PANEL });
    Object.assign(state, { ort: r.ort, session: r.session, backend: r.backend, reports: r.reports });
    return r.note;
  } catch (e) {
    state.reports = e.reports || [];
    throw e;
  }
}

// purpose: with ?selftest=1, show the result of every golden case on every backend tried (pass / fail, differing cells, max probability difference) so it can be screenshotted.
function renderSelfTestPanel() {
  if (!SELFTEST_PANEL) return;
  el('selftest').hidden = false;
  el('selftest').textContent = formatReports(state.reports, 'Self-test ' + new Date().toISOString() + '\nuserAgent: ' + navigator.userAgent);
}

// purpose: paint the current class grid (the model's latest frame) with the shared renderer: tiles, the snake from its direction classes, and one orb per food cell. Nothing is carried over from earlier frames.
function draw() {
  drawBoard(ctx, state.grid, boardPx);
  placeOrbs(el('orbs'), state.grid);
}

// purpose: show the backend, median step time, step count, the model's last ate / game-over probabilities and the game-over banner.
function renderStatus(probs) {
  const sorted = [...state.times].sort((a, b) => a - b);
  el('backend').textContent = state.backend ? state.backend + (state.note ? ' (' + state.note + ')' : '') : 'loading model...';
  el('ms').textContent = sorted.length ? sorted[Math.floor(sorted.length / 2)].toFixed(2) + ' (n=' + sorted.length + ')' : '-';
  el('steps').textContent = String(state.steps);
  if (probs) el('probs').textContent = 'p(ate)=' + probs.ate.toFixed(3) + '  p(game over)=' + probs.over.toFixed(3);
  el('banner').textContent = state.over ? 'Game over (the model says so). Press Space or Restart.' : '';
}

// purpose: draw one cell index from the model's food probabilities with Math.random (inverse CDF). Nothing is checked or resampled.
function sampleCell(probs) {
  let u = Math.random() * probs.reduce((a, b) => a + b, 0);
  for (let i = 0; i < probs.length; i++) { u -= probs[i]; if (u <= 0) return i; }
  return probs.length - 1;
}

// purpose: one model call: send (grid, action), take the model's argmax grid as the next state, write a sampled food cell if p(ate) > 0.5, flag game over if p(game over) > 0.5.
async function step(action) {
  const t0 = performance.now();
  const out = await state.session.run({
    grid: new state.ort.Tensor('int32', state.grid, [1, N, N]),
    action: new state.ort.Tensor('int32', Int32Array.of(action), [1]),
  });
  state.times.push(performance.now() - t0);
  if (state.times.length > 500) state.times.shift();
  state.grid = Int32Array.from(out.next_grid.data, Number); // next_grid is ArgMax's int64 output (BigInt64Array)
  const ate = out.ate.data[0], over = out.over.data[0];
  if (ate > 0.5) state.grid[sampleCell(out.food.data)] = 1;
  if (over > 0.5) state.over = true;
  state.steps++;
  return { ate, over };
}

// purpose: timer callback every 150 ms: consume the pending key (NOOP if none), run one model step, redraw. Skips when a step is still running or the game is over.
async function tick() {
  if (state.busy || state.over || !state.session) return;
  state.busy = true;
  const action = state.pendingAction;
  state.pendingAction = NOOP;
  try {
    renderStatus(await step(action));
    draw();
  } catch (e) {
    el('banner').textContent = 'Model error: ' + String((e && e.message) || e);
    state.over = true;
  }
  renderStatus(null);
  state.busy = false;
}

// purpose: remember an action index (0 UP, 1 DOWN, 2 LEFT, 3 RIGHT) for the next tick; the one place where arrow keys, on-screen buttons and swipes all end up.
function setPending(action) {
  state.pendingAction = action;
}

// purpose: map an arrow key to an action index and remember it for the next tick; Space/Enter restarts.
function onKey(ev) {
  if (ev.key in KEY_TO_ACTION) { setPending(KEY_TO_ACTION[ev.key]); ev.preventDefault(); }
  else if (ev.key === ' ' || ev.key === 'Enter') { ev.preventDefault(); restart(); }
}

// purpose: restart = reload the engine-generated start state from the data file and clear the game-over flag.
function restart() {
  state.grid = Int32Array.from(state.start.grid.flat());
  state.over = false;
  state.pendingAction = NOOP;
  state.steps = 0;
  draw();
  renderStatus(null);
}

// purpose: page start-up: load the start state, create the session (WebGPU first), wire the keyboard, touch controls and restart button, start the 150 ms timer.
async function main() {
  state.start = await loadJSON(START_URL);
  restart();
  try {
    state.note = await createSession();
  } catch (e) {
    el('backend').textContent = 'not playing: ' + errorText(e);
    el('banner').textContent = 'No backend passed the self-test (or could load the model), so the page does not play. Open ?selftest=1 for the details.';
    renderSelfTestPanel();
    return;
  }
  renderStatus(null);
  renderSelfTestPanel();
  window.addEventListener('keydown', onKey);
  installControls({ buttons: document.querySelectorAll('#pad button'), swipeTargets: [canvas], onAction: setPending });
  el('restart').addEventListener('click', restart);
  setInterval(tick, TICK_MS);
}

main();
