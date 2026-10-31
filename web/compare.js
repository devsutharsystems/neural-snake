// Comparison page: runs the model (web/model/snake.onnx) and a REAL hand-written engine (web/engine.js) side by side on the same keys and ticks.
// This is the only page allowed to import engine.js; scripts/audit_web.py fails if index.html or main.js reference it.
import { Engine, RIGHT, NOOP } from './engine.js';
import { installControls } from './controls.js'; // on-screen buttons and swipe: they only call setPending(action), exactly like an arrow key
import { chooseBackend, formatReports, errorText } from './selftest.js'; // runtime self-test: golden cases = the model's own recorded outputs, no game rules
import { fitCanvas, drawBoard, placeOrbs } from './render.js'; // drawing only: paints one frame from the cell classes it is given

const N = 12;
const TICK_MS = 150;
const KEY_TO_ACTION = { ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3 };
const VERBOSE = new URLSearchParams(location.search).get('verbose') === '1';
const BASE = new URL('./', import.meta.url);
const VENDOR = new URL('vendor/', BASE).href;
const MODEL_URL = new URL('model/snake.onnx', BASE).href;
const START_URL = new URL('model/start_state.json', BASE).href;
const SELFTEST_URL = new URL('model/selftest.json', BASE).href;
const SELFTEST_PANEL = new URLSearchParams(location.search).get('selftest') === '1'; // ?selftest=1: show every golden case's result on the page

// purpose: shorthand for document.getElementById.
const el = (id) => document.getElementById(id);
fitCanvas(el('model')); // canvas backing store: 480 px, or 960 px on high-density screens
fitCanvas(el('engine'));
const state = { ort: null, session: null, backend: '', reports: [], start: null, modelGrid: null, engine: null, modelOver: false, busy: false, pendingAction: NOOP,
  steps: 0, firstDivergence: null, mismatched: 0, stopped: false };

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
  const options = { executionProviders: [provider], ...(VERBOSE ? { logSeverityLevel: 0 } : {}) };
  return { ort, session: await ort.InferenceSession.create(MODEL_URL, options) };
}

// purpose: open a session for one named backend (the webgpu bundle for 'webgpu', the wasm bundle for 'wasm'); the self-tested backend chooser calls this.
function openBackend(name) {
  return name === 'webgpu' ? openSession('ort.webgpu.min.mjs', 'webgpu') : openSession('ort.wasm.min.mjs', 'wasm');
}

// purpose: same backend choice as the main page: load the golden cases, WebGPU first and kept only if it passes the self-test, else WASM (also self-tested; if it fails too we throw and the page does not play). The label shows what is really running and why.
async function createSession() {
  const selftest = await loadJSON(SELFTEST_URL);
  try {
    const r = await chooseBackend({ open: openBackend, selftest, forceWasm: new URLSearchParams(location.search).get('backend') === 'wasm', testBoth: SELFTEST_PANEL });
    Object.assign(state, { ort: r.ort, session: r.session, backend: r.backend, reports: r.reports });
    return r.note ? ' (' + r.note + ')' : '';
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

// purpose: paint one 14-class grid on a board with the shared renderer (tiles, snake from its direction classes, food orbs); cells listed in `diff` get a red outline.
function drawGrid(canvas, orbs, grid, diff) {
  drawBoard(canvas.getContext('2d'), grid, canvas.width, diff);
  placeOrbs(orbs, grid);
}

// purpose: one model call on (grid, action); returns the model's argmax grid and its game-over decision. Does no food sampling on this page.
async function modelStep(action) {
  const out = await state.session.run({
    grid: new state.ort.Tensor('int32', state.modelGrid, [1, N, N]),
    action: new state.ort.Tensor('int32', Int32Array.of(action), [1]),
  });
  return { grid: Int32Array.from(out.next_grid.data, Number), ate: out.ate.data[0], over: out.over.data[0] };
}

// purpose: compare the model board with the engine's 14-class grid cell by cell; returns the per-cell difference mask and the count.
function compareGrids(a, b) {
  const diff = new Uint8Array(N * N);
  let n = 0;
  for (let i = 0; i < N * N; i++) if (a[i] !== b[i]) { diff[i] = 1; n++; }
  return { diff, n };
}

// purpose: draw both boards and update the step counter, first divergence, mismatch count and game-over text.
function render(diff, summary) {
  drawGrid(el('model'), el('orbsModel'), state.modelGrid, diff);
  drawGrid(el('engine'), el('orbsEngine'), state.engine.grid14(), null);
  el('overModel').textContent = state.modelOver ? 'Game over (the model says so)' : '';
  el('overEngine').textContent = state.engine.gameOver ? 'Game over (the engine says so)' : '';
  el('backend').textContent = state.backend || 'loading...';
  el('steps').textContent = String(state.steps);
  el('first').textContent = state.firstDivergence === null ? 'none yet' : 'step ' + state.firstDivergence;
  el('mism').textContent = String(state.mismatched) + ' of ' + state.steps;
  el('now').textContent = summary || '-';
  el('over').textContent = state.steps === 0 ? '-' : 'model: ' + (state.modelOver ? 'game over' : 'running') + ' | engine: ' + (state.engine.gameOver ? 'game over' : 'running');
}

// purpose: timer callback every 150 ms: apply the pending key (NOOP if none) to the engine and the model, sync food after an engine eat, compare, render.
async function tick() {
  if (state.busy || state.stopped || !state.session) return;
  state.busy = true;
  const action = state.pendingAction;
  state.pendingAction = NOOP;
  try {
    const eaten = state.engine.step(action);
    const m = await modelStep(action);
    state.modelGrid = m.grid;
    state.modelOver = m.over > 0.5;
    if (eaten && state.engine.food) state.modelGrid[state.engine.food[0] * N + state.engine.food[1]] = 1; // food sync: engine's food replaces the model's
    state.steps++;
    const { diff, n } = compareGrids(state.modelGrid, state.engine.grid14());
    const overDiffers = state.modelOver !== state.engine.gameOver;
    const mismatch = n > 0 || overDiffers;
    if (mismatch) { state.mismatched++; if (state.firstDivergence === null) state.firstDivergence = state.steps; }
    render(diff, (mismatch ? n + ' cell(s) differ' + (overDiffers ? ', game-over flags differ' : '') : 'identical') +
      ' | p(ate)=' + m.ate.toFixed(3) + ' engine ate=' + eaten + ' | p(over)=' + m.over.toFixed(3));
    if (state.modelOver || state.engine.gameOver) state.stopped = true;
  } catch (e) {
    el('now').textContent = 'error: ' + String((e && e.message) || e);
    state.stopped = true;
  }
  state.busy = false;
}

// purpose: remember an action index (0 UP, 1 DOWN, 2 LEFT, 3 RIGHT) for the next tick; both sides get it. Arrow keys, on-screen buttons and swipes all end up here.
function setPending(action) {
  state.pendingAction = action;
}

// purpose: map an arrow key to an action index for the next tick (both sides get the same key); Space/Enter restarts both.
function onKey(ev) {
  if (ev.key in KEY_TO_ACTION) { setPending(KEY_TO_ACTION[ev.key]); ev.preventDefault(); }
  else if (ev.key === ' ' || ev.key === 'Enter') { ev.preventDefault(); restart(); }
}

// purpose: reset both sides to the same engine-generated start state and clear the counters.
function restart() {
  const s = state.start;
  state.modelGrid = Int32Array.from(s.grid.flat());
  state.engine = new Engine({ n: N, body: [s.head], direction: RIGHT, food: s.food });
  Object.assign(state, { modelOver: false, pendingAction: NOOP, steps: 0, firstDivergence: null, mismatched: 0, stopped: false });
  render(null, 'start state');
}

// purpose: page start-up: load the start state, create the model session, wire the keyboard and button, start the 150 ms timer.
async function main() {
  state.start = await loadJSON(START_URL);
  restart();
  try {
    const note = await createSession();
    state.backend += note;
  } catch (e) {
    el('backend').textContent = 'not playing: ' + errorText(e);
    el('now').textContent = 'No backend passed the self-test (or could load the model), so the page does not play. Open ?selftest=1 for the details.';
    renderSelfTestPanel();
    return;
  }
  render(null, 'start state');
  renderSelfTestPanel();
  window.addEventListener('keydown', onKey);
  installControls({ buttons: document.querySelectorAll('#pad button'), swipeTargets: [el('model'), el('engine')], onAction: setPending });
  el('restart').addEventListener('click', restart);
  setInterval(tick, TICK_MS);
}

main();
