// Comparison page: runs the model (web/model/snake.onnx) and a REAL hand-written engine (web/engine.js) side by side on the same keys and ticks.
// This is the only page allowed to import engine.js; scripts/audit_web.py fails if index.html or main.js reference it.
import { Engine, RIGHT, NOOP } from './engine.js';
import { installControls } from './controls.js'; // on-screen buttons and swipe: they only call setPending(action), exactly like an arrow key

const N = 12;
const TICK_MS = 150;
const KEY_TO_ACTION = { ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3 };
const COLORS = ['#f5f5f5', '#e53935', ...Array(4).fill('#1b5e20'), ...Array(4).fill('#43a047'), ...Array(4).fill('#a5d6a7')];
const VERBOSE = new URLSearchParams(location.search).get('verbose') === '1';
const BASE = new URL('./', import.meta.url);
const VENDOR = new URL('vendor/', BASE).href;
const MODEL_URL = new URL('model/snake.onnx', BASE).href;
const START_URL = new URL('model/start_state.json', BASE).href;

// purpose: shorthand for document.getElementById.
const el = (id) => document.getElementById(id);
const state = { ort: null, session: null, backend: '', start: null, modelGrid: null, engine: null, modelOver: false, busy: false, pendingAction: NOOP,
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

// purpose: same backend choice as the main page: WebGPU first, WASM only if that fails or ?backend=wasm; the label shows what is really running.
async function createSession() {
  const forceWasm = new URLSearchParams(location.search).get('backend') === 'wasm';
  let note = forceWasm ? ' (WASM forced by ?backend=wasm)' : '';
  if (!forceWasm) {
    try { Object.assign(state, await openSession('ort.webgpu.min.mjs', 'webgpu'), { backend: 'webgpu' }); return note; }
    catch (e) { note = ' (WebGPU unavailable: ' + String((e && e.message) || e).slice(0, 100) + ')'; }
  }
  Object.assign(state, await openSession('ort.wasm.min.mjs', 'wasm'), { backend: 'wasm' });
  return note;
}

// purpose: paint a 14-class grid on a canvas; cells listed in `diff` get a red outline.
function drawGrid(canvas, grid, diff) {
  const ctx = canvas.getContext('2d'), s = canvas.width / N;
  for (let i = 0; i < N * N; i++) {
    const x = (i % N) * s, y = Math.floor(i / N) * s;
    ctx.fillStyle = COLORS[grid[i]] || '#ff00ff';
    ctx.fillRect(x, y, s - 1, s - 1);
    if (diff && diff[i]) { ctx.strokeStyle = '#d50000'; ctx.lineWidth = 3; ctx.strokeRect(x + 2, y + 2, s - 5, s - 5); }
  }
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
  drawGrid(el('model'), state.modelGrid, diff);
  drawGrid(el('engine'), state.engine.grid14(), null);
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
    el('backend').textContent = 'no backend could load the model: ' + String((e && e.message) || e);
    return;
  }
  render(null, 'start state');
  window.addEventListener('keydown', onKey);
  installControls({ buttons: document.querySelectorAll('#pad button'), swipeTargets: [el('model'), el('engine')], onAction: setPending });
  el('restart').addEventListener('click', restart);
  setInterval(tick, TICK_MS);
}

main();
