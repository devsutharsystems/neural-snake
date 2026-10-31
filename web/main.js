// Neural Snake main page. This file contains NO game rules: every next frame comes from the model (web/model/snake.onnx).
// Every function carries a "// purpose:" line directly above it; scripts/audit_web.py prints them as the "what the JS still does" list.
const N = 12;
const TICK_MS = 150;
const NOOP = 4;
const VERBOSE = new URLSearchParams(location.search).get('verbose') === '1';
const KEY_TO_ACTION = { ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3 };
const COLORS = ['#f5f5f5', '#e53935', ...Array(4).fill('#1b5e20'), ...Array(4).fill('#43a047'), ...Array(4).fill('#a5d6a7')];
const BASE = new URL('./', import.meta.url);
const VENDOR = new URL('vendor/', BASE).href;
const MODEL_URL = new URL('model/snake.onnx', BASE).href;
const START_URL = new URL('model/start_state.json', BASE).href;

// purpose: shorthand for document.getElementById.
const el = (id) => document.getElementById(id);
const canvas = el('board');
const ctx = canvas.getContext('2d');
const state = { ort: null, session: null, backend: null, note: '', grid: null, start: null, over: false, busy: false, pendingAction: NOOP, steps: 0, times: [] };

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

// purpose: try a WebGPU session from the webgpu bundle first; only if that fails (or ?backend=wasm), import the wasm bundle and use WASM. Records which one is really running.
async function createSession() {
  const forceWasm = new URLSearchParams(location.search).get('backend') === 'wasm';
  let note = forceWasm ? 'WASM forced by ?backend=wasm' : '';
  if (!forceWasm) {
    try {
      Object.assign(state, await openSession('ort.webgpu.min.mjs', 'webgpu'), { backend: 'webgpu' });
      return note;
    } catch (e) {
      note = 'WebGPU unavailable (' + String((e && e.message) || e).slice(0, 140) + '), fell back to WASM';
    }
  }
  Object.assign(state, await openSession('ort.wasm.min.mjs', 'wasm'), { backend: 'wasm' });
  return note;
}

// purpose: paint the current class grid onto the canvas, one coloured square per cell (class to colour lookup only).
function draw() {
  const s = canvas.width / N;
  for (let i = 0; i < N * N; i++) {
    ctx.fillStyle = COLORS[state.grid[i]] || '#ff00ff';
    ctx.fillRect((i % N) * s, Math.floor(i / N) * s, s - 1, s - 1);
  }
}

// purpose: show the backend, median step time, step count, the model's last ate / game-over probabilities and the game-over banner.
function renderStatus(probs) {
  const sorted = [...state.times].sort((a, b) => a - b);
  el('backend').textContent = state.backend + (state.note ? ' (' + state.note + ')' : '');
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

// purpose: map an arrow key to an action index and remember it for the next tick; Space/Enter restarts.
function onKey(ev) {
  if (ev.key in KEY_TO_ACTION) { state.pendingAction = KEY_TO_ACTION[ev.key]; ev.preventDefault(); }
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

// purpose: page start-up: load the start state, create the session (WebGPU first), wire the keyboard / button, start the 150 ms timer.
async function main() {
  state.start = await loadJSON(START_URL);
  restart();
  try {
    state.note = await createSession();
  } catch (e) {
    el('backend').textContent = 'no backend could load the model: ' + String((e && e.message) || e);
    return;
  }
  renderStatus(null);
  window.addEventListener('keydown', onKey);
  el('restart').addEventListener('click', restart);
  setInterval(tick, TICK_MS);
}

main();
