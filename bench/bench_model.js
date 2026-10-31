// Fair speed benchmark for web/model/snake.onnx. Lives OUTSIDE web/ so it never deploys. Serve from the project root:
//   python3 -m http.server 8001 --bind 127.0.0.1   ->   http://127.0.0.1:8001/bench/bench_model.html
const N = 12, WARMUP = 50, TIMED = 500;
const VENDOR = new URL('../web/vendor/', import.meta.url).href;
const MODEL = new URL('../web/model/snake.onnx', import.meta.url).href;
const START = new URL('../web/model/start_state.json', import.meta.url).href;
const out = document.getElementById('out'), statusEl = document.getElementById('status');
const lines = [];
const log = (s) => { lines.push(s); out.value = lines.join('\n'); };

// purpose: median of an array of numbers.
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
// purpose: 95th percentile of an array of numbers.
const p95 = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.ceil(0.95 * s.length) - 1)]; };

// purpose: sha256 (hex) of the model bytes, so every result is tied to the exact file that was measured.
async function sha256Hex(buf) {
  const d = new Uint8Array(await crypto.subtle.digest('SHA-256', buf));
  return Array.from(d, (b) => b.toString(16).padStart(2, '0')).join('');
}

// purpose: one closed-loop step exactly like the game page: run the model, take its argmax grid, sample a food cell if p(ate) > 0.5, reset to the start state if p(over) > 0.5.
async function oneStep(ort, session, st, action) {
  const res = await session.run({
    grid: new ort.Tensor('int32', st.grid, [1, N, N]),
    action: new ort.Tensor('int32', Int32Array.of(action), [1]),
  });
  st.grid = Int32Array.from(res.next_grid.data, Number);
  if (res.ate.data[0] > 0.5) {
    const p = res.food.data;
    let u = Math.random() * p.reduce((a, b) => a + b, 0), c = p.length - 1;
    for (let i = 0; i < p.length; i++) { u -= p[i]; if (u <= 0) { c = i; break; } }
    st.grid[c] = 1;
  }
  if (res.over.data[0] > 0.5) { st.grid = Int32Array.from(st.start); st.resets++; }
}

// purpose: benchmark one backend: separate session, create time, 50 warm-up steps (excluded), 500 timed steps; returns the numbers or the error.
async function benchOne(label, bundle, provider, modelBytes, start) {
  const r = { label, provider, error: null };
  let session = null;
  try {
    const ort = await import(new URL(bundle, VENDOR).href);
    ort.env.wasm.wasmPaths = VENDOR;
    ort.env.wasm.numThreads = 1; // page is not cross-origin isolated
    const t0 = performance.now();
    session = await ort.InferenceSession.create(modelBytes, { executionProviders: [provider] });
    r.createMs = performance.now() - t0;
    const st = { grid: Int32Array.from(start), start, resets: 0 };
    let seed = 12345;
    const nextAction = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed % 5; }; // deterministic action sequence, NOOP included
    for (let i = 0; i < WARMUP; i++) await oneStep(ort, session, st, nextAction());
    const times = [];
    for (let i = 0; i < TIMED; i++) { const t = performance.now(); await oneStep(ort, session, st, nextAction()); times.push(performance.now() - t); }
    r.median = median(times); r.p95 = p95(times); r.min = Math.min(...times); r.max = Math.max(...times); r.resets = st.resets;
  } catch (e) { r.error = String((e && e.message) || e); }
  try { if (session) await session.release(); } catch (e) { /* ignore */ }
  return r;
}

// purpose: button handler: print environment info, then run the two backends one after the other in the requested order and print a copyable block.
async function runAll() {
  lines.length = 0; statusEl.textContent = 'running...';
  const order = new URLSearchParams(location.search).get('order') === 'wasm-first' ? ['wasm', 'webgpu'] : ['webgpu', 'wasm'];
  const bytes = await (await fetch(MODEL)).arrayBuffer();
  const start = (await (await fetch(START)).json()).grid.flat();
  log('Neural Snake model benchmark (results are only valid if pasted back by the person who ran this page)');
  log('userAgent: ' + navigator.userAgent);
  log('model: web/model/snake.onnx, ' + bytes.byteLength + ' bytes, sha256 ' + await sha256Hex(bytes));
  log('navigator.gpu present: ' + ('gpu' in navigator) + ' | crossOriginIsolated: ' + self.crossOriginIsolated + ' (wasm threads: 1)');
  if (navigator.gpu) {
    try {
      const a = await navigator.gpu.requestAdapter();
      const info = a ? (a.info || {}) : {};
      log('adapter: ' + (a ? JSON.stringify({ vendor: info.vendor, architecture: info.architecture, description: info.description }) : 'null'));
    } catch (e) { log('adapter error: ' + e); }
  }
  log('order: ' + order.join(' then ') + ' | ' + WARMUP + ' warm-up steps excluded, ' + TIMED + ' timed closed-loop steps each | ' + new Date().toISOString());
  log('');
  const cfg = { webgpu: ['webgpu-only', 'ort.webgpu.min.mjs', 'webgpu'], wasm: ['wasm-only', 'ort.wasm.min.mjs', 'wasm'] };
  for (const key of order) {
    statusEl.textContent = 'running ' + key + '...';
    const r = await benchOne(...cfg[key], bytes, start);
    log(`[${r.label}] executionProviders=['${r.provider}']`);
    if (r.error) log('  ERROR: ' + r.error);
    else {
      log(`  session create: ${r.createMs.toFixed(1)} ms`);
      log(`  median: ${r.median.toFixed(2)} ms/step   p95: ${r.p95.toFixed(2)} ms/step   (min ${r.min.toFixed(2)}, max ${r.max.toFixed(2)}, game-over resets ${r.resets})`);
    }
  }
  statusEl.textContent = 'done';
}

document.getElementById('run').addEventListener('click', runAll);
