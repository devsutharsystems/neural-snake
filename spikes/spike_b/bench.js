// Spike B benchmark. Closed loop: the model's own output (cls argmax -> one-hot planes, order) is written back
// into the next input. Weights are random, so the numbers measure speed only, not game behaviour.
const VENDOR = new URL('../../web/vendor/', import.meta.url).href;
const MODEL = new URL('./random_net.onnx', import.meta.url).href;
const N = 12, CELLS = N * N, CIN = 14;
const out = document.getElementById('out'), statusEl = document.getElementById('status');
const lines = [];
const log = (s) => { lines.push(s); out.value = lines.join('\n'); };

const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
const p95 = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.ceil(0.95 * s.length) - 1)]; };

function initialInput() {
  const x = new Float32Array(CIN * CELLS);
  for (let i = 0; i < x.length; i++) x[i] = (Math.sin(i * 12.9898) * 43758.5453) % 1; // deterministic noise
  return x;
}
function feedBack(x, res) {
  const cls = res.cls.data, order = res.order.data;
  for (let c = 0; c < CELLS; c++) {
    let best = 0;
    for (let k = 1; k < 4; k++) if (cls[k * CELLS + c] > cls[best * CELLS + c]) best = k;
    for (let k = 0; k < 4; k++) x[k * CELLS + c] = k === best ? 1 : 0;
    x[4 * CELLS + c] = Math.max(0, Math.min(1, order[c]));
  }
}

async function runOne(label, moduleFile, ep) {
  const r = { label, ep, error: null };
  try {
    const ort = await import(new URL(moduleFile, VENDOR).href);
    ort.env.wasm.wasmPaths = VENDOR;
    ort.env.wasm.numThreads = 1; // page is not cross-origin isolated
    const t0 = performance.now();
    const sess = await ort.InferenceSession.create(MODEL, { executionProviders: [ep] });
    r.createMs = performance.now() - t0;
    const x = initialInput();
    const step = async () => {
      const res = await sess.run({ x: new ort.Tensor('float32', x, [1, CIN, N, N]) });
      feedBack(x, res);
    };
    for (let i = 0; i < 50; i++) await step();
    const times = [];
    for (let i = 0; i < 500; i++) { const t = performance.now(); await step(); times.push(performance.now() - t); }
    r.median = median(times); r.p95 = p95(times);
  } catch (e) { r.error = String(e && e.message || e); }
  return r;
}

document.getElementById('run').onclick = async () => {
  lines.length = 0; statusEl.textContent = 'running...';
  log('Spike B bench (headed-browser numbers must come from the person who ran this page)');
  log('userAgent: ' + navigator.userAgent);
  log('navigator.gpu present: ' + ('gpu' in navigator));
  if (navigator.gpu) {
    try {
      const a = await navigator.gpu.requestAdapter();
      if (!a) log('requestAdapter(): null');
      else {
        const info = a.info || (a.requestAdapterInfo && await a.requestAdapterInfo()) || {};
        log('adapter: ' + JSON.stringify({ vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description }));
      }
    } catch (e) { log('adapter error: ' + e); }
  }
  log('crossOriginIsolated: ' + self.crossOriginIsolated + ' (wasm threads: 1)');
  log('');
  for (const [label, file, ep] of [['webgpu-only', 'ort.webgpu.min.mjs', 'webgpu'], ['wasm-only', 'ort.wasm.min.mjs', 'wasm']]) {
    const r = await runOne(label, file, ep);
    log(`[${r.label}] executionProviders=['${r.ep}']`);
    if (r.error) log('  ERROR: ' + r.error);
    else {
      log(`  session create: ${r.createMs.toFixed(1)} ms`);
      log(`  50 warm-up + 500 timed closed-loop steps`);
      log(`  median: ${r.median.toFixed(2)} ms/step   p95: ${r.p95.toFixed(2)} ms/step`);
    }
  }
  statusEl.textContent = 'done';
};
