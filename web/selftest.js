// Runtime self-test, shared by index.html and compare.html (so it must never touch the hand-written engine module).
// After a backend creates its inference session, the page runs 8 recorded golden cases (web/model/selftest.json: the model's OWN outputs for fixed
// inputs, recorded with onnxruntime-python). A backend whose outputs differ is discarded and the page falls back to WASM.
// This compares the model with its own recorded outputs. It contains no game rules and never looks at what the snake or the board means.
export const DEFAULT_PROB_TOLERANCE = 0.05;

// purpose: run one golden case on a session and compare with the recorded outputs: differing grid cells, ate / game-over decisions (> 0.5) and the largest probability difference.
export async function runCase(ort, session, c, tolerance = DEFAULT_PROB_TOLERANCE) {
  const out = await session.run({
    grid: new ort.Tensor('int32', Int32Array.from(c.grid), [1, 12, 12]),
    action: new ort.Tensor('int32', Int32Array.of(c.action), [1]),
  });
  const got = Array.from(out.next_grid.data, Number);
  let diffCells = Math.abs(got.length - c.next_grid.length);
  for (let i = 0; i < c.next_grid.length; i++) if (got[i] !== c.next_grid[i]) diffCells++;
  const ate = out.ate.data[0], over = out.over.data[0], food = out.food.data;
  let maxProbDiff = Math.max(Math.abs(ate - c.ate_prob), Math.abs(over - c.over_prob));
  if (food.length !== c.food.length) maxProbDiff = Infinity;
  else for (let i = 0; i < c.food.length; i++) maxProbDiff = Math.max(maxProbDiff, Math.abs(food[i] - c.food[i]));
  const ateOk = (ate > 0.5) === c.ate, overOk = (over > 0.5) === c.over;
  return { name: c.name, pass: diffCells === 0 && ateOk && overOk && maxProbDiff <= tolerance, diffCells, ateOk, overOk, maxProbDiff };
}

// purpose: run every golden case on a session; a case that throws counts as failed. Returns { ok, results }.
export async function runSelfTest(ort, session, selftest) {
  const tolerance = selftest.prob_tolerance ?? DEFAULT_PROB_TOLERANCE;
  const results = [];
  for (const c of selftest.cases) {
    try { results.push(await runCase(ort, session, c, tolerance)); }
    catch (e) { results.push({ name: c.name, pass: false, error: errorText(e) }); }
  }
  return { ok: results.length > 0 && results.length === selftest.cases.length && results.every((r) => r.pass), results };
}

// purpose: short text for an error value (message only, cut to 140 characters).
export function errorText(e) {
  return String((e && e.message) || e).slice(0, 140);
}

// purpose: release an inference session we are throwing away; never fails.
async function discard(session) {
  try { await session.release(); } catch (e) { /* nothing to do */ }
}

// purpose: pick the backend. open(name) creates a session for 'webgpu' or 'wasm'. WebGPU is tried first (unless forced to WASM) and kept only if it passes the self-test; otherwise it is discarded and WASM is used, self-tested too. If no backend passes, throw (the page must not play). testBoth also self-tests WASM when WebGPU passed (for the ?selftest=1 report).
export async function chooseBackend({ open, selftest, forceWasm = false, testBoth = false }) {
  const reports = [];
  let note = forceWasm ? 'WASM forced by ?backend=wasm' : '';
  let chosen = null;
  if (!forceWasm) {
    try {
      const h = await open('webgpu');
      const report = await runSelfTest(h.ort, h.session, selftest);
      reports.push({ backend: 'webgpu', ...report });
      if (report.ok) chosen = { backend: 'webgpu', ...h };
      else { note = 'WebGPU failed the self-test on this device'; await discard(h.session); }
    } catch (e) {
      note = 'WebGPU unavailable (' + errorText(e) + '), fell back to WASM';
      reports.push({ backend: 'webgpu', ok: false, error: errorText(e), results: [] });
    }
  }
  if (chosen === null || testBoth) {
    let w = null;
    try { w = await open('wasm'); }
    catch (e) {
      reports.push({ backend: 'wasm', ok: false, error: errorText(e), results: [] });
      if (chosen === null) throw Object.assign(new Error('no backend could load the model: ' + errorText(e)), { reports });
    }
    if (w) {
      const report = await runSelfTest(w.ort, w.session, selftest);
      reports.push({ backend: 'wasm', ...report });
      if (chosen === null && report.ok) chosen = { backend: 'wasm', ...w };
      else await discard(w.session);
      if (chosen === null) throw Object.assign(new Error('the self-test failed on WebGPU and on WASM: not playing'), { reports });
    }
  }
  return { ...chosen, note, reports };
}

// purpose: turn a list of self-test reports (one per backend tried) into the text shown by ?selftest=1: per case pass / fail, differing cells, decisions, max probability difference.
export function formatReports(reports, header = '') {
  const lines = header ? [header] : [];
  for (const r of reports) {
    const passed = r.results.filter((x) => x.pass).length;
    const verdict = r.results.length === 0 && r.error ? 'UNAVAILABLE (session could not be created)' : `${r.ok ? 'PASS' : 'FAIL'} (${passed}/${r.results.length} cases)`;
    lines.push(`${r.backend}: ${verdict}` + (r.error ? ` error: ${r.error}` : ''));
    for (const x of r.results) {
      lines.push(x.error ? `  ${x.name}: FAIL error: ${x.error}`
        : `  ${x.name}: ${x.pass ? 'pass' : 'FAIL'} | differing cells ${x.diffCells} | ate ${x.ateOk ? 'ok' : 'WRONG'}, game over ${x.overOk ? 'ok' : 'WRONG'} | max prob diff ${x.maxProbDiff.toExponential(2)}`);
    }
  }
  return lines.join('\n');
}
