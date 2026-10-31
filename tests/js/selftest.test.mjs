// Runtime self-test (web/selftest.js) against the REAL model on the vendored WASM backend in Node (not a browser):
//  - it passes on the real model and on every recorded case,
//  - it FAILS when one expected cell / decision / probability in selftest.json is altered,
//  - chooseBackend discards a WebGPU session that returns corrupted frames and falls back to WASM, and refuses to play if every backend fails.
// Run: node --test tests/js/selftest.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { runSelfTest, chooseBackend, formatReports } from '../../web/selftest.js';

const root = new URL('../../', import.meta.url);
const ort = await import(new URL('web/vendor/ort.wasm.min.mjs', root).href);
ort.env.wasm.wasmPaths = new URL('web/vendor/', root).href;
ort.env.wasm.numThreads = 1;
const modelBytes = fs.readFileSync(new URL('web/model/snake.onnx', root));
const selftest = JSON.parse(fs.readFileSync(new URL('web/model/selftest.json', root)));
const realSession = () => ort.InferenceSession.create(modelBytes, { executionProviders: ['wasm'] });
const clone = (x) => JSON.parse(JSON.stringify(x));

// a session that returns the real model's outputs with the next grid corrupted (like the wrong frames seen on a real device: extra food cells)
function corrupted(real) {
  return {
    async run(feeds) {
      const out = await real.run(feeds);
      const g = BigInt64Array.from(out.next_grid.data);
      for (const i of [3, 17, 40, 77, 101, 130]) g[i] = 1n;
      return { ...out, next_grid: { data: g } };
    },
    async release() {},
  };
}

test('selftest.json has 8 cases covering start state, long snake, eat step and game-over step', () => {
  assert.equal(selftest.cases.length, 8);
  const names = selftest.cases.map((c) => c.name).join(' ');
  for (const w of ['start-state', 'long-snake', 'eat-step', 'game-over']) assert.ok(names.includes(w), w);
  assert.ok(selftest.cases.some((c) => c.ate) && selftest.cases.some((c) => c.over));
});

test('the self-test passes on the real model (WASM): all 8 cases, 0 differing cells', async () => {
  const s = await realSession();
  const r = await runSelfTest(ort, s, selftest);
  assert.equal(r.ok, true, formatReports([{ backend: 'wasm', ...r }]));
  assert.equal(r.results.length, 8);
  for (const x of r.results) { assert.equal(x.diffCells, 0); assert.ok(x.maxProbDiff < 1e-4, `${x.name} ${x.maxProbDiff}`); }
});

test('altering ONE expected cell in selftest.json makes exactly that case fail', async () => {
  const s = await realSession();
  const bad = clone(selftest);
  bad.cases[3].next_grid[10] = (bad.cases[3].next_grid[10] + 1) % 14;
  const r = await runSelfTest(ort, s, bad);
  assert.equal(r.ok, false);
  assert.deepEqual(r.results.filter((x) => !x.pass).map((x) => x.name), [bad.cases[3].name]);
  assert.equal(r.results[3].diffCells, 1);
});

test('altering an expected ate decision, game-over decision or a probability beyond the tolerance each fails', async () => {
  const s = await realSession();
  for (const mutate of [(b) => { b.cases[4].ate = !b.cases[4].ate; }, (b) => { b.cases[6].over = !b.cases[6].over; },
                        (b) => { b.cases[1].food[5] += 0.2; }, (b) => { b.cases[2].ate_prob += 0.2; }]) {
    const bad = clone(selftest);
    mutate(bad);
    assert.equal((await runSelfTest(ort, s, bad)).ok, false);
  }
});

test('a small probability wobble inside the tolerance still passes (fp differences between backends are not failures)', async () => {
  const s = await realSession();
  const ok = clone(selftest);
  ok.cases[1].food[5] += 0.01;
  assert.equal((await runSelfTest(ort, s, ok)).ok, true);
});

test('a session that throws counts as a failed case', async () => {
  const r = await runSelfTest(ort, { async run() { throw new Error('boom'); } }, selftest);
  assert.equal(r.ok, false);
  assert.ok(r.results.every((x) => !x.pass && x.error === 'boom'));
});

test('chooseBackend keeps WebGPU when it passes', async () => {
  const r = await chooseBackend({ open: async () => ({ ort, session: await realSession() }), selftest });
  assert.equal(r.backend, 'webgpu');
  assert.equal(r.note, '');
  assert.deepEqual(r.reports.map((x) => x.backend), ['webgpu']);
});

test('chooseBackend discards a WebGPU session with corrupted frames and falls back to a self-tested WASM', async () => {
  const opened = [];
  const open = async (name) => { const real = await realSession(); opened.push(name); return { ort, session: name === 'webgpu' ? corrupted(real) : real }; };
  const r = await chooseBackend({ open, selftest });
  assert.equal(r.backend, 'wasm');
  assert.equal(r.note, 'WebGPU failed the self-test on this device');
  assert.deepEqual(opened, ['webgpu', 'wasm']);
  assert.deepEqual(r.reports.map((x) => [x.backend, x.ok]), [['webgpu', false], ['wasm', true]]);
  assert.ok(r.reports[0].results.every((x) => !x.pass || x.diffCells === 0), 'some cases may still pass, but not all');
  assert.ok(r.reports[0].results.some((x) => x.diffCells > 0));
  const text = formatReports(r.reports);
  assert.ok(text.includes('webgpu: FAIL') && text.includes('wasm: PASS') && text.includes('differing cells'));
});

test('chooseBackend falls back to WASM when WebGPU cannot be created', async () => {
  const open = async (name) => { if (name === 'webgpu') throw new Error('no adapter'); return { ort, session: await realSession() }; };
  const r = await chooseBackend({ open, selftest });
  assert.equal(r.backend, 'wasm');
  assert.ok(r.note.startsWith('WebGPU unavailable (no adapter)'));
});

test('?backend=wasm skips WebGPU entirely', async () => {
  const opened = [];
  const open = async (name) => { opened.push(name); return { ort, session: await realSession() }; };
  const r = await chooseBackend({ open, selftest, forceWasm: true });
  assert.deepEqual(opened, ['wasm']);
  assert.equal(r.backend, 'wasm');
  assert.equal(r.note, 'WASM forced by ?backend=wasm');
});

test('if WASM fails the self-test too, chooseBackend throws (the page then does not play) and carries both reports', async () => {
  const open = async () => ({ ort, session: corrupted(await realSession()) });
  await assert.rejects(chooseBackend({ open, selftest }), (e) => {
    assert.match(e.message, /not playing/);
    assert.deepEqual(e.reports.map((x) => [x.backend, x.ok]), [['webgpu', false], ['wasm', false]]);
    return true;
  });
});

test('if WebGPU is unavailable and WASM cannot be created, chooseBackend throws', async () => {
  await assert.rejects(chooseBackend({ open: async () => { throw new Error('nope'); }, selftest }), /no backend could load the model/);
});

test('testBoth (?selftest=1) also self-tests WASM when WebGPU passed, and keeps WebGPU', async () => {
  const open = async () => ({ ort, session: await realSession() });
  const r = await chooseBackend({ open, selftest, testBoth: true });
  assert.equal(r.backend, 'webgpu');
  assert.deepEqual(r.reports.map((x) => [x.backend, x.ok]), [['webgpu', true], ['wasm', true]]);
});
