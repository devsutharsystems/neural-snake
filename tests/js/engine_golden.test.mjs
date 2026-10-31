// Replays the 50 Python golden traces (tests/golden/traces.json) on web/engine.js with the food spawns injected from the trace.
// Requires byte-identical state (14-class grid string, direction, food, game_over, won, score, eaten) after EVERY step.
// Run: node --test tests/js/
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { Engine } from '../../web/engine.js';

const HEX = '0123456789abcdef';
const enc = (g) => Array.from(g, (v) => HEX[v]).join('');
const golden = JSON.parse(fs.readFileSync(new URL('../golden/traces.json', import.meta.url)));

test('golden file has 50 traces', () => assert.equal(golden.traces.length, 50));

golden.traces.forEach((t, idx) => {
  test(`trace ${idx} (n=${t.n}, seed=${t.seed}, ${t.policy}, ${t.actions.length} steps)`, () => {
    const e = new Engine({ n: t.n, body: t.initial.body, direction: t.initial.direction, food: t.initial.food });
    assert.equal(enc(e.grid14()), t.initial.grid14, 'initial grid');
    let spawnIdx = 0;
    t.actions.forEach((a, i) => {
      const spawn = (empty) => {
        const p = t.spawns[spawnIdx++];
        assert.ok(p !== undefined && p !== null, `step ${i}: engine asked for a spawn the trace does not have`);
        assert.ok(empty.some(([r, c]) => r === p[0] && c === p[1]), `step ${i}: trace spawn ${p} is not an empty cell in the JS engine`);
        return p;
      };
      const eaten = e.step(a, spawn);
      const s = t.states[i];
      assert.equal(eaten, s.eaten, `step ${i}: eaten`);
      assert.equal(enc(e.grid14()), s.grid14, `step ${i}: grid14`);
      assert.equal(e.direction, s.direction, `step ${i}: direction`);
      assert.deepEqual(e.food, s.food, `step ${i}: food`);
      assert.equal(e.gameOver, s.game_over, `step ${i}: game_over`);
      assert.equal(e.won, s.won, `step ${i}: won`);
      assert.equal(e.score, s.score, `step ${i}: score`);
    });
    // spawns the trace recorded but the engine never asked for would mean the engine under-ate; null spawns (board full) are never requested
    assert.equal(spawnIdx, t.spawns.filter((p) => p !== null).length, 'number of spawns consumed');
  });
});
