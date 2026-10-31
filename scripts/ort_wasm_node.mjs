// Runs web/model/snake.onnx with the VENDORED onnxruntime-web wasm bundle in Node (not a browser): evidence that the 'wasm' EP supports every op.
// usage: node scripts/ort_wasm_node.mjs  -> prints JSON [{action, next_grid, ate, over, food_sum}] for the start state and a few grids
import fs from 'node:fs';
import { pathToFileURL } from 'node:url';
const root = new URL('..', import.meta.url);
const ort = await import(new URL('web/vendor/ort.wasm.min.mjs', root).href);
ort.env.wasm.wasmPaths = new URL('web/vendor/', root).href;
ort.env.wasm.numThreads = 1;
const sess = await ort.InferenceSession.create(fs.readFileSync(new URL('web/model/snake.onnx', root)), { executionProviders: ['wasm'] });
const start = JSON.parse(fs.readFileSync(new URL('web/model/start_state.json', root)));
const grid = Int32Array.from(start.grid.flat());
const outs = [];
for (const action of [0, 1, 2, 3, 4]) {
  const r = await sess.run({ grid: new ort.Tensor('int32', grid, [1, 12, 12]), action: new ort.Tensor('int32', Int32Array.of(action), [1]) });
  outs.push({ action, next_grid: Array.from(r.next_grid.data, Number), ate: r.ate.data[0], over: r.over.data[0],
              food_sum: Array.from(r.food.data).reduce((a, b) => a + b, 0), food: Array.from(r.food.data), types: [r.next_grid.type, r.food.type, r.ate.type, r.over.type] });
}
console.log(JSON.stringify(outs));
