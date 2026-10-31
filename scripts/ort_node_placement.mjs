// Shows ORT's node-placement log (session_state.cc "Node placements", logged only when the session logger is VERBOSE) for the vendored WASM backend in Node.
// This is NOT WebGPU and NOT a browser. usage: node scripts/ort_node_placement.mjs
import fs from 'node:fs';
const root = new URL('..', import.meta.url);
const ort = await import(new URL('web/vendor/ort.wasm.min.mjs', root).href);
ort.env.wasm.wasmPaths = new URL('web/vendor/', root).href;
ort.env.wasm.numThreads = 1;
await ort.InferenceSession.create(fs.readFileSync(new URL('web/model/snake.onnx', root)), { executionProviders: ['wasm'], logSeverityLevel: 0 });
