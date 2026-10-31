# Step 1 fact table (checked 2026-10-29)

Quotes were returned by WebFetch's summariser, not read by me on the raw page; treat as secondary until re-read by Dev.

| # | Claim | Source | Short quote | Status |
|---|---|---|---|---|
| a | WorldTUI can play snake | https://simedw.com/2026/10/21/world-model-terminal/ | "play snake"; "most compute-intensive snake in the world" | verified (via fetch summary) |
| b | ORT-web calls WebGPU experimental | installed `node_modules/onnxruntime-web/README.md` line 66 (v1.31.1); docs page https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html | README: "WebGPU backend is still an experimental feature." Docs page: no "experimental" wording found | PARTIAL: README says experimental, docs page does not; wording does NOT match |
| c | WebGPU support | https://web.dev/blog/webgpu-supported-major-browsers | Chrome/Edge desktop 113; Chrome Android 121 (Android 12+, Qualcomm/ARM); Safari macOS/iOS 26; Firefox Windows 141, Firefox macOS 145 (Tahoe 26, ARM64 only); Firefox Android "Not yet" | verified (web.dev). MDN compat table did not render: UNVERIFIED there |
| d | WorldTUI drift fix DAgger-style | same URL | "a page from DAgger, where a model learns from the states it visits" | verified (via fetch summary); "on the model's own predicted states" is my reading |
| e | WorldTUI details (self-reported) | same URL | date 2026-10-21; "about 360M parameters"; "based on Qwen2.5-Coder-0.5B"; int4 "208 MB"; WGSL compute shaders via WebGPU; "terminal sessions recorded from real programs running inside Docker"; "roughly 75 GPU-hours on an RTX 4090"; "still far from a reliable filesystem" | verified (self-reported) |
| e2 | WorldTUI author and date | https://simedw.com/2026/10/21/world-model-terminal/ (byline on the raw page, read by Dev) | "2026-10-21 / Simon Edwardsson" | verified by Dev from the raw page, 2026-10-31 |
| f1 | onnxruntime-web version / dist-tags | `npm view onnxruntime-web version dist-tags` | version 1.31.1; latest 1.31.1; dev 1.31.0-dev.20260918-bc8e7ed75; extensions 1.9.0-extensions | verified |
| f2 | Licence | `node_modules/onnxruntime-web/package.json` | "license": "MIT"; no LICENSE file ships in the package root | partial: field only, no LICENSE text |
| f3 | WebGPU import path | package.json `exports` | `onnxruntime-web/webgpu` → `dist/ort.webgpu.bundle.min.mjs` (default) or `dist/ort.webgpu.min.mjs` (extern wasm) | verified |
| f4 | Files to serve | package.json `exports` | `ort.webgpu.min.mjs` + `ort-wasm-simd-threaded.jsep.mjs/.wasm` (WebGPU); `ort.wasm.min.mjs` + `ort-wasm-simd-threaded.mjs/.wasm` (WASM) | inferred from exports; confirmed by bench in browser (pending) |
| f5 | env.wasm.wasmPaths | not in ep-webgpu docs page | | UNVERIFIED |
| g | webgpu→wasm silent fallback | `node_modules/onnxruntime-common/dist/esm/backend-impl.js` lines 103-149 | unavailable requested EP is dropped with only `console.warn`; next listed backend used | verified from source: no thrown error, so the page must report the backend itself |
| h | Colab free tier | https://research.google.com/colaboratory/faq.html | GPUs "vary over time"; free GPU "heavily restricted"; max 12 h; limits not published | verified (official FAQ, via summary) |
| i | torch.onnx default | `inspect.signature(torch.onnx.export)` on torch 2.15.0 | `dynamo=True`, `opset_version=None` (resolved at export; to be reported in Spike B) | verified default; opset pending |

## Files each backend loads, onnxruntime-web 1.31.1 (added 2026-10-30)
Evidence is in `node_modules/onnxruntime-web/dist/`. Line numbers are from the readable builds, which have the same flags as the `.min.mjs` files bench.js imports. The minified files were checked with grep: `ort.webgpu.min.mjs` names only `ort-wasm-simd-threaded.asyncify.mjs`, and `ort.wasm.min.mjs` names only `ort-wasm-simd-threaded.mjs`.

| Entry bench.js imports | Loads (.mjs) | Loads (.wasm) | Source |
|---|---|---|---|
| `ort.webgpu.min.mjs` (`['webgpu']`) | `ort-wasm-simd-threaded.asyncify.mjs` | `ort-wasm-simd-threaded.asyncify.wasm` | `ort.webgpu.mjs:1576` (module name), `ort.webgpu.mjs:4299` (wasm URL); the jsep and jspi branches are `false` |
| `ort.wasm.min.mjs` (`['wasm']`) | `ort-wasm-simd-threaded.mjs` | `ort-wasm-simd-threaded.wasm` | `ort.wasm.mjs:1576`, `ort.wasm.mjs:3520`; the other three branches are `false` |

- **jsep files:** neither path loads `ort-wasm-simd-threaded.jsep.*` in 1.31.1. My Phase 0 vendoring was wrong; these were guessed from the package.json `exports` list, not from the bundle code.
- **JSPI:** a separate entry point, `onnxruntime-web/jspi` (`ort.jspi.*`, exports at package.json lines 117-120). It loads `ort-wasm-simd-threaded.jspi.mjs` and `.wasm` (`ort.jspi.mjs:1576`, `:4299`). It is chosen only by importing that entry; `ort.webgpu` never switches to it. The JSPI build throws if `"Suspending" in WebAssembly` is false (`ort.jspi.mjs:1781`).
- **Vendored set:** `web/vendor/` now holds exactly these 6 files, 43,799,252 bytes in total. Before this fix it held a different 6 files (the two jsep files instead of the two asyncify ones), 44,566,278 bytes.
