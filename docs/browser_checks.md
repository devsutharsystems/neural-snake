# Browser checks (manual, by Dev)

Results reported by Dev from running the pages in Chrome. They are recorded as given; the assistant did not run a browser. Chrome version and GPU were not provided.

Model file for these checks: `web/model/snake.onnx` as exported BEFORE the Cast fix (397,175 bytes, final `Cast(int64 -> int32)` node after `ArgMax`). Server: `python3 -m http.server 8000 --bind 127.0.0.1 --directory web`.

## Before the Cast fix

| Date | Page | Backend | What was seen |
|---|---|---|---|
| 2026-10-31 | `index.html?verbose=1` | WebGPU | Placement log: `Node(s) placed on [CPUExecutionProvider]. Number of nodes: 1` and `Node(s) placed on [WebGpuExecutionProvider]. Number of nodes: 75` |
| 2026-10-31 | `compare.html` | webgpu (shown on the page) | One 210-step game that ended in game over on both sides. First divergence: "none yet". Mismatched steps: 0 of 210 |

Notes:
- The single CPU node was suspected to be the `Cast` after `ArgMax` (the WebGPU Cast kernel does not list int64; `onnxruntime/core/providers/js/operators/cast.cc:16-22`). The log did not name the node; the re-check below shows no CPU node after the fix.
- The model was then re-exported without that `Cast` (`next_grid` is now `ArgMax`'s int64 output; the pages convert with `Int32Array.from(out.next_grid.data, Number)`). New file size 396,375 bytes (`web/model/export_info.json`).
- Hard-reload the page (Cmd+Shift+R) so Chrome does not use the cached old `snake.onnx`.

## After Cast fix

Re-check by Dev after the fix, with the new 396,375-byte model. Browser: Chrome 156.0.8078.76, arm64, Apple M2.

| Page | Backend | What was seen |
|---|---|---|
| `index.html?verbose=1` | WebGPU | Placement log: `All nodes placed on [WebGpuExecutionProvider]. Number of nodes: 74` |
| `index.html` (default) | `webgpu` | Median 9.10 ms/step (n=127, all steps since page load); a 108-step game ended in model game over |
| `index.html?backend=wasm` | `wasm (WASM forced by ?backend=wasm)` | Median 3.20 ms/step (n=255); a 255-step game ended in model game over |
| `compare.html` | webgpu | 132 steps, 0 of 132 mismatched, game over on both sides. (Earlier run before the fix: 210 steps, 0 of 210) |

Findings:
- The CPU-placed node is gone: every node (74) now runs on WebGPU, where the old model had 1 CPU node and 75 WebGPU nodes. This is consistent with that node having been the `Cast` after `ArgMax`; the logs did not name nodes, so that attribution is an inference.
- For this tiny model WASM is about 3x faster per step than WebGPU (3.20 ms vs 9.10 ms, ratio 2.8). Both are far under the 150 ms tick.
- Caveats on the timing comparison: the two medians come from different sample sizes (n=127 vs n=255), the WebGPU median includes the first steps after page load (shader compilation), and the WASM session runs single-threaded because the page is not cross-origin isolated. Nothing here says one backend is better for the game; it only says both are fast enough.
