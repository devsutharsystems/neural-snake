# web/vendor: onnxruntime-web 1.31.1 (vendored, no CDN)

Version: **onnxruntime-web 1.31.1** (MIT). Nothing here is modified.

## The 6 runtime files
Copied byte-for-byte from the npm package `onnxruntime-web@1.31.1` (`node_modules/onnxruntime-web/dist/`, pinned in `package.json` / `package-lock.json`;
`cmp` against `dist/` reported all six identical on 2026-10-31):

| File | sha256 |
|---|---|
| ort.webgpu.min.mjs | 329cd956772397a1aedd668e911195d39afefb182f6fc14aafa4f08ac8d8f379 |
| ort.wasm.min.mjs | c21d9c326d453a4a61398270c804d5979e118540597cb0c07f7b3d64e6098234 |
| ort-wasm-simd-threaded.mjs | 30ed0b5cd1182ede931fea448afb567d4dc705ec957af09f108a2be5f770dac3 |
| ort-wasm-simd-threaded.wasm | 4b884c203b0fa48d057be3332c65513ff85aa3c5bddd6946acc0fab7ecda001d |
| ort-wasm-simd-threaded.asyncify.mjs | 59cf781530e1a16df2a291d78798e3ec1152467edaec402d6f9599d85047c302 |
| ort-wasm-simd-threaded.asyncify.wasm | ca319e9539cd0f1e291cf56074760121eaf39650f5dbf84b6e52c07f4b2944ec |

`ort.webgpu.min.mjs` loads the `asyncify` pair and `ort.wasm.min.mjs` loads the plain `simd-threaded` pair (grep of the bundles); the page imports the WebGPU bundle first and the WASM bundle only if WebGPU fails.

## License files (from the official microsoft/onnxruntime repository, tag v1.31.1, fetched 2026-10-31)
- `LICENSE` (MIT License, Copyright (c) Microsoft Corporation): https://raw.githubusercontent.com/microsoft/onnxruntime/v1.31.1/LICENSE
- `ThirdPartyNotices.txt`: https://raw.githubusercontent.com/microsoft/onnxruntime/v1.31.1/ThirdPartyNotices.txt

These URLs are documentation only; the page never loads them. `scripts/audit_web.py` allows URLs in this README, `LICENSE` and `ThirdPartyNotices.txt` and nothing else.
