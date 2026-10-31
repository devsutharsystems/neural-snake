"""Audit web/: (1) fail on any external URL in any file, (2) print an inventory of every function in web/main.js with its one-line purpose
(the comment `// purpose: ...` directly above it); a function without a purpose line is a failure.
(3) fail if index.html, main.js or controls.js reference engine.js (only compare.html / compare.js may use the hand-written engine);
    web/controls.js (touch / swipe / button input) is inventoried too and each of its functions needs a purpose line.
Run: .venv/bin/python scripts/audit_web.py [web_dir]   (exit code 1 on any violation)

Allowed URLs (documented, never fetched by the page):
 - web/vendor/README.md, LICENSE, ThirdPartyNotices.txt: documentation / license text (any URL)
 - web/vendor/ort.wasm.min.mjs and ort.webgpu.min.mjs: the single string https://web.dev/cross-origin-isolation-guide/ inside a console warning message
 - the two vendored .wasm binaries: 5 documentation URLs compiled in as string constants (WASM_STRINGS below)
"""
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
URL_RE = re.compile(rb"(?:https?|wss?|ftp)://[^\s\"'`<>)\\\]]+")
DOC_FILES = {"vendor/README.md", "vendor/LICENSE", "vendor/ThirdPartyNotices.txt"}
# Compiled onnxruntime wasm binaries carry 5 documentation URLs as string constants (operator docs / error messages); found by this audit, never fetched.
WASM_STRINGS = {b"https://arxiv.org/abs/1502.03167.", b"https://docs.nvidia.com/cuda/cublas/index.html#cublasLtOrder_t",
                b"https://github.com/google/re2/wiki/Syntax", b"https://ieeexplore.ieee.org/document/1163711", b"https://tinyurl.com/sudb9s96"}
ALLOWED = {"vendor/ort.wasm.min.mjs": {b"https://web.dev/cross-origin-isolation-guide/"},
           "vendor/ort.webgpu.min.mjs": {b"https://web.dev/cross-origin-isolation-guide/"},
           "vendor/ort-wasm-simd-threaded.wasm": WASM_STRINGS, "vendor/ort-wasm-simd-threaded.asyncify.wasm": WASM_STRINGS}
# the main game page must contain no game rules, so it (and the input module it shares with compare.html) may not touch web/engine.js
NO_ENGINE_FILES = ("index.html", "main.js", "controls.js", "selftest.js", "render.js")
# modules shared by the game page and compare.html: each is inventoried and every function needs a purpose line
SHARED_MODULES = {"controls.js": "Touch / swipe / button input (web/controls.js, shared with compare.html; it only reports an action index)",
                  "selftest.js": "Runtime self-test and backend choice (web/selftest.js, shared with compare.html; compares the model with its own recorded outputs)",
                  "render.js": "Board drawing (web/render.js, shared with compare.html; paints one frame from the cell classes it is given, nothing carried between frames)"}
ENGINE_RE = re.compile(r"engine\.js|(?:from|import\s*\()\s*['\"][^'\"]*engine")
FUNC_RE = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(|^\s*(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>")


def external_urls(web_dir):
    """-> list of (relative path, url) violations, and the list of allowed hits (for printing)."""
    bad, allowed_hits = [], []
    for dp, _, files in os.walk(web_dir):
        for f in files:
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, web_dir).replace(os.sep, "/")
            data = open(p, "rb").read()
            for u in sorted(set(URL_RE.findall(data))):
                if rel in DOC_FILES or u in ALLOWED.get(rel, set()):
                    allowed_hits.append((rel, u.decode()))
                else:
                    bad.append((rel, u.decode()))
    return bad, allowed_hits


def inventory(js_path):
    """-> (list of (function name, purpose), list of function names missing a purpose line)."""
    lines = open(js_path, encoding="utf-8").read().split("\n")
    funcs, missing = [], []
    for i, line in enumerate(lines):
        m = FUNC_RE.match(line)
        if not m:
            continue
        name = m.group(1) or m.group(2)
        prev = lines[i - 1].strip() if i else ""
        if prev.startswith("// purpose:"):
            funcs.append((name, prev[len("// purpose:"):].strip()))
        else:
            missing.append(name)
    return funcs, missing


def engine_references(web_dir):
    """-> list of files among NO_ENGINE_FILES that reference engine.js (the hand-written engine is for compare.html / compare.js only)."""
    hits = []
    for f in NO_ENGINE_FILES:
        p = os.path.join(web_dir, f)
        if os.path.exists(p) and ENGINE_RE.search(open(p, encoding="utf-8").read()):
            hits.append(f)
    return hits


def audit(web_dir):
    bad, allowed_hits = external_urls(web_dir)
    js = os.path.join(web_dir, "main.js")
    funcs, missing = inventory(js) if os.path.exists(js) else ([], [])
    cmp_js = os.path.join(web_dir, "compare.js")
    cmp_missing = inventory(cmp_js)[1] if os.path.exists(cmp_js) else []
    modules = {}
    for name in SHARED_MODULES:
        p = os.path.join(web_dir, name)
        modules[name] = inventory(p) if os.path.exists(p) else ([], [])
    engine_refs = engine_references(web_dir)
    return {"external_urls": bad, "allowed_url_hits": allowed_hits, "functions": funcs, "functions_missing_purpose": missing,
            "modules": {n: {"functions": f, "missing_purpose": m} for n, (f, m) in modules.items()},
            "controls_functions": modules["controls.js"][0], "controls_functions_missing_purpose": modules["controls.js"][1],
            "compare_functions_missing_purpose": cmp_missing, "engine_references_in_main_page": engine_refs,
            "ok": not bad and not missing and not cmp_missing and not any(m for _, m in modules.values()) and not engine_refs}


def main():
    web_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "web"))
    r = audit(web_dir)
    print(f"audit of {web_dir}")
    print("allowed URL hits (documentation / one warning string, never fetched):")
    for rel, u in r["allowed_url_hits"]:
        print(f"  {rel}: {u}")
    print("EXTERNAL URL VIOLATIONS:" if r["external_urls"] else "no external URL violations")
    for rel, u in r["external_urls"]:
        print(f"  {rel}: {u}")
    print("\nWhat web/main.js still does (function inventory):")
    for name, purpose in r["functions"]:
        print(f"  {name}() - {purpose}")
    for name in r["functions_missing_purpose"]:
        print(f"  MISSING PURPOSE LINE: {name}()")
    for mod, title in SHARED_MODULES.items():
        print(f"\n{title}:")
        for name, purpose in r["modules"][mod]["functions"]:
            print(f"  {name}() - {purpose}")
        for name in r["modules"][mod]["missing_purpose"]:
            print(f"  MISSING PURPOSE LINE in {mod}: {name}()")
    for name in r["compare_functions_missing_purpose"]:
        print(f"  MISSING PURPOSE LINE in compare.js: {name}()")
    print("\nengine.js references in index.html / main.js / controls.js / selftest.js / render.js: " + (", ".join(r["engine_references_in_main_page"]) + "  <- FORBIDDEN" if r["engine_references_in_main_page"] else "none (compare.html / compare.js may use it)"))
    print("\nAUDIT " + ("PASSED" if r["ok"] else "FAILED"))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
