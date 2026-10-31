import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import audit_web  # noqa: E402

WEB = os.path.join(HERE, "..", "web")


def test_web_dir_passes_audit():
    r = audit_web.audit(WEB)
    assert r["external_urls"] == [] and r["functions_missing_purpose"] == [], r
    assert len(r["functions"]) >= 8


def test_audit_fails_on_external_url_and_missing_purpose(tmp_path):
    (tmp_path / "index.html").write_text('<script src="https://cdn.example.com/x.js"></script>')
    (tmp_path / "main.js").write_text("// purpose: ok\nfunction good() {}\nfunction bad() {}\nconst f = async (x) => x;\n")
    r = audit_web.audit(str(tmp_path))
    assert not r["ok"]
    assert ("index.html", "https://cdn.example.com/x.js") in r["external_urls"]
    assert r["functions_missing_purpose"] == ["bad", "f"]
    assert [n for n, _ in r["functions"]] == ["good"]


def test_audit_allows_urls_only_in_documented_files(tmp_path):
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "LICENSE").write_text("see https://example.org/license")
    (tmp_path / "other.json").write_text('{"u": "https://example.org/license"}')
    r = audit_web.audit(str(tmp_path))
    assert [rel for rel, _ in r["external_urls"]] == ["other.json"]


def test_audit_fails_if_main_page_references_engine(tmp_path):
    for bad_file, text in (("index.html", '<script type="module" src="./engine.js"></script>'),
                           ("main.js", "// purpose: x\nimport { Engine } from './engine.js';\nfunction f() {}\n")):
        d = tmp_path / bad_file.replace(".", "_")
        d.mkdir()
        (d / bad_file).write_text(text)
        r = audit_web.audit(str(d))
        assert bad_file in r["engine_references_in_main_page"] and not r["ok"]


def test_compare_page_may_use_engine(tmp_path):
    (tmp_path / "compare.html").write_text('<script type="module" src="./compare.js"></script>')
    (tmp_path / "compare.js").write_text("import { Engine } from './engine.js';\n// purpose: x\nfunction f() {}\n")
    (tmp_path / "engine.js").write_text("export class Engine {}\n")
    r = audit_web.audit(str(tmp_path))
    assert r["engine_references_in_main_page"] == [] and r["ok"]


def test_compare_functions_need_purpose_lines(tmp_path):
    (tmp_path / "compare.js").write_text("function nope() {}\n")
    assert audit_web.audit(str(tmp_path))["compare_functions_missing_purpose"] == ["nope"]


def test_controls_js_is_inventoried_and_needs_purpose_lines(tmp_path):
    (tmp_path / "controls.js").write_text("// purpose: ok\nexport function good() {}\nexport function bad() {}\n")
    r = audit_web.audit(str(tmp_path))
    assert [n for n, _ in r["controls_functions"]] == ["good"]
    assert r["controls_functions_missing_purpose"] == ["bad"] and not r["ok"]


def test_controls_js_may_not_reference_engine(tmp_path):
    (tmp_path / "controls.js").write_text("// purpose: x\nimport { Engine } from './engine.js';\nexport function f() {}\n")
    r = audit_web.audit(str(tmp_path))
    assert "controls.js" in r["engine_references_in_main_page"] and not r["ok"]


def test_selftest_js_is_inventoried_needs_purpose_lines_and_may_not_reference_engine(tmp_path):
    (tmp_path / "selftest.js").write_text("// purpose: ok\nexport async function good() {}\nexport async function bad() {}\n")
    r = audit_web.audit(str(tmp_path))
    assert [n for n, _ in r["modules"]["selftest.js"]["functions"]] == ["good"]
    assert r["modules"]["selftest.js"]["missing_purpose"] == ["bad"] and not r["ok"]
    (tmp_path / "selftest.js").write_text("// purpose: x\nimport { Engine } from './engine.js';\nexport function f() {}\n")
    r = audit_web.audit(str(tmp_path))
    assert "selftest.js" in r["engine_references_in_main_page"] and not r["ok"]
