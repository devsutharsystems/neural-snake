"""Check that every headline number in README.md matches the committed result files.

Sources: eval/results_heldout.json (held-out), eval/dryrun_run1_validation.json (validation side of the side-by-side),
docs/browser_checks.md (fair benchmark table), web/model/export_info.json, spikes/spike_a/results_{v2,v3,run1,run2}.json, diag_run2.json, run1.log.
Two kinds of checks:
  1. TABLES: the markdown tables built here from the JSON/markdown sources must appear in README.md line for line (verbatim).
     `--print-tables` prints them so the README can be pasted from the same code instead of typed by hand.
  2. FACTS: strings for the numbers quoted in prose must appear in README.md; every function in web/main.js must be listed.
Run: python scripts/check_readme_numbers.py   (exit 1 and a list of problems on any mismatch)
"""
import json
import os
import re
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import audit_web  # noqa: E402

HS = ("50", "100", "200", "500", "1000")
POLICIES = (("BFS", "rollout_bfs"), ("random non-suicidal", "rollout_random_safe"), ("combined", "rollout_all_200"))


def load(rel):
    return json.load(open(os.path.join(ROOT, rel)))


def pct(x, d=3):
    return f"{x * 100:.{d}f}%"


def sources():
    res = load("eval/results_heldout.json")
    return {"held": res["headline_run1_heldout"], "spike": res["spike_a_heldout_seen_once_in_phase0_by_throwaway_models"],
            "val": load("eval/dryrun_run1_validation.json"), "browser": open(os.path.join(ROOT, "docs/browser_checks.md"), encoding="utf-8").read()}


def km_table(res):
    lines = ["| policy | rollouts | mismatched | matched to end | alive at cap | S(50) | S(100) | S(200) | S(500) | S(1000) | KM median |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, key in POLICIES:
        s = res[key]
        h, km = s["headline"], s["headline"]["kaplan_meier_P_no_mismatch_yet"]
        obs = km["rollouts_still_observed_at_step"]
        cells = [str(km["S"][x]) if obs[x] > 0 else "n/a (none observed)" for x in HS]
        lines.append(f"| {label} | {s['n']} | {h['mismatched']} | {h['matched_to_end']} | {h['alive_at_cap_no_mismatch']} | " + " | ".join(cells) + f" | {km['median']} |")
    return lines


def observed_table(res):
    lines = ["| policy | observed at step 50 | 100 | 200 | 500 | 1000 |", "|---|---|---|---|---|---|"]
    for label, key in POLICIES:
        obs = res[key]["headline"]["kaplan_meier_P_no_mismatch_yet"]["rollouts_still_observed_at_step"]
        lines.append(f"| {label} | " + " | ".join(str(obs[x]) for x in HS) + " |")
    return lines


def one_step_table(res):
    ex = res["one_step"]["exact"]
    lines = ["| snake length | rows | model, all steps | copy-previous-frame baseline | model, eat steps | model, non-eat steps |", "|---|---|---|---|---|---|"]
    for b in ("1-3", "4-7", "8-15", "16+", "all"):
        r = ex[b]
        lines.append(f"| {b} | {r['all_steps']['n']:,} | {pct(r['all_steps']['model_exact'])} | {pct(r['all_steps']['copy14'], 2)} | "
                     f"{pct(r['eat_steps']['model_exact'])} | {pct(r['non_eat_steps']['model_exact'])} |")
    return lines


def side_by_side(val, held):
    def row(name, f):
        return f"| {name} | {f(val)} | {f(held)} |"
    km = lambda r, h: r["rollout_all_200"]["headline"]["kaplan_meier_P_no_mismatch_yet"]["S"][h]
    ex = lambda r, b, s: r["one_step"]["exact"][b][s]["model_exact"]
    go = lambda r: f"{r['one_step']['game_over']['predicted']:,} of {r['one_step']['game_over']['n_true']:,}"
    fh = lambda r: f"{r['one_step']['food_head']['head_cross_entropy']} vs floor {r['one_step']['food_head']['uniform_floor_mean_ln_empty']}"
    fs = lambda r: f"{r['rollout_all_200']['food_on_snake']['events']} of {r['rollout_all_200']['food_on_snake']['eat_events']:,} ({r['rollout_all_200']['food_on_snake']['pct_of_eat_events']}%)"
    hl = lambda r: r["rollout_all_200"]["headline"]
    return ["| measure | validation (data/run1, seed % 5 == 1) | held-out (data/run1, seed % 5 == 0) |", "|---|---|---|",
            row("rows", lambda r: f"{r['rows']:,}"),
            row("one-step whole-board exact, all steps", lambda r: pct(ex(r, "all", "all_steps"))),
            row("one-step exact, eat steps", lambda r: pct(ex(r, "all", "eat_steps"))),
            row("one-step exact, snake length 16+", lambda r: pct(ex(r, "16+", "all_steps"))),
            row("game-overs found", go),
            row("ate-head accuracy", lambda r: pct(r["one_step"]["ate_head"]["accuracy"])),
            row("food-head cross-entropy", fh),
            row("rollouts mismatched / matched to end / alive at cap (of 200)", lambda r: f"{hl(r)['mismatched']} / {hl(r)['matched_to_end']} / {hl(r)['alive_at_cap_no_mismatch']}"),
            row("Kaplan-Meier S(200)", lambda r: str(km(r, "200"))),
            row("Kaplan-Meier S(1000)", lambda r: str(km(r, "1000"))),
            row("rollouts reaching a dynamics-invalid state (of 200)", lambda r: str(r["rollout_all_200"]["dynamics_invalid"]["rollouts_with_invalid_state"])),
            row("food-on-snake events / eat events", fs)]


def speed_rows(browser_md):
    lines = browser_md.split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("| Run | Order | Backend | Median | p95 | Create |"))
    out = []
    for l in lines[i:]:
        if not l.startswith("|"):
            break
        out.append(l)
    return out


def speed_medians(rows):
    by = {}
    for l in rows[2:]:
        c = [x.strip() for x in l.strip("|").split("|")]
        by.setdefault(c[2], []).append(float(c[3]))
    return {k: statistics.median(v) for k, v in by.items()}


def tables(src):
    return {"heldout_rollouts": km_table(src["held"]), "heldout_observed": observed_table(src["held"]), "heldout_one_step": one_step_table(src["held"]),
            "spike_a_heldout_rollouts": km_table(src["spike"]), "spike_a_heldout_observed": observed_table(src["spike"]),
            "validation_vs_heldout": side_by_side(src["val"], src["held"]), "speed": speed_rows(src["browser"])}


def facts(src):
    h, s = src["held"], src["spike"]
    ex = h["one_step"]["exact"]["all"]["all_steps"]
    hl = lambda r: r["headline"]
    go = h["one_step"]["game_over"]
    sa_ex = s["one_step"]["exact"]["all"]["all_steps"]
    med = speed_medians(speed_rows(src["browser"]))
    info = load("web/model/export_info.json")
    v3 = load("spikes/spike_a/results_v3.json")
    f = [pct(ex["model_exact"]), pct(ex["copy14"], 2), pct(sa_ex["model_exact"]), pct(sa_ex["copy14"], 2),
         f"{h['rows']:,} rows", f"{s['rows']:,} rows",
         f"{v3['params']:,}", f"{info['file_bytes']:,} bytes",
         f"{hl(h['rollout_all_200'])['mismatched']} mismatched", f"{hl(h['rollout_all_200'])['matched_to_end']} matched to end", f"{hl(h['rollout_all_200'])['alive_at_cap_no_mismatch']} alive at the cap",
         f"{hl(h['rollout_random_safe'])['mismatched']} of 100", f"{hl(h['rollout_bfs'])['mismatched']} of 100",
         f"{go['n_true'] - go['predicted']} of {go['n_true']:,}",
         f"{h['rollout_all_200']['dynamics_invalid']['rollouts_with_invalid_state']} of 200",
         f"{h['rollout_all_200']['food_on_snake']['pct_of_eat_events']:.2f}%",
         f"longest snake {max(h['rollout_all_200']['longest_snake_engine'], h['rollout_all_200']['longest_snake_model_predicted'])}",
         f"{med['WebGPU']:.2f} ms", f"{med['WASM']:.2f} ms", f"{med['WebGPU'] / med['WASM']:.0f}x"]
    # training story, from the committed result files
    f.append(pct(load("spikes/spike_a/results_v2.json")["R1v2"]["val"]["exact"]["all"]["model_exact"], 1))
    f.append(pct(v3["val"]["exact"]["all"]["all_steps"]["model_exact"], 1))
    f.append(pct(load("spikes/spike_a/results_run1.json")["val"]["exact"]["all"]["all_steps"]["model_exact"], 1))
    f.append(pct(load("spikes/spike_a/results_run2.json")["val"]["exact"]["all"]["all_steps"]["model_exact"], 2))
    run1_log = open(os.path.join(ROOT, "spikes/spike_a/run1.log"), encoding="utf-8").read()
    m = re.search(r"t=300s .*?'overall': ([0-9.]+)", run1_log)
    f.append(pct(float(m.group(1)), 2))
    f.append(f"{load('spikes/spike_a/diag_run2.json')['food_uniform_floor_mean_ln_empty']:.2f}")
    f.append(f"{round(load('spikes/spike_a/results_run2.json')['seconds']):,} s")
    f.append(f"{len(load('web/model/selftest.json')['cases'])} recorded cases")  # the self-test case count quoted in the README
    p0 = load("spikes/spike_a/results.json")  # Phase 0 (two throwaway models, data/spike_a held-out rows)
    f += [pct(p0["R1"]["one_step"]["all"]["model_grid_exact"], 1), pct(p0["R2"]["one_step"]["all"]["model_grid_exact"], 1),
          pct(p0["R1"]["one_step"]["all"]["model_full_state_exact(grid+order+dir)"], 1)]
    f.append(f"{h['one_step']['food_head']['spawn_on_snake_pct']:.2f}%")  # one-step share of sampled spawns on the snake
    for rel in ("web/vendor/ort-wasm-simd-threaded.asyncify.wasm", "web/vendor/ort-wasm-simd-threaded.wasm"):
        f.append(f"{os.path.getsize(os.path.join(ROOT, rel)):,} bytes")
    f.append(f"{sum(os.path.getsize(os.path.join(ROOT, 'web/vendor', n)) for n in os.listdir(os.path.join(ROOT, 'web/vendor')) if n.endswith(('.mjs', '.wasm'))):,} bytes")
    return f


# numbers quoted from docs/browser_checks.md (manual Chrome results): the string must be in BOTH that file and the README
BROWSER_FACTS = ("9.10 ms", "3.20 ms", "Number of nodes: 74", "Chrome 156.0.8078.76", "0 of 132",
                 "Samsung S20+ 5G", "33.7 and 38.5 ms/step", "4.40 ms/step", "81-step game", "6 food cells",
                 "4.14e-3", "1.37e-2", "4.97e-7", "5.25e-7", "174 and 110 steps", "5.60 ms/step", "5.40 ms/step", "Vulkan: Disabled")


def check(readme, src=None):
    src = src or sources()
    problems = []
    text_lines = set(readme.split("\n"))
    for name, lines in tables(src).items():
        for l in lines:
            if l not in text_lines:
                problems.append(f"table '{name}': README lacks the line: {l}")
    for s in facts(src):
        if s not in readme:
            problems.append(f"fact: README lacks '{s}'")
    for s in BROWSER_FACTS:
        if s not in src["browser"]:
            problems.append(f"docs/browser_checks.md lacks '{s}'")
        if s not in readme:
            problems.append(f"browser fact: README lacks '{s}'")
    for js in ("main.js", "controls.js", "selftest.js", "render.js"):  # every function of the game page's JS, including the shared input and self-test modules, must be listed
        funcs, _ = audit_web.inventory(os.path.join(ROOT, "web", js))
        for name, _ in funcs:
            if f"{name}()" not in readme:
                problems.append(f"function inventory ({js}): README lacks {name}()")
    return problems


def main():
    src = sources()
    if "--print-tables" in sys.argv:
        for name, lines in tables(src).items():
            print(f"<!-- {name} -->")
            print("\n".join(lines))
            print()
        return 0
    problems = check(open(os.path.join(ROOT, "README.md"), encoding="utf-8").read(), src)
    for p in problems:
        print("PROBLEM:", p)
    print("README numbers: " + ("ALL MATCH" if not problems else f"{len(problems)} PROBLEM(S)"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
