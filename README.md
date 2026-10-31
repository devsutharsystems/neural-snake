# Neural Snake

A Snake game in the browser where **no game rules run**: a 78,177-parameter neural network, trained from scratch on this one game, predicts every next frame from the current frame and your key press.

**Live:** https://devsutharsystems.github.io/neural-snake/ (deployed 2026-10-31). Comparison page with a real engine beside the model: `compare.html`.

## What it is

Each tick (150 ms) the page sends two things to the model: the current 12x12 board, written as a grid of class numbers (empty, food, snake head/body/tail with a direction), and the key you pressed (or "no key"). The model, a small convolutional network exported to ONNX (`web/model/snake.onnx`, 396,375 bytes) and run in the browser with onnxruntime-web, answers with:

- the next board (the most likely class for each cell),
- a probability that the snake just ate food, and a probability for each of the 144 cells that food appears there,
- a probability that the game is over.

The page draws the board it gets back and feeds it into the next tick. Walls, self-collision, growth and "did it eat" are all things the network has learned; none of it is written as code in the page. A separate page, `compare.html`, runs a hand-written Snake engine beside the model, clearly labelled as comparison only.

## What the JavaScript still does

`web/main.js` is the whole game page, plus three small modules that the compare page shares: `web/controls.js` (on-screen buttons and swipe, for phones that have no arrow keys), `web/selftest.js` (a runtime self-test that checks the model's outputs on this device against outputs recorded from the same model, and picks the backend) and `web/render.js` (drawing the board). The look of the pages is plain CSS in `web/style.css` (no images, no web fonts, no libraries). The lists below are printed by `scripts/audit_web.py`, which also fails if a function lacks a purpose line, if any file in `web/` references an external URL, or if `index.html`, `main.js`, `controls.js`, `selftest.js` or `render.js` reference the hand-written engine (`web/engine.js`, used only by `compare.html`).

- `el()`: shorthand for `document.getElementById`.
- `loadJSON()`: fetch a JSON file that ships with the page (the engine-generated start state).
- `openSession()`: load one vendored onnxruntime-web bundle and create an inference session with exactly one execution provider.
- `openBackend()`: open a session for one named backend (the WebGPU bundle or the WASM bundle).
- `createSession()`: load the self-test cases and choose the backend: WebGPU first and only if it passes the self-test, otherwise WASM (also self-tested; if that fails too the page does not play). It records which backend is really running and why, and the page shows it.
- `renderSelfTestPanel()`: with `?selftest=1`, show the result of every self-test case on every backend tried (pass or fail, differing cells, largest probability difference).
- `draw()`: paint the current class grid (the model's latest frame) with the shared renderer: tiles, the snake from its direction classes, and one orb per food cell. Nothing is carried over from earlier frames.
- `renderStatus()`: show the backend, median step time, step count, the model's last ate / game-over probabilities and the game-over banner.
- `sampleCell()`: draw one cell from the model's food probabilities with `Math.random`. Nothing is checked or resampled.
- `step()`: one model call: send (grid, action), take the model's argmax grid as the next state, write a sampled food cell if p(ate) > 0.5, flag game over if p(game over) > 0.5.
- `tick()`: the 150 ms timer callback: use the pending key (or no key), run one model step, redraw.
- `setPending()`: remember an action (0 up, 1 down, 2 left, 3 right) for the next tick. Arrow keys, the on-screen buttons and swipes all end up here.
- `onKey()`: map an arrow key to an action and remember it for the next tick; Space/Enter restarts.
- `restart()`: reload the start state from its data file and clear the game-over flag.
- `main()`: start-up: load the start state, create the session, wire the keyboard, the touch controls and the restart button, start the timer.

Touch and swipe input (`web/controls.js`). It only reports an action number to the callback it is given, the same value an arrow key sets; it never looks at the board, the snake or the model's output:

- `swipeToAction()`: turn a swipe vector in screen pixels into an action (up, down, left, right), or nothing if the swipe is shorter than 24 px.
- `handlePress()`: button press handler: report the pressed button's action.
- `wireButtons()`: wire the four on-screen arrow buttons (pointer press, or Enter / Space on a focused button).
- `wireSwipe()`: wire a swipe on one element: remember where the pointer went down and report the direction on release.
- `installControls()`: install the buttons and the swipe gestures; every input ends in the same callback.

Runtime self-test (`web/selftest.js`, cases in `web/model/selftest.json`). It runs 8 fixed inputs through each backend and compares the outputs with the outputs this same model produced in Python (the next grid, the ate and game-over decisions, and the probabilities within 0.05). It contains no game rules and never interprets the board:

- `runCase()`: run one case on a session and compare with the recorded outputs: differing grid cells, ate / game-over decisions and the largest probability difference.
- `runSelfTest()`: run every case on a session; a case that throws counts as failed.
- `errorText()`: short text for an error value.
- `discard()`: release a session we are throwing away.
- `chooseBackend()`: WebGPU first and kept only if it passes; otherwise discard it and use WASM, also self-tested; if no backend passes, throw so that the page does not play.
- `formatReports()`: turn the reports into the text shown by `?selftest=1`.

Drawing (`web/render.js`). **Honesty rule:** a frame is painted only from the cell classes of the frame it is given, including the model's own direction classes for head, body and tail. There is no interpolation between frames, no state kept from one frame to the next, and no inferring of movement; a wrong cell from the model is drawn as it is. The head's eyes face the head-direction class, the body joins toward the neighbour its direction class points at, and the tail tapers away from its class's neighbour. `tests/js/render.test.mjs` checks these points with a recording stand-in for the canvas. The food orb's gentle pulse is a CSS animation of decoration, switched off under `prefers-reduced-motion`; the orb's position is set directly, never animated.

- `fitCanvas()`: make the canvas backing store 480 px, or 960 px on high-density screens, so the board is sharp.
- `makeLayer()`: create an offscreen drawing surface.
- `rrect()`: add a rounded rectangle to the current path.
- `tileLayer()`: the layer of 144 empty tiles, built once per size; static decoration that does not depend on any frame.
- `chainDepth()`: from this frame's direction classes, how far each snake cell is from the head; used only to shade the body darker toward the tail.
- `bodyColor()`: the colour of a body segment (glossy green, slightly darker toward the tail).
- `pillPath()`: the rounded bar from a body cell to the neighbour its direction class points at.
- `tailTaper()`: the tapered tail end.
- `headPath()`: the head, a larger rounder block stretched along its direction class.
- `drawEyes()`: two eyes facing the head-direction class.
- `drawSnake()`: paint the snake of this frame (shadow, colour, gloss, then head and eyes).
- `shape()`: add the path of one snake part to the context.
- `drawDiff()`: red outlines for cells flagged as different (compare page only).
- `drawBoard()`: paint one frame: the static tiles, the snake and the optional outlines.
- `placeOrbs()`: show one food orb per food cell of this frame, reusing elements so the CSS pulse keeps running.

**Borderline items, stated plainly.** Some people would count these as game logic:

- **The 150 ms tick timer with "no key" each tick.** It is input plumbing, but it is also, in effect, a game loop.
- **The fixed start state** is loaded from `web/model/start_state.json`, which was generated by the Python engine (`scripts/make_start_state.py`, seed 3). The first frame is therefore not produced by the model.
- **Food is sampled in JavaScript**, with `Math.random`, from the probabilities the model returns. The page never checks that the cell is empty and never resamples, so food can land on the snake and the page shows it as produced.
- **The 0.5 thresholds** on the ate and game-over probabilities are applied in JavaScript.
- **The runtime self-test** compares the model with its own recorded outputs and picks the backend; the page shows which backend runs and why. It decides which runtime to trust, not what the game does.
- **Touch input.** The four on-screen buttons and the swipe gesture only set the same pending action as an arrow key. Swiping on the board is captured by the page (the board does not scroll), so the page scrolls only when you touch outside the board.
- **After the model says game over, the page stops stepping** until you restart. That is page state, not a rule, but it is JavaScript making a decision.

## Why now

[WorldTUI](https://simedw.com/2026/10/21/world-model-terminal/) (Simon Edwardsson, 21 Oct 2026) is a model of about 360M parameters (self-reported) fine-tuned to imitate a terminal inside the browser, and it can also play snake. This project asks a narrower question: how small can a model be for a game with exact rules, if it is trained from scratch on that one game? The shipped model has 78,177 parameters. The comparison is **not like-for-like**: WorldTUI imitates a general terminal and this model learns one fixed game, and I measure this model against an exact engine, which has no equivalent for a terminal.

## Results

Everything here comes from `eval/results_heldout.json` (and `eval/results_heldout.md`) unless another file is named. The held-out evaluation was run **once**, in Python with onnxruntime on the shipped `snake.onnx` (the script refuses a second run: `eval/heldout.lock`). The model file's SHA-256 and the script's are recorded in the output.

### Headline: `data/run1` held-out (133,848 rows from 1,901 episodes, never seen by any model or script before the run)

Whole-board exact match for one step (every cell, 14 classes) against a "copy the previous frame" baseline, by snake length:

| snake length | rows | model, all steps | copy-previous-frame baseline | model, eat steps | model, non-eat steps |
|---|---|---|---|---|---|
| 1-3 | 54,530 | 99.996% | 1.30% | 100.000% | 99.996% |
| 4-7 | 25,918 | 100.000% | 0.62% | 100.000% | 100.000% |
| 8-15 | 23,930 | 99.954% | 0.60% | 99.959% | 99.953% |
| 16+ | 29,470 | 99.983% | 0.69% | 99.964% | 99.985% |
| all | 133,848 | 99.987% | 0.91% | 99.981% | 99.987% |

Overall the model gets 99.987% of single steps exactly right, against 0.91% for copying the previous frame.

**200 rollouts** (100 BFS-bot, 100 random-but-non-suicidal), each up to 1000 steps, with the engine run beside the model and food synced to the engine after each eat. A rollout "mismatches" when the model's board first differs from the engine's, or its game-over flag does. Overall: 19 mismatched, 170 matched to end (the engine's game ended with no mismatch) and 11 alive at the cap of 1000 steps with no mismatch.

Kaplan-Meier estimate of P(no mismatch yet): a rollout that ends in game over without a mismatch, or reaches the cap, is censored at that step. The median is "not reached" because the curve never drops below 0.5. I report no median step.

| policy | rollouts | mismatched | matched to end | alive at cap | S(50) | S(100) | S(200) | S(500) | S(1000) | KM median |
|---|---|---|---|---|---|---|---|---|---|---|
| BFS | 100 | 4 | 96 | 0 | 0.9791 | 0.9791 | 0.9791 | n/a (none observed) | n/a (none observed) | not reached |
| random non-suicidal | 100 | 15 | 74 | 11 | 0.9508 | 0.9048 | 0.8432 | 0.7955 | 0.6641 | not reached |
| combined | 200 | 19 | 170 | 11 | 0.9659 | 0.9467 | 0.9222 | 0.8714 | 0.7274 | not reached |

How many rollouts are still being observed at each horizon (the later numbers rest on few rollouts; no BFS rollout lasts to step 500):

| policy | observed at step 50 | 100 | 200 | 500 | 1000 |
|---|---|---|---|---|---|
| BFS | 92 | 85 | 65 | 0 | 0 |
| random non-suicidal | 66 | 52 | 38 | 27 | 11 |
| combined | 158 | 137 | 103 | 27 | 11 |

### Separately: `data/spike_a` held-out (9,569 rows) — seen once in Phase 0 by throwaway models

These rows were evaluated once in Phase 0 by throwaway models, so they are not the headline. One step: 99.979% exact (copy baseline 0.99%). Rollouts:

| policy | rollouts | mismatched | matched to end | alive at cap | S(50) | S(100) | S(200) | S(500) | S(1000) | KM median |
|---|---|---|---|---|---|---|---|---|---|---|
| BFS | 100 | 5 | 95 | 0 | 1.0 | 1.0 | 1.0 | n/a (none observed) | n/a (none observed) | not reached |
| random non-suicidal | 100 | 7 | 76 | 17 | 0.966 | 0.966 | 0.9484 | 0.878 | 0.878 | not reached |
| combined | 200 | 12 | 171 | 17 | 0.9838 | 0.9838 | 0.977 | 0.8568 | 0.8568 | not reached |

| policy | observed at step 50 | 100 | 200 | 500 | 1000 |
|---|---|---|---|---|---|
| BFS | 96 | 91 | 64 | 0 | 0 |
| random non-suicidal | 65 | 59 | 48 | 32 | 17 |
| combined | 161 | 150 | 112 | 32 | 17 |

### Validation vs held-out (same dataset, `data/run1`)

The validation column is the script's dry run on the validation rows (`eval/dryrun_run1_validation.json`); 5,000 of those rows were used to pick the final checkpoint, so it is not a clean test.

| measure | validation (data/run1, seed % 5 == 1) | held-out (data/run1, seed % 5 == 0) |
|---|---|---|
| rows | 135,292 | 133,848 |
| one-step whole-board exact, all steps | 99.987% | 99.987% |
| one-step exact, eat steps | 99.991% | 99.981% |
| one-step exact, snake length 16+ | 99.977% | 99.983% |
| game-overs found | 1,897 of 1,897 | 1,892 of 1,896 |
| ate-head accuracy | 100.000% | 100.000% |
| food-head cross-entropy | 4.8919 vs floor 4.8793 | 4.8909 vs floor 4.8803 |
| rollouts mismatched / matched to end / alive at cap (of 200) | 13 / 172 / 15 | 19 / 170 / 11 |
| Kaplan-Meier S(200) | 0.9367 | 0.9222 |
| Kaplan-Meier S(1000) | 0.8978 | 0.7274 |
| rollouts reaching a dynamics-invalid state (of 200) | 7 | 13 |
| food-on-snake events / eat events | 40 of 2,910 (1.375%) | 23 of 2,741 (0.839%) |

One-step accuracy holds up on held-out data. Rollouts are somewhat worse on held-out data than on validation, but with 200 rollouts per set and only 11 to 15 observed late in the run, I cannot say whether that gap is real or sampling noise.

## Honest limits and failure modes

- **Where it goes wrong.** In the three failure grids saved in `eval/results_heldout.md` (first mismatch of three rollouts), the wrong cell is a tail or a direction arrow: a body arrow pointing the wrong way at snake length 47, a tail one cell off at length 5, and a missing tail cell at length 21. I did not classify the other 16 mismatched rollouts. Tail cells on eat steps are the weakest class in the one-step results (0.009% wrong).
- **Random-style play diverges more**: 15 of 100 random-policy rollouts mismatched against 4 of 100 BFS rollouts.
- **Game over is missed occasionally**: 4 of 1,896 game-overs were not predicted (precision 1.0, recall 0.998).
- **Invalid states**: 13 of 200 rollouts reached a dynamics-invalid state (7 with not exactly one head, 6 with the wrong length, 0 disconnected).
- **Food on the snake**: in rollouts, 0.84% of eat events sampled a food cell that lies on the model's own snake (23 of 2,741). One step at a time it is 0.58% of sampled spawns. The page shows this as produced; it is never corrected.
- **No long snakes**: the training data has no snakes of length 64 or more (the 64+ bucket has 0 rows; `spikes/spike_a/gen_run1_stats.json`), and none appear in testing either (longest snake 57 in the held-out rollouts). Behaviour on longer snakes is unknown.
- **Design choices reused validation rows.** The Spike A validation rows guided several design decisions, and 5,000 `data/run1` validation rows chose the final checkpoint. The held-out numbers are the unbiased ones.
- **Evaluation setup**: the held-out numbers come from Python (onnxruntime, CPU) on the shipped ONNX file. Browser parity was proven separately: 0 mismatches on 1,000 validation rows and on 20 closed-loop rollouts of 200 steps (`web/model/parity_results.json`), and one full game in Chrome on the compare page matched the engine on every step (manual check, `docs/browser_checks.md`).
- **Small samples and one browser.** 200 rollouts per set is small, and the late-horizon survival numbers rest on few rollouts. I only tested in Chrome, on one machine. The touch controls are tested with stub events in Node (`tests/js/controls.test.mjs`) and the 360 px layout was checked in desktop headless Chrome. I did not test on a phone myself; the phone results come from Dev (next bullet).
- **Real-device finding (Samsung S20+ 5G, Android; reported by Dev, `docs/browser_checks.md`).** On this phone WebGPU produced numerically wrong outputs. With the original page (backend "webgpu") the frames were wrong within 6 steps in Chrome (6 food cells, the head separated from the body, model game over at step 6; 33.7 and 38.5 ms/step) and the identical wrong frames appeared in Brave, while Brave with `?backend=wasm` played correctly (a normal snake, one food, an 81-step game, 4.40 ms/step). The same `snake.onnx` is exact on desktop WebGPU and WASM and in Python. The page now runs 8 recorded cases after creating each session, discards a backend that fails them and falls back to WASM. **Confirmed on the phone (Chrome, `?selftest=1`):** WebGPU failed all 8 cases, with 1 to 15 differing cells per case and probability errors from 4.14e-3 to 1.37e-2 (enough to flip cells; the ate and game-over decisions still matched); WASM passed 8 of 8 and matches the recorded outputs to about 5e-7 (4.97e-7 to 5.25e-7). The default page then showed "wasm (WebGPU failed the self-test on this device)" and played normal games of 174 and 110 steps at 5.60 ms/step and 5.40 ms/step. The cause is not determined: `chrome://gpu` on this phone reports "WebGPU: Hardware accelerated" and "Vulkan: Disabled"; whether that is related is unconfirmed. I name no GPU. The self-test is what protects players: other devices with a faulty WebGPU path may fail it differently, and a device that returns correct grids and decisions but probabilities off by more than 0.05 would also be sent to WASM.

## How it was built

Whole-board exact match for one step, on validation rows unless stated:

| stage | result | what happened | source |
|---|---|---|---|
| Phase 0 | R1 43.8%, R2 44.4% (R1 with body order and direction also right: 1.8%) | first spike, two input encodings trained 300 s each; R1 gave the body order as a single number plane. Measured on the `data/spike_a` held-out rows, which these throwaway models saw once. 20-step rollouts diverged after 1 to 2 steps | `spikes/spike_a/results.json`, `results.md` |
| Debug 1 | 50.5% | replaced the scalar body-order input with a 14-class one-hot (head, body and tail with direction) | `spikes/spike_a/results_v2.json` |
| Debug 2 | 81.1% | added the "ate" head and over-sampled eat and game-over steps | `spikes/spike_a/results_v3.json` |
| Run 1 | peak 97.25% at 5 min, then 75.8% at 60 min | the food head memorised random food spawns: its training loss fell below the 4.87 floor (the loss of guessing uniformly among empty cells) while validation loss rose, and its gradient damaged the shared layers | `spikes/spike_a/run1.log`, `results_run1.json`, `diag_run1.json` |
| Run 2 | 99.98% | stopped the food head's gradient from reaching the shared layers, cosine learning-rate decay, best-checkpoint selection | `spikes/spike_a/results_run2.json`, `diag_run2.json` |

The final model (`checkpoints/run2_best.pt`, git-ignored) is Run 2's best checkpoint, exported to ONNX by `scripts/export_onnx.py`.

## Hardware and training

PyTorch on the Apple GPU backend (MPS) of an Apple M2 Mac with 8 GB of memory (`sysctl -n machdep.cpu.brand_string` prints `Apple M2`; the browser checks also ran on an Apple M2). **Google Colab was not used.** Training wall times: Phase 0 two models of 300 s each (`spikes/spike_a/results.json`), Debug 1 two models of 300 s each (`results_v2.json`), Debug 2 600 s (`results_v3.json`), Run 1 3600 s (`results_run1.json`), and the shipped model's Run 2 stopped after 1,503 s (`results_run2.json`).

## Speed

Fair benchmark (`bench/bench_model.html`): two separate sessions per run, 50 warm-up steps excluded, 500 timed closed-loop steps, in Chrome 156.0.8078.76 on an Apple M2, three runs with the order alternated (reported by hand, recorded in `docs/browser_checks.md`). Times are ms per step; "Create" is session-create time in ms.

| Run | Order | Backend | Median | p95 | Create |
|---|---|---|---|---|---|
| 1 | webgpu first | WebGPU | 4.10 | 4.20 | 424.1 |
| 1 | webgpu first | WASM | 0.80 | 0.90 | 141.6 |
| 2 | wasm first | WASM | 0.70 | 0.80 | 62.5 |
| 2 | wasm first | WebGPU | 3.90 | 4.10 | 94.2 |
| 3 | webgpu first | WebGPU | 4.00 | 4.20 | 119.9 |
| 3 | webgpu first | WASM | 0.80 | 0.80 | 54.5 |

Median of the three runs: WebGPU 4.00 ms per step and WASM 0.80 ms per step, so WASM is about 5x faster for this tiny model. Both are far under the 150 ms tick. Caveats: the timer resolution is about 0.1 ms (the page is not cross-origin isolated), so read WASM as "under 1 ms"; WASM runs on one thread; memory pressure was not recorded.

The page's own in-page median, which includes the first steps after page load, was 9.10 ms with WebGPU and 3.20 ms with WASM (`docs/browser_checks.md`). In Chrome's console (`?verbose=1`) ONNX Runtime reported `All nodes placed on [WebGpuExecutionProvider]. Number of nodes: 74`, so no node falls back to the CPU.

Download size: the vendored runtime is about 44 MB (43,799,252 bytes, `docs/facts.md`), but a visitor needs only one `.wasm`: 28,747,738 bytes for the WebGPU path or 14,848,852 bytes for the WASM path, going by which files each bundle references (`docs/facts.md`; I did not measure it in a browser network panel). If WebGPU fails and the page falls back to WASM, both sets may be fetched.

## Requirements

| Req | Requirement | Verification | Evidence | Status |
|---|---|---|---|---|
| 1 | Playable, model-produced frames | manual play in Chrome; `audit_web.py` keeps the engine out of the game page | `docs/browser_checks.md` (compare page: 0 of 132 steps mismatched; earlier 0 of 210), `scripts/audit_web.py` | Done |
| 2 | Tiny, from scratch, parameters reported | parameter count and ONNX size at export | `spikes/spike_a/results_v3.json` (78,177 parameters), `web/model/export_info.json` (396,375 bytes) | Done |
| 3 | PyTorch training | training scripts and logs | `spikes/spike_a/v3.py`, `run1.py`, `run2.py`, `run2.log` | Done |
| 4 | Free hardware, real time stated | training log with device and wall time | `spikes/spike_a/results_run2.json` (MPS, 1,503 s); Colab not used | Done |
| 5 | WebGPU with WASM fallback, page shows the real backend | Chrome placement log, both backends run in Chrome, ONNX parity, runtime self-test on a real phone | `docs/browser_checks.md` (including the phone self-test: WebGPU 0 of 8, WASM 8 of 8), `web/model/parity_results.json`, `web/model/selftest.json` | Done |
| 6 | Free, static, no CDN | file audit, vendored licence files | `scripts/audit_web.py`, `web/vendor/` | Done |
| 7 | A verified WorldTUI sentence | source check | `docs/facts.md` rows a-e (verified through a fetch summary, not re-read on the raw page) | Done |
| 8 | Honest failure evaluation | held-out evaluation run once, failure grids | `eval/results_heldout.json`, `eval/results_heldout.md`, `eval/heldout.lock` | Done |

## Reproduce

Tested with Python 3.12.16, torch 2.15.0, numpy 2.5.4, onnx 1.23.2, onnxruntime 1.31.1, onnxscript 0.7.2, pytest 9.2.0, and Node 24. **What was re-run and what was not.** On 2026-10-31 the export, parity, test and audit commands below were re-run. The data generation and training commands were **not** re-run from a fresh clone; they are the commands that produced the data and the shipped model originally, not re-verified since. I did not test whether training reproduces bit for bit.

```bash
source scripts/env.sh                      # keeps pip/npm/temp caches inside the project
python3.12 -m venv .venv
.venv/bin/pip install torch==2.15.0 numpy onnx onnxruntime==1.31.1 onnxscript pytest
npm ci                                     # onnxruntime-web 1.31.1 (package-lock.json)

.venv/bin/python spikes/spike_a/gen.py     # Phase 0 data  -> data/spike_a/ (git-ignored)
.venv/bin/python spikes/spike_a/gen_run1.py   # more data   -> data/run1/ (git-ignored)
.venv/bin/python spikes/spike_a/run2.py train 1800   # Run 2: up to 30 min on MPS, writes checkpoints/run2_best.pt

.venv/bin/python scripts/export_onnx.py    # -> web/model/snake.onnx
.venv/bin/python scripts/make_start_state.py
.venv/bin/python scripts/make_selftest.py  # golden cases for the in-page self-test; re-run whenever the model is re-exported
.venv/bin/python scripts/parity_onnx.py    # PyTorch vs ONNX: 1,000 rows + 20 x 200-step rollouts
.venv/bin/python scripts/make_golden_traces.py

.venv/bin/python -m pytest tests -q
node --test tests/js/engine_golden.test.mjs tests/js/controls.test.mjs tests/js/selftest.test.mjs
.venv/bin/python scripts/audit_web.py
python3 scripts/check_readme_numbers.py   # this README's numbers against the result files

python3 -m http.server 8000 --bind 127.0.0.1 --directory web   # then open http://127.0.0.1:8000/
```

The data and the checkpoint are git-ignored, so the model-dependent tests skip on a fresh clone. `scripts/eval_heldout.py dryrun` re-runs the evaluation on validation rows. **The held-out evaluation cannot be re-run**: `eval/heldout.lock` makes the script refuse a second run, by design.

## Licences

onnxruntime-web 1.31.1 is MIT licensed; its `LICENSE` and `ThirdPartyNotices.txt` (from the official `microsoft/onnxruntime` repository, tag v1.31.1) are in `web/vendor/`, with sources and checksums in `web/vendor/README.md`. This project's own code is MIT licensed (`LICENSE`, Copyright (c) 2026 Dev Suthar).
