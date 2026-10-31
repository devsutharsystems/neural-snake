# Spike A debug iteration (validation only)
Produced by `.venv/bin/python spikes/spike_a/v2.py full 300` (log run_v2.log, numbers results_v2.json, failures failures_v2.md). Overfit runs: `v2.py overfit <rep> 180` (overfit_*.json). Info limit: info_limit_val.json.
Split: split_v2.json. Train = episode seed %5 in {2,3,4} (30,907 transitions), validation = seed %5 == 1 (10,374). Held-out seeds (%5==0) were dropped on load (common.py `load_trainval`) and never evaluated or printed in this step.
Models: C=32 channels, receptive field 33 cells + two global-pool paths. R1v2 75,253 params, 3,747 steps in 300 s; R2 73,195 params, 4,087 steps in 300 s; MPS. Game-over pos_weight = 73.30 (neg/pos on train).

## Overfit test (1,000 train transitions, 180 s each)
| encoding | steps | grid exact | stricter metric | reached 99% |
|---|---|---|---|---|
| R1v2 (C=64, first run) | 1029 | 89.3% | type-collapsed 90.7% | NO |
| R1v2 (C=32) | 1777 | 90.5% | 91.9% | NO |
| R1v2 (C=32, lr decay) | 2033 | 77.0% | 80.0% | NO |
| R1old (scalar order/144) | 2052 | 43.0% | grid+order+dir 4.6% | NO |
Diagnosis of a 150 s R1v2 fit (87.2% exact): errors concentrate on the rare events: 85% of eat steps and 82% of game-over steps are wrong, and 161 of 174 wrong cells are cells whose target equals the input (e.g. the tail moved when it should stay). Not a pipeline bug (tests 1000/1000). It is under-fitting of the eat/game-over events within about 2k steps; the lr-decay run was worse, so it is not an lr-schedule fix. NOT confirmed whether more steps would reach 99%.

## One-step exact match, validation
| length | n | copy baseline (4-class) | R1v2 strict (14-class) | R1v2 type only | R2 (4-class) |
|---|---|---|---|---|---|
| 1-3 | 3964 | 2.0% | 67.4% | 70.5% | 74.1% |
| 4-7 | 1928 | 1.2% | 55.5% | 68.2% | 49.9% |
| 8-15 | 1799 | 1.2% | 46.0% | 59.3% | 36.3% |
| 16+ | 2683 | 0.7% | 25.1% | 41.4% | 16.9% |
| all | 10374 | 1.4% | 50.5% | 60.6% | 48.3% |
(14-class copy baseline: 0.9% overall.)

## Per-cell error by true class (% of cells wrong)
| true class | cells | R1v2 14-class | R1v2 type only | R2 |
|---|---|---|---|---|
| empty | 1,372,399 | 0.15 | 0.15 | 0.30 |
| food | 9,535 | 11.7 | 11.7 | 13.8 |
| head | 10,374 | 5.4 | 5.4 | 7.0 |
| body | 93,518 | 4.5 | 0.9 | 3.6 |
| tail | 8,030 | 27.9 | 18.5 | 17.8 |

## Game over (141 true)
R1v2 precision 91.6%, recall 92.9% (143 predicted). R2 precision 90.1%, recall 90.1%. Food spawn on empty cell: R1v2 100%, R2 99.9% (839 eat steps).

## 20-step rollouts (50 BFS + 50 random-non-suicidal, validation start states, food synced)
| model | policy | first mismatch median/p10 | matched to end | first invalid median/p10 | invalid reasons |
|---|---|---|---|---|---|
| R1v2 | BFS | 1.5 / 1 (type-only 2 / 1) | 0 | 2 / 1 | wrong_length 29, heads!=1 21 |
| R1v2 | random | 1 / 1 | 0 | 2 / 1 | heads!=1 35, wrong_length 14 |
| R2 | BFS | 1 / 1 | 0 | 1 / 1 | wrong_length 26, heads!=1 23, disconnected 1 |
| R2 | random | 1.5 / 1 | 0 | 2 / 1 | heads!=1 29, wrong_length 19 |
Censored: 1 R1v2 and 2 R2 random rollouts had no mismatch in 20 steps. Failure grids: failures_v2.md (R1v2 loses the tail cell; R2 adds a spurious body cell or moves the tail wrongly).

## R2 information limit (replaced)
Method: for each validation transition, take the snake cells of frame t and enumerate every 4-connected self-avoiding path through exactly those cells that starts at the head, has the previous head as its neck, and (if the snake did not grow) ends next to the cell that left the snake between t-1 and t. If valid paths end in more than one cell, frames t-1 and t do not determine which cell vacates at t+1.
Result: 278 of 10,374 (2.68%) ambiguous; a further 200 (1.93%) hit the 20,000-node search cap and are undetermined, so the ambiguity is between 2.68% and 4.61%. So R2's information limit is not what holds it down at ~48%.
