# Spike A results (held-out episode seeds only)
Produced by: `.venv/bin/python spikes/spike_a/gen.py` then `.venv/bin/python spikes/spike_a/train_eval.py 300` (log: run.log; raw numbers: results.json; failure grids: failures.md).
Device: MPS. Each representation trained 300 s wall-clock, batch 256: R1 2074 steps, R2 2623 steps. Params: R1 157,019; R2 156,587. Train-test split: episode seed % 5 == 0 held out (split.json, written before data generation). 41,281 train / 9,569 held-out transitions.
Receptive field: 7 dilated 3x3 convs (1,2,4,1,2,4,1) = 31 cells wide, plus a global mean/max pooled feature after layer 3.

**Headline: both models are poor. Neither passes a 20-step rollout; median first mismatch is 1 to 2 steps.**

## Median real episode length per policy (gen_stats.json, cap 500)
random 30 | greedy 213 | bfs 356 | bfs_eps20 61 | greedy_eps30 45.5

## Snake-length histogram (transitions)
| split | 1-3 | 4-7 | 8-15 | 16+ |
|---|---|---|---|---|
| train | 16387 | 7747 | 7393 | 9754 |
| held-out | 3693 | 1749 | 2181 | 1946 |
(Rebalancing was done by sampling weights ~1/sqrt(bucket count) during training, not by changing the dataset.)

## One-step whole-board exact match of the deterministic grid
| length | n | copy-previous baseline | R1 model | R2 model |
|---|---|---|---|---|
| 1-3 | 3693 | 2.3% | 47.7% | 63.5% |
| 4-7 | 1749 | 1.0% | 51.3% | 42.1% |
| 8-15 | 2181 | 1.0% | 39.8% | 30.5% |
| 16+ | 1946 | 0.9% | 33.9% | 25.5% |
| all | 9569 | 1.5% | 43.8% | 44.4% |
R1 also needs order and direction right: full-state exact match (grid+order+dir) is only 1.8% overall.

## Game over (141 true game-over transitions)
R1 precision 0.0, recall 0.0 (never fired). R2 precision 92.2%, recall 83.7%.

## Food head (801 eat steps, sampled)
Spawn lands on an empty cell of the true grid: R1 100%, R2 100% (model's own grid: 99.9% / 100%). This only shows the food head avoids the snake when the grid is right; it does not show uniform placement.

## 20-step rollouts (50 BFS-bot + 50 random-non-suicidal; start states from held-out transitions, 12-13 per length bucket; food synced to engine)
| model | policy | first mismatch median / p10 | matched to end | first INVALID median / p10 | invalid reasons |
|---|---|---|---|---|---|
| R1 | BFS | 2 / 1 | 0 | 2 / 1 | wrong_length 42, heads!=1 8 |
| R1 | random | 1 / 1 | 0 | 1 / 1 | wrong_length 43, heads!=1 7 |
| R2 | BFS | 1 / 1 | 0 | 1 / 1 | wrong_length 45, heads!=1 5 |
| R2 | random | 1 / 1 | 0 | 1 / 1 | wrong_length 37, heads!=1 13 |
Compare-with-engine checks the class grid only (empty/food/head/body), not R1's order plane.

## R2 information limit
Same R2 input (2 frames + action) with different true next state: 0 conflicting inputs. Held-out: 132 of 9,430 distinct inputs repeat; all data: 1,598 of 48,497 repeat. **Weak evidence**: almost no input repeats, so 0 conflicts mostly means the check has little to compare. (results.json: R2_information_limit_*.)

## Failure examples
See failures.md: in the R1 BFS rollouts the model leaves an extra body cell (tail not removed; grid longer than the engine's). R2 shows the same extra-cell error.

## Caveats
- Single 5-minute run per representation, one seed, no tuning, no repeat. Differences between R1 and R2 on one-step accuracy (43.8 vs 44.4) are not meaningful.
- Possible causes I did NOT test (UNVERIFIED): 5x class weight on non-empty cells may push the model to over-predict body cells; too few training steps (2-3k); the tail-removal step needs sharp use of the order plane.
- Rollout start states are mid-game states reconstructed from stored order grids, not fresh games.
