# Held-out evaluation (run once)

Shipped model `web/model/snake.onnx` (sha256 baf94b18913641846a448ced2c298c9ce707869d24aed1eefbb4b4bf474fd740), onnxruntime 1.31.1, script sha256 3b10e81e5204d6b415bf98a267542573e6f2e6ebe71f1b7fc0ada282280b2ace, git HEAD cf6d67958a2a009109d487a6fee345b11a7f1980, 2026-10-31T06:10:30Z.

## HEADLINE: data/run1 held-out (seed % 5 == 0, never seen by any model or script before this run)

dataset `data/run1`, split `heldout`, 133848 rows from 1901 episodes (51.0 s).

### One-step whole-board exact match (model / copy-previous-frame baseline; n)

| length | all steps | eat steps | non-eat steps |
|---|---|---|---|
| 1-3 | 0.99996 / 0.01298 (n=54530) | 1.0 / 0.0 (n=2960) | 0.99996 / 0.01373 (n=51570) |
| 4-7 | 1.0 / 0.00625 (n=25918) | 1.0 / 0.0 (n=2506) | 1.0 / 0.00692 (n=23412) |
| 8-15 | 0.99954 / 0.00598 (n=23930) | 0.99959 / 0.0 (n=2438) | 0.99953 / 0.00665 (n=21492) |
| 16+ | 0.99983 / 0.00685 (n=29470) | 0.99964 / 0.0 (n=2795) | 0.99985 / 0.00757 (n=26675) |
| 16-31 | 0.99982 / 0.0059 (n=22536) | 0.99955 / 0.0 (n=2205) | 0.99985 / 0.00654 (n=20331) |
| 32-63 | 0.99986 / 0.00995 (n=6934) | 1.0 / 0.0 (n=590) | 0.99984 / 0.01088 (n=6344) |
| 64+ | None / None (n=0) | None / None (n=0) | None / None (n=0) |
| all | 0.99987 / 0.00908 (n=133848) | 0.99981 / 0.0 (n=10699) | 0.99987 / 0.00987 (n=123149) |

### Per-class cell error (%)

| class | cells | wrong (14-class) | type only | eat steps | non-eat steps |
|---|---|---|---|---|---|
| empty | 17887414 | 0.0 | 0.0 | 0.0 | 0.0 |
| food | 123149 | 0.0032 | 0.0032 | None | 0.0032 |
| head | 133848 | 0.0015 | 0.0015 | 0.0 | 0.0016 |
| body | 1029080 | 0.0012 | 0.0001 | 0.0009 | 0.0012 |
| tail | 100621 | 0.007 | 0.005 | 0.0093 | 0.0067 |

Game over: {'n_true': 1896, 'predicted': 1892, 'precision': 1.0, 'recall': 0.99789}

Ate head: {'accuracy': 1.0, 'n_eat': 10699, 'precision': 1.0, 'recall': 1.0, 'accuracy_if_always_no': 0.92007}

Food head vs floor: {'eat_rows_with_spawn': 10699, 'head_cross_entropy': 4.8909, 'uniform_floor_mean_ln_empty': 4.8803, 'ce_minus_floor': 0.0107, 'mean_prob_mass_on_snake_cells': 0.00589, 'spawn_valid_pct_sampled_on_empty_cell': 99.421, 'spawn_on_snake_pct': 0.579}

Implied median ln(0.5)/ln(p) from one-step p: {'overall_exact': {'p': 0.99987, 'median_steps': 5331.55}, 'non_eat_exact': {'p': 0.99987, 'median_steps': 5331.55}, 'bucket_balanced_exact': {'p': 0.99983, 'median_steps': 4137.85}}

### Rollouts (cap 1000, food synced)

Headline: counts, and the Kaplan-Meier estimate of P(no mismatch yet). A rollout that ends in game over without a mismatch, or reaches the cap, is censored at that step. KM median is "not reached" if the curve never drops below 0.5.

| policy | rollouts | mismatched | matched to end | alive at cap | S(50) | S(100) | S(200) | S(500) | S(1000) | KM median |
|---|---|---|---|---|---|---|---|---|---|---|
| BFS | 100 | 4 | 96 | 0 | 0.9791 | 0.9791 | 0.9791 | n/a (none observed) | n/a (none observed) | not reached |
| random non-suicidal | 100 | 15 | 74 | 11 | 0.9508 | 0.9048 | 0.8432 | 0.7955 | 0.6641 | not reached |
| combined | 200 | 19 | 170 | 11 | 0.9659 | 0.9467 | 0.9222 | 0.8714 | 0.7274 | not reached |

Rollouts still under observation at step h (not yet mismatched, ended or capped): BFS: 50: 92, 100: 85, 200: 65, 500: 0, 1000: 0; random non-suicidal: 50: 66, 100: 52, 200: 38, 500: 27, 1000: 11; combined: 50: 158, 100: 137, 200: 103, 500: 27, 1000: 11

| policy | rollouts with a dynamics-invalid state | reasons (heads != 1 / wrong length / disconnected) | food-on-snake events / eat events | rollouts with a food-on-snake event | longest snake (engine / model) | rollouts with engine length >= 64 | snake length at first mismatch (median / max) |
|---|---|---|---|---|---|---|---|
| BFS | 3 | 2 / 1 / 0 | 23 / 2559 (0.899%) | 23 | 56 / 56 | 0 | 30.5 / 47 |
| random non-suicidal | 10 | 5 / 5 / 0 | 0 / 182 (0.0%) | 0 | 57 / 57 | 0 | 7.0 / 38 |
| combined | 13 | 7 / 6 / 0 | 23 / 2741 (0.839%) | 23 | 57 / 57 | 0 | 11.0 / 47 |

Dynamics-invalid (heads, length, connectivity) and food-on-snake are reported separately. Food-on-snake = the cell sampled from the model's food head lies on a snake cell of the model's own predicted grid.
The censored-at-end-step median, the cap+1 median and the rollout-hazard implied median are in the JSON under `conventions_not_headline` only.

### Failure grids (first mismatch; engine left, model right)

bfs rollout 11, step 322, action 0, engine snake length 47
```
engine (head ^v<>, body arrows, tail T)   model
............   ............
...........F   ...........F
..........↓←   ..........↓↑
..........↓↑   ..........↓↑
..........↓↑   ..........↓↑
..........↓↑   ..........↓↑
..↓←..↓←←←←↑   ..↓←..↓←←←←↑
..↓↑..→→→↓.↑   ..↓↑..→→→↓.↑
..↓↑←...↓←.↑   ..↓↑←...↓←.↑
..↓.↑←←←←..↑   ..↓.↑←←←←..↑
.^↓..T→→→→→↑   .^↓..T→→→→→↑
.↑←.........   .↑←.........
```

random rollout 0, step 782, action 2, engine snake length 5
```
engine (head ^v<>, body arrows, tail T)   model
..........→>   ..........→>
..........↑←   ..........↑T
...........T   ...........T
............   ............
............   ............
............   ............
............   ............
............   ............
............   ............
............   ............
............   ............
.....F......   .....F......
```

bfs rollout 19, step 32, action 2, engine snake length 21
```
engine (head ^v<>, body arrows, tail T)   model
............   ............
............   ............
............   ............
............   ............
............   ............
............   ............
..........↓←   ..........↓←
..........↓↑   ..........↓↑
......↓←←←←↑   ......↓←←←←↑
.....↓←....↑   .....↓←....↑
...F<←T→→→→↑   ...F<←→→→→→↑
............   ............
```

## data/spike_a held-out (seen once in Phase 0 by throwaway models)

dataset `data/spike_a`, split `heldout`, 9569 rows from 141 episodes (26.7 s).

### One-step whole-board exact match (model / copy-previous-frame baseline; n)

| length | all steps | eat steps | non-eat steps |
|---|---|---|---|
| 1-3 | 1.0 / 0.01571 (n=3693) | 1.0 / 0.0 (n=209) | 1.0 / 0.01665 (n=3484) |
| 4-7 | 1.0 / 0.00572 (n=1749) | 1.0 / 0.0 (n=181) | 1.0 / 0.00638 (n=1568) |
| 8-15 | 0.99908 / 0.0055 (n=2181) | 1.0 / 0.0 (n=225) | 0.99898 / 0.00613 (n=1956) |
| 16+ | 1.0 / 0.00771 (n=1946) | 1.0 / 0.0 (n=186) | 1.0 / 0.00852 (n=1760) |
| 16-31 | 1.0 / 0.00558 (n=1614) | 1.0 / 0.0 (n=157) | 1.0 / 0.00618 (n=1457) |
| 32-63 | 1.0 / 0.01807 (n=332) | 1.0 / 0.0 (n=29) | 1.0 / 0.0198 (n=303) |
| 64+ | None / None (n=0) | None / None (n=0) | None / None (n=0) |
| all | 0.99979 / 0.00993 (n=9569) | 1.0 / 0.0 (n=801) | 0.99977 / 0.01083 (n=8768) |

### Per-class cell error (%)

| class | cells | wrong (14-class) | type only | eat steps | non-eat steps |
|---|---|---|---|---|---|
| empty | 1281890 | 0.0 | 0.0 | 0.0 | 0.0 |
| food | 8768 | 0.0114 | 0.0114 | None | 0.0114 |
| head | 9569 | 0.0 | 0.0 | 0.0 | 0.0 |
| body | 70304 | 0.0014 | 0.0 | 0.0 | 0.0016 |
| tail | 7405 | 0.0 | 0.0 | 0.0 | 0.0 |

Game over: {'n_true': 141, 'predicted': 141, 'precision': 1.0, 'recall': 1.0}

Ate head: {'accuracy': 1.0, 'n_eat': 801, 'precision': 1.0, 'recall': 1.0, 'accuracy_if_always_no': 0.91629}

Food head vs floor: {'eat_rows_with_spawn': 801, 'head_cross_entropy': 4.8936, 'uniform_floor_mean_ln_empty': 4.8851, 'ce_minus_floor': 0.0085, 'mean_prob_mass_on_snake_cells': 0.00555, 'spawn_valid_pct_sampled_on_empty_cell': 99.501, 'spawn_on_snake_pct': 0.499}

Implied median ln(0.5)/ln(p) from one-step p: {'overall_exact': {'p': 0.99979, 'median_steps': 3300.35}, 'non_eat_exact': {'p': 0.99977, 'median_steps': 3013.34}, 'bucket_balanced_exact': {'p': 0.99977, 'median_steps': 3013.34}}

### Rollouts (cap 1000, food synced)

Headline: counts, and the Kaplan-Meier estimate of P(no mismatch yet). A rollout that ends in game over without a mismatch, or reaches the cap, is censored at that step. KM median is "not reached" if the curve never drops below 0.5.

| policy | rollouts | mismatched | matched to end | alive at cap | S(50) | S(100) | S(200) | S(500) | S(1000) | KM median |
|---|---|---|---|---|---|---|---|---|---|---|
| BFS | 100 | 5 | 95 | 0 | 1.0 | 1.0 | 1.0 | n/a (none observed) | n/a (none observed) | not reached |
| random non-suicidal | 100 | 7 | 76 | 17 | 0.966 | 0.966 | 0.9484 | 0.878 | 0.878 | not reached |
| combined | 200 | 12 | 171 | 17 | 0.9838 | 0.9838 | 0.977 | 0.8568 | 0.8568 | not reached |

Rollouts still under observation at step h (not yet mismatched, ended or capped): BFS: 50: 96, 100: 91, 200: 64, 500: 0, 1000: 0; random non-suicidal: 50: 65, 100: 59, 200: 48, 500: 32, 1000: 17; combined: 50: 161, 100: 150, 200: 112, 500: 32, 1000: 17

| policy | rollouts with a dynamics-invalid state | reasons (heads != 1 / wrong length / disconnected) | food-on-snake events / eat events | rollouts with a food-on-snake event | longest snake (engine / model) | rollouts with engine length >= 64 | snake length at first mismatch (median / max) |
|---|---|---|---|---|---|---|---|
| BFS | 2 | 2 / 0 / 0 | 33 / 2556 (1.291%) | 26 | 57 / 57 | 0 | 38.0 / 48 |
| random non-suicidal | 2 | 2 / 0 / 0 | 0 / 185 (0.0%) | 0 | 40 / 40 | 0 | 10.0 / 20 |
| combined | 4 | 4 / 0 / 0 | 33 / 2741 (1.204%) | 26 | 57 / 57 | 0 | 16.0 / 48 |

Dynamics-invalid (heads, length, connectivity) and food-on-snake are reported separately. Food-on-snake = the cell sampled from the model's food head lies on a snake cell of the model's own predicted grid.
The censored-at-end-step median, the cap+1 median and the rollout-hazard implied median are in the JSON under `conventions_not_headline` only.

### Failure grids (first mismatch; engine left, model right)

bfs rollout 8, step 399, action 0, engine snake length 47
```
engine (head ^v<>, body arrows, tail T)   model
↓←←←←←←←....   ↓←←←←←←←....
↓......↑....   ↓......↑....
↓....FT↑....   ↓....FT↑....
↓....→↓.....   ↓....↑↓.....
↓....↑↓...^.   ↓....↑↓...^.
↓....↑→→→→↑.   ↓....↑→→→→↑.
↓....↑......   ↓....↑......
↓....↑......   ↓....↑......
↓....↑......   ↓....↑......
→→→↓.↑←←←...   →→→↓.↑←←←...
...→→→→→↑...   ...→→→→→↑...
............   ............
```

random rollout 5, step 124, action 1, engine snake length 4
```
engine (head ^v<>, body arrows, tail T)   model
..........→↓   ..........→↓
..........Tv   ..........↑T
............   ............
............   ............
............   ............
............   ............
............   ............
............   ............
............   ............
............   ............
............   ............
...........F   ...........F
```

bfs rollout 89, step 435, action 4, engine snake length 48
```
engine (head ^v<>, body arrows, tail T)   model
.....↓←←←...   .....↓←←←...
.....↓→→↑...   .....↓→→↑...
.....↓↑←←←←.   .....↓↑←←←←.
.....↓....↑.   .....↓....↑.
..↓←←←....↑.   ..↓←←←....↑.
..↓↓←.....↑.   ..↓↓←.....↑.
..↓↓T.....↑.   ..↓↓T.....↑.
↓←←→→→↓.F.↑.   ↓←←→→→↓.F.↑.
↓<←...→→→→↑.   ↓<↑...→→→→↑.
→→↑.........   →→↑.........
............   ............
............   ............
```
