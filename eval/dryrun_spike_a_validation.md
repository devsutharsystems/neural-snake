## DRY RUN on VALIDATION rows: data/spike_a

dataset `data/spike_a`, split `validation`, 10374 rows from 141 episodes (43.5 s).

### One-step whole-board exact match (model / copy-previous-frame baseline; n)

| length | all steps | eat steps | non-eat steps |
|---|---|---|---|
| 1-3 | 1.0 / 0.01312 (n=3964) | 1.0 / 0.0 (n=218) | 1.0 / 0.01388 (n=3746) |
| 4-7 | 0.99948 / 0.00519 (n=1928) | 0.99479 / 0.0 (n=192) | 1.0 / 0.00576 (n=1736) |
| 8-15 | 1.0 / 0.00723 (n=1799) | 1.0 / 0.0 (n=189) | 1.0 / 0.00807 (n=1610) |
| 16+ | 0.99963 / 0.00596 (n=2683) | 1.0 / 0.0 (n=240) | 0.99959 / 0.00655 (n=2443) |
| 16-31 | 1.0 / 0.00497 (n=1811) | 1.0 / 0.0 (n=171) | 1.0 / 0.00549 (n=1640) |
| 32-63 | 0.99885 / 0.00803 (n=872) | 1.0 / 0.0 (n=69) | 0.99875 / 0.00872 (n=803) |
| 64+ | None / None (n=0) | None / None (n=0) | None / None (n=0) |
| all | 0.99981 / 0.00877 (n=10374) | 0.99881 / 0.0 (n=839) | 0.9999 / 0.00954 (n=9535) |

### Per-class cell error (%)

| class | cells | wrong (14-class) | type only | eat steps | non-eat steps |
|---|---|---|---|---|---|
| empty | 1372399 | 0.0 | 0.0 | 0.0 | 0.0 |
| food | 9535 | 0.0 | 0.0 | None | 0.0 |
| head | 10374 | 0.0 | 0.0 | 0.0 | 0.0 |
| body | 93518 | 0.0011 | 0.0 | 0.0 | 0.0012 |
| tail | 8030 | 0.0125 | 0.0125 | 0.1192 | 0.0 |

Game over: {'n_true': 141, 'predicted': 141, 'precision': 1.0, 'recall': 1.0}

Ate head: {'accuracy': 1.0, 'n_eat': 839, 'precision': 1.0, 'recall': 1.0, 'accuracy_if_always_no': 0.91912}

Food head vs floor: {'eat_rows_with_spawn': 839, 'head_cross_entropy': 4.8794, 'uniform_floor_mean_ln_empty': 4.8722, 'ce_minus_floor': 0.0072, 'mean_prob_mass_on_snake_cells': 0.00648, 'spawn_valid_pct_sampled_on_empty_cell': 99.523, 'spawn_on_snake_pct': 0.477}

Implied median ln(0.5)/ln(p) from one-step p: {'overall_exact': {'p': 0.99981, 'median_steps': 3647.8}, 'non_eat_exact': {'p': 0.9999, 'median_steps': 6931.13}, 'bucket_balanced_exact': {'p': 0.99978, 'median_steps': 3114.92}}

### Rollouts (cap 1000, food synced)

Headline: counts, and the Kaplan-Meier estimate of P(no mismatch yet). A rollout that ends in game over without a mismatch, or reaches the cap, is censored at that step. KM median is "not reached" if the curve never drops below 0.5.

| policy | rollouts | mismatched | matched to end | alive at cap | S(50) | S(100) | S(200) | S(500) | S(1000) | KM median |
|---|---|---|---|---|---|---|---|---|---|---|
| BFS | 100 | 3 | 97 | 0 | 1.0 | 1.0 | 0.9855 | 0.932 | n/a (none observed) | not reached |
| random non-suicidal | 100 | 8 | 78 | 14 | 0.9573 | 0.9274 | 0.9108 | 0.9108 | 0.8744 | not reached |
| combined | 200 | 11 | 175 | 14 | 0.9792 | 0.9666 | 0.9515 | 0.9257 | 0.8901 | not reached |

Rollouts still under observation at step h (not yet mismatched, ended or capped): BFS: 50: 96, 100: 86, 200: 65, 500: 1, 1000: 0; random non-suicidal: 50: 73, 100: 59, 200: 44, 500: 27, 1000: 14; combined: 50: 169, 100: 145, 200: 109, 500: 28, 1000: 14

| policy | rollouts with a dynamics-invalid state | reasons (heads != 1 / wrong length / disconnected) | food-on-snake events / eat events | rollouts with a food-on-snake event | longest snake (engine / model) | rollouts with engine length >= 64 | snake length at first mismatch (median / max) |
|---|---|---|---|---|---|---|---|
| BFS | 2 | 1 / 1 / 0 | 29 / 2531 (1.146%) | 27 | 57 / 57 | 0 | 31.0 / 38 |
| random non-suicidal | 1 | 0 / 1 / 0 | 0 / 172 (0.0%) | 0 | 49 / 49 | 0 | 21.0 / 49 |
| combined | 3 | 1 / 2 / 0 | 29 / 2703 (1.073%) | 27 | 57 / 57 | 0 | 27.0 / 49 |

Dynamics-invalid (heads, length, connectivity) and food-on-snake are reported separately. Food-on-snake = the cell sampled from the model's food head lies on a snake cell of the model's own predicted grid.
The censored-at-end-step median, the cap+1 median and the rollout-hazard implied median are in the JSON under `conventions_not_headline` only.

### Failure grids (first mismatch; engine left, model right)

bfs rollout 29, step 299, action 4, engine snake length 38
```
engine (head ^v<>, body arrows, tail T)   model
............   ............
............   ............
............   ............
............   ............
............   ............
↓←←←←←......   ↓←←←←←......
↓→→↓.↑......   ↓→→↓.↑......
↓↑←→→↑.F....   ↓↑←→→↑.F....
→↓↑.........   →↓↑.........
.↓↑.......T.   .↓↑.......T.
↓←↑.......↓.   ↓←↑.......↓.
→>↑←←←←←←←←.   →^↑←←←←←←←←.
```

random rollout 15, step 32, action 0, engine snake length 38
```
engine (head ^v<>, body arrows, tail T)   model
............   ............
.....F......   .....F......
............   ............
............   ............
.........→→↓   .........→→↓
...→→→→↓.↑←↓   ...→→→→↓.↑←↓
..→↑...→→→↑↓   ..→↑...→→→↑↓
..↑.......↓←   ..↑.......↓←
..↑.......↓^   ..↑.......↓^
.T↑.......↓↑   ..↑.......↓↑
↓←↑.......↓↑   ↓T↑.......↓↑
→→↑.......→↑   →→↑.......→↑
```

random rollout 87, step 4, action 3, engine snake length 49
```
engine (head ^v<>, body arrows, tail T)   model
......→→→→↓.   ......→→→→↓.
......↑←←←↓.   ......↑←←←↓.
.....→→→→↑↓.   .....→→↓→↑↓.
.....↑.→>.↓.   .....↑.→>.↓.
→→↓..↑.↑..↓.   →→↓..↑.↑..↓.
↑.→→→↑.↑..↓.   ↑.→→→↑.↑..↓.
↑.F....↑←.↓.   ↑.F....↑←.↓.
↑←......↑.↓.   ↑←......↑.↓.
.↑......↑←←.   .↑......↑←←.
.↑..........   .↑..........
.↑..........   .↑..........
.↑T.........   .↑T.........
```
