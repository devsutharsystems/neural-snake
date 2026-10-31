## DRY RUN on VALIDATION rows: data/run1

dataset `data/run1`, split `validation`, 135292 rows from 1901 episodes (82.7 s).

### One-step whole-board exact match (model / copy-previous-frame baseline; n)

| length | all steps | eat steps | non-eat steps |
|---|---|---|---|
| 1-3 | 0.99991 / 0.01305 (n=53728) | 1.0 / 0.0 (n=2955) | 0.9999 / 0.01381 (n=50773) |
| 4-7 | 0.99992 / 0.00678 (n=25219) | 1.0 / 0.0 (n=2455) | 0.99991 / 0.00751 (n=22764) |
| 8-15 | 0.99984 / 0.00459 (n=25483) | 1.0 / 0.0 (n=2597) | 0.99983 / 0.00511 (n=22886) |
| 16+ | 0.99977 / 0.0068 (n=30862) | 0.99966 / 0.0 (n=2914) | 0.99979 / 0.00751 (n=27948) |
| 16-31 | 0.99979 / 0.0054 (n=23530) | 1.0 / 0.0 (n=2327) | 0.99976 / 0.00599 (n=21203) |
| 32-63 | 0.99973 / 0.01132 (n=7332) | 0.9983 / 0.0 (n=587) | 0.99985 / 0.01231 (n=6745) |
| 64+ | None / None (n=0) | None / None (n=0) | None / None (n=0) |
| all | 0.99987 / 0.00886 (n=135292) | 0.99991 / 0.0 (n=10921) | 0.99986 / 0.00964 (n=124371) |

### Per-class cell error (%)

| class | cells | wrong (14-class) | type only | eat steps | non-eat steps |
|---|---|---|---|---|---|
| empty | 18049395 | 0.0 | 0.0 | 0.0 | 0.0 |
| food | 124371 | 0.0024 | 0.0024 | None | 0.0024 |
| head | 135292 | 0.0007 | 0.0007 | 0.0 | 0.0008 |
| body | 1070443 | 0.0006 | 0.0001 | 0.0 | 0.0006 |
| tail | 102547 | 0.0029 | 0.001 | 0.0092 | 0.0022 |

Game over: {'n_true': 1897, 'predicted': 1897, 'precision': 1.0, 'recall': 1.0}

Ate head: {'accuracy': 1.0, 'n_eat': 10921, 'precision': 1.0, 'recall': 1.0, 'accuracy_if_always_no': 0.91928}

Food head vs floor: {'eat_rows_with_spawn': 10921, 'head_cross_entropy': 4.8919, 'uniform_floor_mean_ln_empty': 4.8793, 'ce_minus_floor': 0.0126, 'mean_prob_mass_on_snake_cells': 0.00596, 'spawn_valid_pct_sampled_on_empty_cell': 99.405, 'spawn_on_snake_pct': 0.595}

Implied median ln(0.5)/ln(p) from one-step p: {'overall_exact': {'p': 0.99987, 'median_steps': 5331.55}, 'non_eat_exact': {'p': 0.99986, 'median_steps': 4950.7}, 'bucket_balanced_exact': {'p': 0.99986, 'median_steps': 4950.7}}

### Rollouts (cap 1000, food synced)

Headline: counts, and the Kaplan-Meier estimate of P(no mismatch yet). A rollout that ends in game over without a mismatch, or reaches the cap, is censored at that step. KM median is "not reached" if the curve never drops below 0.5.

| policy | rollouts | mismatched | matched to end | alive at cap | S(50) | S(100) | S(200) | S(500) | S(1000) | KM median |
|---|---|---|---|---|---|---|---|---|---|---|
| BFS | 100 | 3 | 97 | 0 | 1.0 | 1.0 | 0.989 | n/a (none observed) | n/a (none observed) | not reached |
| random non-suicidal | 100 | 10 | 75 | 15 | 0.9765 | 0.9343 | 0.8641 | 0.8444 | 0.8444 | not reached |
| combined | 200 | 13 | 172 | 15 | 0.9892 | 0.9713 | 0.9367 | 0.8978 | 0.8978 | not reached |

Rollouts still under observation at step h (not yet mismatched, ended or capped): BFS: 50: 97, 100: 95, 200: 71, 500: 0, 1000: 0; random non-suicidal: 50: 73, 100: 61, 200: 45, 500: 29, 1000: 15; combined: 50: 170, 100: 156, 200: 116, 500: 29, 1000: 15

| policy | rollouts with a dynamics-invalid state | reasons (heads != 1 / wrong length / disconnected) | food-on-snake events / eat events | rollouts with a food-on-snake event | longest snake (engine / model) | rollouts with engine length >= 64 | snake length at first mismatch (median / max) |
|---|---|---|---|---|---|---|---|
| BFS | 3 | 0 / 3 / 0 | 40 / 2722 (1.47%) | 32 | 54 / 51 | 0 | 43.0 / 45 |
| random non-suicidal | 4 | 2 / 2 / 0 | 0 / 188 (0.0%) | 0 | 51 / 51 | 0 | 9.0 / 28 |
| combined | 7 | 2 / 5 / 0 | 40 / 2910 (1.375%) | 32 | 54 / 51 | 0 | 14.0 / 45 |

Dynamics-invalid (heads, length, connectivity) and food-on-snake are reported separately. Food-on-snake = the cell sampled from the model's food head lies on a snake cell of the model's own predicted grid.
The censored-at-end-step median, the cap+1 median and the rollout-hazard implied median are in the JSON under `conventions_not_headline` only.

### Failure grids (first mismatch; engine left, model right)

bfs rollout 35, step 134, action 3, engine snake length 43
```
engine (head ^v<>, body arrows, tail T)   model
............   ............
............   ............
...→→→↓.....   ...→→→↓.....
...↑..↓.....   ...↑..↓.....
→→↓↑←←↓.....   →→↓↑←←←.....
↑←↓..↑→>.T..   ↑←↓..↑→>.T..
.↑→→→↑...↓..   .↑→→→↑...↓..
.↑.......↓..   .↑.......↓..
.↑.......↓..   .↑.......↓..
.↑.......↓..   .↑.......↓..
.↑.......↓..   .↑.......↓..
.↑←←←←←←←←..   .↑←←←←←←←←..
```

random rollout 0, step 162, action 2, engine snake length 2
```
engine (head ^v<>, body arrows, tail T)   model
............   ............
............   ............
............   ............
............   ............
............   ............
............   ............
............   ............
............   ............
............   ............
..........<T   ..........<.
............   ............
..........F.   ..........F.
```

bfs rollout 50, step 350, action 0, engine snake length 45
```
engine (head ^v<>, body arrows, tail T)   model
............   ............
...→→→↓....^   ...→→→↓....^
...↑..→→→→→↑   ...↑..→→→→→↑
..F↑←....T→↓   ..F↑←....T→↑
....↑.....↓←   ....↑.....↓←
....↑.....↓.   ....↑.....↓.
....↑.....↓.   ....↑.....↓.
....↑.....↓.   ....↑.....↓.
....↑..↓←←←.   ....↑..↓←←←.
....↑..→→→↓.   ....↑..→→→↓.
....↑.↓←←←←.   ....↑.↓←←←←.
....↑←←.....   ....↑←←.....
```
