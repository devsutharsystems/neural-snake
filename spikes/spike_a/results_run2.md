# run2 results (VALIDATION, seed % 5 == 1, Spike A rows)

Command: `run2.py train 1800.0`; raw numbers in `results_run2.json`.

Train rows 431551, steps 43074, 1503.4 s, 7335 samples/s.

## Whole-board exact match by snake length (model / copy-previous-frame baseline)

| length | all steps | eat steps | non-eat steps |
|---|---|---|---|
| 1-3 | 1.0 / 0.0131 (n=3964) | 1.0 / 0.0 (n=218) | 1.0 / 0.0139 (n=3746) |
| 4-7 | 0.9995 / 0.0052 (n=1928) | 0.9948 / 0.0 (n=192) | 1.0 / 0.0058 (n=1736) |
| 8-15 | 1.0 / 0.0072 (n=1799) | 1.0 / 0.0 (n=189) | 1.0 / 0.0081 (n=1610) |
| 16+ | 0.9996 / 0.006 (n=2683) | 1.0 / 0.0 (n=240) | 0.9996 / 0.0065 (n=2443) |
| all | 0.9998 / 0.0088 (n=10374) | 0.9988 / 0.0 (n=839) | 0.9999 / 0.0095 (n=9535) |

## Per-class cell error (%, 14-class; type-only; eat / non-eat steps)

| class | cells | wrong | type-only | eat | non-eat |
|---|---|---|---|---|---|
| empty | 1372399 | 0.0 | 0.0 | 0.0 | 0.0 |
| food | 9535 | 0.0 | 0.0 | None | 0.0 |
| head | 10374 | 0.0 | 0.0 | 0.0 | 0.0 |
| body | 93518 | 0.001 | 0.0 | 0.0 | 0.001 |
| tail | 8030 | 0.012 | 0.012 | 0.119 | 0.0 |

Game over: {'n_true': 141, 'predicted': 141, 'precision': 1.0, 'recall': 1.0}

Ate head: {'accuracy': 1.0, 'n_eat': 839, 'precision': 1.0, 'recall': 1.0, 'accuracy_always_no': 0.9191}

Food spawn on empty cell: 99.52% over 839 eat steps

## Rollouts (cap 200, food synced)

- bfs n=100: first mismatch median 201.0, p10 201.0, matched to end 38, no mismatch within cap 99, first invalid median 201.0, invalid reasons {'wrong_length': 1, 'food_on_snake': 15}
- random n=100: first mismatch median 201.0, p10 201.0, matched to end 47, no mismatch within cap 91, first invalid median 201.0, invalid reasons {'wrong_length': 2, 'heads!=1': 1}

Implied median ln(0.5)/ln(p): {"overall_exact": {"p": 0.9998, "median_steps": 3465.39}, "non_eat_exact": {"p": 0.9999, "median_steps": 6931.13}, "bucket_balanced_exact": {"p": 0.9998, "median_steps": 3080.31}}

## Failure grids

bfs rollout 67 step 173 action 3
```
engine (head ^v<>, body arrows, tail T)   model
............   ............
.↓←←←←←←↓←←←   .↓←←←←←←↓←←←
.→↓→↓..↑↓..↑   .→↓→↓..↑↓..↑
..→↑↓..↑↓..↑   ..→↑↓..↑↓..↑
...↓←..↑←..↑   ...↓←..↑←..↑
...↓..T→→→→↑   ...↓T.T→→→→↑
...↓→>...F..   ...↓→>...F..
...↓↑.......   ...↓↑.......
...→↑.......   ...→↑.......
............   ............
............   ............
............   ............
```

random rollout 9 step 154 action 2
```
engine (head ^v<>, body arrows, tail T)   model
..F↓←T......   ..T↓←T......
.<←←........   .<←←........
............   ............
............   ............
............   ............
............   ............
............   ............
............   ............
............   ............
............   ............
............   ............
............   ............
```
