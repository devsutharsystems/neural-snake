import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.snake import Snake, UP, DOWN, LEFT, RIGHT, NOOP  # noqa: E402


def make(body, direction, food, n=12):
    s = Snake(seed=0, n=n)
    s.body = list(body)
    s.direction = direction
    s.food = food
    s.game_over = False
    return s


def test_movement():
    s = make([(5, 5)], RIGHT, (0, 0))
    s.step(UP)
    assert s.body == [(4, 5)]
    s.step(LEFT)
    assert s.body == [(4, 4)]
    s.step(DOWN)
    assert s.body == [(5, 4)]
    s.step(RIGHT)
    assert s.body == [(5, 5)]


def test_growth_and_new_food():
    s = make([(5, 5)], RIGHT, (5, 6))
    assert s.step(RIGHT) is True
    assert len(s.body) == 2 and s.body[0] == (5, 6) and s.score == 1
    assert s.food is not None and s.food not in s.body


def test_wall_death():
    s = make([(0, 3)], UP, (9, 9))
    s.step(UP)
    assert s.game_over
    s = make([(3, 11)], RIGHT, (9, 9))
    s.step(NOOP)
    assert s.game_over


def test_self_death():
    # head at (5,5) moving UP into body cell (4,5), not the tail
    body = [(5, 5), (5, 4), (4, 4), (4, 5), (4, 6)]
    s = make(body, LEFT, (9, 9))
    s.direction = RIGHT
    s.step(UP)
    assert s.game_over


def test_reverse_is_noop():
    s = make([(5, 5), (5, 4), (5, 3)], RIGHT, (9, 9))
    s.step(LEFT)
    assert not s.game_over
    assert s.body[0] == (5, 6) and s.direction == RIGHT


def test_noop_continues():
    s = make([(5, 5)], DOWN, (0, 0))
    s.step(NOOP)
    s.step(NOOP)
    assert s.body == [(7, 5)]


def test_food_never_on_snake():
    s = Snake(seed=3)
    for i in range(2000):
        if s.game_over:
            break
        s.step([UP, RIGHT, DOWN, LEFT, NOOP][i % 5])
        if s.food is not None:
            assert s.food not in s.body


def test_determinism_by_seed():
    acts = [RIGHT, DOWN, NOOP, LEFT, UP, NOOP, RIGHT] * 20

    def run(seed):
        s = Snake(seed=seed)
        out = []
        for a in acts:
            s.step(a)
            out.append((tuple(s.body), s.food, s.game_over))
        return out

    assert run(7) == run(7)
    assert Snake(seed=1).food != Snake(seed=2).food or Snake(seed=1).food != Snake(seed=3).food


def test_tail_chase_legal():
    # 2x2 loop: head (5,5), body (5,6)... tail at (6,5); moving DOWN enters tail cell vacated this step
    body = [(5, 5), (5, 6), (6, 6), (6, 5)]
    s = make(body, LEFT, (0, 0))
    s.direction = LEFT
    s.step(DOWN)
    assert not s.game_over
    assert s.body[0] == (6, 5)


def test_tail_chase_while_eating_is_collision():
    body = [(5, 5), (5, 6), (6, 6), (6, 5)]
    s = make(body, LEFT, (6, 5))
    s.step(DOWN)
    assert s.game_over


def test_board_full():
    # 2x2 board: snake of 3 + food in the last cell; eating fills the board
    s = Snake(seed=0, n=2)
    s.body = [(0, 1), (0, 0), (1, 0)]
    s.direction = DOWN
    s.food = (1, 1)
    assert s.step(DOWN) is True
    assert s.game_over and s.won and s.food is None
    assert len(s.body) == 4
