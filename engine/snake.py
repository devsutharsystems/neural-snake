"""Deterministic 12x12 Snake engine (reference rules; training-data generator and test oracle).

Conventions
- Cells are (row, col); row 0 is the top. UP = row-1, DOWN = row+1, LEFT = col-1, RIGHT = col+1.
- Actions: UP, DOWN, LEFT, RIGHT, NOOP. NOOP continues in the current direction.
- Reversing into the neck is treated as NOOP (keeps current direction).
- Wall or self collision ends the game. The game-over state is terminal: further steps change nothing.
- Eating grows the snake by 1 (tail does not move that step) and spawns food uniformly
  on an empty cell using the seeded RNG.
- Tail-chase: moving into the cell the tail vacates THIS step is legal, unless the snake
  eats on that step (then the tail stays, so the cell is occupied and it is a collision).
- Board-full: if after eating no empty cell remains, the game ends with food=None and
  won=True (game_over is also True). No food spawn is attempted.
"""
import random

N = 12
UP, DOWN, LEFT, RIGHT, NOOP = 0, 1, 2, 3, 4
ACTIONS = (UP, DOWN, LEFT, RIGHT, NOOP)
DELTA = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


class Snake:
    def __init__(self, seed=0, n=N):
        self.n = n
        self.rng = random.Random(seed)
        mid = n // 2
        self.body = [(mid, mid)]  # body[0] = head, body[-1] = tail
        self.direction = RIGHT
        self.game_over = False
        self.won = False
        self.score = 0
        self.food = None
        self.food = self._spawn_food()

    def _spawn_food(self):
        occupied = set(self.body)
        empty = [(r, c) for r in range(self.n) for c in range(self.n) if (r, c) not in occupied]
        if not empty:
            return None
        return self.rng.choice(empty)

    def step(self, action):
        """Advance one tick. Returns True if food was eaten this step."""
        if self.game_over:
            return False
        if action in DELTA and not (len(self.body) > 1 and action == OPPOSITE[self.direction]):
            self.direction = action
        dr, dc = DELTA[self.direction]
        hr, hc = self.body[0]
        new = (hr + dr, hc + dc)
        if not (0 <= new[0] < self.n and 0 <= new[1] < self.n):
            self.game_over = True
            return False
        eating = new == self.food
        # Tail cell is vacated this step unless we are growing.
        blocking = self.body if eating else self.body[:-1]
        if new in blocking:
            self.game_over = True
            return False
        self.body.insert(0, new)
        if eating:
            self.score += 1
            self.food = self._spawn_food()
            if self.food is None:
                self.game_over = True
                self.won = True
        else:
            self.body.pop()
        return eating

    def grid(self):
        """Return n x n list of ints: 0 empty, 1 food, 2 head, 3 body."""
        g = [[0] * self.n for _ in range(self.n)]
        if self.food is not None:
            g[self.food[0]][self.food[1]] = 1
        for r, c in self.body[1:]:
            g[r][c] = 3
        g[self.body[0][0]][self.body[0][1]] = 2
        return g
