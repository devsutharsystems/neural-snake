// Hand-written Snake engine: a line-by-line port of engine/snake.py. Used ONLY by compare.html and the golden-trace test.
// The main page (index.html / main.js) must never import this file (scripts/audit_web.py enforces it).
// Food spawn is injected: step(action, spawn) calls spawn(emptyCells) -> [row, col] when food is eaten, so tests can replay Python's spawn positions.
export const UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3, NOOP = 4;
const DELTA = [[-1, 0], [1, 0], [0, -1], [0, 1]];
const OPPOSITE = [DOWN, UP, RIGHT, LEFT];

export class Engine {
  // body[0] = head, body[last] = tail; cells are [row, col]; row 0 is the top.
  constructor({ n = 12, body, direction = RIGHT, food = null }) {
    this.n = n;
    this.body = body.map((c) => [c[0], c[1]]);
    this.direction = direction;
    this.food = food ? [food[0], food[1]] : null;
    this.gameOver = false;
    this.won = false;
    this.score = 0;
  }

  // Row-major list of cells not occupied by the body (same order as engine/snake.py _spawn_food).
  emptyCells() {
    const occ = new Set(this.body.map(([r, c]) => r * this.n + c));
    const out = [];
    for (let r = 0; r < this.n; r++) for (let c = 0; c < this.n; c++) if (!occ.has(r * this.n + c)) out.push([r, c]);
    return out;
  }

  // Default spawn: uniform over the empty cells with Math.random.
  static randomSpawn(empty) {
    return empty[Math.floor(Math.random() * empty.length)];
  }

  // Advance one tick; returns true if food was eaten. Rules and order of checks are exactly those of engine/snake.py Snake.step.
  step(action, spawn = Engine.randomSpawn) {
    if (this.gameOver) return false;
    if (action >= 0 && action <= 3 && !(this.body.length > 1 && action === OPPOSITE[this.direction])) this.direction = action;
    const [dr, dc] = DELTA[this.direction];
    const [hr, hc] = this.body[0];
    const nr = hr + dr, nc = hc + dc;
    if (nr < 0 || nr >= this.n || nc < 0 || nc >= this.n) { this.gameOver = true; return false; }
    const eating = this.food !== null && nr === this.food[0] && nc === this.food[1];
    const blocking = eating ? this.body : this.body.slice(0, -1); // the tail cell is vacated this step unless we grow
    if (blocking.some(([r, c]) => r === nr && c === nc)) { this.gameOver = true; return false; }
    this.body.unshift([nr, nc]);
    if (eating) {
      this.score += 1;
      const empty = this.emptyCells();
      this.food = empty.length ? spawn(empty) : null;
      if (this.food === null) { this.gameOver = true; this.won = true; }
    } else {
      this.body.pop();
    }
    return eating;
  }

  // 14-class grid, row-major Int32Array(n*n): 0 empty, 1 food, 2-5 head (direction of travel U,D,L,R), 6-9 body, 10-13 tail
  // (body/tail class = 6 or 10 + direction from that segment to the next segment toward the head). Port of spikes/spike_a/common.py grid14.
  grid14() {
    const n = this.n, g = new Int32Array(n * n);
    if (this.food) g[this.food[0] * n + this.food[1]] = 1;
    const L = this.body.length;
    this.body.forEach(([r, c], i) => {
      if (i === 0) { g[r * n + c] = 2 + this.direction; return; }
      const [pr, pc] = this.body[i - 1];
      const d = DELTA.findIndex(([a, b]) => a === pr - r && b === pc - c);
      g[r * n + c] = (i === L - 1 ? 10 : 6) + d;
    });
    return g;
  }
}
