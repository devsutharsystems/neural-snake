// Board renderer, shared by index.html and compare.html. DRAWING ONLY.
// Honesty rule: a frame is painted from the 14 cell classes of THE FRAME IT IS GIVEN (including the model's own direction classes for head, body and tail):
// 0 empty, 1 food, 2-5 head, 6-9 body, 10-13 tail; direction index 0 UP, 1 DOWN, 2 LEFT, 3 RIGHT.
// No interpolation between frames, no state kept from one frame to the next, no inferring of movement. The only thing cached is a static
// decoration layer (the empty tiles) that does not depend on any frame. It must never touch the hand-written engine module.
const N = 12;
const BASE_PX = 480;
const DR = [-1, 1, 0, 0];
const DC = [0, 0, -1, 1];
let tileCache = null; // { size, layer }: the 144 empty tiles, identical for every frame

// purpose: make the canvas backing store 480 px, or 960 px on high-density screens, so the board is sharp; returns the pixel size.
export function fitCanvas(canvas) {
  const ratio = Math.min(2, Math.max(1, Math.ceil((typeof window !== 'undefined' && window.devicePixelRatio) || 1)));
  canvas.width = BASE_PX * ratio;
  canvas.height = BASE_PX * ratio;
  return canvas.width;
}

// purpose: create an offscreen drawing surface of size x size pixels (OffscreenCanvas when available, otherwise a detached canvas element).
function makeLayer(size) {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(size, size);
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  return c;
}

// purpose: add a rounded rectangle to the current path (roundRect where the browser has it, a plain rectangle otherwise).
function rrect(ctx, x, y, w, h, r) {
  if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
  else ctx.rect(x, y, w, h);
}

// purpose: the layer of 144 soft bevelled empty tiles, built once per canvas size; static decoration that does not depend on any frame.
function tileLayer(size) {
  if (tileCache && tileCache.size === size) return tileCache.layer;
  const layer = makeLayer(size), g = layer.getContext('2d'), s = size / N, inset = 0.05 * s;
  g.fillStyle = '#0b1120';
  g.fillRect(0, 0, size, size);
  for (let i = 0; i < N * N; i++) {
    const x = (i % N) * s + inset, y = Math.floor(i / N) * s + inset, w = s - 2 * inset;
    const fill = g.createLinearGradient(x, y, x + w, y + w);
    fill.addColorStop(0, '#222d45');
    fill.addColorStop(1, '#151c2e');
    g.beginPath();
    rrect(g, x, y, w, w, 0.2 * s);
    g.fillStyle = fill;
    g.fill();
    const edge = g.createLinearGradient(x, y, x + w, y + w);
    edge.addColorStop(0, 'rgba(255,255,255,0.16)');
    edge.addColorStop(0.5, 'rgba(255,255,255,0)');
    edge.addColorStop(1, 'rgba(0,0,0,0.45)');
    g.lineWidth = Math.max(1, 0.035 * s);
    g.strokeStyle = edge;
    g.stroke();
  }
  tileCache = { size, layer };
  return layer;
}

// purpose: from THIS frame's classes, walk from the head along the model's own direction classes (a body or tail cell points at its neighbour toward the head) and return each cell's distance from the head, -1 if not reached; used only to shade the body darker toward the tail.
function chainDepth(grid) {
  const depth = new Int16Array(N * N).fill(-1);
  const incoming = Array.from({ length: N * N }, () => []);
  let head = -1;
  for (let i = 0; i < N * N; i++) {
    const c = grid[i];
    if (c >= 2 && c <= 5 && head < 0) head = i;
    if (c >= 6 && c <= 13) {
      const d = (c - 6) % 4, r = Math.floor(i / N) + DR[d], col = (i % N) + DC[d];
      if (r >= 0 && r < N && col >= 0 && col < N) incoming[r * N + col].push(i);
    }
  }
  if (head < 0) return depth;
  depth[head] = 0;
  const queue = [head];
  for (let k = 0; k < queue.length; k++) for (const j of incoming[queue[k]]) if (depth[j] < 0) { depth[j] = depth[queue[k]] + 1; queue.push(j); }
  return depth;
}

// purpose: colour of a body segment at fraction t (0 next to the head, 1 at the far end): a glossy green that is slightly darker toward the tail.
function bodyColor(t) {
  const k = Math.min(1, Math.max(0, t));
  return `rgb(${Math.round(56 - 46 * k)},${Math.round(224 - 104 * k)},${Math.round(123 - 56 * k)})`;
}

// purpose: path of a pill (a rounded bar) from the centre of cell (r, c) to the centre of its neighbour in direction d: the body joins toward the neighbour that the model's own direction class points at.
function pillPath(ctx, r, c, d, half, s) {
  const cx = (c + 0.5) * s, cy = (r + 0.5) * s, ex = cx + DC[d] * s, ey = cy + DR[d] * s;
  ctx.beginPath();
  rrect(ctx, Math.min(cx, ex) - half, Math.min(cy, ey) - half, Math.abs(ex - cx) + 2 * half, Math.abs(ey - cy) + 2 * half, half);
}

// purpose: add the tapered tail end to the current path: full width where it joins the neighbour (direction d of the tail class), narrowing to a rounded point on the opposite side.
function tailTaper(ctx, r, c, d, half, s) {
  const cx = (c + 0.5) * s, cy = (r + 0.5) * s, ax = DC[d], ay = DR[d], nx = -ay, ny = ax;
  const tx = cx - ax * 0.46 * s, ty = cy - ay * 0.46 * s, q = 0.2 * half;
  ctx.moveTo(cx + nx * half, cy + ny * half);
  ctx.bezierCurveTo(cx + nx * half - ax * 0.22 * s, cy + ny * half - ay * 0.22 * s, tx + nx * q + ax * 0.1 * s, ty + ny * q + ay * 0.1 * s, tx, ty);
  ctx.bezierCurveTo(tx - nx * q + ax * 0.1 * s, ty - ny * q + ay * 0.1 * s, cx - nx * half - ax * 0.22 * s, cy - ny * half - ay * 0.22 * s, cx - nx * half, cy - ny * half);
  ctx.closePath();
}

// purpose: path of the head: a larger, rounder block than the body, stretched forward along the model's head-direction class.
function headPath(ctx, r, c, d, s) {
  const cx = (c + 0.5) * s + DC[d] * 0.03 * s, cy = (r + 0.5) * s + DR[d] * 0.03 * s;
  const w = (DC[d] !== 0 ? 0.96 : 0.88) * s, h = (DR[d] !== 0 ? 0.96 : 0.88) * s;
  ctx.beginPath();
  rrect(ctx, cx - w / 2, cy - h / 2, w, h, 0.3 * s);
}

// purpose: two eyes set forward on the head and looking along the model's head-direction class.
function drawEyes(ctx, r, c, d, s) {
  const cx = (c + 0.5) * s + DC[d] * 0.15 * s, cy = (r + 0.5) * s + DR[d] * 0.15 * s, nx = -DR[d], ny = DC[d];
  for (const side of [-1, 1]) {
    const ex = cx + nx * side * 0.2 * s, ey = cy + ny * side * 0.2 * s;
    ctx.beginPath();
    ctx.arc(ex, ey, 0.105 * s, 0, 2 * Math.PI);
    ctx.fillStyle = '#f4fff8';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(ex + DC[d] * 0.04 * s, ey + DR[d] * 0.04 * s, 0.055 * s, 0, 2 * Math.PI);
    ctx.fillStyle = '#10261a';
    ctx.fill();
  }
}

// purpose: paint the snake of this frame from its classes 2-13 only: a shadow pass, a colour pass from the tail end to the head, a gloss pass, then the head with its eyes on top.
function drawSnake(ctx, grid, s) {
  const depth = chainDepth(grid);
  let maxDepth = 1;
  for (const v of depth) if (v > maxDepth) maxDepth = v;
  const half = 0.36 * s, parts = [];
  for (let i = 0; i < N * N; i++) {
    const c = grid[i];
    if (c < 2 || c > 13) continue;
    parts.push({ r: Math.floor(i / N), c: i % N, kind: c <= 5 ? 'head' : c <= 9 ? 'body' : 'tail', d: c <= 5 ? c - 2 : (c - 6) % 4, t: depth[i] < 0 ? 0.5 : depth[i] / maxDepth });
  }
  if (parts.length === 0) return; // no snake cells in this frame: nothing to paint
  parts.sort((a, b) => b.t - a.t); // far end first, so segments nearer the head overlap them
  // purpose: add the path of one snake part (head block, body pill, or tail pill plus taper) to the context.
  const shape = (p) => {
    if (p.kind === 'head') { headPath(ctx, p.r, p.c, p.d, s); return; }
    pillPath(ctx, p.r, p.c, p.d, half, s);
    if (p.kind === 'tail') tailTaper(ctx, p.r, p.c, p.d, half, s);
  };
  ctx.save();
  ctx.shadowColor = 'rgba(0,0,0,0.6)';
  ctx.shadowBlur = 0.16 * s;
  ctx.shadowOffsetY = 0.08 * s;
  ctx.fillStyle = '#0a5c33';
  for (const p of parts) { shape(p); ctx.fill(); }
  ctx.restore();
  for (const p of parts) {
    if (p.kind === 'head') continue;
    shape(p);
    ctx.fillStyle = bodyColor(p.kind === 'tail' ? 1 : p.t);
    ctx.fill();
  }
  ctx.beginPath(); // gloss: ONE path for every highlight, so overlapping highlights do not add up
  for (const p of parts) {
    if (p.kind === 'head') continue;
    const cx = (p.c + 0.5) * s, cy = (p.r + 0.5) * s, ex = cx + DC[p.d] * s, ey = cy + DR[p.d] * s, hw = 0.13 * s, off = 0.12 * s;
    rrect(ctx, Math.min(cx, ex) - hw - off, Math.min(cy, ey) - hw - off, Math.abs(ex - cx) + 2 * hw, Math.abs(ey - cy) + 2 * hw, hw);
  }
  ctx.fillStyle = 'rgba(255,255,255,0.2)';
  ctx.fill();
  for (const p of parts) { // the head goes on top of the body and its gloss
    if (p.kind !== 'head') continue;
    const x = (p.c + 0.5) * s, y = (p.r + 0.5) * s, g = ctx.createLinearGradient(x - 0.4 * s, y - 0.5 * s, x + 0.3 * s, y + 0.5 * s);
    g.addColorStop(0, '#9dffc4');
    g.addColorStop(1, '#1fbf68');
    shape(p);
    ctx.fillStyle = g;
    ctx.fill();
    drawEyes(ctx, p.r, p.c, p.d, s);
  }
}

// purpose: draw red outlines around the cells flagged in diff (compare page only: where the model's cell differs from the engine's).
function drawDiff(ctx, diff, s) {
  ctx.lineWidth = Math.max(2, 0.07 * s);
  ctx.strokeStyle = '#ff4d5e';
  for (let i = 0; i < N * N; i++) {
    if (!diff[i]) continue;
    ctx.beginPath();
    rrect(ctx, (i % N) * s + 0.06 * s, Math.floor(i / N) * s + 0.06 * s, 0.88 * s, 0.88 * s, 0.2 * s);
    ctx.stroke();
  }
}

// purpose: paint ONE frame: the static tiles, then the snake from this frame's classes, then red outlines for flagged cells (compare page). Food is shown by placeOrbs.
export function drawBoard(ctx, grid, size, diff = null) {
  ctx.clearRect(0, 0, size, size);
  ctx.drawImage(tileLayer(size), 0, 0);
  drawSnake(ctx, grid, size / N);
  if (diff) drawDiff(ctx, diff, size / N);
}

// purpose: show one food orb per food cell (class 1) of THIS frame as an element over the board; existing elements are reused, and their position is set directly (no transition), so nothing is interpolated and the CSS pulse keeps running.
export function placeOrbs(container, grid) {
  const cells = [];
  for (let i = 0; i < N * N; i++) if (grid[i] === 1) cells.push(i);
  while (container.children.length < cells.length) {
    const o = container.ownerDocument.createElement('div');
    o.className = 'orb';
    container.appendChild(o);
  }
  while (container.children.length > cells.length) container.removeChild(container.lastChild);
  cells.forEach((cell, k) => {
    const st = container.children[k].style;
    st.left = ((cell % N) * 100 / N) + '%';
    st.top = (Math.floor(cell / N) * 100 / N) + '%';
  });
}
