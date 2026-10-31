// web/render.js honesty rule, tested with a recording stub canvas context (no browser):
// a frame is painted ONLY from the cell classes of the frame it is given (including the model's direction classes for head, body and tail);
// nothing is carried from one frame to the next, nothing is interpolated, nothing is inferred.
// Run: node --test tests/js/render.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';

const calls = [];   // every call and property set on the MAIN context
function makeCtx(log) {
  const gradient = { addColorStop: (...a) => log.push(['addColorStop', ...a]) };
  return new Proxy({}, {
    get(_, prop) {
      if (prop === 'createLinearGradient') return (...a) => { log.push([prop, ...a]); return gradient; };
      return (...a) => log.push([String(prop), ...a]);
    },
    set(_, prop, value) { log.push(['set', String(prop), String(value)]); return true; },
  });
}
// the static tile layer is built through document.createElement('canvas') when OffscreenCanvas does not exist (Node)
globalThis.document = { createElement: () => ({ width: 0, height: 0, getContext: () => makeCtx([]) }) };
const { drawBoard, placeOrbs } = await import('../../web/render.js');

const SIZE = 480, S = SIZE / 12, HALF = 0.36 * S;
const frame = (cells) => { const g = new Int32Array(144); for (const [r, c, cls] of cells) g[r * 12 + c] = cls; return g; };
const draw = (grid, diff = null) => { const log = []; drawBoard(makeCtx(log), grid, SIZE, diff); return log; };
const names = (log) => log.map((x) => x[0]);

test('an empty frame paints only the static tiles: no snake, no fill at all', () => {
  const log = draw(frame([]));
  assert.deepEqual(names(log).filter((n) => n === 'fill' || n === 'stroke'), []);
  assert.ok(names(log).includes('drawImage'));
});

test('food cells draw nothing on the canvas (they are orbs) and placeOrbs makes one orb per food cell', () => {
  const g = frame([[2, 3, 1], [9, 9, 1], [5, 5, 1]]);
  assert.deepEqual(names(draw(g)).filter((n) => n === 'fill'), []);
  const container = { children: [], ownerDocument: { createElement: () => ({ style: {}, className: '' }) }, appendChild(e) { this.children.push(e); }, removeChild(e) { this.children.splice(this.children.indexOf(e), 1); }, get lastChild() { return this.children[this.children.length - 1]; } };
  placeOrbs(container, g);
  assert.equal(container.children.length, 3);
  assert.deepEqual(container.children.map((o) => [o.style.left, o.style.top]), [['25%', '16.666666666666668%'], ['41.666666666666664%', '41.666666666666664%'], ['75%', '75%']]);
});

test('placeOrbs reuses the elements (so the CSS pulse keeps running) and sets positions directly: no transition, no interpolation', () => {
  const made = [];
  const container = { children: [], ownerDocument: { createElement: () => { const o = { style: {}, className: '' }; made.push(o); return o; } }, appendChild(e) { this.children.push(e); }, removeChild(e) { this.children.splice(this.children.indexOf(e), 1); }, get lastChild() { return this.children[this.children.length - 1]; } };
  placeOrbs(container, frame([[1, 1, 1], [2, 2, 1]]));
  const before = [...container.children];
  placeOrbs(container, frame([[8, 8, 1], [9, 9, 1]]));
  assert.deepEqual(container.children, before, 'same elements');
  assert.equal(made.length, 2);
  assert.equal(container.children[0].style.left, `${8 * 100 / 12}%`);
  assert.ok(container.children.every((o) => !('transition' in o.style) && !('animation' in o.style)));
  placeOrbs(container, frame([]));
  assert.equal(container.children.length, 0);
});

test('no state is carried between frames: drawing B after A gives exactly the calls of drawing B on its own', () => {
  const A = frame([[5, 5, 5], [5, 4, 9], [5, 3, 9], [5, 2, 12]]);
  const B = frame([[2, 7, 3], [3, 7, 6], [4, 7, 10]]);
  const ctx = makeCtx([]); drawBoard(ctx, A, SIZE);       // first frame goes to a different context
  const afterA = draw(B), alone = draw(B);
  assert.deepEqual(afterA, alone);
  assert.deepEqual(draw(A), draw(A), 'same frame, same drawing');
  assert.notDeepEqual(draw(A), draw(B));
});

test('the body joins toward the neighbour that the cell\'s own direction class points at', () => {
  // a body cell with class 6+d is drawn as a pill from its centre to the centre of the neighbour in direction d
  const cx = 5.5 * S, cy = 5.5 * S;
  const pills = (cls) => draw(frame([[5, 5, cls]])).filter((x) => x[0] === 'roundRect');
  const [, x, y, w, h] = pills(6 + 3)[0];   // RIGHT
  assert.ok(Math.abs(x - (cx - HALF)) < 1e-9 && Math.abs(w - (S + 2 * HALF)) < 1e-9 && Math.abs(y - (cy - HALF)) < 1e-9 && Math.abs(h - 2 * HALF) < 1e-9);
  const [, x2, , w2] = pills(6 + 2)[0];     // LEFT
  assert.ok(Math.abs(x2 - (cx - S - HALF)) < 1e-9 && Math.abs(w2 - (S + 2 * HALF)) < 1e-9);
  const [, , y3, , h3] = pills(6 + 0)[0];   // UP
  assert.ok(Math.abs(y3 - (cy - S - HALF)) < 1e-9 && Math.abs(h3 - (S + 2 * HALF)) < 1e-9);
  const [, , y4, , h4] = pills(6 + 1)[0];   // DOWN
  assert.ok(Math.abs(y4 - (cy - HALF)) < 1e-9 && Math.abs(h4 - (S + 2 * HALF)) < 1e-9);
});

test('head and tail differ from the body by SHAPE: the tail adds a taper (bezier curves), the head is a larger block with eyes', () => {
  const count = (log, n) => log.filter((x) => x[0] === n).length;
  const body = draw(frame([[5, 5, 7]])), tail = draw(frame([[5, 5, 11]])), head = draw(frame([[5, 5, 4]]));
  assert.equal(count(body, 'bezierCurveTo'), 0);
  assert.ok(count(tail, 'bezierCurveTo') >= 2, 'tail is tapered');
  assert.equal(count(head, 'bezierCurveTo'), 0);
  assert.equal(count(head, 'arc'), 4, 'two eyes, each a white and a pupil');
  assert.equal(count(body, 'arc') + count(tail, 'arc'), 0);
  const headRect = head.filter((x) => x[0] === 'roundRect').pop(), bodyRect = body.filter((x) => x[0] === 'roundRect')[0];
  assert.ok(Math.min(headRect[3], headRect[4]) > 2 * HALF, 'the head is wider than the body');
  assert.ok(bodyRect[4] > 0);
});

test('the eyes face the model\'s head-direction class (class 2+d: UP, DOWN, LEFT, RIGHT)', () => {
  const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  dirs.forEach(([dr, dc], d) => {
    const arcs = draw(frame([[5, 5, 2 + d]])).filter((x) => x[0] === 'arc');
    assert.equal(arcs.length, 4);
    for (let k = 0; k < 4; k += 2) {
      const white = arcs[k], pupil = arcs[k + 1];
      assert.ok(Math.abs(pupil[1] - white[1] - dc * 0.04 * S) < 1e-9 && Math.abs(pupil[2] - white[2] - dr * 0.04 * S) < 1e-9, `pupil shifted along direction ${d}`);
    }
    const eyesCentreX = (arcs[0][1] + arcs[2][1]) / 2, eyesCentreY = (arcs[0][2] + arcs[2][2]) / 2;
    assert.ok(Math.abs(eyesCentreX - (5.5 * S + dc * 0.15 * S)) < 1e-9 && Math.abs(eyesCentreY - (5.5 * S + dr * 0.15 * S)) < 1e-9, 'eyes set forward');
  });
});

test('the tail taper points away from the neighbour its class names (tail class 10+d)', () => {
  const tipX = (cls) => { const bz = draw(frame([[5, 5, cls]])).filter((x) => x[0] === 'bezierCurveTo'); return [bz[0][5], bz[0][6]]; };
  const cx = 5.5 * S, cy = 5.5 * S;
  assert.ok(Math.abs(tipX(10 + 3)[0] - (cx - 0.46 * S)) < 1e-9, 'RIGHT: tip on the left');
  assert.ok(Math.abs(tipX(10 + 2)[0] - (cx + 0.46 * S)) < 1e-9, 'LEFT: tip on the right');
  assert.ok(Math.abs(tipX(10 + 0)[1] - (cy + 0.46 * S)) < 1e-9, 'UP: tip below');
  assert.ok(Math.abs(tipX(10 + 1)[1] - (cy - 0.46 * S)) < 1e-9, 'DOWN: tip above');
});

test('a model error is drawn as produced: a body cell whose class points off the board still draws its own stub, and a snake with no head still draws', () => {
  assert.doesNotThrow(() => draw(frame([[0, 0, 6 + 0]])));
  const headless = draw(frame([[3, 3, 8], [3, 4, 7]]));
  assert.ok(names(headless).includes('fill'));
  assert.equal(names(headless).filter((n) => n === 'arc').length, 0, 'no head, so no eyes');
});

test('red diff outlines are drawn only for flagged cells', () => {
  const diff = new Uint8Array(144); diff[13] = 1; diff[100] = 1;
  const withDiff = draw(frame([]), diff).filter((x) => x[0] === 'stroke').length, without = draw(frame([])).filter((x) => x[0] === 'stroke').length;
  assert.equal(withDiff - without, 2);
});
