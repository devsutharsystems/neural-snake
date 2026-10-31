// web/controls.js reports only an action index 0..3 (UP, DOWN, LEFT, RIGHT) to a callback: the same value an arrow key sets.
// Tested with stub elements (no browser): button presses, swipes, cancelled swipes, short swipes.
// Run: node --test tests/js/controls.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { swipeToAction, installControls, SWIPE_MIN_PX } from '../../web/controls.js';

const stub = (dataset = {}) => {
  const handlers = {};
  return { dataset, handlers, addEventListener(type, fn) { handlers[type] = fn; } };
};

test('swipe direction follows the dominant axis (UP 0, DOWN 1, LEFT 2, RIGHT 3)', () => {
  assert.equal(swipeToAction(40, 5), 3);
  assert.equal(swipeToAction(-40, 5), 2);
  assert.equal(swipeToAction(3, 40), 1);
  assert.equal(swipeToAction(3, -40), 0);
  assert.equal(swipeToAction(30, -31), 0, 'vertical wins when |dy| > |dx|');
  assert.equal(swipeToAction(30, 30), 3, 'a tie goes to the horizontal axis');
});

test('a swipe shorter than the minimum is ignored', () => {
  assert.equal(SWIPE_MIN_PX, 24);
  assert.equal(swipeToAction(0, 0), null);
  assert.equal(swipeToAction(23, 23), null);
  assert.equal(swipeToAction(24, 0), 3);
});

test('swipeToAction only ever returns null or 0..3', () => {
  for (let dx = -60; dx <= 60; dx += 7) for (let dy = -60; dy <= 60; dy += 7) {
    const a = swipeToAction(dx, dy);
    assert.ok(a === null || [0, 1, 2, 3].includes(a), `(${dx},${dy}) -> ${a}`);
  }
});

test('each on-screen button reports its data-action on pointerdown and on click, and stops the default action', () => {
  const buttons = ['0', '1', '2', '3'].map((a) => stub({ action: a }));
  const got = [];
  installControls({ buttons, swipeTargets: [], onAction: (a) => got.push(a) });
  let prevented = 0;
  for (const b of buttons) {
    b.handlers.pointerdown({ preventDefault() { prevented++; }, currentTarget: b });
    b.handlers.click({ preventDefault() { prevented++; }, currentTarget: b });
  }
  assert.deepEqual(got, [0, 0, 1, 1, 2, 2, 3, 3]);
  assert.equal(prevented, 8);
});

test('swipe on a target: down then up reports a direction; stray up, cancelled and short swipes report nothing', () => {
  const t = stub();
  const got = [];
  installControls({ buttons: [], swipeTargets: [t], onAction: (a) => got.push(a) });
  t.handlers.pointerdown({ clientX: 100, clientY: 100 });
  t.handlers.pointerup({ clientX: 100, clientY: 60 });
  assert.deepEqual(got, [0]);
  t.handlers.pointerup({ clientX: 300, clientY: 100 });                         // no pointerdown before it
  t.handlers.pointerdown({ clientX: 100, clientY: 100 });
  t.handlers.pointercancel();
  t.handlers.pointerup({ clientX: 160, clientY: 100 });                         // swipe was cancelled
  t.handlers.pointerdown({ clientX: 100, clientY: 100 });
  t.handlers.pointerup({ clientX: 110, clientY: 105 });                         // too short (a tap)
  assert.deepEqual(got, [0]);
  t.handlers.pointerdown({ clientX: 50, clientY: 50 });
  t.handlers.pointerup({ clientX: 10, clientY: 55 });
  assert.deepEqual(got, [0, 2]);
});

test('several swipe targets (compare page: both boards) all report through the same callback', () => {
  const a = stub(), b = stub();
  const got = [];
  installControls({ buttons: [], swipeTargets: [a, b], onAction: (x) => got.push(x) });
  a.handlers.pointerdown({ clientX: 0, clientY: 0 }); a.handlers.pointerup({ clientX: 50, clientY: 0 });
  b.handlers.pointerdown({ clientX: 0, clientY: 0 }); b.handlers.pointerup({ clientX: 0, clientY: 50 });
  assert.deepEqual(got, [3, 1]);
});
