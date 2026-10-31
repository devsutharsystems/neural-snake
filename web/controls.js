// Touch, swipe and on-screen-button input, shared by index.html and compare.html (so it must never touch the hand-written engine module).
// It does ONE thing: report an action index (0 UP, 1 DOWN, 2 LEFT, 3 RIGHT) to the callback it is given: the same value an arrow key sets.
// No game logic here: it never looks at the board, the snake or the model's output.
export const SWIPE_MIN_PX = 24;

// purpose: turn a swipe vector (dx, dy in screen pixels) into an action index (0 UP, 1 DOWN, 2 LEFT, 3 RIGHT), or null if the swipe is shorter than 24 px.
export function swipeToAction(dx, dy) {
  if (Math.max(Math.abs(dx), Math.abs(dy)) < SWIPE_MIN_PX) return null;
  if (Math.abs(dx) >= Math.abs(dy)) return dx > 0 ? 3 : 2;
  return dy > 0 ? 1 : 0;
}

// purpose: button press handler: report the pressed button's data-action (0..3) to onAction.
function handlePress(ev, onAction) {
  ev.preventDefault();
  onAction(Number(ev.currentTarget.dataset.action));
}

// purpose: wire the four on-screen arrow buttons so that pressing (or Enter / Space on a focused button) reports their action.
function wireButtons(buttons, onAction) {
  for (const b of buttons) {
    b.addEventListener('pointerdown', (ev) => handlePress(ev, onAction));
    b.addEventListener('click', (ev) => handlePress(ev, onAction));
  }
}

// purpose: wire a swipe on one element: remember where the pointer went down, and on release report the swipe direction (if long enough).
function wireSwipe(target, onAction) {
  let start = null;
  target.addEventListener('pointerdown', (ev) => { start = { x: ev.clientX, y: ev.clientY }; });
  target.addEventListener('pointerup', (ev) => {
    if (!start) return;
    const action = swipeToAction(ev.clientX - start.x, ev.clientY - start.y);
    start = null;
    if (action !== null) onAction(action);
  });
  target.addEventListener('pointercancel', () => { start = null; });
}

// purpose: install the on-screen buttons and the swipe gestures; every input ends in onAction(actionIndex).
export function installControls({ buttons, swipeTargets, onAction }) {
  wireButtons(buttons, onAction);
  for (const t of swipeTargets) wireSwipe(t, onAction);
}
