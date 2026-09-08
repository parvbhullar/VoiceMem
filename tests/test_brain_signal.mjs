/* Brain signal cascade: the animation must trace the real pipeline, staged.
 *
 * Loads the actual VMBrain module out of web/voicemem.html against a stub
 * canvas/DOM, then inspects the scheduled beams. Run: node tests/test_brain_signal.mjs
 */
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import assert from 'assert';

const html = readFileSync(join(dirname(fileURLToPath(import.meta.url)),
                               '..', 'web', 'voicemem.html'), 'utf8');
const a = html.indexOf('window.VMBrain = (function(){');
assert.ok(a > -1, 'VMBrain module not found');
const b = html.indexOf('\n})();', a);
assert.ok(b > -1, 'VMBrain module end not found');
const src = html.slice(a, b + 6);

// ── stubs: enough canvas/DOM for the module to construct ─────────────────────
const ctx2d = new Proxy({}, { get: (_, k) =>
  k === 'canvas' ? {} : (k === 'createRadialGradient'
    ? () => ({ addColorStop() {} })
    : () => {}) });
function stubEl(tag) {
  return { tagName: tag, style: {}, classList: { add(){}, remove(){}, toggle(){}, contains(){return false;} },
           width: 900, height: 500, complete: true, naturalWidth: 1672,
           getContext: () => ctx2d,
           getBoundingClientRect: () => ({ width: 900, height: 500, left: 0, top: 0 }),
           addEventListener(){}, appendChild(){}, querySelector(){ return null; } };
}
const byId = { brainwrap: stubEl('div'), brain: stubEl('img'), fx: stubEl('canvas'),
               card: stubEl('div'), 'c-sw': stubEl('span'), 'c-kd': stubEl('span'),
               'c-id': stubEl('span'), 'c-body': stubEl('div'), 'c-src': stubEl('span') };
const win = {
  VMBrain: null,
  devicePixelRatio: 1,
  matchMedia: () => ({ matches: false }),
  ResizeObserver: class { observe(){} },
  requestAnimationFrame: () => 0,
  performance: { now: () => 0 },
  addEventListener(){},
};
const document = { getElementById: (id) => byId[id] || stubEl('div'),
                   createElement: stubEl, addEventListener(){} };

new Function('window', 'document', 'requestAnimationFrame', 'performance',
             'matchMedia', 'ResizeObserver', 'i18n', src)(
  win, document, win.requestAnimationFrame, win.performance,
  win.matchMedia, win.ResizeObserver, (k) => k);

const B = win.VMBrain;
assert.ok(B && B.cascade, 'VMBrain.cascade missing');

// Build a small graph: left facts under two clusters, one right-brain node.
B.reset();
const w1 = B.add('L', 'entity', 'works at Unpod', 'work', {});
const w2 = B.add('L', 'entity', 'salary jump',    'work', {});
const h1 = B.add('L', 'entity', 'knee pain',      'health', {});
const r1 = B.add('R', 'emotion', 'anxious',       'emotion', {});
const dump = () => B.__beams();

let pass = 0;
const t = (name, f) => { f(); pass++; console.log('  ok  ' + name); };

console.log('brain signal cascade');

t('a turn with slots fires you → hub → memory, staged', () => {
  B.clear();
  B.cascade({ slots: ['work'], left: [w1, w2], right: [] });
  const beams = dump();
  const hub = B.clusterNode('work');
  assert.ok(hub >= 0, 'work cluster has a label node');

  const first = beams.filter(x => x.hue === 'in');
  const later = beams.filter(x => x.hue === 'recall');
  assert.ok(first.length, 'input leg fired');
  assert.ok(later.length, 'recall leg fired');
  assert.deepStrictEqual([...new Set(first.map(x => x.b))], [hub],
                         'input leg lands on the cluster hub');
  assert.ok(later.every(x => x.a === hub), 'recall leg starts at the hub');
  assert.ok(Math.min(...later.map(x => x.t)) < Math.min(...first.map(x => x.t)),
            'recall starts later than input (more negative t = longer delay)');
});

t('right-brain hits fire after the left ones', () => {
  B.clear();
  B.cascade({ slots: ['work'], left: [w1], right: [r1] });
  const beams = dump();
  const toRight = beams.filter(x => x.b === r1);
  const toLeft  = beams.filter(x => x.b === w1);
  assert.ok(toRight.length && toLeft.length);
  assert.ok(toRight[0].t < toLeft[0].t, 'right hemisphere is the later stage');
});

t('with no slots it still reaches the hits, straight from you', () => {
  B.clear();
  B.cascade({ slots: [], left: [h1], right: [] });
  const beams = dump();
  assert.ok(beams.length, 'typed turns classify no slot but must still animate');
  assert.ok(beams.every(x => x.a === B.userNode()), 'starts at you');
});

t('a turn that matched nothing schedules no beams but still sparks', () => {
  B.clear();
  B.cascade({ slots: [], left: [], right: [] });
  assert.strictEqual(dump().length, 0, 'nothing to travel to');
  assert.ok(B.userLit(), 'you still pulses, so "it is thinking" stays visible');
});

t('the answer leg flows back into you', () => {
  B.clear();
  B.answer([w1, h1]);
  const beams = dump();
  assert.ok(beams.length);
  assert.ok(beams.every(x => x.b === B.userNode()), 'all legs end at you');
  assert.ok(beams.every(x => x.hue === 'out'), 'answer leg has its own colour');
});

t('each leg carries its own clock, so stages cannot share one timeline', () => {
  B.clear();
  B.cascade({ slots: ['work'], left: [w1], right: [r1] });
  const ts = new Set(dump().map(x => x.t));
  assert.ok(ts.size > 1, 'staged beams must not all start at the same t');
});

t('fan-out is capped so a big hit set cannot white out the canvas', () => {
  B.clear();
  const many = [w1, w2, h1, r1, w1, w2, h1, r1];
  B.cascade({ slots: [], left: many, right: [] });
  assert.ok(dump().length <= 4, 'at most 4 targets per source');
});

console.log(`\n${pass} passed`);
