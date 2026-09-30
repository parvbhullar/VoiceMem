/* Compare panel rendering: the labels and the "injected into" badge must follow
   the server's per-arm memory flag, never the local checkbox.
 *
 * Runs the real functions, pulled out of web/supermem.html, against a stub DOM --
 * so it needs no browser. Run: node tests/test_compare_ui.mjs
 */
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import assert from 'assert';

const html = readFileSync(join(dirname(fileURLToPath(import.meta.url)),
                               '..', 'web', 'supermem.html'), 'utf8');

function slice(startMarker, endMarker) {
  const a = html.indexOf(startMarker);
  assert.ok(a > -1, `not found in supermem.html: ${startMarker}`);
  const b = html.indexOf(endMarker, a);
  assert.ok(b > -1, `end not found: ${endMarker}`);
  return html.slice(a, b);
}

// The real source under test.
const src = slice('const CMP={ on:false', 'async function cmpLoad(');

// ── stub DOM ──────────────────────────────────────────────────────────────────
const EN = { cmpWith: 'With SuperMem', cmpWithout: 'Without SuperMem',
             cmpWithCart: 'With SuperMem · KV cartridge',
             cmpWaiting: 'waiting for this turn…', cmpCtxEmpty: 'Nothing retrieved',
             cmpCtxNobody: 'neither arm has memory on' };
const nodes = {};
function el(id) {
  const n = { id, textContent: '', className: '', checked: false, value: '',
              hidden: false, placeholder: '',
              classList: { _s: new Set(),
                toggle(c, on) { on ? this._s.add(c) : this._s.delete(c); },
                contains(c) { return this._s.has(c); } },
              _lab: null,
              querySelector(sel) { if (sel === '.lab') return this._lab; return null; } };
  nodes[id] = n; return n;
}
for (const u of ['A', 'B']) {
  el('cmpTxt' + u); el('cmpMs' + u); el('cmpMem' + u); el('cmpCart' + u); el('cmpModel' + u);
  el('cmpUrl' + u); el('cmpKey' + u); el('cmpAdv' + u); el('cmpGear' + u);
  const col = el('cmpCol' + u);
  col._lab = { textContent: '' };
}
el('cmpCtx'); el('cmpCtxN'); el('cmpTg'); el('cmp'); el('reply');

const ctx = {
  $: (id) => nodes[id],
  i18n: (k) => EN[k] ?? k,
  toast: () => {},
  fetch: async () => ({ ok: true, json: async () => ({}) }),
  document: { querySelector: (s) => nodes[s.replace('.', '')] || el('x' + s),
              activeElement: null },
};
ctx.window = ctx;

// eval the real code in that context
const fn = new Function('$', 'i18n', 'toast', 'fetch', 'document',
                        src + '\nreturn {CMP, renderCmp, cmpReset, cmpApply, kvGrid, kvPer, latBar};');
const { CMP, renderCmp, cmpReset, cmpApply, kvGrid, kvPer, latBar } =
  fn(ctx.$, ctx.i18n, ctx.toast, ctx.fetch, ctx.document);

const labelOf = (u) => nodes['cmpCol' + u]._lab.textContent;
const isMemStyled = (u) => nodes['cmpCol' + u].classList.contains('mem');

// Mirrors what web/supermem.html does in the ws 'cmp_start' case.
function serverStart(panel, memory) {
  const st = CMP[panel];
  st.text = ''; st.err = ''; st.ms = 0; st.memory = !!memory;
  nodes['cmpMem' + panel.toUpperCase()].checked = !!memory;
  renderCmp();
}

let pass = 0;
function t(name, f) {
  f(); pass++; console.log('  ok  ' + name);
}

console.log('compare panel rendering');

t('default: A is the memory arm, B is the baseline', () => {
  renderCmp();
  assert.strictEqual(labelOf('A'), 'With SuperMem');
  assert.strictEqual(labelOf('B'), 'Without SuperMem');
  assert.strictEqual(isMemStyled('A'), true);
  assert.strictEqual(isMemStyled('B'), false);
});

t('server flipping the arms flips the labels too', () => {
  serverStart('a', false);
  serverStart('b', true);
  assert.strictEqual(labelOf('A'), 'Without SuperMem');
  assert.strictEqual(labelOf('B'), 'With SuperMem');
  assert.strictEqual(isMemStyled('A'), false);
  assert.strictEqual(isMemStyled('B'), true);
});

t('the badge names the arm the server actually gave the context to', () => {
  CMP.ctx = 'x'.repeat(212);
  serverStart('a', false);
  serverStart('b', true);
  assert.strictEqual(nodes.cmpCtxN.textContent, '212 chars → B');
});

t('a stale checkbox cannot make the badge lie', () => {
  // This was the bug: the page said "→ A" because A's box was ticked, while the
  // server had handed the memory to B.
  CMP.ctx = 'y'.repeat(100);
  nodes.cmpMemA.checked = true;          // stale UI state
  nodes.cmpMemB.checked = false;
  serverStart('a', false);               // server: A has no memory
  serverStart('b', true);                // server: B has it
  assert.strictEqual(nodes.cmpCtxN.textContent, '100 chars → B');
  assert.strictEqual(nodes.cmpMemA.checked, false, 'checkbox corrected to server truth');
  assert.strictEqual(nodes.cmpMemB.checked, true);
});

t('both arms on names both', () => {
  CMP.ctx = 'z'.repeat(50);
  serverStart('a', true); serverStart('b', true);
  assert.strictEqual(nodes.cmpCtxN.textContent, '50 chars → A+B');
});

t('neither arm on says so instead of naming a panel', () => {
  CMP.ctx = 'z'.repeat(50);
  serverStart('a', false); serverStart('b', false);
  assert.strictEqual(nodes.cmpCtxN.textContent, '50 chars → neither arm has memory on');
});

t('cmpReset keeps the arm identity but clears the turn', () => {
  serverStart('a', true); serverStart('b', false);
  CMP.a.text = 'old reply'; CMP.a.ms = 900;
  cmpReset();
  assert.strictEqual(CMP.a.text, '');
  assert.strictEqual(CMP.a.ms, 0);
  assert.strictEqual(CMP.a.memory, true, 'identity survives the reset');
  assert.strictEqual(CMP.b.memory, false);
});

t('cmpApply from GET/POST also updates the arm identity', () => {
  cmpApply({ enabled: true, arms: [{ label: 'a', memory: false, model: '', base_url: '' },
                                   { label: 'b', memory: true, model: '', base_url: '' }] });
  assert.strictEqual(CMP.a.memory, false);
  assert.strictEqual(CMP.b.memory, true);
  assert.strictEqual(labelOf('A'), 'Without SuperMem');
  assert.strictEqual(labelOf('B'), 'With SuperMem');
});

t('a cartridge arm is titled as one, but only while it has memory', () => {
  cmpApply({ enabled: true, arms: [{ label: 'a', memory: true, cartridge: true, model: '', base_url: '' },
                                   { label: 'b', memory: false, cartridge: true, model: '', base_url: '' }] });
  assert.strictEqual(labelOf('A'), 'With SuperMem · KV cartridge');
  assert.strictEqual(labelOf('B'), 'Without SuperMem', 'no memory -> no cartridge to carry');
  assert.strictEqual(CMP.b.cartridge, false);
  assert.strictEqual(nodes.cmpCartA.checked, true);
});

t('KV grid: one cell per block, reused prefix blue, unknown reuse outlined', () => {
  const cls = (h) => [...h.matchAll(/<i class="([^"]*)"/g)].map(m => m[1]);
  const txt = (h) => h.replace(/<[^>]+>/g, '');
  // The screenshot turn: 1,920 prompt, 1,024 reused, 16 tok per cell -> 64 hit + 56 prefilled.
  const g = kvGrid({ prompt_tokens: 1920, cached_tokens: 1024 }, 16);
  const c = cls(g);
  assert.strictEqual(c.length, 120);
  assert.strictEqual(c.filter(x => x === 'hit').length, 64);
  assert.strictEqual(c.filter(x => x === '').length, 56);
  assert.ok(txt(g).endsWith('1,024 / 1,920 reused · 896 prefilled (46.7%)'));
  assert.ok(g.includes('1 cell = 16 tok'), 'scale is on the grid tooltip');
  // A block only partly served from cache is its own shade, not rounded either way.
  assert.deepStrictEqual(cls(kvGrid({ prompt_tokens: 40, cached_tokens: 24 }, 16)), ['hit', 'part', '']);
  const u = kvGrid({ prompt_tokens: 40, cached_tokens: null }, 16);
  assert.deepStrictEqual(cls(u), ['unk', 'unk', 'unk'], 'unreported reuse is unknown, not zero');
  assert.ok(u.includes('reuse not reported'));
  assert.strictEqual(kvGrid({ prompt_tokens: 0, cached_tokens: 0 }, 16), null);
  assert.strictEqual(kvGrid(null, 16), null);
});

t('KV grid scale: 16-token blocks, coarser only when the prompt would overflow', () => {
  assert.strictEqual(kvPer(0), 16);
  assert.strictEqual(kvPer(1920), 16);
  assert.strictEqual(kvPer(15280), 96);
  assert.ok(Math.ceil(15280 / kvPer(15280)) <= 168);
});

t('latency bars share one scale; a missing first token is not drawn as instant', () => {
  const w = (h) => [...h.matchAll(/width:([\d.]+)%/g)].map(m => +m[1]);
  const max = 1674;
  assert.deepStrictEqual(w(latBar({ ttfb: 1189, ms: 1674 }, max)), [71.0, 29.0]);
  const b = latBar({ ttfb: 924, ms: 1040 }, max);
  assert.deepStrictEqual(w(b), [55.2, 6.9]);
  assert.ok(b.includes('<b>924 ms</b> first token · 1040 ms total'));
  assert.deepStrictEqual(w(latBar({ ttfb: 924, ms: null }, max)), [55.2, 0], 'mid-stream: TTFT only');
  assert.ok(latBar({ ttfb: null, ms: 800 }, max).includes('no first token · 800 ms total'));
  assert.strictEqual(latBar({ ttfb: null, ms: null }, max), null);
});

console.log(`\n${pass} passed`);
