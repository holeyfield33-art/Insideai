/* Regression tests exercise the actual Zustand event reducer, including old
 * recordings. TypeScript is already a project dependency; no test runner added. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const root = path.resolve(__dirname, '..');
const temp = fs.mkdtempSync(path.join(root, 'node_modules', '.telemetry-test-'));
try {
  for (const name of ['palette', 'store']) {
    const source = fs.readFileSync(path.join(root, 'src/lib', name + '.ts'), 'utf8');
    const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText.replace('require("./palette")', 'require("./palette.cjs")');
    fs.writeFileSync(path.join(temp, name + '.cjs'), compiled);
  }
  const { useSimStore: store } = require(path.join(temp, 'store.cjs'));
  const apply = store.getState().applyEvent;
  const anomaly = (step, layer, extra = {}) => ({ type:'anomaly', step, layer, zeta_proxy:.72, flagged:false, source:'unitarity-lab', ...extra });
  for (let layer=0; layer<24; layer++) apply(anomaly(0, layer));
  assert.equal(store.getState().anomalyHistory.length, 1, '24-layer forward pass must produce ONE time point');
  assert.equal(store.getState().anomalyLatest.filter(Boolean).length, 24, 'keep layer coverage');
  assert.equal(store.getState().anomalyHistory[0].spectralGap, null, 'old recordings must not invent spectral gap');
  for (let layer=0; layer<24; layer++) apply(anomaly(1, layer, {spectral_gap:.18,calibrated:true,threshold:.3,calibration_scope:'backend_session'}));
  assert.deepEqual(store.getState().anomalyHistory.map(p=>p.step), [0,1]);
  assert.equal(store.getState().anomalyHistory[1].spectralGap, .18);
  assert.equal(store.getState().anomalyHistory[1].threshold, .3);
  apply(anomaly(0,0));
  assert.deepEqual(store.getState().anomalyHistory.map(p=>p.step), [0,1], 'late frame must not move time backward');
  apply(anomaly(2,0,{zeta_proxy:null,spectral_gap:null,flagged:null}));
  assert.equal(store.getState().anomalyHistory.at(-1).zetaProxy, null);
  apply(anomaly(3,0,{zeta_proxy:NaN,spectral_gap:Infinity}));
  assert.equal(store.getState().anomalyHistory.at(-1).spectralGap, null);
  for(let step=4;step<605;step++) apply(anomaly(step,0));
  assert.equal(store.getState().anomalyHistory.length,600);
  store.getState().setReplaying(true);
  apply({type:'generation_start',prompt:'test',prompt_tokens:1,truncated:false,template:false,speed:0,params:{max_new_tokens:3,temperature:0,top_k:0,top_p:1,repetition_penalty:1,seed:42,strategy:'greedy'}});
  assert.equal(store.getState().anomalyHistory.length,0,'new live or replay run must clear prior history');
  assert.equal(store.getState().replaying,true,'recorded run must keep its replay label');
  const camera = store.getState().focusNonce;
  store.getState().setCinematic(true);
  store.getState().setFocusZone('tower');
  assert.equal(store.getState().cinematic,false);
  assert.ok(store.getState().focusNonce>camera);
  console.log('PASS: step deduplication, old replay compatibility, null signals, bounds, reset, replay label, camera navigation.');
} finally { fs.rmSync(temp,{recursive:true,force:true}); }
