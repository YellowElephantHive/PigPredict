import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {Decoder, deltaE, search, labToRgb, validateTarget} from '../static/studio/engine.js';
const dataDir = new URL('../static/studio/data/', import.meta.url);
const metadata = JSON.parse(fs.readFileSync(new URL('model.json', dataDir)));
const history = JSON.parse(fs.readFileSync(new URL('recipes.json', dataDir)));
const bytes = fs.readFileSync(new URL('weights.bin', dataDir));
const decoder = new Decoder(metadata, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));

test('analytic dye derivatives match independent central differences', () => {
  const dyes=[7,15,18], concentrations=[.23,.41,.62];
  const state=decoder.forward(decoder.vector(dyes,concentrations,0),true);
  const jac=decoder.jacobian(state.activations,dyes);
  dyes.forEach((_,i)=>{
    const plus=[...concentrations],minus=[...concentrations],epsilon=1e-6;
    plus[i]+=epsilon;minus[i]-=epsilon;
    const a=decoder.predict(dyes,plus,0),b=decoder.predict(dyes,minus,0);
    a.forEach((v,k)=>assert.ok(Math.abs((v-b[k])/(2*epsilon)-jac[k][i])<1e-4));
  });
});
test('inverse search is deterministic, bounded, sorted and fabric-specific',()=>{
  for(let fabric=0;fabric<3;fabric++){
    const row=history.find(r=>r.fabric===fabric&&r.dyes.length===3);
    const results=search(decoder,history,row.lab,fabric);
    assert.deepEqual(results,search(decoder,history,row.lab,fabric));
    assert.ok(results.length>1&&results.length<=8);
    results.forEach((r,i)=>{
      assert.ok(r.concentrations.every(v=>v>=0&&v<=5));
      assert.ok(history.some(h=>h.fabric===fabric&&h.dyes.join(',')===r.dyes.join(',')));
      assert.ok(Math.abs(deltaE(decoder.predict(r.dyes,r.concentrations,fabric),row.lab)-r.delta)<1e-10);
      assert.ok(r.trace.every((v,j)=>j===0||v<=r.trace[j-1]));
      if(i)assert.ok(r.delta>=results[i-1].delta);
    });
  }
});
test('invalid targets and recipes fail instead of producing a plausible color',()=>{
  for(const lab of [[NaN,0,0],[101,0,0],[50,129,0],[],[50,0,Infinity]]) assert.throws(()=>validateTarget(lab,0,metadata));
  assert.throws(()=>validateTarget([50,0,0],3,metadata));
  for(const c of [[-1],[6],[NaN],[]])assert.throws(()=>decoder.predict([0],c,0));
  assert.throws(()=>decoder.predict([0,0],[1,1],0));
  assert.throws(()=>decoder.predict([44],[1],0));
  assert.throws(()=>search(decoder,[],[50,0,0],0));
});
test('display conversion produces neutral black and white',()=>{
  assert.deepEqual(labToRgb([0,0,0]),[0,0,0]);
  assert.deepEqual(labToRgb([100,0,0]),[255,255,255]);
});
