// Run after: python -m scripts.export_public
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {predict,parseCSV,mergeHistory,buildInput,bday} from '../web/browser_engine.mjs';
const bundle=JSON.parse(fs.readFileSync(new URL('../output/huggingface/space/bundle.json',import.meta.url),'utf8'));
const lines=fs.readFileSync(new URL('../artifacts/brent_v1/predictions_test.csv',import.meta.url),'utf8').trim().split(/\r?\n/);
const headers=lines.shift().split(',');let count=0,maximum=0;
for(const line of lines){
 const row=Object.fromEntries(line.split(',').map((v,i)=>[headers[i],v]));const h=Number(row.horizon_bdays);
 if(bundle.models[h].architecture!==row.architecture)continue;
 const got=predict(bundle.history,row.origin_date,h,bundle,'2026-11-20');
 const error=Math.abs(got.predicted_usd-Number(row.predicted_usd));maximum=Math.max(maximum,error);count++;
 assert.ok(error<.005,`Diferencia ${error} en ${row.origin_date}, horizonte ${h}`);
}
assert.equal(count,2737);
for(const demo of JSON.parse(fs.readFileSync(new URL('../reports/demo_app.json',import.meta.url),'utf8'))){
 const got=predict(bundle.history,demo.origin_date,demo.horizon_bdays,bundle,'2026-11-20');
 for(const key of ['last_available_usd','input_imputed_fraction','predicted_usd','range_low_usd','range_high_usd'])assert.ok(Math.abs(got[key]-demo[key])<.005,key);
 for(const key of ['last_observation_date','target_date'])assert.equal(got[key],demo[key]);
}
for(const csv of ['Date,brent\n2026-11-06,1\n2026-11-06,2','Date,brent\n2026-11-07,1','Date,brent\n2099-01-01,1','Date,brent\n2026-11-06,NaN','Date,brent\n2026-02-30,3'])assert.throws(()=>parseCSV(csv,'2026-11-20'));
const update=mergeHistory(bundle.history,parseCSV('Date,brent\n2026-11-06,110','2026-11-20'));
assert.equal(update.added,1);assert.equal(mergeHistory(update.history,[['2026-11-06',110]]).added,0);
assert.throws(()=>mergeHistory(update.history,[['2026-11-06',999]]));
assert.throws(()=>predict(bundle.history,'2026-11-23',5,bundle,'2026-11-20'));
const meta=bundle.models['5'].metadata,origin='2026-11-06';
const before=buildInput(bundle.history,origin,meta,'2026-11-20');
// Observations not yet available at the origin cannot change the input.
assert.deepEqual(buildInput(update.history,origin,meta,'2026-11-20'),before);
// A gap longer than the causal filling limit must stop inference.
const cutoff=bday(origin,-meta.preparation_config.publication_lag_bdays);
const withGap=bundle.history.filter(([d])=>d<bday(cutoff,-6)||d>cutoff);
assert.throws(()=>buildInput(withGap,origin,meta,'2026-11-20'));
const result={predictions_checked:count,max_absolute_difference_usd:maximum,tolerance_usd:.005,demo_horizons:3,csv_checks:'passed',causality_and_gaps:'passed'};
fs.writeFileSync(new URL('../reports/browser_verification.json',import.meta.url),JSON.stringify(result,null,2)+'\n');
console.log(result);
