/* Inferencia de las redes seleccionadas. Sin entrenamiento ni llamadas externas. */
export function madridToday(){return new Intl.DateTimeFormat('en-CA',{timeZone:'Europe/Madrid',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date());}
export function isoDate(s){if(typeof s!=='string'||!/^\d{4}-\d{2}-\d{2}$/.test(s))throw Error('Usa fechas AAAA-MM-DD.');const d=new Date(s+'T00:00:00Z');if(!Number.isFinite(d.getTime())||d.toISOString().slice(0,10)!==s)throw Error('Fecha no válida.');return d;}
const iso=d=>d.toISOString().slice(0,10);
const weekday=d=>![0,6].includes(d.getUTCDay());
export function bday(s,n){const d=isoDate(s);while(n){d.setUTCDate(d.getUTCDate()+(n>0?1:-1));if(weekday(d))n+=n>0?-1:1;}return iso(d);}
export function parseCSV(content,today=madridToday()){
 if(new TextEncoder().encode(content).length>5000000)throw Error('El CSV supera el límite de 5 MB.');
 const rows=content.replace(/^\uFEFF/,'').trim().split(/\r?\n/).map(r=>r.split(',').map(v=>v.trim().replace(/^"([^"\n]*)"$/,'$1')));
 const header=rows.shift();if(header.length!==2||new Set(header).size!==2||!header.includes('Date')||!header.includes('brent')||!rows.length)throw Error('Usa exactamente las columnas Date,brent con datos.');
 const seen=new Set(),result=[];
 for(const row of rows){if(row.length!==2)throw Error('Cada fila debe contener fecha y precio separados por una coma.');const s=row[header.indexOf('Date')],v=row[header.indexOf('brent')],d=isoDate(s);
 if(!weekday(d))throw Error('Se admiten observaciones de lunes a viernes.');if(s>today)throw Error('El CSV contiene observaciones con fecha futura.');if(seen.has(s))throw Error('El CSV contiene fechas duplicadas.');seen.add(s);
 if(!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(v)||!Number.isFinite(Number(v)))throw Error('Los precios deben ser números finitos con punto decimal.');result.push([s,Number(v)]);}
 return result.sort((a,b)=>a[0].localeCompare(b[0]));
}
export function mergeHistory(history,incoming){const map=new Map(history);let count=0;for(const [d,v] of incoming){if(d<history[0][0])throw Error('La actualización no puede preceder al histórico.');if(map.has(d)&&Math.abs(map.get(d)-v)>1e-8)throw Error('El CSV modifica precios ya guardados. Revisa la fuente.');if(!map.has(d)){map.set(d,v);count++;}}return {history:[...map].sort((a,b)=>a[0].localeCompare(b[0])),added:count};}
export const historyCSV=history=>'Date,brent\n'+history.map(r=>r.join(',')).join('\n')+'\n';
export function buildInput(history,origin,meta,today=madridToday()){
 const d=isoDate(origin);if(!weekday(d)||origin>today)throw Error('La fecha de predicción debe ser de lunes a viernes y no puede ser futura.');
 const map=new Map(history),config=meta.preparation_config;let current=bday(origin,-config.publication_lag_bdays),values=[],sources=[],filled=0;
 for(let i=0;i<meta.window;i++){let source=current,value=map.get(source),gap=0;
 while(value===undefined&&gap<config.max_forward_fill_bdays){source=bday(source,-1);value=map.get(source);gap++;}
 if(current<config.start||source<config.start||value===undefined)throw Error('No hay una ventana completa de 60 días disponibles. Añade observaciones; los huecos largos no se rellenan.');
 values.unshift(value);sources.unshift(source);if(gap)filled++;current=bday(current,-1);}
 const input=values.map(v=>Math.fround((v-meta.scaler.mean)/meta.scaler.scale));if(input.some(v=>!Number.isFinite(v)))throw Error('Los precios producen una entrada no válida.');
 return {input,info:{origin_date:origin,last_observation_date:sources.at(-1),last_available_usd:values.at(-1),input_imputed_fraction:filled/meta.window}};
}
export function recurrent(input,model){
 const w=model.tensors,n=model.metadata.candidate.hidden_size;let hidden=new Float64Array(n),cell=new Float64Array(n);const sigmoid=x=>1/(1+Math.exp(-x));
 for(const x of input){const next=new Float64Array(n),gates=new Float64Array(w['recurrent.bias_ih_l0'].length);
 for(let i=0;i<gates.length;i++){let v=w['recurrent.weight_ih_l0'][i][0]*x+w['recurrent.bias_ih_l0'][i]+w['recurrent.bias_hh_l0'][i];for(let j=0;j<n;j++)v+=w['recurrent.weight_hh_l0'][i][j]*hidden[j];gates[i]=v;}
 if(model.architecture==='RNN'){for(let j=0;j<n;j++)next[j]=Math.tanh(gates[j]);}
 else if(model.architecture==='LSTM'){for(let j=0;j<n;j++){cell[j]=sigmoid(gates[n+j])*cell[j]+sigmoid(gates[j])*Math.tanh(gates[2*n+j]);next[j]=sigmoid(gates[3*n+j])*Math.tanh(cell[j]);}}
 else throw Error('Arquitectura no compatible con esta exportación.');hidden=next;}
 let value=w['output.bias'][0];for(let j=0;j<n;j++)value+=w['output.weight'][0][j]*hidden[j];return value;
}
export function predict(history,origin,horizon,bundle,today=madridToday()){
 if(![5,63,252].includes(horizon))throw Error('Horizonte no válido.');const model=bundle.models[String(horizon)],meta=model.metadata,{input,info}=buildInput(history,origin,meta,today);
 const value=recurrent(input,model)*meta.scaler.scale+meta.scaler.mean;if(!Number.isFinite(value))throw Error('La red produjo una predicción no válida.');const change=value-info.last_available_usd;
 return {...info,architecture:model.architecture,horizon_bdays:horizon,target_date:bday(origin,horizon),predicted_usd:value,change_usd:change,change_pct:info.last_available_usd===0?null:100*change/info.last_available_usd,range_low_usd:value+model.range.residual_low,range_high_usd:value+model.range.residual_high,range_test_coverage:model.range.test_coverage};
}
export function snapshot(history,bundle,changed=false,today=madridToday()){
 let limit=today;while(!weekday(isoDate(limit))){const d=isoDate(limit);d.setUTCDate(d.getUTCDate()-1);limit=iso(d);}
 const latest=bday(history.at(-1)[0],bundle.models['5'].metadata.preparation_config.publication_lag_bdays);
 return {today,origin:latest<limit?latest:limit,last_date:history.at(-1)[0],observations:history.length,changed,history,models:Object.fromEntries(Object.entries(bundle.models).map(([h,m])=>[h,{architecture:m.architecture,coverage:m.range.test_coverage}]))};
}
