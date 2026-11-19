'use strict';
const $ = id => document.getElementById(id);
let state = null, result = null, horizon = 5, page = 'forecast', evaluation = null, sequence = 0;
const colors = {PERSISTENCE:'#8b98ae',RNN:'#7155df',LSTM:'#008b80',GRU:'#d78b42'};
const amount = (n, digits=2) => Number(n).toLocaleString('es-ES',{minimumFractionDigits:digits,maximumFractionDigits:digits});
const dateLabel = s => new Date(s+'T00:00:00Z').toLocaleDateString('es-ES',{timeZone:'UTC',day:'2-digit',month:'2-digit',year:'numeric'});
const escapeHTML = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(path, body, raw=false) {
  const options=body===undefined?{}:{method:'POST',headers:{'X-Brent-Client':'1','Content-Type':raw?'text/csv':'application/json'},body:raw?body:JSON.stringify(body)};
  const response=await fetch(path,options);
  let data; try { data=await response.json(); } catch { throw new Error('No se pudo leer la respuesta del servidor Python.'); }
  if(!response.ok) throw new Error(data.error||'No se pudo completar la operación.');
  return data;
}
function message(text, success=false) { $('global-message').textContent=text; $('global-message').hidden=!text; $('global-message').classList.toggle('success',success); }
function saveStatus() { ['save','save-top'].forEach(id=>$(id).disabled=!state?.changed); $('save-status').textContent=state?.changed?'Cambios sin guardar en este ordenador':'Histórico conservado · sin cambios pendientes'; }
function applyState(next) {
  state=next; $('observation-count').textContent=amount(state.observations,0)+' observaciones';
  $('last-date').textContent='Último dato: '+dateLabel(state.last_date);
  $('origin').max=state.today; $('origin').value=state.origin; saveStatus();
}
function switchPage(next) {
  page=next;
  document.querySelectorAll('.page').forEach(el=>{el.hidden=el.id!=='page-'+page;el.classList.toggle('active',!el.hidden);});
  document.querySelectorAll('.nav-item').forEach(el=>{const active=el.dataset.page===page;el.classList.toggle('active',active);if(active)el.setAttribute('aria-current','page');else el.removeAttribute('aria-current');});
  $('page-title').textContent={forecast:'PREDICCIÓN',evidence:'RESULTADOS REALES',history:'HISTORIA DEL BRENT',method:'METODOLOGÍA'}[page];
  if(state) { if(page==='forecast'&&result) drawForecast(); if(page==='evidence') loadEvaluation(); if(page==='history') drawHistory(); }
  window.scrollTo({top:0,behavior:'instant'});
}
async function chooseHorizon(h) {
  horizon=Number(h);
  document.querySelectorAll('[data-horizon]').forEach(b=>{let selected=Number(b.dataset.horizon)===horizon;b.classList.toggle('selected',selected);b.setAttribute('aria-pressed',String(selected));});
  if(!state)return;
  if(page==='evidence') loadEvaluation();
  await calculate();
}
async function calculate() {
  const current=++sequence;
  $('calculate').disabled=true; $('calculate').textContent='Calculando…';
  $('forecast-content').setAttribute('aria-busy','true'); $('download-prediction').disabled=true;
  result=null; $('forecast-chart').replaceChildren();
  ['predicted','last-price','change','range','coverage'].forEach(id=>$(id).textContent='—');
  $('target').textContent='Calculando con el modelo guardado'; $('explanation').textContent='Esperando al modelo.';
  message('');
  try {
    const fresh=await api('/api/forecast',{horizon,origin:$('origin').value});
    if(current!==sequence)return;
    result=fresh; renderForecast();
  } catch(error) {
    if(current===sequence){message(error.message);$('target').textContent='Predicción no disponible';$('model-tag').textContent='Revisa los datos';}
  } finally {
    if(current===sequence){$('calculate').disabled=false;$('calculate').textContent='Calcular predicción';$('forecast-content').setAttribute('aria-busy','false');}
  }
}
function renderForecast() {
  const r=result;
  $('model-tag').textContent=r.architecture+' · '+horizon+' días';
  $('predicted').innerHTML=amount(r.predicted_usd)+'<span class="currency">USD</span>';
  $('last-price').innerHTML=amount(r.last_available_usd)+'<span class="currency">USD</span>';
  $('change').innerHTML=(r.change_usd>=0?'+':'')+amount(r.change_usd)+'<span class="currency">USD</span>';
  $('target').textContent='Objetivo: '+dateLabel(r.target_date);
  $('last-used').textContent='Observación: '+dateLabel(r.last_observation_date);
  $('change-pct').textContent=(r.change_pct===null?'Variación porcentual no calculable':(r.change_pct>=0?'+':'')+amount(r.change_pct,1)+' % respecto al último precio');
  $('range').textContent=amount(r.range_low_usd)+' — '+amount(r.range_high_usd);
  $('coverage').textContent=amount(r.range_test_coverage*100,1)+' %';
  $('stale').hidden=r.origin_date===state.today;
  $('stale').textContent='Escenario fechado el '+dateLabel(r.origin_date)+', no a fecha de hoy. Incorpora observaciones recientes para actualizarlo.';
  $('explanation').textContent='La red sitúa el precio '+(r.change_usd>0?'por encima':r.change_usd<0?'por debajo':'al mismo nivel')+' del último dato disponible. Es una estimación puntual, no una subida o bajada continua hasta la fecha objetivo.'+(horizon===252?' A largo plazo, las redes produjeron valores casi constantes y no anticiparon los picos.':'');
  $('download-prediction').disabled=false;
  if(page==='forecast')drawForecast();
}
function chart(containerId, lines, point=null, markers=[]) {
  const el=$(containerId);el.replaceChildren();
  const W=Math.max(280,el.clientWidth||700),H=300,p={l:45,r:17,t:28,b:37};
  const all=lines.flatMap(l=>l.data).concat(point?[[point.date,point.low],[point.date,point.high]]:[]);
  if(!all.length)return;
  const toTime=s=>Date.parse(s+'T00:00:00Z');
  const times=all.map(d=>toTime(d[0])),values=all.map(d=>d[1]);
  let xmin=Math.min(...times),xmax=Math.max(...times),ymin=Math.min(...values),ymax=Math.max(...values);
  const padding=Math.max((ymax-ymin)*.16,1); ymin-=padding;ymax+=padding;
  if(xmin===xmax)xmax+=86400000;
  const X=d=>p.l+(toTime(d)-xmin)/(xmax-xmin)*(W-p.l-p.r),Y=v=>H-p.b-(v-ymin)/(ymax-ymin)*(H-p.t-p.b);
  let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${escapeHTML(containerId==='history-chart'?'Histórico del Brent con extremos señalados':'Precios en dólares por barril según la fecha')}"><text x="${p.l}" y="13" fill="#91a0b3" font-size="10">USD / BARRIL</text>`;
  for(let i=0;i<=4;i++){let v=ymin+(ymax-ymin)*i/4,y=Y(v);svg+=`<line x1="${p.l}" y1="${y}" x2="${W-p.r}" y2="${y}" stroke="#edf1f7"/><text x="${p.l-9}" y="${y+4}" fill="#8b99ad" font-size="11" text-anchor="end">${amount(v,0)}</text>`;}
  const ticks=W<500?3:5;
  for(let i=0;i<ticks;i++){const t=xmin+(xmax-xmin)*i/(ticks-1),x=p.l+(W-p.l-p.r)*i/(ticks-1);const label=new Date(t).toLocaleDateString('es-ES',{timeZone:'UTC',month:'short',year:'2-digit'});svg+=`<text x="${x}" y="${H-10}" fill="#8b99ad" font-size="11" text-anchor="${i===0?'start':i===ticks-1?'end':'middle'}">${escapeHTML(label)}</text>`;}
  for(const l of lines){const d=l.data.map((q,i)=>(i?'L':'M')+X(q[0]).toFixed(2)+','+Y(q[1]).toFixed(2)).join(' ');svg+=`<path d="${d}" fill="none" stroke="${l.color}" stroke-width="${lines.length>1?1.6:2.3}" stroke-linecap="round" stroke-linejoin="round"/>`;}
  if(point){const x=X(point.date);svg+=`<line x1="${x}" y1="${Y(point.low)}" x2="${x}" y2="${Y(point.high)}" stroke="#7155df" stroke-width="5" opacity=".35"/><circle cx="${x}" cy="${Y(point.value)}" r="5" fill="#7155df" stroke="white" stroke-width="2"><title>Predicción: ${amount(point.value)} USD · ${dateLabel(point.date)}</title></circle>`;}
  for(const m of markers){const x=X(m.date),y=Y(m.value);svg+=`<circle cx="${x}" cy="${y}" r="4.5" fill="#d78b42" stroke="white" stroke-width="1.5"><title>${escapeHTML(m.label)} · ${dateLabel(m.date)} · ${amount(m.value)} USD</title></circle><text x="${Math.min(Math.max(x,65),W-40)}" y="${y-12}" text-anchor="middle" fill="#9c661f" font-size="10">${escapeHTML(m.short)}</text>`;}
  svg+='</svg>';el.innerHTML=svg;
  const tip=document.createElement('div');tip.className='chart-tooltip';tip.hidden=true;el.append(tip);
  const graphic=el.querySelector('svg');
  graphic.addEventListener('pointermove',event=>{const box=graphic.getBoundingClientRect(),px=(event.clientX-box.left)*W/box.width;if(px<p.l||px>W-p.r){tip.hidden=true;return;}const t=xmin+(px-p.l)/(W-p.l-p.r)*(xmax-xmin),data=lines[0].data;let nearest=data.reduce((a,b)=>Math.abs(toTime(a[0])-t)<Math.abs(toTime(b[0])-t)?a:b);tip.textContent=dateLabel(nearest[0])+' · '+amount(nearest[1])+' USD';tip.hidden=false;tip.style.left=Math.max(0,Math.min(event.clientX-box.left-60,box.width-tip.offsetWidth))+'px';tip.style.top=Math.max(0,event.clientY-box.top-42)+'px';});
  graphic.addEventListener('pointerleave',()=>tip.hidden=true);
}
function drawForecast(){if(!result)return;chart('forecast-chart',[{color:'#008b80',data:state.history.filter(d=>d[0]<=result.last_observation_date).slice(-260)}],{date:result.target_date,value:result.predicted_usd,low:result.range_low_usd,high:result.range_high_usd});}
async function loadEvaluation(){const requested=horizon;try {const data=await api('/api/evaluation/'+requested);if(requested!==horizon)return;evaluation=data;const maximum=Math.max(...data.metrics.map(r=>r.rmse_usd));$('error-bars').innerHTML=data.metrics.map(r=>`<div class="bar-row"><span>${escapeHTML(r.architecture)}</span><div class="bar-track"><div class="bar-fill" style="width:${r.rmse_usd/maximum*100}%;background:${colors[r.architecture]}"></div></div><strong>${amount(r.rmse_usd,3)}</strong></div>`).join('');$('metrics-table').innerHTML=data.metrics.map(r=>`<tr><td>${escapeHTML(r.architecture)}</td>${[r.r2,r.mae_usd,r.mse_usd2,r.rmse_usd].map(n=>'<td>'+amount(n,3)+'</td>').join('')}<td>${amount(r.improvement_rmse_pct,2)} %</td></tr>`).join('');$('selection-note').textContent='Selección previa en validación: '+data.selected+' a '+requested+' días. Los modelos seleccionados no superaron a persistencia en prueba.';$('evidence-model').textContent=data.selected+' · '+requested+' días';if(page==='evidence')drawEvaluation();}catch(error){message(error.message);}}
function drawEvaluation(){if(!evaluation)return;chart('evidence-chart',[{color:'#008b80',data:evaluation.series.map(r=>[r[0],r[1]])},{color:'#7155df',data:evaluation.series.map(r=>[r[0],r[2]])}]);}
const episodes=[
 {year:'2008',start:'2008-01-01',end:'2008-07-31',max:true,short:'2008',title:'Máximo y crisis financiera',text:'El máximo fue seguido por una fuerte caída en el contexto de la crisis financiera y la contracción de la demanda.',link:'https://www.eia.gov/finance/performanceprofiles/overview.php'},
 {year:'2008',start:'2008-08-01',end:'2008-12-31',max:false,short:'Caída',title:'Caída de 2008',text:'',link:''},
 {year:'2014–16',start:'2014-01-01',end:'2016-12-31',max:false,short:'2016',title:'Exceso de oferta',text:'La oferta superior a la demanda y la acumulación de inventarios presionaron los precios a la baja.',link:'https://www.eia.gov/TODAYINENERGY/detail.php?id=24432'},
 {year:'2020',start:'2020-01-01',end:'2020-12-31',max:false,short:'2020',title:'El impacto de la pandemia',text:'Las restricciones redujeron la demanda de petróleo. El futuro WTI llegó a cotizar en negativo; no debe confundirse con el Brent de este gráfico.',link:'https://www.eia.gov/tODAyinenergy/detail.php?id=46336'},
 {year:'2022',start:'2022-01-01',end:'2022-12-31',max:true,short:'2022',title:'Tensiones en el suministro',text:'La invasión de Ucrania y los bajos inventarios contribuyeron a las subidas. Después pesaron las preocupaciones por la demanda.',link:'https://www.eia.gov/todayinenergy/detail.php?id=55079'},
 {year:'2026',start:'2026-01-01',end:'2026-12-31',max:true,short:'Reciente',title:'Movimientos recientes',text:'Los picos de esta copia coinciden con grandes errores de predicción. No se atribuyen aquí a un acontecimiento concreto sin un análisis adicional de fuentes.',link:''}
];
function drawHistory(){const marks=[];for(const e of episodes){const rows=state.history.filter(d=>d[0]>=e.start&&d[0]<=e.end);if(rows.length){const q=rows.reduce((a,b)=>(e.max?a[1]>b[1]:a[1]<b[1])?a:b);marks.push({date:q[0],value:q[1],label:e.title,short:e.short});}}chart('history-chart',[{color:'#008b80',data:state.history}],null,marks);$('timeline').innerHTML=episodes.filter(e=>e.text).map(e=>`<article><span class="year">${e.year}</span><h2>${e.title}</h2><p>${e.text}</p>${e.link?`<a href="${e.link}" target="_blank" rel="noopener noreferrer">Consultar fuente EIA</a>`:''}</article>`).join('');}
function download(name,content){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([content],{type:'text/csv;charset=utf-8'}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
$('forecast-form').addEventListener('submit',e=>{e.preventDefault();calculate();});
document.querySelectorAll('[data-horizon]').forEach(b=>b.addEventListener('click',()=>chooseHorizon(b.dataset.horizon)));
document.querySelectorAll('[data-page]').forEach(b=>b.addEventListener('click',()=>switchPage(b.dataset.page)));
document.querySelectorAll('.upload-open').forEach(b=>b.addEventListener('click',()=>{$('upload-message').hidden=true;$('upload-dialog').showModal();}));
$('close-dialog').addEventListener('click',()=>$('upload-dialog').close());
$('csv-file').addEventListener('change',()=>$('import-csv').disabled=!$('csv-file').files.length);
$('import-csv').addEventListener('click',async()=>{const file=$('csv-file').files[0];if(!file)return;$('import-csv').disabled=true;try{if(file.size>5000000)throw new Error('El CSV supera los 5 MB.');const data=await api('/api/upload',await file.text(),true);applyState(data.state);$('upload-dialog').close();await calculate();message(data.added+' observaciones incorporadas. Pulsa Guardar cambios para conservarlas.',true);if(page==='history')drawHistory();}catch(error){$('upload-message').textContent=error.message;$('upload-message').hidden=false;}finally{$('import-csv').disabled=false;}});
['save','save-top'].forEach(id=>$(id).addEventListener('click',async()=>{try{await api('/api/save',{});state.changed=false;saveStatus();message('Histórico guardado en este ordenador.',true);}catch(error){message(error.message);}}));
$('download-prediction').addEventListener('click',()=>{if(result)download('prediccion_brent_'+horizon+'.csv',Object.keys(result).join(',')+'\n'+Object.values(result).map(v=>v??'').join(',')+'\n');});
let resizing;window.addEventListener('resize',()=>{clearTimeout(resizing);resizing=setTimeout(()=>{if(page==='forecast')drawForecast();if(page==='evidence')drawEvaluation();if(page==='history'&&state)drawHistory();},150);});
(async()=>{try{applyState(await api('/api/state'));await calculate();}catch(error){message('No se puede conectar con los modelos. Abre la aplicación con «python app.py» y entra en http://127.0.0.1:8502. '+error.message);$('calculate').disabled=true;}})();
