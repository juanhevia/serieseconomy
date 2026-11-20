import {parseCSV,mergeHistory,historyCSV,predict,snapshot} from './browser_engine.mjs';
const response=await fetch('./bundle.json');if(!response.ok)throw Error('No se pudo cargar el paquete de modelos.');
const bundle=await response.json();let history=bundle.history,changed=false;
const storageKey='brent-observations-v1-'+bundle.source_sha256;
let recoveryWarning='';
try{const saved=localStorage.getItem(storageKey);if(saved)history=mergeHistory(history,parseCSV(saved)).history;}catch(error){recoveryWarning='No se recuperó el histórico guardado: '+error.message;}
window.brentClient={async request(path,body){
 if(path==='/api/state')return snapshot(history,bundle,changed);
 if(path==='/api/forecast')return predict(history,body.origin,body.horizon,bundle);
 if(path==='/api/upload'){const merged=mergeHistory(history,parseCSV(body));history=merged.history;changed=changed||merged.added>0;return {added:merged.added,state:snapshot(history,bundle,changed)};}
 if(path==='/api/save'){try{localStorage.setItem(storageKey,historyCSV(history));changed=false;return {saved:true};}catch{throw Error('El navegador no permite guardar. Descarga el CSV para conservarlo.');}}
 if(path.startsWith('/api/evaluation/'))return bundle.evaluation[path.split('/').at(-1)];
 throw Error('Acción no disponible.');
}};
document.addEventListener('click',e=>{const a=e.target.closest('a');if(!a)return;const p=a.getAttribute('href');if(!['/api/history.csv','/api/metrics.csv'].includes(p))return;e.preventDefault();const content=p.includes('history')?historyCSV(history):bundle.metrics_csv;const blob=URL.createObjectURL(new Blob([content],{type:'text/csv;charset=utf-8'}));const link=document.createElement('a');link.href=blob;link.download=a.download;link.click();setTimeout(()=>URL.revokeObjectURL(blob),1000);});
await import('./app.js');
if(recoveryWarning){const message=document.getElementById('global-message');message.textContent=recoveryWarning;message.hidden=false;}
