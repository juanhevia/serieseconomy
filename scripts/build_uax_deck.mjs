import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {PresentationFile,FileBlob}=require(process.env.ARTIFACT_TOOL_PATH || '@oai/artifact-tool');
const root=process.cwd(),dir=path.join(root,'tmp/entrega_template');
await fs.mkdir(dir,{recursive:true});
const p=await PresentationFile.importPptx(await FileBlob.load(path.join(root,'docs/assignment/PPT_Template.pptx')));
const cover=p.slides.items[0], blank=p.slides.items[1];
p.delete(p.slides.items[3].id);p.delete(p.slides.items[2].id);
for(let i=0;i<4;i++)p.slides.addCopy(blank.copy());
const blue='#0760AB',ink='#173247',muted='#536878';
function txt(s,t,x,y,w,h,size=22,color=ink,bold=false){const z=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});z.text=t;z.text.style={fontFamily:'Muli',fontSize:size,color,bold,verticalAlignment:'top',wrap:true};return z;}
function base(i,title){const s=p.slides.items[i];txt(s,title,48,106,1180,62,36,blue,true);txt(s,'Juan Hevia Losa · Aprendizaje automático · Noviembre 2026',48,680,1080,22,13,muted);txt(s,`${i} / 5`,1165,680,70,22,13,muted);return s;}
function table(s,values,y,widths,height=350,size=17){const t=s.tables.add({rows:values.length,columns:values[0].length,left:48,top:y,width:1184,height,columnWidths:widths,values});t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({margins:{top:3,bottom:3,left:8,right:8}});t.borders.assign({fill:'#D7E3ED',width:.5,style:'solid'});for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){const cell=t.getCell(r,c);cell.fill=r===0?blue:r%2?'#F0F5FA':'#FFFFFF';cell.text.style={fontFamily:'Muli',fontSize:size,color:r===0?'#FFFFFF':ink,bold:r===0};}return t;}
function csv(t){const lines=t.trim().split(/\r?\n/).map(x=>x.split(','));return lines.slice(1).map(v=>Object.fromEntries(lines[0].map((k,i)=>[k,v[i]])));}
const fmt=(v,n=2)=>Number(v).toFixed(n).replace('.',',');
const met=csv(await fs.readFile('artifacts/brent_v1/metrics_test.csv','utf8'));
const costs=csv(await fs.readFile('artifacts/brent_v1/metrics_validation.csv','utf8'));
const demo=JSON.parse(await fs.readFile('reports/demo_app_2026-11-20.json','utf8'));
const selected={5:'RNN',63:'LSTM',252:'RNN'};
for(const z of cover.shapes.items){if(z.text?.toString().includes('xxxxxxxx')){z.text='Predicción del Brent\nRNN, LSTM y GRU';z.position={left:90,top:275,width:810,height:150};z.text.style={fontFamily:'Muli',fontSize:43,bold:true,color:'#FFFFFF',alignment:'right'};}else if(z.text?.toString().includes('Fecha')){z.text='20 noviembre 2026';z.position={left:560,top:505,width:340,height:50};z.text.style={fontFamily:'Muli',fontSize:23,color:'#FFFFFF',alignment:'right'};}}
txt(cover,'Juan Hevia Losa',430,443,470,52,25,'#FFFFFF',true);txt(cover,'Aprendizaje automático · Caso individual',110,176,780,50,22,'#FFFFFF');
let s=base(1,'Planificación energética ante un precio volátil');
txt(s,'100.000 USD',48,190,510,75,52,blue,true);txt(s,'de impacto por cada 1 USD/barril en una compra de 100.000 barriles.',48,278,480,95,25);txt(s,'Ejemplo ilustrativo, sin costes adicionales.',48,375,490,35,16,muted);
txt(s,'La oportunidad',48,440,470,40,25,blue,true);txt(s,'Comparar escenarios de compra con su error histórico y su incertidumbre antes de comprometer un presupuesto.',48,487,480,125,24);
const raw=csv(await fs.readFile('data/raw/brent_2000-01-01_2026-11-05.csv','utf8')).filter(r=>r.brent&&r.brent!=='.'&&Number.isFinite(+r.brent));const monthly=[];for(const r of raw){if(!monthly.length||monthly.at(-1).Date.slice(0,7)!==r.Date.slice(0,7))monthly.push(r);else monthly[monthly.length-1]=r;}
s.charts.add('line',{position:{left:584,top:205,width:636,height:260},categories:monthly.map(r=>r.Date.slice(0,7)),series:[{name:'Brent USD/barril',values:monthly.map(r=>+r.brent),fill:blue}],hasLegend:false,title:'Brent observado · cierre mensual',xAxis:{tickLabelFontSize:10},yAxis:{numberFormatCode:'0'}});
txt(s,'2008: crisis financiera · 2014–16: exceso de oferta\n2020: pandemia · 2022: invasión de Ucrania',584,484,640,66,19);
txt(s,'2026: las restricciones de oferta en Oriente Medio y los inventarios reducidos presionan los precios, según EIA (noviembre).',584,559,640,75,19);
txt(s,'Fuentes: EIA/FRED, DCOILBRENTEU (copia 05/11; último precio 03/11). EIA, STEO noviembre 2026.',48,642,1185,30,13,muted);
s.speakerNotes.text='Contexto reciente: https://www.eia.gov/outlooks/steo/archives/Nov26.pdf . Histórico EIA vía FRED: https://fred.stlouisfed.org/series/DCOILBRENTEU . Los datos del experimento permanecen congelados. Los acontecimientos contextualizan la serie, no prueban causalidad individual.';
s=base(2,'Una comparación temporal y una selección previa al test');
const blocks=[['Preparación','60 observaciones por ventana. Escalador ajustado solo en entrenamiento. Relleno causal limitado a 5 días. Retraso de publicación supuesto: 3 días.'],['Comparación justa','RNN, LSTM y GRU con iguales datos y particiones. Tres folds temporales y dos tamaños de red. 54 ajustes de búsqueda y 9 principales.'],['Evaluación independiente','Elección en validación. Pesos y escaladores fijos al abrir prueba. Referencia: mantener el último precio disponible. Sin reelección tras ver test.']];
for(let i=0;i<3;i++){const x=48+i*405;txt(s,blocks[i][0],x,187,375,45,26,blue,true);txt(s,blocks[i][1],x,243,370,158,21);}
txt(s,'Modelos integrados y motivo de la selección',48,418,1180,42,26,blue,true);
table(s,[['Días','Elegido','RMSE validación','Parámetros','Motivo'],['5 días','RNN','5,428 USD','321','Menor RMSE; coste bajo'],['63 días','LSTM','15,523 USD','4.513','Mejor precisión de validación'],['252 días','RNN','26,505 USD','1.153','Menor RMSE; menor tamaño que LSTM']],470,[110,110,190,150,624],140,18);
txt(s,'Orígenes: entrenamiento 2000–2018, validación 2018–2022, test 2022–2026; límites purgados por horizonte. Una semilla: estabilidad no demostrada.',48,624,1184,44,15,muted);
s.speakerNotes.text='Fuentes: artifacts/brent_v1/metrics_validation.csv y metadatos de runs/laboratorio. Ventana y calendario lunes-viernes. Selección por RMSE, coste descrito como criterio complementario, no criterio retrospectivo.';
s=base(3,'Las redes seleccionadas no superan a persistencia');
const vals=[['Días','Modelo','MAE','MSE','RMSE','R²','Mejora %','Parám.','Tiempo s']];
for(const r of met){const co=costs.find(c=>c.architecture===r.architecture&&c.horizon_bdays===r.horizon_bdays);vals.push([r.horizon_bdays,r.architecture+(selected[r.horizon_bdays]===r.architecture?' *':''),fmt(r.mae_usd),fmt(r.mse_usd2),fmt(r.rmse_usd),fmt(r.r2,3),fmt(r.improvement_rmse_pct),co.parameters,fmt(co.fit_seconds)]);}
const t=table(s,vals,186,[65,175,115,125,115,100,145,120,224],354,17);
for(let r=1;r<vals.length;r++)if(vals[r][1].includes('*'))for(let c=0;c<9;c++)t.getCell(r,c).fill='#DCEBFA';
txt(s,'* Elegido en validación. MAE/RMSE: USD/barril; MSE: USD². Mejora positiva = menor RMSE que persistencia.',48,580,1180,25,14,muted);txt(s,'Tiempo: ajuste principal en GPU, sin búsqueda. N de test: 1.010 / 955 / 772. Fuente: métricas del experimento.',48,605,1180,25,14,muted);
txt(s,'GRU logra el menor RMSE en test; no se reelige con ese bloque. Las redes elegidas empeoran un 3,33 %, 2,04 % y 2,42 %.',48,640,1180,30,17);
s.speakerNotes.text='Fuentes: artifacts/brent_v1/metrics_test.csv y metrics_validation.csv. N test 1010,955,772. Tiempos de ajuste principal medidos en GPU del laboratorio, excluyen búsqueda. No se ha demostrado significación estadística de las diferencias.';
s=base(4,'La aplicación muestra precio, variación y rango');
txt(s,'Demostración: 20/11/2026 · Último precio disponible: 92,94 USD (17/11/2026)',48,181,1180,45,22);
const dv=[['Horizonte / red','Fecha objetivo','Precio USD','Cambio USD','Rango orientativo USD']];for(const r of demo)dv.push([`${r.horizon_bdays} días / ${r.architecture}`,r.target_date.split('-').reverse().join('/'),fmt(r.predicted_usd),(r.change_usd>0?'+':'')+fmt(r.change_usd),`${fmt(r.range_low_usd)} a ${fmt(r.range_high_usd)}`]);
table(s,dv,246,[220,200,190,190,384],185,23);
txt(s,'Uso de la aplicación',48,468,540,42,26,blue,true);txt(s,'Cargar brent_2026-11-20.csv y elegir el 20/11/2026. Consultar histórico, predicción y errores. Los pesos permanecen fijos.',48,519,540,110,22);
txt(s,'Lectura del rango',678,468,554,42,26,blue,true);txt(s,'Percentiles 10 y 90 del error de validación. Cobertura en test: 75,0 %, 77,3 % y 90,4 %. Son rangos orientativos, sin garantía futura del 80 %.',678,519,554,110,22);
txt(s,'Demo pública: huggingface.co/spaces/juanhevia/brent-observatorio · Actualizar el dato no reentrena la red.',48,647,1184,25,15,muted);
s.speakerNotes.text='Fuente: reports/demo_app_2026-11-20.json, inferencia real con los pesos guardados. Actualización de diez precios EIA/FRED descargados el 20/11/2026. Inferencia con los mismos pesos, sin reentrenamiento. Selección y calibración comparten validación. App https://huggingface.co/spaces/juanhevia/brent-observatorio';
s=base(5,'Una herramienta de análisis con límites explícitos');
txt(s,'Valor disponible',48,187,555,45,28,blue,true);txt(s,'Histórico, escenarios y evidencia del error en una misma consulta.\n\nActualización manual mediante CSV sin coste de reentrenamiento.\n\nCódigo y nueve redes publicados para reproducir los resultados.',48,245,555,270,24);
txt(s,'Condiciones para ampliar el uso',678,187,554,45,28,blue,true);txt(s,'Versión univariante: las variables externas exigidas aún no entran al modelo y falta la cesta OPEP.\n\nSiguiente experimento: incorporar las variables, comprobar su disponibilidad temporal y evaluar en un nuevo período.',678,245,554,246,23);
txt(s,'Conclusión para compras y dirección',48,535,1184,45,27,blue,true);txt(s,'Útil para explorar escenarios y entender el riesgo. El test actual no demuestra una mejora frente a mantener el último precio ni justifica automatizar compras o inversiones.',48,588,1184,72,24);
s.speakerNotes.text='Limitaciones: test conocido tras auditoría, por lo que una versión nueva necesita nueva evaluación independiente; no cambiar las redes seleccionadas usando este test. Una semilla y calibración compartida con selección. Código Python disponible; demo pública equivalente en JavaScript.';
for(let i=0;i<p.slides.items.length;i++){const s=p.slides.items[i];const b=await p.export({slide:s,format:'png',scale:1});await fs.writeFile(path.join(dir,`final-${i}.png`),new Uint8Array(await b.arrayBuffer()));}
await (await PresentationFile.exportPptx(p)).save(path.join(dir,'candidate.pptx'));
console.log('BUILT',p.slides.items.length);
