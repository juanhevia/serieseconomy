"""PDF de entrega con identidad UAX y colores del observatorio; no entrena."""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import csv,json
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf/01_presentacion_brent.pdf'
FONT=Path('/System/Library/Fonts/Supplemental')
for name,file in [('Body','Arial.ttf'),('Bold','Arial Bold.ttf')]:pdfmetrics.registerFont(TTFont(name,str(FONT/file)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
with ZipFile(ROOT/'docs/assignment/PPT_Template.pptx') as z:
 logos={i:ImageReader(BytesIO(z.read(f'ppt/media/image{i}.png'))) for i in [1,2,3,4]}
W,H=1280,720
INK='#16343d';TEAL='#237265';LIME='#d3ef77';PURPLE='#7155cf';BLUE='#0760AB';MUTED='#526b75';PALE='#eef5f2';LAVENDER='#eeeafa'
c=canvas.Canvas(str(OUT),pagesize=(W,H));c.setTitle('Predicción del Brent - Juan Hevia Losa - UAX');c.setAuthor('Juan Hevia Losa')
def rect(x,y,w,h,color):
 c.setFillColor(HexColor(color));c.rect(x,H-y-h,w,h,fill=1,stroke=0)
def text(t,x,y,size=20,color=INK,font='Body'):
 c.setFillColor(HexColor(color));c.setFont(font,size);c.drawString(x,H-y-size*.82,t)
def para(t,x,y,w,size=20,color=INK):
 p=Paragraph(t,ParagraphStyle('p',fontName='Body',fontSize=size,leading=size*1.3,textColor=HexColor(color)))
 _,h=p.wrap(w,H);p.drawOn(c,x,H-y-h);return h
def line(x,y,x2,y2,col='#d9e3e1',width=1):
 c.setStrokeColor(HexColor(col));c.setLineWidth(width);c.line(x,H-y,x2,H-y2)
def logo(x=43.6,y=27.4,w=196.8,h=43.2):c.drawImage(logos[1],x,H-y-h,w,h,mask='auto')
def base(n,title,label):
 rect(0,0,W,H,'#FFFFFF');logo();text(label.upper(),850,44,13,TEAL,'Bold');rect(48,94,62,5,LIME);rect(110,94,30,5,PURPLE)
 text(title,48,119,35,INK,'Bold');line(48,673,1232,673);text('Juan Hevia Losa · Aprendizaje automático · Noviembre 2026',48,690,12,MUTED);text(f'{n} / 5',1185,690,12,MUTED)
def tab(values,x,y,widths,rowheights,size=18,highlight=(),rowcolors=None):
 t=Table(values,colWidths=widths,rowHeights=rowheights)
 st=[('FONTNAME',(0,0),(-1,-1),'Body'),('FONTNAME',(0,0),(-1,0),'Bold'),('FONTSIZE',(0,0),(-1,-1),size),('TEXTCOLOR',(0,0),(-1,-1),HexColor(INK)),('BACKGROUND',(0,0),(-1,0),HexColor(INK)),('TEXTCOLOR',(0,0),(-1,0),HexColor('#ffffff')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,-1),.5,HexColor('#dce6e3'))]
 for r in range(1,len(values)):
  st.append(('BACKGROUND',(0,r),(-1,r),HexColor(LAVENDER if r in highlight else ('#f1f6f4' if r%2 else '#ffffff'))))
  if rowcolors:st.extend([('TEXTCOLOR',(0,r),(0,r),HexColor(rowcolors[r-1])),('FONTNAME',(0,r),(0,r),'Bold')])
 t.setStyle(TableStyle(st));_,h=t.wrap(W,H);t.drawOn(c,x,H-y-h)
def chart(rows,x,y,w,h):
 lo,hi=0,160
 for v in [0,40,80,120,160]:
  yy=y+h-v/hi*h;line(x,yy,x+w,yy,'#dfe8e5',.6);text(str(v),x-33,yy-6,11,MUTED)
 pts=[(x+i/(len(rows)-1)*w,H-(y+h-v/hi*h)) for i,(_,v) in enumerate(rows)]
 p=c.beginPath();p.moveTo(x,H-y-h)
 for xx,yy in pts:p.lineTo(xx,yy)
 p.lineTo(x+w,H-y-h);p.close();c.setFillColor(HexColor('#e0f0e8'));c.drawPath(p,fill=1,stroke=0)
 p=c.beginPath()
 for i,(xx,yy) in enumerate(pts):(p.moveTo if i==0 else p.lineTo)(xx,yy)
 c.setStrokeColor(HexColor(TEAL));c.setLineWidth(2);c.drawPath(p)
 for year in ['2000','2008','2014','2020','2026']:
  i=next(i for i,r in enumerate(rows) if r[0].startswith(year));xx=x+i/(len(rows)-1)*w;text(year,min(xx,x+w-30),y+h+12,11,MUTED)
 c.setFillColor(HexColor(PURPLE));c.circle(*pts[-1],4,fill=1,stroke=0)
fmt=lambda v,n=2:f'{float(v):.{n}f}'.replace('.',',')
# Portada: misma diagonal y logotipos de la plantilla UAX.
rect(0,0,W,H,'#ffffff');p=c.beginPath();p.moveTo(0,H);p.lineTo(W,H);p.lineTo(779,0);p.lineTo(0,0);p.close();c.setFillColor(HexColor(BLUE));c.drawPath(p,fill=1,stroke=0)
c.drawImage(logos[3],0,0,367.16,138.49,mask='auto');logo(1044,660,200,44)
text('Aprendizaje automático · Caso individual',110,177,22,'#ffffff')
text('Predicción del Brent',225,290,58,'#ffffff','Bold');text('RNN, LSTM y GRU',270,365,48,LIME,'Bold')
text('Juan Hevia Losa',465,460,25,'#ffffff','Bold');text('20 noviembre 2026',650,519,21,'#ffffff');c.showPage()
base(1,'Planificación energética ante un precio volátil','Problema y oportunidad')
rect(48,199,495,210,INK);text('100.000 USD',72,228,49,LIME,'Bold');para('de impacto por cada 1 USD/barril en una compra de 100.000 barriles.',72,303,445,24,'#ffffff');text('Ejemplo ilustrativo, sin costes adicionales.',72,382,15,'#c1d5d0')
text('La oportunidad',48,454,25,TEAL,'Bold');para('Comparar escenarios de compra con su error histórico y su incertidumbre antes de comprometer un presupuesto.',48,499,495,25)
raw=[]
with (ROOT/'data/raw/brent_2000-01-01_2026-11-05.csv').open() as f:
 for r in csv.DictReader(f):
  try:raw.append((r['Date'],float(r['brent'])))
  except ValueError:pass
monthly={d[:7]:(d,v) for d,v in raw};text('Brent observado · USD/barril',609,201,19,TEAL,'Bold');chart(list(monthly.values()),623,250,587,216)
para('<b>2008</b> Crisis financiera · <b>2014–16</b> Exceso de oferta<br/><b>2020</b> Pandemia · <b>2022</b> Invasión de Ucrania',589,507,638,18)
para('<b>2026.</b> Restricciones de oferta en Oriente Medio e inventarios reducidos presionan los precios, según EIA.',589,570,638,18)
text('Fuentes: EIA/FRED (copia 05/11, último precio 03/11; último dato por mes). EIA, STEO noviembre 2026.',48,645,12,MUTED)
c.linkURL('https://www.eia.gov/outlooks/steo/archives/Nov26.pdf',(48,57,1230,80),relative=0);c.showPage()
base(2,'Una selección previa al test, bajo las mismas condiciones','Solución y metodología')
blocks=[('Preparación','60 observaciones por ventana. Escalado solo con entrenamiento. Relleno causal de hasta 5 días. Retraso de publicación supuesto: 3 días.'),('Comparación justa','RNN, LSTM y GRU con los mismos datos y particiones. Tres folds temporales y dos tamaños de red. 54 ajustes de búsqueda y 9 principales.'),('Evaluación independiente','Selección en validación. Pesos y escaladores fijos al abrir prueba. Referencia: mantener el último precio. Sin reelección tras ver el test.')]
for i,(title,body) in enumerate(blocks):
 x=48+i*405;col=[TEAL,PURPLE,TEAL][i];text(f'0{i+1}',x,194,18,col,'Bold');text(title,x,228,24,col,'Bold');para(body,x,272,370,21)
text('Modelos integrados y motivo de selección',48,440,25,INK,'Bold')
tab([['Días','Elegido','RMSE validación','Parámetros','Motivo'],['5','RNN','5,428 USD','321','Menor RMSE; coste bajo'],['63','LSTM','15,523 USD','4.513','Mejor precisión en validación'],['252','RNN','26,505 USD','1.153','Menor RMSE; menor tamaño que LSTM']],48,488,[85,115,205,155,624],[33]*4,18,rowcolors=[TEAL,PURPLE,TEAL])
para('Orígenes: entrenamiento 2000–2018, validación 2018–2022 y test 2022–2026; límites ajustados por horizonte. Una semilla: estabilidad no demostrada.',48,638,1184,14,MUTED);c.showPage()
base(3,'Las redes elegidas no superan a persistencia','Comparativa fuera de muestra')
metrics=list(csv.DictReader((ROOT/'artifacts/brent_v1/metrics_test.csv').open()));costs={(r['architecture'],r['horizon_bdays']):r for r in csv.DictReader((ROOT/'artifacts/brent_v1/metrics_validation.csv').open())};sel={5:'RNN',63:'LSTM',252:'RNN'}
rows=[['Días','Modelo','MAE','MSE','RMSE','R²','Mejora %','Parám.','Tiempo s']];high=[]
for r in metrics:
 co=costs[(r['architecture'],r['horizon_bdays'])];chosen=sel[int(r['horizon_bdays'])]==r['architecture'];rows.append([r['horizon_bdays'],r['architecture']+(' *' if chosen else ''),fmt(r['mae_usd']),fmt(r['mse_usd2']),fmt(r['rmse_usd']),fmt(r['r2'],3),fmt(r['improvement_rmse_pct']),co['parameters'],fmt(co['fit_seconds'])]);
 if chosen:high.append(len(rows)-1)
tab(rows,48,189,[65,175,110,120,110,105,145,125,229],[32]+[27]*12,17,high)
text('* Elegido en validación. MAE/RMSE: USD/barril; MSE: USD². Mejora positiva = menor RMSE que persistencia.',48,564,14,MUTED);text('Tiempo: ajuste principal en GPU, sin búsqueda. N de test: 1.010 / 955 / 772. Fuente: métricas del experimento.',48,589,14,MUTED)
rect(48,624,1184,35,LAVENDER);text('GRU logra el menor RMSE en test; ese bloque no se reutiliza para cambiar la selección.',62,634,18,PURPLE,'Bold');c.showPage()
base(4,'Precio, variación y rango para los tres horizontes','Demostración de la aplicación')
para('Demostración del <b>20/11/2026</b>. Último precio disponible: <b>92,94 USD</b> (17/11/2026).',48,188,1184,22)
demo=json.loads((ROOT/'reports/demo_app_2026-11-20.json').read_text());colors=[TEAL,PURPLE,TEAL]
for i,r in enumerate(demo):
 x=48+i*405;rect(x,246,374,7,[TEAL,PURPLE,LIME][i]);text(f"{r['horizon_bdays']} DÍAS · {r['architecture']}",x,278,18,colors[i],'Bold');text(fmt(r['predicted_usd']),x,322,53,INK,'Bold');text('USD/barril',x,385,18,MUTED)
 text('Objetivo: '+r['target_date'].split('-')[2]+'/'+r['target_date'].split('-')[1]+'/'+r['target_date'].split('-')[0],x,426,18)
 text('Variación: '+('+' if r['change_usd']>0 else '')+fmt(r['change_usd'])+' USD',x,458,19,colors[i],'Bold')
 text('Rango: '+fmt(r['range_low_usd'])+' a '+fmt(r['range_high_usd']),x,492,18)
line(48,538,1232,538)
para('<b>Uso de la app</b><br/>Cargar brent_2026-11-20.csv y elegir el 20/11/2026. Consultar histórico, predicción y errores. Pesos fijos.',48,561,555,19)
para('<b>Lectura del rango</b><br/>Percentiles 10/90 del error de validación. Cobertura en test: 75,0 %, 77,3 % y 90,4 %. Sin garantía futura del 80 %.',678,561,554,19)
c.linkURL('https://huggingface.co/spaces/juanhevia/brent-observatorio',(48,60,600,160),relative=0);c.showPage()
base(5,'Una herramienta de análisis con límites explícitos','Propuesta de valor')
text('Valor disponible',48,204,28,TEAL,'Bold');para('Histórico, escenarios y evidencia del error en una misma consulta.<br/><br/>Actualización manual mediante CSV sin coste de reentrenamiento.<br/><br/>Código y nueve redes publicados para reproducir los resultados.',48,261,546,24)
text('Condiciones para ampliar el uso',678,204,28,PURPLE,'Bold');para('Versión univariante: las variables externas exigidas aún no entran al modelo y falta la cesta OPEP.<br/><br/>Siguiente experimento: incorporar las variables, comprobar su disponibilidad temporal y evaluar en un nuevo período.',678,261,546,23)
rect(48,550,1184,108,INK);text('Conclusión para compras y dirección',69,568,24,LIME,'Bold');para('Útil para explorar escenarios y entender el riesgo. El test actual no demuestra una mejora frente a mantener el último precio ni justifica automatizar compras o inversiones.',69,605,1140,19,'#ffffff');c.showPage();c.save();print(OUT)
