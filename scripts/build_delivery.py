"""Genera los dos PDF de entrega a partir de resultados verificados.

Ejecutar con ReportLab instalado; no entrena. Solo genera el PDF de enlaces
cuando output/project_links.json contiene los tres destinos verificados.
"""
import csv
import json
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf'
OUT.mkdir(parents=True, exist_ok=True)
FONT = Path('/System/Library/Fonts/Supplemental')
for name, filename in [('Body', 'Arial.ttf'), ('Bold', 'Arial Bold.ttf'), ('Serif', 'Georgia.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(FONT / filename)))
pdfmetrics.registerFontFamily('Body', normal='Body', bold='Bold', italic='Body', boldItalic='Bold')
INK='#16343d'; TEAL='#237265'; LIME='#d3ef77'; VIOLET='#6653c7'; MUTED='#526b75'; PAPER='#f4f7f8'
W,H=1280,720
c=canvas.Canvas(str(OUT/'01_presentacion_brent.pdf'),pagesize=(W,H))
c.setTitle('Brent - Predicción con redes recurrentes');c.setAuthor('Juan Hevia Losa')

def text(s,x,y,size=20,color=INK,font='Body'):
    c.setFillColor(HexColor(color));c.setFont(font,size);c.drawString(x,H-y-size*.82,s)

def para(s,x,y,w,size=20,color=INK,leading=None):
    p=Paragraph(s,ParagraphStyle('p',fontName='Body',fontSize=size,leading=leading or size*1.36,textColor=HexColor(color)))
    _,h=p.wrap(w,H);p.drawOn(c,x,H-y-h);return h

def line(x,y,x2,y2,color='#cfdddf',width=1):
    c.setStrokeColor(HexColor(color));c.setLineWidth(width);c.line(x,H-y,x2,H-y2)

def base(n,title,eyebrow):
    c.setFillColor(HexColor(PAPER));c.rect(0,0,W,H,fill=1,stroke=0)
    text('BRENT / SERIES ECONOMY',54,26,12,MUTED,'Bold');text('JUAN HEVIA LOSA',1050,26,12,MUTED)
    line(54,58,1226,58);text(eyebrow.upper(),54,85,13,TEAL,'Bold');text(title,54,118,38,INK,'Bold')
    line(54,666,1226,666);text('Proyecto académico · RNN, LSTM y GRU · Noviembre 2026',54,684,11,MUTED)
    text(f'{n} / 5',1173,682,13,MUTED)

def chart(rows,x,y,w,h,color=TEAL,second=None):
    vals=[v for _,v in rows]+([v for _,v in second] if second else [])
    lo=min(vals);hi=max(vals);pad=max((hi-lo)*.12,1);lo-=pad;hi+=pad
    for i in range(4):
        val=lo+(hi-lo)*i/3; yy=y+h-i*h/3
        line(x,yy,x+w,yy,'#dbe5e7',.6);text(f'{val:.0f}',x-34,yy-6,11,MUTED)
    for data,col in [(rows,color)]+([(second,VIOLET)] if second else []):
        p=c.beginPath()
        for i,(_,v) in enumerate(data):
            xx=x+i/max(1,len(data)-1)*w; yy=H-(y+h-(v-lo)/(hi-lo)*h)
            (p.moveTo if i==0 else p.lineTo)(xx,yy)
        c.setStrokeColor(HexColor(col));c.setLineWidth(1.5);c.drawPath(p)
    text(rows[0][0][:4],x,y+h+12,11,MUTED);text(rows[-1][0][:4],x+w-28,y+h+12,11,MUTED)

# Portada, no computa en las cinco diapositivas.
c.setFillColor(HexColor(INK));c.rect(0,0,W,H,fill=1,stroke=0)
text('BRENT / SERIES ECONOMY',65,62,16,LIME,'Bold')
text('El petróleo,',65,178,76,'#ffffff','Bold')
text('bajo la lupa de los datos',65,265,62,LIME,'Serif')
para('Predicción del Brent con RNN, LSTM y GRU.<br/>Una aplicación para explorar escenarios y contrastar su fiabilidad.',68,390,1020,25,'#d1e1e4')
line(68,585,1210,585,'#46636a');text('Juan Hevia Losa',68,615,22,'#ffffff','Bold')
text('Aprendizaje automático · Caso práctico individual',68,653,16,'#b4cdd1')
text('Noviembre 2026',1040,653,16,'#b4cdd1');c.showPage()

# 1 Problema y oportunidad.
base(1,'Planificar compras en un mercado volátil','Problema y oportunidad')
para('El coste del petróleo altera los presupuestos de transporte y energía. Una previsión útil debe mostrar también cuánto puede equivocarse.',54,185,650,24)
text('100.000 USD',54,304,50,TEAL,'Bold')
para('de diferencia por cada 1 USD/barril en una compra de 100.000 barriles. Ejemplo ilustrativo, sin costes adicionales.',56,365,570,20)
text('La oportunidad',54,475,22,INK,'Bold')
para('Reunir histórico, escenarios y evidencia de error en una misma herramienta. Ayudar a comparar supuestos antes de decidir una compra o cobertura.',54,515,610,21)
raw=[]
with (ROOT/'data/raw/brent_2000-01-01_2026-11-05.csv').open() as f:
    for r in csv.DictReader(f):
        try: raw.append((r['Date'],float(r['brent'])))
        except ValueError: pass
text('Brent observado · USD/barril',760,197,17,INK,'Bold');chart(raw,780,250,425,275)
para('6.501 precios observados desde 2000.<br/>Último dato de esta copia: 03/11/2026.',760,566,450,17,MUTED)
text('Fuente: EIA vía FRED, DCOILBRENTEU. Copia del experimento descargada el 05/11/2026.',54,637,12,MUTED);c.showPage()

# 2 Solución y metodología.
base(2,'Tres horizontes, una evaluación temporal','La solución')
for x,num,label in [(54,'5','días · corto plazo'),(452,'63','días · medio plazo'),(850,'252','días · largo plazo')]:
    text(num,x,190,62,TEAL,'Serif');text(label,x,270,21,INK,'Bold')
line(54,321,1226,321)
para('<b>Datos y preparación</b><br/>60 entradas del Brent; escalador ajustado solo en entrenamiento. Relleno causal de hasta 5 días y retraso supuesto de publicación de 3 días.',54,350,350,21)
para('<b>Comparación en igualdad</b><br/>RNN, LSTM y GRU con las mismas particiones. Tres bloques temporales de búsqueda, dos tamaños de red y 63 ajustes en total.',452,350,350,21)
para('<b>Prueba posterior</b><br/>Selección en validación; después se congelan pesos y escaladores. Se contrasta con mantener el último precio disponible: persistencia.',850,350,350,21)
para('<b>Orígenes principales:</b> entrenamiento 2000–2018 · validación 2018–2022 · prueba 2022–2026. Los límites efectivos dependen del horizonte para evitar que los objetivos crucen bloques.',54,551,1170,18)
para('Calendario de lunes a viernes, incluidos festivos. Versión univariante: las variables externas solicitadas en el enunciado siguen pendientes.',54,617,1170,16,MUTED);c.showPage()

# 3 Tabla requerida, con el coste real del ajuste principal.
base(3,'Las redes elegidas no superan a persistencia','Comparativa fuera de muestra')
with (ROOT/'artifacts/brent_v1/metrics_test.csv').open() as f: metrics=list(csv.DictReader(f))
with (ROOT/'artifacts/brent_v1/metrics_validation.csv').open() as f: costs={(r['architecture'],r['horizon_bdays']):r for r in csv.DictReader(f)}
selected={5:'RNN',63:'LSTM',252:'RNN'}
fmt=lambda v,n=2:f'{float(v):.{n}f}'.replace('.',',')
rows=[['Días','Modelo','MAE','MSE','RMSE','R²','Mejora RMSE','Parámetros','Ajuste (s)']]
styles=[]
for i,r in enumerate(metrics,1):
    h=int(r['horizon_bdays']);a=r['architecture'];co=costs[(a,str(h))]
    rows.append([str(h),a+(' *' if selected[h]==a else ''),fmt(r['mae_usd']),fmt(r['mse_usd2']),fmt(r['rmse_usd']),fmt(r['r2'],3),fmt(r['improvement_rmse_pct'])+' %',co['parameters'],fmt(co['fit_seconds'])])
    if selected[h]==a:styles.append(('BACKGROUND',(0,i),(-1,i),HexColor('#e9e5f6')))
    if i in [4,8]:styles.append(('LINEBELOW',(0,i),(-1,i),1,HexColor('#a9bec6')))
t=Table(rows,colWidths=[55,170,105,125,105,105,165,150,132],rowHeights=[31]+[27]*12)
t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Body'),('FONTSIZE',(0,0),(-1,-1),15),('FONTNAME',(0,0),(-1,0),'Bold'),('TEXTCOLOR',(0,0),(-1,-1),HexColor(INK)),('BACKGROUND',(0,0),(-1,0),HexColor('#dce8e9')),('ALIGN',(2,0),(-1,-1),'RIGHT'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,-1),.3,HexColor('#dce5e7'))]+styles));t.wrapOn(c,1172,500);t.drawOn(c,54,H-190-355)
para('* Modelo integrado, elegido en validación. N de prueba: 1.010 / 955 / 772. MAE y RMSE en USD/barril; MSE en USD². Mejora positiva = menor RMSE que persistencia.',54,559,1170,15,MUTED)
para('GRU obtiene el menor RMSE en prueba, pero elegirla ahora reutilizaría la prueba para seleccionar. Las mejoras no son uniformes por año ni tienen significación estadística demostrada.',54,600,1170,17,INK)
text('Coste: segundos del ajuste principal de cada red en la GPU del laboratorio; no incluye búsqueda. Fuente: tablas guardadas del experimento.',54,649,11,MUTED);c.showPage()

# 4 Demostración con valores efectivamente calculados por los pesos guardados.
base(4,'Una predicción con fecha, rango y contexto','Demostración de la aplicación')
demo=json.loads((ROOT/'reports/demo_app.json').read_text())
para('Ejemplo emitido el <b>06/11/2026</b>, usando el último precio disponible de <b>107,03 USD/barril</b> (03/11/2026). Es un escenario fechado; no una previsión actualizada a hoy.',54,180,1170,21)
for x,r in zip([54,452,850],demo):
    text(f"{r['horizon_bdays']} DÍAS · {r['architecture']}",x,281,16,TEAL,'Bold')
    text(fmt(r['predicted_usd']),x,324,53,INK,'Bold');text('USD/barril',x,390,19,MUTED)
    from datetime import date
    target=date.fromisoformat(r['target_date']).strftime('%d/%m/%Y')
    text('Objetivo: '+target,x,432,18,INK)
    text('Variación: '+('+' if r['change_usd']>=0 else '')+fmt(r['change_usd'])+' USD',x,466,18,INK)
    text('Rango: '+fmt(r['range_low_usd'])+' – '+fmt(r['range_high_usd']),x,500,18,TEAL,'Bold')
    text('Cobertura en prueba: '+fmt(100*r['range_test_coverage'],1)+' %',x,535,16,MUTED)
line(54,577,1226,577)
para('El usuario elige horizonte y fecha, consulta gráficos y carga un CSV para actualizar datos sin reentrenar. Los rangos usan los percentiles 10 y 90 de los errores de validación; son orientativos, sin garantía futura del 80 %.',54,596,1170,18)
c.showPage()

# 5 Propuesta viable, condicionada por la evidencia real.
base(5,'Valor para analizar; límites para decidir','Propuesta de valor y próximos pasos')
text('Qué aporta hoy',54,193,26,TEAL,'Bold')
para('Un punto de consulta para comparar escenarios, revisar el histórico y descargar resultados.<br/><br/>La app hace explícitos el error, la referencia ingenua y la incertidumbre. Incorpora datos manualmente y mantiene los pesos fijos.',54,244,532,23)
text('Qué no ha demostrado',681,193,26,INK,'Bold')
para('Los modelos elegidos empeoran el RMSE de persistencia un <b>3,33 %, 2,04 % y 2,42 %</b>.<br/><br/>A corto plazo siguen el nivel con retraso; a 252 días las predicciones son casi constantes. Los picos recientes generan errores grandes.',681,244,545,23)
line(54,495,1226,495)
para('<b>Uso propuesto:</b> apoyo exploratorio para compras y planificación, acompañado de escenarios y criterio profesional. No se justifica una señal autónoma de inversión ni una ventaja de rentabilidad.',54,522,1170,22)
para('<b>Antes de ampliar el uso:</b> incorporar las variables externas exigidas, reservar un nuevo período de evaluación y una calibración independiente, y comprobar estabilidad con varias semillas. La versión actual es univariante; el rango comparte validación con la selección.',54,600,1170,17,MUTED)
c.showPage();c.save()

# El documento de enlaces solo se emite después de comprobar los destinos.
links_path=ROOT/'output/project_links.json'
if links_path.exists():
    links=json.loads(links_path.read_text())
    if not all(links.get(k,'').startswith('https://') for k in ['github','models','app']):
        raise ValueError('Faltan enlaces públicos verificados.')
    c=canvas.Canvas(str(OUT/'02_enlaces_proyecto.pdf'),pagesize=(595.28,841.89));W,H=595.28,841.89
    c.setTitle('Brent - Enlaces del proyecto');c.setAuthor('Juan Hevia Losa')
    text('BRENT / SERIES ECONOMY',44,48,12,TEAL,'Bold');text('Enlaces del proyecto',44,93,31,INK,'Bold')
    text('Juan Hevia Losa · Entrega individual',44,145,15,MUTED)
    for y,key,title,desc in [(218,'github','Repositorio de GitHub','Código, instrucciones de ejecución y resultados del experimento.'),(350,'models','Modelos en Hugging Face','Pesos entrenados, configuración y documentación para reproducir la inferencia.'),(482,'app','Aplicación','Predicciones a 5, 63 y 252 días, gráficos y actualización manual de datos.')]:
        line(44,y-18,550,y-18);text(title,44,y,20,INK,'Bold');para(desc,44,y+37,505,15,MUTED)
        u=links[key];para(f'<link href="{u}" color="{TEAL}"><u>{u}</u></link>',44,y+82,505,14,TEAL)
    para('Los tres destinos se han comprobado sin iniciar sesión. Los modelos se conservan tal como se seleccionaron en validación; no se reentrenan al incorporar un CSV.',44,656,505,15,MUTED)
    text('Comprobación de acceso: '+links['verified_date'],44,758,12,MUTED);c.save()
print('PDF generados en',OUT)
