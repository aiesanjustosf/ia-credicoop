
import io, re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import streamlit as st
import pandas as pd

# --- Helpers ---
SEP_CHARS = r"\.\u00A0\u202F\u2007 "
ALLOWED_IN_AMOUNT = re.compile(rf"^[\d,{SEP_CHARS}\(\)\-\$]+$")
MONEY_RE = re.compile(rf"^\(?\$?\s*\d{{1,3}}(?:[{SEP_CHARS}]\d{{3}})*,\d{{2}}\)?$")
DATE_RE = [re.compile(r"^\d{2}/\d{2}/\d{2}$"), re.compile(r"^\d{2}/\d{2}/\d{4}$")]
DATE_CHARS = re.compile(r"^[0-9/]+$")

def q(s): return Decimal(s).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def parse_money_es(s: str) -> Decimal:
    s = (s or "").strip()
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True; s = s[1:-1]
    s = s.replace("$","").strip()
    for ch in ["\u00A0","\u202F","\u2007"," "]:
        s = s.replace(ch, "")
    s = s.replace(".", "").replace(",", ".")
    try:
        d = Decimal(s)
    except InvalidOperation:
        return Decimal("0")
    return (-d if neg else d).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def is_date_string(txt): return any(r.match((txt or "").strip()) for r in DATE_RE)

def group_rows_by_top(words, tol=2.0):
    rows = {}
    for w in words:
        k = round(w["top"]/tol)*tol
        rows.setdefault(k, []).append(w)
    return [rows[k] for k in sorted(rows.keys())]

def detect_runs(words, allowed_regex, max_gap=12.0):
    rows = group_rows_by_top(words)
    def center(w): return (w["x0"]+w["x1"])/2.0
    assembled_rows = []
    for row in rows:
        row = sorted(row, key=lambda w: w["x0"])
        runs, cur, last = [], [], None
        for w in row:
            t = w["text"]
            ok = allowed_regex.match(t)
            if ok:
                if last is None or (w["x0"]-last)<=max_gap: cur.append(w); last = w["x1"]
                else: runs.append(cur); cur=[w]; last=w["x1"]
            else:
                if cur: runs.append(cur); cur=[]; last=None
        if cur: runs.append(cur)
        amts = []
        for r in runs:
            txt = "".join(w["text"] for w in r).strip()
            x0=min(w["x0"] for w in r); x1=max(w["x1"] for w in r)
            amts.append({"text":txt,"x0":x0,"x1":x1,"top":min(w["top"] for w in r),"bottom":max(w["bottom"] for w in r)})
        assembled_rows.append((row,amts))
    return assembled_rows

# --- PDF/ OCR ---
PDF=True; OCR=True
try:
    import pdfplumber
except Exception: PDF=False
try:
    from pdf2image import convert_from_bytes
    import pytesseract
except Exception: OCR=False

def extract_words_pdf(b):
    pages=[]
    with pdfplumber.open(io.BytesIO(b)) as pdf:
        for p in pdf.pages:
            ws = p.extract_words(use_text_flow=True, keep_blank_chars=False, extra_attrs=["x0","x1","top","bottom"]) or []
            pages.append(ws)
    return pages

def extract_words_ocr(b, dpi=300, lang="spa"):
    pages=[]
    for img in convert_from_bytes(b, dpi=dpi):
        data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
        n=len(data["text"]); ws=[]
        for i in range(n):
            t=(data["text"][i] or "").strip()
            if not t: continue
            ws.append({"text":t,"x0":float(data["left"][i]),"x1":float(data["left"][i]+data["width"][i]),"top":float(data["top"][i]),"bottom":float(data["top"][i]+data["height"][i])})
        pages.append(ws)
    return pages

def detect_columns(pages):
    xs=[]
    for words in pages:
        _, amts_rows = None, detect_runs(words, ALLOWED_IN_AMOUNT)
        for row, amts in amts_rows:
            for a in amts:
                if MONEY_RE.match(a["text"]):
                    xs.append((a["x0"]+a["x1"])/2.0)
    if not xs: return None
    xs=sorted(xs)
    # 3 clusters by simple thirds (robust + cheap)
    step=max(1,len(xs)//3); cents=[xs[min(i*step,len(xs)-1)] for i in range(3)]
    cents=sorted(cents)
    return {"debito":cents[0],"credito":cents[1],"saldo":cents[2]}

def compute_bands(colmap):
    xD,xC,xS = colmap.get("debito"), colmap.get("credito"), colmap.get("saldo")
    return {"borde_D":(xD+xC)/2.0,"borde_C":(xC+xS)/2.0,"xD":xD,"xC":xC,"xS":xS}

def classify(x,b):
    return "debito" if x<=b["borde_D"] else ("credito" if x<=b["borde_C"] else "saldo")

def detect_date(row, left_limit, max_gap=6.0):
    left=[w for w in row if w["x1"]<left_limit]
    # rebuild [0-9/] runs
    runs,cur,last=[],[],None
    for w in left:
        t=w["text"]
        if DATE_CHARS.match(t):
            if last is None or (w["x0"]-last)<=max_gap: cur.append(w); last=w["x1"]
            else: runs.append(cur); cur=[w]; last=w["x1"]
        else:
            if cur: runs.append(cur); cur=[]; last=None
    if cur: runs.append(cur)
    best=None
    for r in runs:
        txt="".join(w["text"] for w in r)
        if is_date_string(txt):
            x0=min(w["x0"] for w in r)
            if best is None or x0<best[1]: best=(txt,x0,r)
    if not best: return None, left
    date,_,tokens=best
    left_for_desc=[w for w in left if w not in tokens]
    return date, left_for_desc

def leer_saldos(pages, colmap):
    xS=colmap.get("saldo")
    saldo_ant=saldo_fin=None; fecha_fin=None
    for words in pages:
        rows=group_rows_by_top(words)
        for i,row in enumerate(rows):
            line=" ".join(w["text"] for w in sorted(row,key=lambda w:w["x0"])).upper()
            if ("SALDO ANTERIOR" in line) or ("SALDO AL" in line):
                cand=[]
                for r in (row, rows[i+1] if i+1<len(rows) else []):
                    rr=sorted(r,key=lambda w:w["x0"])
                    # assemble amounts
                    runs,cur,last=[],[],None
                    for w in rr:
                        t=w["text"]
                        if ALLOWED_IN_AMOUNT.match(t):
                            if last is None or (w["x0"]-last)<=12: cur.append(w); last=w["x1"]
                            else: runs.append(cur); cur=[w]; last=w["x1"]
                        else:
                            if cur: runs.append(cur); cur=[]; last=None
                    if cur: runs.append(cur)
                    for run in runs:
                        txt="".join(w["text"] for w in run).strip()
                        if MONEY_RE.match(txt):
                            cx=(min(w["x0"] for w in run)+max(w["x1"] for w in run))/2.0
                            cand.append((abs(cx-xS), txt))
                    if "SALDO AL" in line and fecha_fin is None:
                        joined = " ".join(w["text"] for w in rr)
                        m=re.search(r"\b(\d{2}/\d{2}/\d{4})\b", joined)
                        if m: fecha_fin=m.group(1)
                if cand:
                    monto=min(cand,key=lambda t:t[0])[1]
                    if "SALDO ANTERIOR" in line: saldo_ant=parse_money_es(monto)
                    else: saldo_fin=parse_money_es(monto)
    return saldo_ant, saldo_fin, fecha_fin

st.set_page_config(page_title="Extractor Credicoop v3.7", page_icon="📄")
st.title("📄 Extractor Credicoop v3.7 (bandas + control)")
f = st.file_uploader("Subí tu PDF", type=["pdf"])

if f:
    b=f.read()
    pages = extract_words_pdf(b) if PDF else []
    if not pages or all(len(p)==0 for p in pages):
        pages = extract_words_ocr(b) if OCR else []

    if not pages:
        st.error("No pude leer texto del PDF.")
    else:
        colmap = detect_columns(pages)
        if not colmap:
            st.error("No pude detectar columnas de montos.")
        else:
            bands = compute_bands(colmap)
            # textos/ montos por fila
            rows_mov=[]; cont=0; ignored=0
            # left limit para separar zona texto de importes
            mov_cols=[colmap["debito"], colmap["credito"]]
            left_limit=min(mov_cols)-20 if mov_cols else 999999

            for words in pages:
                assembled = detect_runs(words, ALLOWED_IN_AMOUNT)
                for row, amts in assembled:
                    up=" ".join(w["text"] for w in row).upper()
                    if any(k in up for k in ["USTED PUEDE","TOTALES","SALDO ANTERIOR","SALDO AL"]): 
                        continue
                    # clasificar montos por banda (tomar TODOS)
                    assigned={"debito":[], "credito":[], "saldo":[]}
                    for a in amts:
                        if MONEY_RE.match(a["text"]):
                            cx=(a["x0"]+a["x1"])/2.0
                            assigned[classify(cx,bands)].append(parse_money_es(a["text"]))
                    # fecha / descripción
                    date, left_for_desc = detect_date(row, left_limit)
                    if not date:
                        # continuación o saldo del día
                        if (not assigned["debito"]) and (not assigned["credito"]) and assigned["saldo"]:
                            ignored+=1
                        elif left_for_desc and (not assigned["debito"]) and (not assigned["credito"]):
                            if rows_mov:
                                extra=" ".join(w["text"] for w in left_for_desc).strip()
                                if extra:
                                    rows_mov[-1]["descripcion"] = (rows_mov[-1]["descripcion"]+" | "+extra).strip()
                                    cont+=1
                        continue
                    desc=" ".join(w["text"] for w in left_for_desc).strip()
                    rows_mov.append({"fecha":date,"descripcion":desc,"debito":sum(assigned["debito"]) if assigned["debito"] else Decimal("0"),"credito":sum(assigned["credito"]) if assigned["credito"] else Decimal("0")})

            saldo_ini, saldo_fin, fecha_fin = leer_saldos(pages, colmap)
            rows_out=[]
            if saldo_ini is not None:
                first_date = rows_mov[0]["fecha"] if rows_mov else None
                rows_out.append({"tipo":"saldo_inicial","fecha":first_date,"descripcion":"SALDO ANTERIOR","debito":Decimal("0"),"credito":Decimal("0"),"saldo":saldo_ini})
            rows_out.extend({"tipo":"movimiento", **r, "saldo":None} for r in rows_mov)
            if saldo_fin is not None:
                last_date = fecha_fin or (rows_mov[-1]["fecha"] if rows_mov else None)
                rows_out.append({"tipo":"saldo_final","fecha":last_date,"descripcion":"SALDO AL","debito":Decimal("0"),"credito":Decimal("0"),"saldo":saldo_fin})

            df=pd.DataFrame(rows_out, columns=["tipo","fecha","descripcion","debito","credito","saldo"])

            # Control
            deb=df.loc[df["tipo"]=="movimiento","debito"].sum() if not df.empty else Decimal("0")
            cre=df.loc[df["tipo"]=="movimiento","credito"].sum() if not df.empty else Decimal("0")
            si = df.loc[df["tipo"]=="saldo_inicial","saldo"]; si=si.iloc[0] if len(si)>0 else None
            sf = df.loc[df["tipo"]=="saldo_final","saldo"]; sf=sf.iloc[0] if len(sf)>0 else None
            calc = (si - deb + cre).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) if si is not None else None
            diff = (calc - sf).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) if (calc is not None and sf is not None) else None
            ok = (diff is not None) and (abs(diff) <= Decimal("0.01"))

            col1,col2,col3=st.columns(3)
            col1.metric("Saldo anterior", str(si) if si is not None else "—")
            col2.metric("Débitos", str(deb))
            col3.metric("Créditos", str(cre))
            col4,col5,col6=st.columns(3)
            col4.metric("Saldo final (inf.)", str(sf) if sf is not None else "—")
            col5.metric("Saldo final (calc.)", str(calc) if calc is not None else "—")
            col6.metric("Diferencia", str(diff) if diff is not None else "—")
            st.dataframe(df)

            if ok:
                out=io.BytesIO()
                with pd.ExcelWriter(out, engine="openpyxl") as w:
                    dfx=df.copy()
                    for col in ["debito","credito","saldo"]:
                        dfx[col+"_num"]=dfx[col].apply(lambda x: float(x) if x is not None else 0.0)
                        dfx[col+"_centavos"]=dfx[col].apply(lambda x: int((x*100).to_integral_value(rounding=ROUND_HALF_UP)) if x is not None else 0)
                        dfx[col]=dfx[col].astype(str)
                    dfx.to_excel(w, index=False, sheet_name="Tabla")
                st.download_button("⬇️ Descargar Excel (v3.7)", data=out.getvalue(), file_name="credicoop_movimientos_v3_7.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.error("La conciliación no cierra (export deshabilitado).")

else:
    st.info("Subí un PDF para comenzar.")
