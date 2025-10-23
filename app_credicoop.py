# Extractor Credicoop — encabezados anclados + bandas + control
# Reglas:
# - Movimiento = fila con FECHA; la descripción puede continuar en renglones sin fecha.
# - Débito = banda izquierda, Crédito = banda central, Saldo (derecha) se IGNORA en filas.
# - “SALDO ANTERIOR” y “SALDO AL dd/mm/aaaa” se leen (línea y siguiente) y se agregan como filas especiales.
# - Control: saldo_inicial − ΣDébitos + ΣCréditos == saldo_final (±$0,01). Si NO cierra, no hay export.

import io, re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import List, Tuple, Dict
import streamlit as st
import pandas as pd
import pdfplumber

st.set_page_config(page_title="Extractor Credicoop", page_icon="📄")
st.title("📄 Extractor Credicoop (encabezados anclados)")

# ------------ utilidades ------------
SEP_CHARS = r"\.\u00A0\u202F\u2007 "            # miles: . NBSP etc
MONEY_RE = re.compile(rf"^\(?\$?\s*\d{{1,3}}(?:[{SEP_CHARS}]\d{{3}})*,\d{{2}}\)?$")
DATE_RES = [re.compile(r"^\d{2}/\d{2}/\d{2}$"), re.compile(r"^\d{2}/\d{2}/\d{4}$")]
DATE_CHARS = re.compile(r"^[0-9/]+$")

def q2(x: Decimal) -> Decimal:
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

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
        return Decimal("0.00")
    return q2(-d if neg else d)

def is_date_string(txt: str) -> bool:
    t = (txt or "").strip()
    return any(r.match(t) for r in DATE_RES)

def group_rows_by_top(words, tol=2.0):
    rows = {}
    for w in words:
        key = round(w["top"]/tol)*tol
        rows.setdefault(key, []).append(w)
    return [rows[k] for k in sorted(rows.keys())]

def detect_amount_runs(row_sorted, max_gap=12.0):
    # Une tokens de monto contiguos (dígitos, ., NBSP, ',', '(', ')', '$', '-')
    ALLOWED = re.compile(rf"^[\d,{SEP_CHARS}\(\)\-\$]+$")
    runs, cur, last = [], [], None
    for w in row_sorted:
        t = w["text"]
        if ALLOWED.match(t):
            if last is None or (w["x0"] - last) <= max_gap:
                cur.append(w); last = w["x1"]
            else:
                runs.append(cur); cur = [w]; last = w["x1"]
        else:
            if cur: runs.append(cur); cur = []; last = None
    if cur: runs.append(cur)
    out = []
    for r in runs:
        txt = "".join(w["text"] for w in r).strip()
        x0 = min(w["x0"] for w in r); x1 = max(w["x1"] for w in r)
        out.append({"text": txt, "x0": x0, "x1": x1, "top": min(w["top"] for w in r), "bottom": max(w["bottom"] for w in r)})
    return out

def rebuild_date_from_left(row_sorted, left_limit, max_gap=6.0) -> Tuple[str, list]:
    left = [w for w in row_sorted if w["x1"] < left_limit]
    runs, cur, last = [], [], None
    for w in left:
        t = w["text"]
        if DATE_CHARS.match(t):
            if last is None or (w["x0"]-last) <= max_gap:
                cur.append(w); last = w["x1"]
            else:
                runs.append(cur); cur = [w]; last = w["x1"]
        else:
            if cur: runs.append(cur); cur = []; last = None
    if cur: runs.append(cur)
    best = None
    for r in runs:
        txt = "".join(w["text"] for w in r)
        if is_date_string(txt):
            x0 = min(w["x0"] for w in r)
            if (best is None) or (x0 < best[1]): best = (txt, x0, r)
    if not best:
        return None, left
    date, _, tokens = best
    left_desc = [w for w in left if w not in tokens]
    return date, left_desc

# ------------ anclaje de columnas por encabezado ------------
def normalize(s: str) -> str:
    return (s or "").replace(".", "").replace("́","").replace("’","").strip().upper()

def find_column_centers_by_header(words) -> Dict[str, float]:
    # Busca palabras "DEBITO", "CREDITO", "SALDO" y toma el centro X de cada una
    centers = {}
    for w in words:
        t = normalize(w["text"])
        if t in ("DEBITO","DÉBITO") and "debito" not in centers:
            centers["debito"] = (w["x0"] + w["x1"]) / 2.0
        elif t in ("CREDITO","CRÉDITO") and "credito" not in centers:
            centers["credito"] = (w["x0"] + w["x1"]) / 2.0
        elif t == "SALDO" and "saldo" not in centers:
            centers["saldo"] = (w["x0"] + w["x1"]) / 2.0
    # Si falta alguno, no seguimos; el PDF de Credicoop trae estos textos
    return centers if len(centers) == 3 else {}

def compute_bands(c):
    # bandas: [.. debito .. | .. credito .. | .. saldo ..]
    return {
        "borde_D": (c["debito"] + c["credito"]) / 2.0,
        "borde_C": (c["credito"] + c["saldo"]) / 2.0,
        "xD": c["debito"], "xC": c["credito"], "xS": c["saldo"]
    }

def classify_by_band(x, bands):
    if x <= bands["borde_D"]: return "debito"
    elif x <= bands["borde_C"]: return "credito"
    else: return "saldo"

# ------------ lectura de saldos (resumen) ------------
def pick_summary_amount(row, next_row, xS) -> str:
    # toma importe(s) de row y la siguiente y elige el más cercano a la columna SALDO
    rows = []
    for r in (row, next_row):
        if not r: continue
        rr = sorted(r, key=lambda w: w["x0"])
        amts = detect_amount_runs(rr)
        for a in amts:
            if MONEY_RE.match(a["text"]):
                cx = (a["x0"] + a["x1"]) / 2.0
                rows.append((abs(cx - xS), a["text"]))
    if not rows: return None
    return min(rows, key=lambda t: t[0])[1]

def read_summary_balances(pages_words, centers):
    saldo_ant = None; saldo_fin = None; fecha_fin = None
    xS = centers["saldo"]
    for words in pages_words:
        rows = group_rows_by_top(words)
        for i, row in enumerate(rows):
            txt = " ".join(w["text"] for w in sorted(row, key=lambda w: w["x0"])).upper()
            if "SALDO ANTERIOR" in txt:
                amt = pick_summary_amount(row, rows[i+1] if i+1 < len(rows) else [], xS)
                if amt: saldo_ant = parse_money_es(amt)
            if "SALDO AL" in txt:
                amt = pick_summary_amount(row, rows[i+1] if i+1 < len(rows) else [], xS)
                if amt: saldo_fin = parse_money_es(amt)
                # fecha final si viene
                joined = " ".join(w["text"] for w in sorted(row, key=lambda w: w["x0"]))
                m = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", joined)
                if m: fecha_fin = m.group(1)
    return saldo_ant, saldo_fin, fecha_fin

# ------------ parser principal ------------
def parse_pdf(pdf_bytes: bytes):
    pages_words = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for p in pdf.pages:
            ws = p.extract_words(use_text_flow=True, keep_blank_chars=False,
                                 extra_attrs=["x0","x1","top","bottom"]) or []
            pages_words.append(ws)

    # 1) anclar columnas por encabezado en la primera página que las tenga
    centers = {}
    for ws in pages_words:
        centers = find_column_centers_by_header(ws)
        if centers: break
    if not centers:
        raise RuntimeError("No se detectaron encabezados DEBITO/CREDITO/SALDO.")

    bands = compute_bands(centers)
    movs = []

    # 2) saldos del resumen
    saldo_ant, saldo_fin, fecha_fin = read_summary_balances(pages_words, centers)

    # 3) movimientos por fecha (ignorando columna SALDO en las filas)
    for words in pages_words:
        if not words: continue
        left_limit = bands["borde_D"] - 20  # separa zona de texto vs importes
        current = None

        for row in group_rows_by_top(words):
            row_sorted = sorted(row, key=lambda w: w["x0"])
            up = " ".join(w["text"] for w in row_sorted).upper()
            if any(k in up for k in ["USTED PUEDE","TOTALES","SALDO ANTERIOR","SALDO AL"]):
                continue

            # Montos: tomar TODOS los importes y clasificar por banda
            amts = [a for a in detect_amount_runs(row_sorted) if MONEY_RE.match(a["text"])]
            assigned = {"debito": [], "credito": [], "saldo": []}
            for a in amts:
                cx = (a["x0"] + a["x1"]) / 2.0
                assigned[classify_by_band(cx, bands)].append(parse_money_es(a["text"]))

            # Fecha y descripción
            date_txt, left_desc_tokens = rebuild_date_from_left(row_sorted, left_limit)
            if not date_txt:
                # sin fecha: continuación de descripción, siempre que NO haya monto en débito/crédito
                if (not assigned["debito"]) and (not assigned["credito"]):
                    if current:
                        extra = " ".join(w["text"] for w in left_desc_tokens).strip()
                        if extra:
                            current["descripcion"] = (current["descripcion"] + " | " + extra).strip()
                continue

            # con fecha: crear movimiento nuevo (saldo de la fila se ignora)
            descripcion = " ".join(w["text"] for w in left_desc_tokens).strip()
            deb = sum(assigned["debito"]) if assigned["debito"] else Decimal("0.00")
            cre = sum(assigned["credito"]) if assigned["credito"] else Decimal("0.00")

            if current:
                movs.append(current)
            current = {"fecha": date_txt, "descripcion": descripcion,
                       "debito": deb, "credito": cre}

        if current:
            movs.append(current)

    # 4) armar tabla con saldos inicial/final como filas especiales
    rows_out = []
    if saldo_ant is not None:
        first_date = movs[0]["fecha"] if movs else None
        rows_out.append({"tipo":"saldo_inicial","fecha":first_date,"descripcion":"SALDO ANTERIOR",
                         "debito":Decimal("0.00"),"credito":Decimal("0.00"),"saldo":saldo_ant})
    rows_out.extend({"tipo":"movimiento", **m, "saldo":None} for m in movs)
    if saldo_fin is not None:
        last_date = fecha_fin or (movs[-1]["fecha"] if movs else None)
        rows_out.append({"tipo":"saldo_final","fecha":last_date,"descripcion":"SALDO AL",
                         "debito":Decimal("0.00"),"credito":Decimal("0.00"),"saldo":saldo_fin})

    df = pd.DataFrame(rows_out, columns=["tipo","fecha","descripcion","debito","credito","saldo"])
    return df

def reconcile(df: pd.DataFrame):
    deb = df.loc[df["tipo"]=="movimiento","debito"].sum() if not df.empty else Decimal("0.00")
    cre = df.loc[df["tipo"]=="movimiento","credito"].sum() if not df.empty else Decimal("0.00")
    si_s = df.loc[df["tipo"]=="saldo_inicial","saldo"]; si = si_s.iloc[0] if len(si_s)>0 else None
    sf_s = df.loc[df["tipo"]=="saldo_final","saldo"]; sf = sf_s.iloc[0] if len(sf_s)>0 else None
    calc = q2(si - deb + cre) if si is not None else None
    diff = q2(calc - sf) if (calc is not None and sf is not None) else None
    ok = (diff is not None) and (abs(diff) <= Decimal("0.01"))
    return ok, {
        "saldo_anterior": str(si) if si is not None else None,
        "debito_total": str(deb),
        "credito_total": str(cre),
        "saldo_final_informe": str(sf) if sf is not None else None,
        "saldo_final_calculado": str(calc) if calc is not None else None,
        "diferencia": str(diff) if diff is not None else None,
        "n_movimientos": int((df["tipo"]=="movimiento").sum()) if not df.empty else 0,
    }

# ------------ UI ------------
pdf = st.file_uploader("Subí tu PDF del Banco Credicoop", type=["pdf"])
if not pdf:
    st.info("Esperando un PDF…")
    st.stop()

df = parse_pdf(pdf.read())
ok, resumen = reconcile(df)

st.subheader("Conciliación")
c = st.columns(3)
c[0].metric("Saldo anterior", resumen["saldo_anterior"] or "—")
c[1].metric("Débitos", resumen["debito_total"])
c[2].metric("Créditos", resumen["credito_total"])
c2 = st.columns(3)
c2[0].metric("Saldo final (informado)", resumen["saldo_final_informe"] or "—")
c2[1].metric("Saldo final (calculado)", resumen["saldo_final_calculado"] or "—")
c2[2].metric("Diferencia", resumen["diferencia"] or "—")

st.subheader("Movimientos")
st.dataframe(df)

if ok:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        dfx = df.copy()
        for col in ["debito","credito","saldo"]:
            dfx[col+"_num"] = dfx[col].apply(lambda x: float(x) if x is not None else 0.0)
            dfx[col+"_centavos"] = dfx[col].apply(lambda x: int((x*100).to_integral_value(rounding=ROUND_HALF_UP)) if x is not None else 0)
            dfx[col] = dfx[col].astype(str)
        dfx.to_excel(w, index=False, sheet_name="Tabla")
    st.download_button("⬇️ Descargar Excel", data=out.getvalue(),
                       file_name="credicoop_movimientos.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.error("❌ La conciliación NO cierra (±$0,01). Revisá que Débitos/Créditos estén en la banda correcta.")
