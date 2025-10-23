# Extractor Credicoop — determinista (vFinal)
# Reglas:
# • Movimiento = fila con FECHA válida (dd/mm/aa|aaaa) a la izquierda de la columna Débito.
# • En una fila con fecha, el monto del movimiento es SIEMPRE el MÁS IZQUIERDO.
# • Clasificación por alineación (bandas): Izq=Débito, Centro=Crédito, Der=Saldo.
# • El "Saldo del día" de cada fila se IGNORA SIEMPRE.
# • "SALDO ANTERIOR" y "SALDO AL dd/mm/aaaa" se agregan como filas especiales.
# • Conciliación obligatoria: saldo_inicial − ΣDébitos + ΣCréditos == saldo_final (±$0,01).

import io, re, unicodedata
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Dict, List, Tuple

import streamlit as st
import pandas as pd
import pdfplumber

st.set_page_config(page_title="Extractor Credicoop", page_icon="📄")
st.title("📄 Extractor Credicoop")

# ---------------- Utilidades ----------------

SEP_CHARS = r"\.\u00A0\u202F\u2007 "                               # ., NBSP, NARROW_NBSP, FIGURE_SPACE, space
MONEY_RE = re.compile(rf"^\(?\$?\s*\d{{1,3}}(?:[{SEP_CHARS}]\d{{3}})*,\d{{2}}\)?$")
AMT_TXT = r"\d{1,3}(?:[.\u00A0\u202F\u2007]\d{3})*,\d{2}"
DATE_PATTS = [re.compile(r"^\d{2}/\d{2}/\d{2}$"), re.compile(r"^\d{2}/\d{2}/\d{4}$")]
DATE_STRICT = re.compile(r"^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/(\d{2}|\d{4})$")
DATE_CHARS = re.compile(r"^[0-9/]+$")

_PDF_BYTES_CACHE = b""  # para lectura de texto en saldos

def q2(x: Decimal) -> Decimal:
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def wtext(w):
    # Siempre devolver string (hotfix "expected str instance, dict found")
    t = w.get("text", "")
    return t if isinstance(t, str) else str(t)

def parse_money_es(s: str) -> Decimal:
    s = (s or "").strip()
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True; s = s[1:-1]
    s = s.replace("$", "").strip()
    for ch in ["\u00A0", "\u202F", "\u2007", " "]:
        s = s.replace(ch, "")
    s = s.replace(".", "").replace(",", ".")
    try:
        d = Decimal(s)
    except InvalidOperation:
        return Decimal("0.00")
    return q2(-d if neg else d)

def is_date_string(txt: str) -> bool:
    t = (txt or "").strip()
    return any(p.match(t) for p in DATE_PATTS)

def group_rows_by_top(words, tol: float = 2.0):
    rows = {}
    for w in words:
        rows.setdefault(round(w["top"]/tol)*tol, []).append(w)
    return [rows[k] for k in sorted(rows.keys())]

def detect_amount_runs(row_sorted, max_gap: float = 12.0):
    # Une tokens contiguos que parecen montos (dígitos, separadores, paréntesis, $)
    ALLOWED = re.compile(rf"^[\d,{SEP_CHARS}\(\)\-\$]+$")
    runs, cur, last = [], [], None
    for w in row_sorted:
        t = wtext(w)
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
        txt = "".join(wtext(w) for w in r).strip()
        x0 = min(w["x0"] for w in r); x1 = max(w["x1"] for w in r)
        out.append({"text": txt, "x0": x0, "x1": x1,
                    "top": min(w["top"] for w in r), "bottom": max(w["bottom"] for w in r)})
    return out

def normalize_token(t: str) -> str:
    # Mayúsculas, sin tildes, sólo letras (para detectar encabezados aunque estén espaciados)
    t = unicodedata.normalize("NFD", t or "").upper()
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^A-Z]", "", t)

# ---------------- Columnas (encabezado tolerante + fallback por montos) ----------------

def centers_from_headers(words) -> Dict[str, float]:
    """Detecta centros por encabezados tolerantes (DEBITO/CREDITO/SALDO, con o sin espacios/acentos)."""
    rows = group_rows_by_top(words)
    found = {}
    for row in rows:
        row_sorted = sorted(row, key=lambda w: w["x0"])
        toks = [normalize_token(wtext(w)) for w in row_sorted]
        joined = "".join(toks)  # p. ej. D E B I T O -> "DEBITO"

        def find_seq(label: str):
            idx = joined.find(label)
            if idx == -1:
                return None
            acc = ""; take = []
            for w, t in zip(row_sorted, toks):
                if len(acc) < idx and t:
                    acc += t; continue
                if len(acc) >= idx and len("".join(normalize_token(wtext(x)) for x in take)) < len(label) and t:
                    take.append(w)
                    if len("".join(normalize_token(wtext(x)) for x in take)) >= len(label):
                        break
            if not take: return None
            return (min(w["x0"] for w in take) + max(w["x1"] for w in take)) / 2.0

        for lab, key in [("DEBITO","debito"), ("CREDITO","credito"), ("SALDO","saldo")]:
            if key not in found:
                cx = find_seq(lab)
                if cx is not None:
                    found[key] = cx

    return found if len(found) == 3 else {}

def centers_from_amounts(pages_words) -> Dict[str, float]:
    """Fallback robusto: terciles por página y promedio global."""
    per_page_centers = []
    for words in pages_words:
        xs = []
        for row in group_rows_by_top(words):
            for a in detect_amount_runs(sorted(row, key=lambda w: w["x0"])):
                if MONEY_RE.match(a["text"]):
                    xs.append((a["x0"]+a["x1"])/2.0)
        if len(xs) >= 3:
            xs = sorted(xs); n = len(xs)
            c = [xs[n//6], xs[n//2], xs[5*n//6]]  # terciles robustos
            per_page_centers.append(c)
    if not per_page_centers:
        return {}
    d = sum(c[0] for c in per_page_centers) / len(per_page_centers)
    c = sum(c[1] for c in per_page_centers) / len(per_page_centers)
    s = sum(c[2] for c in per_page_centers) / len(per_page_centers)
    d, c, s = sorted([d, c, s])
    return {"debito": d, "credito": c, "saldo": s}

def compute_bands(c: Dict[str, float]):
    return {
        "borde_D": (c["debito"] + c["credito"]) / 2.0,
        "borde_C": (c["credito"] + c["saldo"]) / 2.0,
        "xD": c["debito"], "xC": c["credito"], "xS": c["saldo"]
    }

def classify_by_band(x: float, b) -> str:
    # guarda por si bordes vienen mal
    if b["borde_D"] >= b["borde_C"]:
        # fallback directo a medias entre centros
        if x <= (b["xD"] + b["xC"]) / 2.0: return "debito"
        if x <= (b["xC"] + b["xS"]) / 2.0: return "credito"
        return "saldo"
    return "debito" if x <= b["borde_D"] else ("credito" if x <= b["borde_C"] else "saldo")

# ---------------- Fecha estricta (sólo zona izquierda) ----------------

def detect_date_strict(row_sorted, bands, max_gap: float = 6.0):
    # Zona de texto = bien a la izquierda de la columna Débito
    left_limit = bands["xD"] - 40
    left = [w for w in row_sorted if w["x1"] <= left_limit]

    runs, cur, last = [], [], None
    for w in left:
        t = wtext(w)
        if DATE_CHARS.match(t):
            if last is None or (w["x0"] - last) <= max_gap:
                cur.append(w); last = w["x1"]
            else:
                runs.append(cur); cur = [w]; last = w["x1"]
        else:
            if cur: runs.append(cur); cur = []; last = None
    if cur: runs.append(cur)

    best = None
    for run in runs:
        txt = "".join(wtext(w) for w in run)
        if DATE_STRICT.fullmatch(txt):
            x0 = min(w["x0"] for w in run)
            if best is None or x0 < best[0]:
                best = (x0, txt, run)

    if not best:
        return None, []
    _, txt, toks = best
    left_desc = [w for w in left if w not in toks]
    return txt, left_desc

# ---------------- Resumen (SALDO ANTERIOR / SALDO AL ...) ----------------

def _find_last_amount_in_text(s: str) -> str|None:
    cands = list(re.finditer(AMT_TXT, s))
    return cands[-1].group(0) if cands else None

def read_summary_balances_from_text(pdf_bytes: bytes):
    saldo_ant = None; saldo_fin = None; fecha_fin = None
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for p in pdf.pages:
            txt = (p.extract_text(x_tolerance=1, y_tolerance=1) or "")
            lines = [ln.strip().upper() for ln in txt.splitlines() if ln.strip()]
            for i, ln in enumerate(lines):
                if "SALDO ANTERIOR" in ln and saldo_ant is None:
                    amt = _find_last_amount_in_text(ln) or (_find_last_amount_in_text(lines[i+1]) if i+1 < len(lines) else None)
                    if amt: saldo_ant = parse_money_es(amt)
                if "SALDO AL" in ln and saldo_fin is None:
                    m = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", ln)
                    if m: fecha_fin = m.group(1)
                    amt = _find_last_amount_in_text(ln) or (_find_last_amount_in_text(lines[i+1]) if i+1 < len(lines) else None)
                    if amt: saldo_fin = parse_money_es(amt)
    return saldo_ant, saldo_fin, fecha_fin

# ---------------- Parser principal ----------------

def parse_pdf(pdf_bytes: bytes) -> pd.DataFrame:
    global _PDF_BYTES_CACHE
    _PDF_BYTES_CACHE = pdf_bytes

    pages_words = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for p in pdf.pages:
            ws = p.extract_words(use_text_flow=True, keep_blank_chars=False,
                                 extra_attrs=["x0","x1","top","bottom"]) or []
            pages_words.append(ws)

    # 1) Columnas: encabezado tolerante; si no, fallback por montos
    centers = {}
    for ws in pages_words:
        centers = centers_from_headers(ws)
        if centers: break
    if not centers:
        centers = centers_from_amounts(pages_words)
    if not centers:
        raise RuntimeError("No pude detectar columnas (encabezado ni montos).")

    bands = compute_bands(centers)

    # 2) Saldos del resumen (por TEXTO)
    saldo_ant, saldo_fin, fecha_fin = read_summary_balances_from_text(pdf_bytes)

    # 3) Movimientos por fecha (estricta). "Saldo del día" ignorado. Flush inmediato.
    movs = []
    for words in pages_words:
        if not words: continue
        current = None
        for row in group_rows_by_top(words):
            row_sorted = sorted(row, key=lambda w: w["x0"])
            up = " ".join(wtext(w) for w in row_sorted).upper()
            if any(k in up for k in ["USTED PUEDE", "TOTALES", "SALDO ANTERIOR", "SALDO AL"]):
                continue

            # Fecha obligatoria (sólo zona izquierda)
            date_txt, left_desc = detect_date_strict(row_sorted, bands)
            if not date_txt:
                # Sólo complemento descripción anterior; jamás creo movimiento sin fecha.
                if current and left_desc:
                    extra = " ".join(wtext(w) for w in left_desc).strip()
                    if extra:
                        current["descripcion"] = (current["descripcion"] + " | " + extra).strip()
                continue

            # FLUSH del anterior para no “comer” el primero
            if current:
                movs.append(current)

            # Montos de la fila (si hay): el MOVIMIENTO es el MÁS IZQUIERDO.
            amts = [a for a in detect_amount_runs(row_sorted) if MONEY_RE.match(a["text"])]
            amts.sort(key=lambda a: a["x0"])
            deb = cre = Decimal("0.00")
            if amts:
                mov = amts[0]  # izquierda → derecha
                cx = (mov["x0"] + mov["x1"]) / 2.0
                col = classify_by_band(cx, bands)
                val = parse_money_es(mov["text"])
                if col == "debito":
                    deb = val
                elif col == "credito":
                    cre = val
                # El resto de montos (incluido SALDO) se ignoran

            current = {
                "fecha": date_txt,
                "descripcion": " ".join(wtext(w) for w in left_desc).strip(),
                "debito": deb,
                "credito": cre,
            }

        # fin de página → push si quedó algo abierto
        if current:
            movs.append(current)
            current = None

    # 4) Salida con filas especiales de saldo
    rows = []
    if saldo_ant is not None:
        first_date = movs[0]["fecha"] if movs else None
        rows.append({"tipo":"saldo_inicial","fecha":first_date,"descripcion":"SALDO ANTERIOR",
                     "debito":Decimal("0.00"),"credito":Decimal("0.00"),"saldo":saldo_ant})
    rows.extend({"tipo":"movimiento", **m, "saldo":None} for m in movs)
    if saldo_fin is not None:
        last_date = fecha_fin or (movs[-1]["fecha"] if movs else None)
        rows.append({"tipo":"saldo_final","fecha":last_date,"descripcion":"SALDO AL",
                     "debito":Decimal("0.00"),"credito":Decimal("0.00"),"saldo":saldo_fin})

    return pd.DataFrame(rows, columns=["tipo","fecha","descripcion","debito","credito","saldo"])

def reconcile(df: pd.DataFrame):
    deb = df.loc[df["tipo"]=="movimiento","debito"].sum() if not df.empty else Decimal("0.00")
    cre = df.loc[df["tipo"]=="movimiento","credito"].sum() if not df.empty else Decimal("0.00")
    si_s = df.loc[df["tipo"]=="saldo_inicial","saldo"]; si = si_s.iloc[0] if len(si_s)>0 else None
    sf_s = df.loc[df["tipo"]=="saldo_final","saldo"]; sf = sf_s.iloc[0] if len(sf_s)>0 else None
    calc = q2(si - deb + cre) if si is not None else None
    diff = q2(calc - sf) if (calc is not None and sf is not None) else None
    ok = (diff is not None) and (abs(diff) <= Decimal("0.01"))
    resumen = {
        "saldo_anterior": str(si) if si is not None else None,
        "debito_total": str(deb),
        "credito_total": str(cre),
        "saldo_final_informe": str(sf) if sf is not None else None,
        "saldo_final_calculado": str(calc) if calc is not None else None,
        "diferencia": str(diff) if diff is not None else None,
        "n_movimientos": int((df["tipo"]=="movimiento").sum()) if not df.empty else 0,
    }
    return ok, resumen

# ---------------- UI ----------------

pdf = st.file_uploader("Subí tu PDF del Banco Credicoop", type=["pdf"])
if not pdf:
    st.info("Esperando un PDF…")
    st.stop()

try:
    df = parse_pdf(pdf.read())
except Exception as e:
    st.error(f"Error al parsear: {e}")
    st.stop()

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
st.dataframe(df, use_container_width=True)

if ok:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        dfx = df.copy()
        for col in ["debito","credito","saldo"]:
            dfx[col + "_num"] = dfx[col].apply(lambda x: float(x) if x is not None else 0.0)
            dfx[col + "_centavos"] = dfx[col].apply(
                lambda x: int((x*100).to_integral_value(rounding=ROUND_HALF_UP)) if x is not None else 0
            )
            dfx[col] = dfx[col].astype(str)
        dfx.to_excel(w, index=False, sheet_name="Tabla")
    st.download_button("⬇️ Descargar Excel",
                       data=out.getvalue(),
                       file_name="credicoop_movimientos.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.error("❌ La conciliación NO cierra (±$0,01). Revisá que los montos caigan en la banda correcta y que el PDF no tenga separadores rotos.")
