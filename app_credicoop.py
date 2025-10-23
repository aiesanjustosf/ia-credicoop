
# app_credicoop.py (v3.4) — reglas exactas por fila/fecha
# Reglas:
# 1) Cada fila válida TIENE FECHA (dd/mm/yy o dd/mm/yyyy). Sin fecha => continuación de descripción.
# 2) Débito=columna izquierda, Crédito=columna del medio. "Saldo" (derecha) se IGNORA SIEMPRE.
# 3) "SALDO ANTERIOR" y "SALDO AL dd/mm/aaaa" no son movimientos (sólo resumen).
# 4) Líneas "saldo del día" (importe en col. SALDO, con o sin fecha) se ignoran.
# 5) Export seguro para montos grandes.

import io, re, os
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import List, Dict, Optional, Tuple

import streamlit as st
import pandas as pd

PDF_AVAILABLE = True
OCR_AVAILABLE = True
try:
    import pdfplumber
except Exception:
    PDF_AVAILABLE = False
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    from PIL import Image
except Exception:
    OCR_AVAILABLE = False

st.set_page_config(page_title="Extractor Credicoop Online — v3.4", page_icon="📄")
st.title("📄 Extractor Credicoop Online — v3.4 (fecha obligatoria, saldo omitido)")
st.caption("Cada movimiento tiene FECHA. Descripciones sin fecha se anexan arriba. Saldo (columna derecha) siempre se ignora.")

DEBUG = st.checkbox("Modo debug", False)

# Separadores y regex robustos
SEP_CHARS = r"\.\u00A0\u202F\u2007 "  # .  NBSP  NARROW_NBSP  FIGURE_SPACE  space
ALLOWED_IN_AMOUNT = re.compile(rf"^[\d,{SEP_CHARS}\(\)\-\$]+$")
MONEY_RE = re.compile(rf"^\(?\$?\s*\d{{1,3}}(?:[{SEP_CHARS}]\d{{3}})*,\d{{2}}\)?$")
DATE_RES = [re.compile(r"^\d{2}/\d{2}/\d{2}$"), re.compile(r"^\d{2}/\d{2}/\d{4}$")]

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
    if neg: d = -d
    return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def is_date_token(tok: str) -> bool:
    t = (tok or "").strip()
    for r in DATE_RES:
        if r.match(t):
            return True
    return False

def cx(w): return (w["x0"] + w["x1"]) / 2.0

def group_rows_by_top(words, tolerance=2.0):
    rows = {}
    for w in words:
        key = round(w["top"]/tolerance)*tolerance
        rows.setdefault(key, []).append(w)
    return [rows[k] for k in sorted(rows.keys())]

def words_to_lines(words) -> List[str]:
    lines = []
    for row in group_rows_by_top(words, tolerance=2.0):
        row_sorted = sorted(row, key=lambda w: w["x0"])
        lines.append(" ".join(w["text"] for w in row_sorted))
    return lines

def kmeans_1d(xs, k=3, iters=60):
    xs = sorted(float(x) for x in xs)
    if len(xs) < k:
        cents = xs + [xs[-1]]*(k-len(xs)) if xs else [0.0, 1.0, 2.0]
        return cents
    step = max(1, len(xs)//k)
    cents = [xs[min(i*step, len(xs)-1)] for i in range(k)]
    for _ in range(iters):
        buckets = {i: [] for i in range(k)}
        for x in xs:
            i = min(range(k), key=lambda j: abs(x - cents[j]))
            buckets[i].append(x)
        new_c = []
        for i in range(k):
            if buckets[i]:
                new_c.append(sum(buckets[i])/len(buckets[i]))
            else:
                new_c.append(cents[i])
        if all(abs(new_c[i]-cents[i]) < 1e-3 for i in range(k)):
            break
        cents = new_c
    return sorted(cents)

def extract_words_pdf(pdf_bytes):
    pages_words, pages_lines = [], []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False, extra_attrs=["x0","x1","top","bottom"]) or []
            pages_words.append(words)
            text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
            pages_lines.append(text.splitlines())
    return pages_words, pages_lines

def extract_words_ocr(pdf_bytes, dpi=300, lang="spa"):
    pages_words, pages_lines = [], []
    images = convert_from_bytes(pdf_bytes, dpi=dpi)
    for img in images:
        data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
        n = len(data["text"])
        words = []
        for i in range(n):
            txt = (data["text"][i] or "").strip()
            if not txt: continue
            try:
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                words.append({"text": txt, "x0": float(x), "x1": float(x+w), "top": float(y), "bottom": float(y+h)})
            except Exception:
                continue
        pages_words.append(words)
        lines = words_to_lines(words)
        pages_lines.append(lines)
    return pages_words, pages_lines

def detect_amount_runs(row_sorted, max_gap=12.0):
    runs, current, last_x1 = [], [], None
    for w in row_sorted:
        t = w["text"]
        if ALLOWED_IN_AMOUNT.match(t):
            if last_x1 is None or (w["x0"] - last_x1) <= max_gap:
                current.append(w); last_x1 = w["x1"]
            else:
                runs.append(current); current = [w]; last_x1 = w["x1"]
        else:
            if current: runs.append(current); current = []; last_x1 = None
    if current: runs.append(current)
    assembled = []
    for run in runs:
        text = "".join(w["text"] for w in run).strip()
        x0 = min(w["x0"] for w in run); x1 = max(w["x1"] for w in run)
        assembled.append({"text": text, "x0": x0, "x1": x1, "top": min(w["top"] for w in run), "bottom": max(w["bottom"] for w in run)})
    return assembled

def detect_columns_fixed(pages_words):
    # Fija: izq=debito, medio=credito, der=saldo (por clúster global de X de montos)
    all_x = []
    for words in pages_words:
        for row in group_rows_by_top(words):
            row_sorted = sorted(row, key=lambda w: w["x0"])
            runs = detect_amount_runs(row_sorted)
            for r in runs:
                if MONEY_RE.match(r["text"]):
                    all_x.append((r["x0"] + r["x1"]) / 2.0)
    if not all_x: return None
    cents = kmeans_1d(all_x, k=3, iters=60)
    uniq = []
    for c in cents:
        if not uniq or abs(c - uniq[-1]) > 10:
            uniq.append(c)
    if len(uniq) >= 3:
        return {"debito": uniq[0], "credito": uniq[1], "saldo": uniq[2]}
    elif len(uniq) == 2:
        return {"debito": uniq[0], "credito": None, "saldo": uniq[1]}
    else:
        return {"debito": uniq[0], "credito": None, "saldo": None}

def assign_amount_to_col(x, colmap):
    order = ["debito","credito","saldo"]
    best, best_d = None, 1e18
    for k in order:
        v = (colmap or {}).get(k)
        if v is None: continue
        d = abs(x - v)
        if d < best_d:
            best, best_d = k, d
    return best or "monto"

def to_centavos(dec: Decimal) -> int:
    return int((dec * 100).to_integral_value(rounding=ROUND_HALF_UP))

def parse_movements(pages_words):
    # Leer SALDOS de resumen (para conciliación, no como movimientos)
    saldo_anterior, saldo_final = None, None
    for words in pages_words:
        for line in words_to_lines(words):
            up = line.upper()
            if "SALDO ANTERIOR" in up:
                m = re.findall(MONEY_RE, line)
                if m: saldo_anterior = parse_money_es(m[-1])
            if "SALDO AL" in up:
                m = re.findall(MONEY_RE, line)
                if m: saldo_final = parse_money_es(m[-1])

    colmap = detect_columns_fixed(pages_words)
    if DEBUG: st.write("Columnas X (centros):", colmap or "—")

    rows_out = []
    cont_desc_count = 0
    ignored_saldo_dia = 0

    for words in pages_words:
        if not words: continue
        # ubicación que separa "texto" de "importes"
        mov_cols = [v for k,v in (colmap or {}).items() if k in ("debito","credito") and v is not None]
        left_limit = min(mov_cols) - 20 if mov_cols else 999999

        current = None
        for row in group_rows_by_top(words):
            row_sorted = sorted(row, key=lambda w: w["x0"])
            up = " ".join(w["text"] for w in row_sorted).strip().upper()

            # Ignorar encabezados y resumen
            if ("FECHA" in up and "DESCRIPCION" in up) or "USTED PUEDE" in up:
                continue
            if "SALDO ANTERIOR" in up or "SALDO AL" in up:
                continue

            # Montos (sólo a la derecha del bloque de importes)
            runs = detect_amount_runs(row_sorted)
            amts = [r for r in runs if MONEY_RE.match(r["text"]) and r["x1"] >= left_limit]
            assigned = {"debito": None, "credito": None, "saldo": None}
            for r in amts:
                col = assign_amount_to_col((r["x0"] + r["x1"]) / 2.0, colmap or {})
                if col in assigned and assigned[col] is None:
                    assigned[col] = parse_money_es(r["text"])

            # Texto a la izquierda (fecha/combte/desc)
            left_tokens = [w for w in row_sorted if w["x1"] < left_limit]
            left_texts = [w["text"] for w in left_tokens]

            # Si NO hay fecha al inicio -> es continuación de descripción / "saldo del día"
            date_tok, combte_tok = None, None
            if left_texts and is_date_token(left_texts[0]):
                date_tok = left_texts[0]
                if len(left_texts) > 1 and left_texts[1].isdigit():
                    combte_tok = left_texts[1]

            # Caso A: no hay fecha => continuación (NUNCA crea movimiento)
            if not date_tok:
                # Si sólo hay importe en 'saldo' (y sin deb/cred), es "saldo del día" => ignorar
                if (assigned["debito"] is None and assigned["credito"] is None) and assigned["saldo"] is not None:
                    ignored_saldo_dia += 1
                    continue
                # anexar texto a descripción actual
                if left_texts and (assigned["debito"] is None and assigned["credito"] is None):
                    if current:
                        extra = " ".join(left_texts).strip()
                        if extra:
                            current["descripcion"] = (current["descripcion"] + " | " + extra) if current["descripcion"] else extra
                            cont_desc_count += 1
                    continue
                # si hay importes en deb/cred sin fecha (no debería pasar): los ignoramos
                continue

            # Caso B: hay fecha => NUEVO movimiento
            # Descripción: todo lo que queda a la izquierda tras quitar fecha y combte
            desc_tokens = left_texts[:]
            if desc_tokens and desc_tokens[0] == date_tok: desc_tokens.pop(0)
            if combte_tok and desc_tokens and desc_tokens[0] == combte_tok: desc_tokens.pop(0)
            description = " ".join(desc_tokens).strip()

            # Tomar débito o crédito (saldo se ignora siempre)
            debit_val = assigned["debito"] or Decimal("0")
            credit_val = assigned["credito"] or Decimal("0")

            # Cerrar movimiento anterior y abrir este
            if current:
                rows_out.append(current); current = None
            current = {
                "fecha": date_tok,
                "comprobante": combte_tok,
                "descripcion": description,
                "debito": debit_val,
                "credito": credit_val,
            }

        if current:
            rows_out.append(current)

    df = pd.DataFrame(rows_out, columns=["fecha","comprobante","descripcion","debito","credito"])
    # Drop filas vacías de importes si las hubiera (pero todas tienen fecha)
    if not df.empty:
        df["debito"] = df["debito"].fillna(Decimal("0"))
        df["credito"] = df["credito"].fillna(Decimal("0"))
        # Al menos uno de los dos puede ser 0, pero conservamos la fila porque tiene fecha y descripción

    stats = {"continuaciones_agregadas": cont_desc_count, "saldos_dia_ignorados": ignored_saldo_dia}
    return df, saldo_anterior, saldo_final, stats, colmap

def reconcile(df, saldo_anterior, saldo_final):
    deb_total = df["debito"].sum() if not df.empty else Decimal("0")
    cred_total = df["credito"].sum() if not df.empty else Decimal("0")
    calc_final, diff = None, None
    if saldo_anterior is not None:
        calc_final = saldo_anterior - deb_total + cred_total
    if calc_final is not None and saldo_final is not None:
        diff = calc_final - saldo_final
    resumen = {
        "saldo_anterior": str(saldo_anterior) if saldo_anterior is not None else None,
        "debito_total": str(deb_total),
        "credito_total": str(cred_total),
        "saldo_calculado_final": str(calc_final) if calc_final is not None else None,
        "saldo_final_informe": str(saldo_final) if saldo_final is not None else None,
        "diferencia": str(diff) if diff is not None else None,
        "n_registros": int(len(df)),
    }
    return df, resumen

# ---- UI ----
uploaded = st.file_uploader("Subí tu PDF del Banco Credicoop", type=["pdf"])

if uploaded:
    pdf_bytes = uploaded.read()

    pages_words, pages_lines = [], []
    if PDF_AVAILABLE:
        try:
            with st.spinner("Leyendo texto del PDF…"):
                pages_words, pages_lines = extract_words_pdf(pdf_bytes)
        except Exception:
            pages_words, pages_lines = [], []

    # Si no hay palabras, intentar OCR (ultimo recurso)
    if not pages_words or all(len(p)==0 for p in pages_words):
        if OCR_AVAILABLE:
            with st.spinner("OCR en progreso…"):
                try:
                    pages_words, pages_lines = extract_words_ocr(pdf_bytes, dpi=300, lang="spa")
                except Exception as e:
                    st.error(f"OCR no disponible o falló: {e}")
                    pages_words = []

    df, s_ant, s_fin, stats, colmap = parse_movements(pages_words)
    df, resumen = reconcile(df, s_ant, s_fin)

    st.subheader("Conciliación")
    c = st.columns(3)
    c[0].metric("Saldo anterior", resumen["saldo_anterior"] or "—")
    c[1].metric("Débitos", resumen["debito_total"])
    c[2].metric("Créditos", resumen["credito_total"])
    c2 = st.columns(3)
    c2[0].metric("Saldo final (informado)", resumen["saldo_final_informe"] or "—")
    c2[1].metric("Saldo final (calculado)", resumen["saldo_calculado_final"] or "—")
    c2[2].metric("Diferencia", resumen["diferencia"] or "—")

    st.subheader("Movimientos (fecha obligatoria)")
    if df.empty:
        st.warning("No se detectaron movimientos con fecha. Mandame un ejemplo para ajustar reglas.")
    else:
        st.dataframe(df)

        # Export seguro (Decimal + num + str + centavos)
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df_x = df.copy()
            for col in ["debito","credito"]:
                df_x[col + "_num"] = df_x[col].apply(lambda x: float(x) if x is not None else 0.0)
                df_x[col + "_str"] = df_x[col].apply(lambda d: f"{d:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
                df_x[col + "_centavos"] = df_x[col].apply(lambda d: int((d*100).to_integral_value(rounding=ROUND_HALF_UP)))
            for col in ["debito","credito"]:
                df_x[col] = df_x[col].astype(str)
            df_x.to_excel(writer, index=False, sheet_name="Movimientos")
            pd.DataFrame([resumen]).to_excel(writer, index=False, sheet_name="Resumen")
        st.download_button("⬇️ Descargar Excel", data=out.getvalue(), file_name="credicoop_movimientos_v3_4.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if DEBUG:
        st.divider()
        st.write("Columnas X (centros):", colmap or "—")
        st.write("Stats:", stats)

else:
    st.info("Subí un PDF para comenzar.")
