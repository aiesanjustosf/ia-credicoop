
# app_credicoop.py (v3.2) — grandes montos: export seguro
# - Alineación fija (izq=Débito, medio=Crédito, der=Saldo)
# - Parser usa Decimal (precisión arbitraria)
# - Excel exporta numérico + string exacto + centavos (int)

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

st.set_page_config(page_title="Extractor Credicoop Online — v3.2", page_icon="📄")
st.title("📄 Extractor Credicoop Online — v3.2 (grandes montos)")
st.caption("Alineación fija. Montos en Decimal. Export a Excel: numérico + string exacto + centavos.")

MONEY_RE = re.compile(r"\d{1,3}(?:\.\d{3})*,\d{2}")
DATE_RES = [re.compile(r"^\d{2}/\d{2}/\d{2}$"), re.compile(r"^\d{2}/\d{2}/\d{4}$")]

def parse_money_es(s: str) -> Decimal:
    s = (s or "").strip()
    if s == "":
        return Decimal("0")
    s = s.replace(".", "").replace(",", ".")
    try:
        d = Decimal(s)
    except InvalidOperation:
        return Decimal("0")
    # normalizar a 2 decimales por seguridad
    return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def is_date_token(tok: str) -> bool:
    t = (tok or "").strip()
    for r in DATE_RES:
        if r.match(t):
            return True
    return False

def center_x(w): return (w["x0"] + w["x1"]) / 2.0

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

def detect_columns_fixed(pages_words):
    all_x = []
    for words in pages_words:
        for w in words:
            tt = w["text"].strip()
            if MONEY_RE.fullmatch(tt):
                all_x.append(center_x(w))
    if not all_x:
        return None
    cents = kmeans_1d(all_x, k=3, iters=60)
    uniq = []
    for c in sorted(cents):
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

def parse_with_alignment_fixed(pages_words):
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
    rows_all = []

    for words in pages_words:
        if not words: continue
        mov_cols = [v for k,v in (colmap or {}).items() if k in ("debito","credito") and v is not None]
        left_limit = min(mov_cols) - 20 if mov_cols else 999999

        current = None
        for row in group_rows_by_top(words):
            row_sorted = sorted(row, key=lambda w: w["x0"])
            up = " ".join(w["text"] for w in row_sorted).strip().upper()

            if any(h in up for h in ("FECHA","COMBTE","DEBITO","CREDITO","SALDO")) and len(row_sorted) <= 8:
                continue
            if "SALDO ANTERIOR" in up or "SALDO AL" in up:
                continue
            if "USTED PUEDE" in up or "TOTALES" in up:
                continue

            amts = [w for w in row_sorted if MONEY_RE.fullmatch(w["text"].strip()) and w["x1"] >= left_limit]
            assigned = {"debito": None, "credito": None, "saldo": None}
            for w in amts:
                col = assign_amount_to_col(center_x(w), colmap or {})
                if col in assigned and assigned[col] is None:
                    assigned[col] = parse_money_es(w["text"])

            left_tokens = [w for w in row_sorted if w["x1"] < left_limit]
            left_texts = [w["text"] for w in left_tokens]
            date_tok, combte_tok = None, None
            if left_texts and is_date_token(left_texts[0]):
                date_tok = left_texts[0]
                if len(left_texts) > 1 and left_texts[1].isdigit():
                    combte_tok = left_texts[1]
            desc_tokens = left_texts[:]
            if date_tok and desc_tokens and desc_tokens[0] == date_tok: desc_tokens.pop(0)
            if combte_tok and desc_tokens and desc_tokens[0] == combte_tok: desc_tokens.pop(0)
            description = " ".join(desc_tokens).strip()

            debit_val = assigned["debito"] or Decimal("0")
            credit_val = assigned["credito"] or Decimal("0")

            has_movement = (debit_val > 0) or (credit_val > 0)
            is_continuation = (not has_movement) and bool(description)

            if date_tok or has_movement:
                if current:
                    rows_all.append(current); current = None
                current = {
                    "fecha": date_tok,
                    "comprobante": combte_tok,
                    "descripcion": description,
                    "debito": debit_val,
                    "credito": credit_val,
                }
            elif is_continuation and current:
                current["descripcion"] = (current["descripcion"] + " | " + description) if current["descripcion"] else description

        if current:
            rows_all.append(current)

    df = pd.DataFrame(rows_all, columns=["fecha","comprobante","descripcion","debito","credito"])
    if not df.empty:
        df["debito"] = df["debito"].fillna(Decimal("0"))
        df["credito"] = df["credito"].fillna(Decimal("0"))
        df = df[(df["debito"] > 0) | (df["credito"] > 0) | (df["descripcion"].fillna("").str.len() > 0)]
    return df, saldo_anterior, saldo_final, colmap

def to_centavos(dec: Decimal) -> int:
    return int((dec * 100).to_integral_value(rounding=ROUND_HALF_UP))

def reconcile(df, saldo_anterior, saldo_final):
    deb_total = df["debito"].sum() if not df.empty else Decimal("0")
    cred_total = df["credito"].sum() if not df.empty else Decimal("0")
    calc_final = None
    diff = None
    if saldo_anterior is not None:
        calc_final = saldo_anterior - deb_total + cred_total
    if calc_final is not None and saldo_final is not None:
        diff = calc_final - saldo_final

    if saldo_anterior is not None and not df.empty:
        running = []
        bal = saldo_anterior
        for _, r in df.iterrows():
            bal = bal - r["debito"] + r["credito"]
            running.append(bal)
        df["saldo_calculado"] = running

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
    used_strategy = None

    pages_words, pages_lines = [], []
    if PDF_AVAILABLE:
        try:
            with st.spinner("Leyendo texto del PDF…"):
                pages_words, pages_lines = extract_words_pdf(pdf_bytes)
        except Exception:
            pages_words, pages_lines = [], []

    if pages_words and any(len(w)>0 for w in pages_words):
        df, s_ant, s_fin, colmap = parse_with_alignment_fixed(pages_words)
        used_strategy = "texto-posicional"
    else:
        if OCR_AVAILABLE:
            with st.spinner("OCR en progreso…"):
                try:
                    pages_words, pages_lines = extract_words_ocr(pdf_bytes, dpi=300, lang="spa")
                    df, s_ant, s_fin, colmap = parse_with_alignment_fixed(pages_words)
                    used_strategy = "ocr-posicional"
                except Exception as e:
                    st.error(f"OCR no disponible o falló: {e}")
                    df, s_ant, s_fin, colmap = pd.DataFrame(), None, None, None
        else:
            st.error("No se detectó texto en el PDF y OCR no está habilitado en este entorno.")
            df, s_ant, s_fin, colmap = pd.DataFrame(), None, None, None

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

    st.subheader("Movimientos")
    if df.empty:
        st.warning("No se detectaron movimientos. Compartí un PDF ejemplo para ajustar reglas si fuese necesario.")
    else:
        st.dataframe(df)

        # ---- Export: numeric + string + centavos ----
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df_x = df.copy()

            # Numeric (float) columns for convenience
            for col in ["debito","credito","saldo_calculado"]:
                if col in df_x.columns:
                    df_x[col + "_num"] = df_x[col].apply(lambda x: float(x) if x is not None else 0.0)

            # Exact string columns (Spanish formatting)
            for col in ["debito","credito","saldo_calculado"]:
                if col in df_x.columns:
                    df_x[col + "_str"] = df_x[col].apply(lambda d: f"{d:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

            # Centavos integer columns (lossless)
            for col in ["debito","credito","saldo_calculado"]:
                if col in df_x.columns:
                    df_x[col + "_centavos"] = df_x[col].apply(lambda d: to_centavos(d))

            # Keep original Decimal columns too (as text)
            for col in ["debito","credito","saldo_calculado"]:
                if col in df_x.columns:
                    df_x[col] = df_x[col].astype(str)

            df_x.to_excel(writer, index=False, sheet_name="Movimientos")

            # Resumen + centavos
            res = resumen.copy()
            # centavos extra
            for k in ["saldo_anterior","debito_total","credito_total","saldo_calculado_final","saldo_final_informe","diferencia"]:
                if res.get(k) not in (None, "None"):
                    try:
                        dec = Decimal(str(res[k]))
                        res[k + "_centavos"] = to_centavos(dec)
                    except Exception:
                        pass
            pd.DataFrame([res]).to_excel(writer, index=False, sheet_name="Resumen")

        st.download_button("⬇️ Descargar Excel (preciso para montos grandes)", data=out.getvalue(), file_name="credicoop_movimientos_v3_2.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.caption(f"Estrategia: {used_strategy or '—'} | Columnas X: {colmap or '—'}")

else:
    st.info("Subí un PDF para comenzar.")

st.divider()
with st.expander("Notas técnicas"):
    st.markdown("""
- Parser mantiene **Decimal** en memoria (sin límites prácticos para montos).
- Excel incluye **numérico**, **string exacto** y **centavos (int)** para evitar pérdidas con montos muy grandes.
- Alineación fija: **Débito (izq)**, **Crédito (medio)**, **Saldo (der)**. OCR solo si no hay texto.
""")
