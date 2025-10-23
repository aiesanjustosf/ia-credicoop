
# app_credicoop.py
# Streamlit app - Online extractor for Banco Credicoop statements (PDF) with OCR fallback.
# Upload any Credicoop PDF. The app extracts movements, reconciles balances, and lets you export to Excel.

import io, re, os, math
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import streamlit as st
import pandas as pd

# Try to import heavy deps if available in the environment
OCR_AVAILABLE = True
PDF_AVAILABLE = True
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

st.set_page_config(page_title="Extractor Credicoop Online", page_icon="📄")
st.title("📄 Extractor Credicoop Online")
st.caption("Subí tu PDF del Banco Credicoop. La app intentará extraer movimientos, conciliar saldos y exportar a Excel.")

# ---- Utilities ----
MONEY_RE = re.compile(r"\d{1,3}(?:\.\d{3})*,\d{2}")
DATE_RE  = re.compile(r"^\d{2}/\d{2}/\d{2}$")
HEADER_KEYS = ("DEBITO", "CREDITO", "SALDO")

def parse_money_es(s: str) -> Decimal:
    s = (s or "").strip()
    if s == "":
        return Decimal("0")
    s = s.replace(".", "").replace(",", ".")
    try:
        return Decimal(s)
    except InvalidOperation:
        return Decimal("0")

def to_float_or_none(x):
    try:
        return float(x)
    except Exception:
        return None

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

def find_header_positions(words) -> Optional[Dict[str, float]]:
    found = {}
    for w in words:
        t = w["text"].strip().upper()
        if t in HEADER_KEYS and t not in found:
            found[t] = (w["x0"] + w["x1"]) / 2.0
    if all(k in found for k in HEADER_KEYS):
        return found
    return None

def classify_amount(word, col_x_map):
    cx = (word["x0"] + word["x1"]) / 2.0
    nearest = min(col_x_map.items(), key=lambda kv: abs(cx - kv[1]))[0]
    return nearest.lower()

def extract_words_pdfplumber(pdf_bytes) -> List[List[dict]]:
    pages_words = []
    if not PDF_AVAILABLE:
        return pages_words
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False, extra_attrs=["x0","x1","top","bottom"])
            pages_words.append(words or [])
    return pages_words

def extract_words_ocr(pdf_bytes, dpi=300, lang="spa"):
    pages_words = []
    if not OCR_AVAILABLE:
        return pages_words
    images = convert_from_bytes(pdf_bytes, dpi=dpi)
    for img in images:
        data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
        n = len(data["text"])
        page_words = []
        for i in range(n):
            txt = (data["text"][i] or "").strip()
            if not txt:
                continue
            try:
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                page_words.append({"text": txt, "x0": float(x), "x1": float(x+w), "top": float(y), "bottom": float(y+h)})
            except Exception:
                continue
        pages_words.append(page_words)
    return pages_words

def parse_pages(pages_words):
    all_rows = []
    saldo_anterior = None
    saldo_final = None

    for words in pages_words:
        if not words:
            continue
        # detect headers
        colmap = find_header_positions(words)

        # find 'SALDO ANTERIOR' / 'SALDO AL'
        lines = words_to_lines(words)
        for line in lines:
            up = line.upper()
            if "SALDO ANTERIOR" in up:
                moneys = MONEY_RE.findall(line)
                if moneys:
                    saldo_anterior = parse_money_es(moneys[-1])
            if "SALDO AL" in up:
                moneys = MONEY_RE.findall(line)
                if moneys:
                    saldo_final = parse_money_es(moneys[-1])

        if not colmap:
            # No detailed movement table recognized on this page
            continue

        # left boundary before amounts
        left_amount_threshold = min(colmap["DEBITO"], colmap["CREDITO"]) - 20

        # Iterate row-by-row
        current = None
        for row in group_rows_by_top(words, tolerance=2.0):
            row_sorted = sorted(row, key=lambda w: w["x0"])
            line_text = " ".join(w["text"] for w in row_sorted).strip().replace("\u00a0"," ")
            up = line_text.upper()

            # Skip headers/footers
            if any(h in up for h in ("FECHA", "COMBTE", "DEBITO", "CREDITO", "SALDO")) and len(row_sorted) <= 8:
                continue
            if "SALDO ANTERIOR" in up or "SALDO AL" in up:
                continue
            if any(k in up for k in ("DEBITOS AUTOMATICOS", "CABAL DEBITO", "TRANSFERENCIAS PESOS", "USTED PUEDE", "TOTALES")):
                # Estos bloques suelen tener totales/auxiliares; se omiten
                pass

            # Amounts in this row
            amount_words = [w for w in row_sorted if MONEY_RE.fullmatch(w["text"].strip())]
            amt_map = {"debito": None, "credito": None, "saldo": None}
            for w in amount_words:
                col = classify_amount(w, colmap)
                if amt_map[col] is None:
                    amt_map[col] = parse_money_es(w["text"])

            # Left tokens (date/combte/desc)
            left_tokens = [w for w in row_sorted if w["x1"] < left_amount_threshold]
            left_texts = [w["text"] for w in left_tokens]
            date_tok = None
            combte_tok = None
            if left_texts and DATE_RE.match(left_texts[0]):
                date_tok = left_texts[0]
                if len(left_texts) > 1 and left_texts[1].isdigit():
                    combte_tok = left_texts[1]

            # Build description
            desc_tokens = left_texts[:]
            if date_tok and desc_tokens and desc_tokens[0] == date_tok:
                desc_tokens.pop(0)
            if combte_tok and desc_tokens and desc_tokens[0] == combte_tok:
                desc_tokens.pop(0)
            description = " ".join(desc_tokens).strip()

            continuation_only = (amt_map["debito"] is None and amt_map["credito"] is None and description)

            if date_tok or (amt_map["debito"] is not None or amt_map["credito"] is not None):
                if current:
                    all_rows.append(current)
                    current = None
                current = {
                    "fecha": date_tok,
                    "comprobante": combte_tok,
                    "descripcion": description,
                    "debito": amt_map["debito"] if amt_map["debito"] is not None else Decimal("0"),
                    "credito": amt_map["credito"] if amt_map["credito"] is not None else Decimal("0"),
                }
            elif continuation_only and current:
                if description:
                    current["descripcion"] = (current["descripcion"] + " | " + description) if current["descripcion"] else description
            else:
                pass

        if current:
            all_rows.append(current)

    df = pd.DataFrame(all_rows, columns=["fecha","comprobante","descripcion","debito","credito"])
    # Limpieza básica
    if not df.empty:
        df["debito"] = df["debito"].fillna(Decimal("0"))
        df["credito"] = df["credito"].fillna(Decimal("0"))
        # Eliminar filas vacías
        df = df[(df["debito"] > 0) | (df["credito"] > 0) | (df["descripcion"].fillna("").str.len() > 0)]
    return df, saldo_anterior, saldo_final

def reconcile(df, saldo_anterior, saldo_final):
    deb_total = df["debito"].sum() if not df.empty else Decimal("0")
    cred_total = df["credito"].sum() if not df.empty else Decimal("0")
    calc_final = None
    diff = None
    if saldo_anterior is not None:
        calc_final = saldo_anterior - deb_total + cred_total
    if calc_final is not None and saldo_final is not None:
        diff = calc_final - saldo_final

    # saldo calculado por fila (opcional)
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
force_ocr = st.toggle("Forzar OCR (para PDF escaneado)", value=False, help="Usá esta opción si tu PDF es una imagen o no se detecta texto.")

if uploaded:
    pdf_bytes = uploaded.read()

    # 1) Intento con texto (pdfplumber)
    pages_words = []
    if not force_ocr and PDF_AVAILABLE:
        try:
            pages_words = extract_words_pdfplumber(pdf_bytes)
        except Exception as e:
            st.info("No se pudo extraer texto directamente. Probando con OCR…")
            pages_words = []

    # 2) Fallback OCR si no hay palabras o si el usuario lo fuerza
    if force_ocr or (not pages_words or all(len(p)==0 for p in pages_words)):
        if OCR_AVAILABLE:
            with st.spinner("Ejecutando OCR…"):
                try:
                    pages_words = extract_words_ocr(pdf_bytes, dpi=300, lang="spa")
                except Exception as e:
                    st.error(f"OCR no disponible o falló: {e}")
                    pages_words = []
        else:
            st.error("OCR no disponible en este entorno. Activalo agregando Tesseract/Poppler (ver README).")

    df, saldo_anterior, saldo_final = parse_pages(pages_words)
    df, resumen = reconcile(df, saldo_anterior, saldo_final)

    st.subheader("Conciliación")
    cols = st.columns(3)
    cols[0].metric("Saldo anterior", resumen["saldo_anterior"] or "—")
    cols[1].metric("Débitos", resumen["debito_total"])
    cols[2].metric("Créditos", resumen["credito_total"])

    cols2 = st.columns(3)
    cols2[0].metric("Saldo final (informado)", resumen["saldo_final_informe"] or "—")
    cols2[1].metric("Saldo final (calculado)", resumen["saldo_calculado_final"] or "—")
    cols2[2].metric("Diferencia", resumen["diferencia"] or "—")

    if resumen["diferencia"] is not None:
        try:
            diff = Decimal(resumen["diferencia"])
            if abs(diff) < Decimal("0.01"):
                st.success("✅ Conciliación OK (diferencia < $0,01).")
            else:
                st.warning("⚠️ La conciliación no cierra. Revisá filas faltantes o lectura del PDF.")
        except Exception:
            pass

    st.subheader("Movimientos")
    if df.empty:
        st.info("No se detectaron movimientos. Si tu PDF es escaneado, activá **Forzar OCR**.")
    else:
        st.dataframe(df)

        # Exportar a Excel
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df_x = df.copy()
            # Convertir Decimal a float
            for col in ["debito","credito","saldo_calculado"]:
                if col in df_x.columns:
                    df_x[col] = df_x[col].apply(lambda x: float(x) if x is not None else 0.0)
            df_x.to_excel(writer, index=False, sheet_name="Movimientos")
            resumen_df = pd.DataFrame([resumen])
            resumen_df.to_excel(writer, index=False, sheet_name="Resumen")
        st.download_button("⬇️ Descargar Excel", data=out.getvalue(), file_name="credicoop_movimientos.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("Subí un PDF para comenzar.")

st.divider()
with st.expander("Ayuda / Notas técnicas"):
    st.markdown("""
- Si tu PDF **no contiene texto** (es una imagen escaneada), activá **Forzar OCR**.  
- Para un despliegue 100% online, usá **Streamlit Community Cloud** o **Hugging Face Spaces**.  
- Este proyecto incluye `packages.txt` para instalar **Tesseract** y **Poppler** en el servidor.
- OCR utiliza `pytesseract` (Tesseract) y `pdf2image` (Poppler).
""")
