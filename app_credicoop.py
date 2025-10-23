
# app_credicoop.py (v2)
# Streamlit app - Robust online extractor for Banco Credicoop statements (PDF) with automatic strategy + OCR fallback.
# Strategies: words -> lines -> tables -> OCR. Amount columns via 1D k-means clustering.

import io, re, os, math, statistics, random
from decimal import Decimal, InvalidOperation
from typing import List, Dict, Optional, Tuple

import streamlit as st
import pandas as pd

# Optional heavy deps
PDF_AVAILABLE = True
OCR_AVAILABLE = True
TABLE_AVAILABLE = True

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

st.set_page_config(page_title="Extractor Credicoop Online — v2", page_icon="📄")
st.title("📄 Extractor Credicoop Online — v2 (auto)")
st.caption("Subí tu PDF del Banco Credicoop. La app usa múltiples estrategias (texto/tabla/OCR) y concilia saldos.")

# ---- Utils ----
MONEY_RE = re.compile(r"\d{1,3}(?:\.\d{3})*,\d{2}")
DATE_RES = [
    re.compile(r"^\d{2}/\d{2}/\d{2}$"),      # dd/mm/yy
    re.compile(r"^\d{2}/\d{2}/\d{4}$"),     # dd/mm/yyyy
]

def parse_money_es(s: str) -> Decimal:
    s = (s or "").strip()
    if s == "":
        return Decimal("0")
    s = s.replace(".", "").replace(",", ".")
    try:
        return Decimal(s)
    except InvalidOperation:
        return Decimal("0")

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

def kmeans_1d(xs: List[float], k=3, iters=20) -> List[float]:
    """Simple 1D k-means returning sorted centroids."""
    xs = [float(x) for x in xs]
    if len(xs) < k:
        return sorted(xs + [xs[-1]]*(k-len(xs))) if xs else [0.0, 1.0, 2.0]
    # init centroids by quantiles
    qs = [i/(k) for i in range(1,k+1)]
    centroids = []
    for q in qs:
        idx = int(q*len(xs)) - 1
        idx = max(0, min(idx, len(xs)-1))
        centroids.append(xs[idx])
    for _ in range(iters):
        buckets = {i: [] for i in range(k)}
        for x in xs:
            i = min(range(k), key=lambda j: abs(x - centroids[j]))
            buckets[i].append(x)
        new_c = []
        for i in range(k):
            if buckets[i]:
                new_c.append(sum(buckets[i])/len(buckets[i]))
            else:
                new_c.append(centroids[i])
        if all(abs(new_c[i]-centroids[i]) < 1e-3 for i in range(k)):
            break
        centroids = new_c
    return sorted(centroids)

def classify_by_clusters(x, clusters_sorted):
    # debito < credito < saldo by X position
    if len(clusters_sorted) < 3:
        return "monto"
    c0, c1, c2 = clusters_sorted[0], clusters_sorted[1], clusters_sorted[2]
    d = [abs(x-c0), abs(x-c1), abs(x-c2)]
    idx = d.index(min(d))
    return ["debito","credito","saldo"][idx]

def parse_words_strategy(pages_words):
    """Parse using positioned words + clustering."""
    rows_all = []
    saldo_anterior = None
    saldo_final = None

    for words in pages_words:
        if not words: 
            continue

        # extract SALDO ANTERIOR / SALDO AL from lines
        for line in words_to_lines(words):
            up = line.upper()
            if "SALDO ANTERIOR" in up:
                m = MONEY_RE.findall(line)
                if m: saldo_anterior = parse_money_es(m[-1])
            if "SALDO AL" in up:
                m = MONEY_RE.findall(line)
                if m: saldo_final = parse_money_es(m[-1])

        # get amount centers to cluster
        amount_words = [w for w in words if MONEY_RE.fullmatch(w["text"].strip())]
        if len(amount_words) < 3:
            # not enough signals for this page
            continue
        xs = [center_x(w) for w in amount_words]
        clusters = kmeans_1d(xs, k=3)

        left_limit = min(clusters[0], clusters[1]) - 20

        current = None
        for row in group_rows_by_top(words):
            row_sorted = sorted(row, key=lambda w: w["x0"])
            line_text = " ".join(w["text"] for w in row_sorted).strip().replace("\u00a0"," ")
            up = line_text.upper()

            if any(h in up for h in ("FECHA","COMBTE","DEBITO","CREDITO","SALDO")) and len(row_sorted) <= 8:
                continue
            if "SALDO ANTERIOR" in up or "SALDO AL" in up:
                continue
            if any(k in up for k in ("USTED PUEDE", "TOTALES")):
                continue

            amts = [w for w in row_sorted if MONEY_RE.fullmatch(w["text"].strip())]
            amt_map = {"debito": None, "credito": None, "saldo": None}
            for w in amts:
                col = classify_by_clusters(center_x(w), clusters)
                if col in ("debito","credito","saldo") and amt_map[col] is None:
                    amt_map[col] = parse_money_es(w["text"])

            left_tokens = [w for w in row_sorted if w["x1"] < left_limit]
            left_texts = [w["text"] for w in left_tokens]

            date_tok = None
            combte_tok = None
            if left_texts and is_date_token(left_texts[0]):
                date_tok = left_texts[0]
                if len(left_texts) > 1 and left_texts[1].isdigit():
                    combte_tok = left_texts[1]

            desc_tokens = left_texts[:]
            if date_tok and desc_tokens and desc_tokens[0] == date_tok: desc_tokens.pop(0)
            if combte_tok and desc_tokens and desc_tokens[0] == combte_tok: desc_tokens.pop(0)
            description = " ".join(desc_tokens).strip()

            continuation_only = (amt_map["debito"] is None and amt_map["credito"] is None and description)

            if date_tok or (amt_map["debito"] is not None or amt_map["credito"] is not None):
                if current:
                    rows_all.append(current); current = None
                current = {
                    "fecha": date_tok,
                    "comprobante": combte_tok,
                    "descripcion": description,
                    "debito": amt_map["debito"] if amt_map["debito"] is not None else Decimal("0"),
                    "credito": amt_map["credito"] if amt_map["credito"] is not None else Decimal("0"),
                }
            elif continuation_only and current:
                current["descripcion"] = (current["descripcion"] + " | " + description) if current["descripcion"] else description

        if current:
            rows_all.append(current)

    df = pd.DataFrame(rows_all, columns=["fecha","comprobante","descripcion","debito","credito"])
    if not df.empty:
        df["debito"] = df["debito"].fillna(Decimal("0"))
        df["credito"] = df["credito"].fillna(Decimal("0"))
        df = df[(df["debito"] > 0) | (df["credito"] > 0) | (df["descripcion"].fillna("").str.len() > 0)]
    return df, saldo_anterior, saldo_final

def parse_lines_strategy(pages_lines: List[List[str]]):
    """Fallback: treat each line as text, parse trailing amounts and leading date/comprobante."""
    rows_all = []
    saldo_anterior = None
    saldo_final = None

    for lines in pages_lines:
        for line in lines:
            up = line.upper()
            if "SALDO ANTERIOR" in up:
                m = MONEY_RE.findall(line)
                if m: saldo_anterior = parse_money_es(m[-1]); continue
            if "SALDO AL" in up:
                m = MONEY_RE.findall(line)
                if m: saldo_final = parse_money_es(m[-1]); continue

            # Identify up to 3 money values at end of line
            m = list(MONEY_RE.finditer(line))
            if not m:
                continue
            # keep last two for debito/credito (saldo ignorado)
            last_vals = [mm.group() for mm in m[-3:]]  # at most last 3
            # Heuristic: when there are 3, assume [debito, credito, saldo]. When 2: [debito, credito] or only one: unclear.
            deb, cred = Decimal("0"), Decimal("0")
            if len(last_vals) == 3:
                deb = parse_money_es(last_vals[0]); cred = parse_money_es(last_vals[1])
            elif len(last_vals) == 2:
                # If one is zero-like, guess which is deb/cred by context: if "DEB" word in line? else use order.
                deb = parse_money_es(last_vals[0]); cred = parse_money_es(last_vals[1])
            else:
                # single amount lines are often headers/continuations
                continue

            # remove trailing amounts to get left side text
            cut = m[-1].start()
            left = line[:cut].strip()

            # Date + optional comp
            parts = left.split()
            date_tok = None
            combte_tok = None
            if parts and is_date_token(parts[0]):
                date_tok = parts[0]
                if len(parts) > 1 and parts[1].isdigit():
                    combte_tok = parts[1]
                desc = " ".join(parts[2:] if combte_tok else parts[1:]).strip()
            else:
                desc = left

            if date_tok or deb > 0 or cred > 0 or desc:
                rows_all.append({
                    "fecha": date_tok,
                    "comprobante": combte_tok,
                    "descripcion": desc,
                    "debito": deb,
                    "credito": cred,
                })

    df = pd.DataFrame(rows_all, columns=["fecha","comprobante","descripcion","debito","credito"])
    if not df.empty:
        df["debito"] = df["debito"].fillna(Decimal("0"))
        df["credito"] = df["credito"].fillna(Decimal("0"))
        df = df[(df["debito"] > 0) | (df["credito"] > 0) | (df["descripcion"].fillna("").str.len() > 0)]
    return df, saldo_anterior, saldo_final

def extract_words_pdf(pdf_bytes):
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages_words, pages_lines = [], []
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
            if not txt:
                continue
            try:
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                words.append({"text": txt, "x0": float(x), "x1": float(x+w), "top": float(y), "bottom": float(y+h)})
            except Exception:
                continue
        pages_words.append(words)
        # Simple line reconstruction by y
        lines = words_to_lines(words)
        pages_lines.append(lines)
    return pages_words, pages_lines

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

    pages_words, pages_lines = [], []
    used_strategy = None

    # Strategy 1: vector words
    if PDF_AVAILABLE:
        try:
            pages_words, pages_lines = extract_words_pdf(pdf_bytes)
        except Exception:
            pages_words, pages_lines = [], []

    df, s_ant, s_fin = pd.DataFrame(), None, None

    if pages_words and any(len(w)>0 for w in pages_words):
        df, s_ant, s_fin = parse_words_strategy(pages_words)
        used_strategy = "words"
    # Strategy 2: lines fallback (still vector text)
    if (df.empty or df.shape[0]==0) and pages_lines and any(len(l)>0 for l in pages_lines):
        df, s_ant2, s_fin2 = parse_lines_strategy(pages_lines)
        used_strategy = "lines"
        s_ant = s_ant or s_ant2
        s_fin = s_fin or s_fin2
    # Strategy 3: OCR
    if (df.empty or df.shape[0]==0) and OCR_AVAILABLE:
        with st.spinner("Ejecutando OCR…"):
            try:
                pages_words, pages_lines = extract_words_ocr(pdf_bytes, dpi=300, lang="spa")
                df, s_ant3, s_fin3 = parse_words_strategy(pages_words)
                used_strategy = "ocr-words"
                if df.empty:
                    df, s_ant3, s_fin3 = parse_lines_strategy(pages_lines)
                    used_strategy = "ocr-lines"
                s_ant = s_ant or s_ant3
                s_fin = s_fin or s_fin3
            except Exception as e:
                st.error(f"OCR no disponible o falló: {e}")

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

    if resumen["diferencia"] is not None:
        try:
            from decimal import Decimal
            diff = Decimal(resumen["diferencia"])
            if abs(diff) < Decimal("0.01"):
                st.success("✅ Conciliación OK (diferencia < $0,01).")
            else:
                st.warning("⚠️ La conciliación no cierra. Probá con otra estrategia o revisá el PDF.")
        except Exception:
            pass

    st.subheader("Movimientos")
    if df.empty:
        st.info("No se detectaron movimientos con las estrategias actuales.")
    else:
        st.dataframe(df)

        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df_x = df.copy()
            for col in ["debito","credito","saldo_calculado"]:
                if col in df_x.columns:
                    df_x[col] = df_x[col].apply(lambda x: float(x) if x is not None else 0.0)
            df_x.to_excel(writer, index=False, sheet_name="Movimientos")
            pd.DataFrame([resumen]).to_excel(writer, index=False, sheet_name="Resumen")
        st.download_button("⬇️ Descargar Excel", data=out.getvalue(), file_name="credicoop_movimientos.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.caption(f"Estrategia utilizada: {used_strategy or '—'}")

else:
    st.info("Subí un PDF para comenzar.")

st.divider()
with st.expander("Ayuda / despliegue"):
    st.markdown("""
- El sistema decide automáticamente entre texto vectorial, líneas y OCR.  
- Para despliegue 100% online, usá Streamlit Cloud o Hugging Face Spaces.  
- Incluimos `packages.txt` que instala **Tesseract** y **Poppler** en el server.
- Si seguís sin ver movimientos, compartí un PDF de ejemplo para afinar reglas específicas.
""")
