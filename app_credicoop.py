
# app_credicoop.py (v3.6) — saldo inicial/final como filas + control de conciliación
# Cambios clave:
# 1) Detecta **SALDO ANTERIOR** (primera fila) y **SALDO AL ...** (última fila):
#    - Busca el importe en la línea del texto o la siguiente, y toma el más cercano al centro de la columna SALDO.
#    - Crea filas especiales: tipo='saldo_inicial' / 'saldo_final' con columna 'saldo' (debito=credito=0).
# 2) Movimientos: fecha obligatoria; descripción puede continuar; Débito=izq, Crédito=medio; SALDO (derecha) se ignora.
#    - Se eliminó el filtro por left_limit para no perder Débitos.
# 3) Control de conciliación: exige que saldo_inicial - sum(debitos) + sum(creditos) ≈ saldo_final (±$0.01) antes de exportar.

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

st.set_page_config(page_title="Extractor Credicoop Online — v3.6", page_icon="📄")
st.title("📄 Extractor Credicoop Online — v3.6")
st.caption("Saldo inicial/final como filas. Fecha obligatoria en movimientos. Alineación fija (Débito/Crédito). Control de conciliación.")

DEBUG = st.checkbox("Modo debug", False)

# Regex y separadores
SEP_CHARS = r"\.\u00A0\u202F\u2007 "  # .  NBSP  NARROW_NBSP  FIGURE_SPACE  space
ALLOWED_IN_AMOUNT = re.compile(rf"^[\d,{SEP_CHARS}\(\)\-\$]+$")
MONEY_RE = re.compile(rf"^\(?\$?\s*\d{{1,3}}(?:[{SEP_CHARS}]\d{{3}})*,\d{{2}}\)?$")
DATE_RES = [re.compile(r"^\d{2}/\d{2}/\d{2}$"), re.compile(r"^\d{2}/\d{2}/\d{4}$")]
DATE_CHARS = re.compile(r"^[0-9/]+$")

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

def is_date_string(txt: str) -> bool:
    t = (txt or "").strip()
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
    # Une tokens contiguos de posibles montos (dígitos, separadores, $, paréntesis)
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
        assembled.append({"text": text, "x0": x0, "x1": x1, "top": min(w["top"] for w in run), "bottom": max(w["bottom"] for w in run), "tokens": run})
    return assembled

def detect_date_run(left_tokens, max_gap=6.0):
    runs, current, last_x1 = [], [], None
    for w in left_tokens:
        t = w["text"]
        if DATE_CHARS.match(t):
            if last_x1 is None or (w["x0"] - last_x1) <= max_gap:
                current.append(w); last_x1 = w["x1"]
            else:
                runs.append(current); current = [w]; last_x1 = w["x1"]
        else:
            if current: runs.append(current); current = []; last_x1 = None
    if current: runs.append(current)
    best = None
    for run in runs:
        txt = "".join(w["text"] for w in run)
        if is_date_string(txt):
            x0 = min(w["x0"] for w in run)
            if best is None or x0 < best[2]:
                best = (txt, run, x0)
    if best:
        return best[0], best[1]
    return None, []

def detect_columns_fixed(pages_words):
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

def leer_saldos_resumen(pages_words, colmap):
    """Detecta SALDO ANTERIOR y SALDO AL ... buscando el importe en la línea del texto o la siguiente.
       Prioriza el importe más cercano a la columna 'saldo'."""
    saldo_ant = None
    saldo_fin = None
    fecha_fin = None

    for words in pages_words:
        rows = group_rows_by_top(words)
        for i, row in enumerate(rows):
            row_sorted = sorted(row, key=lambda w: w["x0"])
            line_txt = " ".join(w["text"] for w in row_sorted).upper()
            if ("SALDO ANTERIOR" in line_txt) or ("SALDO AL" in line_txt):
                candidates = []
                for r in (row, rows[i+1] if i+1 < len(rows) else []):
                    if not r: continue
                    r_sorted = sorted(r, key=lambda w: w["x0"])
                    runs = detect_amount_runs(r_sorted)
                    for run in runs:
                        if MONEY_RE.match(run["text"]):
                            cx_run = (run["x0"] + run["x1"]) / 2.0
                            dist = abs(cx_run - ((colmap or {}).get("saldo") or 1e9))
                            candidates.append((dist, run["text"]))
                    # extra: fecha final si aparece en misma línea (después de "SALDO AL")
                    if "SALDO AL" in line_txt:
                        # buscamos un patrón dd/mm/aaaa en la misma línea
                        joined = " ".join(w["text"] for w in r_sorted)
                        m = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", joined)
                        if m:
                            fecha_fin = m.group(1)
                if candidates:
                    monto_txt = min(candidates, key=lambda t: t[0])[1]
                    if "SALDO ANTERIOR" in line_txt:
                        saldo_ant = parse_money_es(monto_txt)
                    else:
                        saldo_fin = parse_money_es(monto_txt)
    return saldo_ant, saldo_fin, fecha_fin

def parse_movements(pages_words):
    colmap = detect_columns_fixed(pages_words)
    if DEBUG: st.write("Columnas X (centros):", colmap or "—")

    # 1) Leer saldos de resumen con método robusto
    saldo_anterior, saldo_final, fecha_final = leer_saldos_resumen(pages_words, colmap)

    rows_mov, cont_desc_count, ignored_saldo_dia = [], 0, 0

    for words in pages_words:
        if not words: continue
        mov_cols = [v for k,v in (colmap or {}).items() if k in ("debito","credito") and v is not None]
        left_limit = min(mov_cols) - 20 if mov_cols else 999999

        current = None
        for row in group_rows_by_top(words):
            row_sorted = sorted(row, key=lambda w: w["x0"])
            up = " ".join(w["text"] for w in row_sorted).strip().upper()

            # Ignorar líneas de resumen / encabezados
            if "USTED PUEDE" in up or "TOTALES" in up: continue
            if ("SALDO ANTERIOR" in up) or ("SALDO AL" in up): continue

            # Montos detectados (tomar TODOS y dejar que la alineación decida)
            runs = detect_amount_runs(row_sorted)
            amts = [r for r in runs if MONEY_RE.match(r["text"])]
            # Asignación por columna
            assigned = {"debito": [], "credito": [], "saldo": []}
            for r in amts:
                col = assign_amount_to_col((r["x0"] + r["x1"]) / 2.0, colmap or {})
                if col in assigned:
                    assigned[col].append(parse_money_es(r["text"]))

            # Texto izquierdo
            left_tokens = [w for w in row_sorted if w["x1"] < left_limit]
            # Reconstruir FECHA
            date_txt, date_tokens = detect_date_run(left_tokens, max_gap=6.0)
            date_tok = date_txt if date_txt else None
            left_tokens_for_desc = [w for w in left_tokens if w not in date_tokens]

            # Combte opcional
            combte_tok = None
            if date_tokens:
                last_x1 = max(w["x1"] for w in date_tokens)
                candidates = [w for w in left_tokens_for_desc if w["x0"] >= last_x1 - 1 and w["text"].isdigit()]
                if candidates:
                    near = min(candidates, key=lambda w: w["x0"])
                    if (near["x0"] - last_x1) <= 12.0:
                        combte_tok = near["text"]
                        left_tokens_for_desc = [w for w in left_tokens_for_desc if w is not near]

            # Sin fecha => continuación o saldo del día; no crea movimiento
            if not date_tok:
                if (not assigned["debito"]) and (not assigned["credito"]) and assigned["saldo"]:
                    ignored_saldo_dia += 1
                    continue
                if left_tokens_for_desc and (not assigned["debito"]) and (not assigned["credito"]):
                    if current:
                        extra = " ".join(w["text"] for w in left_tokens_for_desc).strip()
                        if extra:
                            current["descripcion"] = (current["descripcion"] + " | " + extra) if current["descripcion"] else extra
                            cont_desc_count += 1
                    continue
                continue

            # Con fecha => nuevo movimiento
            description = " ".join(w["text"] for w in left_tokens_for_desc).strip()

            # Elegir ÚNICO monto por fila (débito o crédito). Si hay más de uno por columna, sumarlos (poco común).
            debit_val = sum(assigned["debito"]) if assigned["debito"] else Decimal("0")
            credit_val = sum(assigned["credito"]) if assigned["credito"] else Decimal("0")

            if current:
                rows_mov.append(current); current = None
            current = {
                "fecha": date_tok,
                "comprobante": combte_tok,
                "descripcion": description,
                "debito": debit_val,
                "credito": credit_val,
            }

        if current:
            rows_mov.append(current)

    # 2) Insertar filas especiales de saldo inicial/final
    rows_out = []
    if saldo_anterior is not None:
        # Fecha para saldo inicial: preferimos la primera fecha de movimientos; si no hay, dejamos None
        first_date = rows_mov[0]["fecha"] if rows_mov else None
        rows_out.append({"tipo": "saldo_inicial", "fecha": first_date, "comprobante": None, "descripcion": "SALDO ANTERIOR", "debito": Decimal("0"), "credito": Decimal("0"), "saldo": saldo_anterior})
    rows_out.extend({"tipo": "movimiento", **r, "saldo": None} for r in rows_mov)
    if saldo_final is not None:
        # Fecha para saldo final: usar la fecha detectada en "SALDO AL ..." si se pudo; si no, última fecha de movimiento
        last_date = fecha_final or (rows_mov[-1]["fecha"] if rows_mov else None)
        rows_out.append({"tipo": "saldo_final", "fecha": last_date, "comprobante": None, "descripcion": "SALDO AL", "debito": Decimal("0"), "credito": Decimal("0"), "saldo": saldo_final})

    df = pd.DataFrame(rows_out, columns=["tipo","fecha","comprobante","descripcion","debito","credito","saldo"])
    stats = {"continuaciones_agregadas": cont_desc_count, "saldos_dia_ignorados": ignored_saldo_dia}
    return df, stats, colmap

def reconciliar_control(df):
    """Devuelve (ok:bool, resumen:dict, diff:Decimal)"""
    # Totales
    deb_total = df.loc[df["tipo"]=="movimiento", "debito"].sum() if not df.empty else Decimal("0")
    cred_total = df.loc[df["tipo"]=="movimiento", "credito"].sum() if not df.empty else Decimal("0")
    saldo_ini = df.loc[df["tipo"]=="saldo_inicial", "saldo"]
    saldo_fin = df.loc[df["tipo"]=="saldo_final", "saldo"]
    saldo_ini = saldo_ini.iloc[0] if len(saldo_ini)>0 else None
    saldo_fin = saldo_fin.iloc[0] if len(saldo_fin)>0 else None

    calc_fin = None
    diff = None
    if saldo_ini is not None:
        calc_fin = saldo_ini - deb_total + cred_total
    if calc_fin is not None and saldo_fin is not None:
        diff = (calc_fin - saldo_fin).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    ok = (diff is not None) and (abs(diff) <= Decimal("0.01"))
    resumen = {
        "saldo_anterior": str(saldo_ini) if saldo_ini is not None else None,
        "debito_total": str(deb_total),
        "credito_total": str(cred_total),
        "saldo_final_informe": str(saldo_fin) if saldo_fin is not None else None,
        "saldo_final_calculado": str(calc_fin) if calc_fin is not None else None,
        "diferencia": str(diff) if diff is not None else None,
        "n_movimientos": int((df["tipo"]=="movimiento").sum()) if not df.empty else 0,
    }
    return ok, resumen, diff

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

    if not pages_words or all(len(p)==0 for p in pages_words):
        if OCR_AVAILABLE:
            with st.spinner("OCR en progreso…"):
                try:
                    pages_words, pages_lines = extract_words_ocr(pdf_bytes, dpi=300, lang="spa")
                except Exception as e:
                    st.error(f"OCR no disponible o falló: {e}")
                    pages_words = []

    df, stats, colmap = parse_movements(pages_words)

    # Control de conciliación
    ok, resumen, diff = reconciliar_control(df)

    st.subheader("Control de conciliación")
    c = st.columns(3)
    c[0].metric("Saldo anterior", resumen["saldo_anterior"] or "—")
    c[1].metric("Débitos", resumen["debito_total"])
    c[2].metric("Créditos", resumen["credito_total"])
    c2 = st.columns(3)
    c2[0].metric("Saldo final (informado)", resumen["saldo_final_informe"] or "—")
    c2[1].metric("Saldo final (calculado)", resumen["saldo_final_calculado"] or "—")
    c2[2].metric("Diferencia", resumen["diferencia"] or "—")

    if not ok:
        st.error("❌ La conciliación NO cierra. Revisá 'Saldo anterior', 'SALDO AL' y que los montos estén en la columna correcta. No se habilita la exportación hasta que cierre ±$0,01.")
    else:
        st.success("✅ Conciliación correcta (±$0,01).")

    st.subheader("Tabla (incluye saldo inicial/final como filas)")
    if df.empty:
        st.warning("No se detectaron filas. Activá debug y compartí la salida.")
    else:
        st.dataframe(df)

        if ok:
            # Export
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="openpyxl") as writer:
                df_x = df.copy()
                # Columnas numéricas auxiliares
                for col in ["debito","credito","saldo"]:
                    if col in df_x.columns:
                        df_x[col + "_num"] = df_x[col].apply(lambda x: float(x) if x is not None else 0.0)
                        df_x[col + "_centavos"] = df_x[col].apply(lambda d: int((d*100).to_integral_value(rounding=ROUND_HALF_UP)) if d is not None else 0)
                        df_x[col] = df_x[col].astype(str)
                df_x.to_excel(writer, index=False, sheet_name="Tabla")
                pd.DataFrame([resumen]).to_excel(writer, index=False, sheet_name="Resumen")
            st.download_button("⬇️ Descargar Excel (v3.6)", data=out.getvalue(), file_name="credicoop_movimientos_v3_6.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if DEBUG:
        st.divider()
        st.write("Columnas X (centros):", colmap or "—")
        st.write("Stats:", stats)

else:
    st.info("Subí un PDF para comenzar.")
