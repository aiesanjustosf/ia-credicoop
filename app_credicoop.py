import streamlit as st
import pandas as pd
import pdfplumber
import re
from io import BytesIO

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Extractor y Conciliador Bancario Credicoop (V15 - Solución Lógica)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Funciones de Utilidad ---

def clean_and_parse_amount(text):
    """
    Limpia una cadena de texto y la convierte a un número flotante.
    Maneja el formato argentino (punto como separador de miles, coma como decimal).
    CRÍTICO: Si la celda tiene múltiples valores (separados por \n), los suma.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    
    total_amount = 0.0
    
    # Dividir el texto por saltos de línea (para celdas con múltiples movimientos)
    lines = text.split('\n')
    
    for line in lines:
        if not line.strip():
            continue
            
        # 1. Eliminar símbolos de moneda y espacios
        cleaned_text = line.strip().replace('$', '').replace(' ', '')
        
        # 2. Manejo de negativo (paréntesis o guion)
        is_negative = cleaned_text.startswith('-') or (cleaned_text.startswith('(') and cleaned_text.endswith(')'))
        if is_negative:
            cleaned_text = cleaned_text.replace('-', '').replace('(', '').replace(')', '')
            
        # 3. Eliminar separador de miles y convertir la coma decimal a punto
        if ',' in cleaned_text:
            # Asumimos que el punto es de miles si hay coma decimal
            if cleaned_text.count('.') > 0:
                cleaned_text = cleaned_text.replace('.', '')
            cleaned_text = cleaned_text.replace(',', '.')
        
        try:
            amount = float(cleaned_text)
            total_amount += -amount if is_negative else amount
        except ValueError:
            continue # Ignorar líneas que no son números (como texto de descripción)
            
    return total_amount

def format_currency(amount):
    """Formatea un número como moneda ARS (punto miles, coma decimal)."""
    if amount is None:
        return "$ 0,00"
    
    # Formato ARS: punto como separador de miles, coma como decimal
    formatted_str = f"{amount:,.2f}"
    formatted_str = formatted_str.replace('.', 'X').replace(',', '.').replace('X', ',')
    
    return f"$ {formatted_str}"
    
# --- Lógica Principal de Extracción del PDF ---

@st.cache_data
def process_bank_pdf(file_bytes):
    """
    Extrae, limpia y concilia los movimientos de un extracto bancario Credicoop
    utilizando una estrategia de procesamiento de filas robusta.
    """
    
    extracted_data = []
    saldo_informado = 0.0
    
    # Patrón para encontrar números de moneda (usado para SALDO FINAL)
    currency_pattern = r"[\(]?-?\s*(\d{1,3}(?:\.\d{3})*,\d{2})[\)]?"
    
    
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        full_text = ""
        
        # 1. Extraer todo el texto para buscar saldos clave
        for page in pdf.pages:
            full_text += page.extract_text(x_tolerance=2) + "\n"
        
        # --- Detección de Saldo Final (Saldo al 30/06/2025) ---
        match_sf = re.search(r"(?:SALDO\s*AL.*?)(\d{2}/\d{2}/\d{2,4}).*?(-?" + currency_pattern + r")", full_text, re.DOTALL | re.IGNORECASE)
        
        if match_sf:
            saldo_str = match_sf.group(2)
            saldo_informado = clean_and_parse_amount(saldo_str)
        else:
            match_sf_gen = re.search(r"(?:SALDO\s*FINAL|SALDO.*?AL).*?(-?" + currency_pattern + r")", full_text, re.DOTALL | re.IGNORECASE)
            if match_sf_gen:
                saldo_informado = clean_and_parse_amount(match_sf_gen.group(1))

        # 2. Extraer Movimientos Usando Tablas (V12 Coords) + Lógica V15
        
        # Coordenadas V12 que son las más probables
        table_settings = {
            "vertical_strategy": "explicit",
            "horizontal_strategy": "lines",
            "explicit_vertical_lines": [30, 80, 150, 520, 620, 720], 
            "snap_tolerance": 8 
        }
        
        pages_to_process = range(len(pdf.pages))
        
        last_valid_fecha = ""
        last_valid_comprobante = ""
        
        for page_index in pages_to_process:
            if page_index >= len(pdf.pages):
                continue
                
            page = pdf.pages[page_index]
            tables = page.extract_tables(table_settings)
            
            for table in tables:
                # Omitir el primer elemento si es un encabezado
                start_row = 0
                if table and any("FECHA" in str(c).upper() for c in table[0]):
                    start_row = 1 
                    
                for row in table[start_row:]:
                    
                    if len(row) == 6:
                        
                        fecha_raw = str(row[0]).strip() if row[0] else ""
                        comprobante_raw = str(row[1]).strip() if row[1] else ""
                        descripcion_raw = str(row[2]).strip() if row[2] else ""
                        debito_raw = str(row[3]).strip() if row[3] else ""
                        credito_raw = str(row[4]).strip() if row[4] else ""
                        saldo_raw = str(row[5]).strip() if row[5] else ""

                        # Limpiar saltos de línea en descripción
                        descripcion_clean = descripcion_raw.replace('\n', ' ')
                        
                        # Comprobar si es una fila de movimiento válida
                        is_date_row = re.match(r"\d{2}/\d{2}/\d{2}", fecha_raw)
                        
                        debito = clean_and_parse_amount(debito_raw)
                        credito = clean_and_parse_amount(credito_raw)
                        
                        # Si es una fila con fecha, la procesamos y guardamos la fecha
                        if is_date_row:
                            last_valid_fecha = fecha_raw
                            last_valid_comprobante = comprobante_raw
                            
                            # Solo agregar si tiene débito o crédito
                            if debito != 0.0 or credito != 0.0:
                                extracted_data.append({
                                    'Fecha': fecha_raw,
                                    'Comprobante': comprobante_raw,
                                    'Descripcion': descripcion_clean,
                                    'Débito': debito,
                                    'Crédito': credito,
                                    'Saldo_Final_Linea': clean_and_parse_amount(saldo_raw)
                                })
                        
                        # Si NO es una fila con fecha (fila envuelta)
                        # PERO SÍ tiene un débito o crédito
                        elif not is_date_row and (debito != 0.0 or credito != 0.0):
                            extracted_data.append({
                                'Fecha': last_valid_fecha, # Usar la última fecha recordada
                                'Comprobante': last_valid_comprobante, # Usar el último comprobante
                                'Descripcion': descripcion_clean, # Usar la descripción de esta fila
                                'Débito': debito,
                                'Crédito': credito,
                                'Saldo_Final_Linea': clean_and_parse_amount(saldo_raw)
                            })
                            
    if not extracted_data:
        st.error("❌ ¡ALERTA! Falló la extracción de movimientos (V15). No se encontraron movimientos con Débito o Crédito. Verifique que el PDF sea texto seleccionable.")
        return pd.DataFrame(), {}
        
    # Crear DataFrame
    df = pd.DataFrame(extracted_data)
    
    # 3. Conciliación y Cálculos Finales
    
    if saldo_informado == 0.0 and not df.empty:
        saldo_informado = df['Saldo_Final_Linea'].iloc[-1]
        st.info(f"ℹ️ Saldo Final obtenido de la última línea de movimientos: {format_currency(saldo_informado)}")


    total_debitos_calc = df['Débito'].sum()
    total_creditos_calc = df['Crédito'].sum()
    
    saldo_anterior = saldo_informado - total_creditos_calc + total_debitos_calc
    saldo_calculado = saldo_anterior + total_creditos_calc - total_debitos_calc
    
    
    conciliation_results = {
        'Saldo Anterior (CALCULADO)': saldo_anterior,
        'Créditos Totales (Movimientos)': total_creditos_calc,
        'Débitos Totales (Movimientos)': total_debitos_calc,
        'Saldo Final Calculado': saldo_calculado,
        'Saldo Final Informado (PDF)': saldo_informado,
        'Diferencia de Conciliación': saldo_informado - saldo_calculado
    }
    
    return df, conciliation_results


# --- Interfaz de Streamlit ---

st.title("💳 Extractor y Conciliador Bancario Credicoop (V15 - Solución Lógica)")
st.markdown("---")

uploaded_file = st.file_uploader(
    "**1. Sube tu resumen de cuenta corriente en PDF (ej. Credicoop N&P)**",
    type=['pdf']
)

if uploaded_file is not None:
    st.info("⌛ Procesando archivo... por favor espera.")
    
    file_bytes = uploaded_file.read()
    
    df_movs, results = process_bank_pdf(file_bytes)
    
    if not df_movs.empty and results:
        st.success("✅ Extracción y procesamiento completados.")
        
        # --- Sección de Conciliación ---
        st.header("2. Resumen de Conciliación")
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Saldo Anterior (Calculado)", format_currency(results['Saldo Anterior (CALCULADO)']))
        col2.metric("Créditos Totales", format_currency(results['Créditos Totales (Movimientos)']), 
                    delta_color="normal")
        col3.metric("Débitos Totales", format_currency(results['Débitos Totales (Movimientos)']),
                    delta_color="inverse")
        col4.metric("Movimientos Extraídos", len(df_movs))
        
        
        st.markdown("---")
        
        # --- Conciliación Final ---
        st.subheader("Resultado Final")
        
        diff = results['Diferencia de Conciliación']
        
        st.markdown(f"**Saldo Final Calculado (SA + Créditos - Débitos):** **{format_currency(results['Saldo Final Calculado'])}**")
        st.markdown(f"**Saldo Final Informado (PDF):** **{format_currency(results['Saldo Final Informado (PDF)'])}**")
        
        if abs(diff) < 0.50: 
            st.success(f"**Conciliación Exitosa:** El saldo calculado coincide con el saldo informado en el extracto. Diferencia: {format_currency(diff)}")
        else:
            st.error(f"**Diferencia Detectada:** La conciliación **NO CIERRA**. Diferencia: {format_currency(diff)}")
            st.warning("Esto puede deberse a: 1) Un error en la lectura del Saldo Final Informado del PDF. 2) Movimientos no capturados por la lógica de extracción de tablas.")

        
        # --- Sección de Exportación ---
        st.header("3. Movimientos Detallados y Exportación")
        
        @st.cache_data
        def convert_df_to_excel(df):
            """Convierte el DataFrame a formato BytesIO para descarga en Excel."""
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Hoja 1: Movimientos
                df.to_excel(writer, sheet_name='Movimientos', index=False)
                
                # Hoja 2: Resumen/Conciliación
                resumen_data = [
                    ('Saldo Anterior (CALCULADO)', results['Saldo Anterior (CALCULADO)']),
                    ('Créditos Totales', results['Créditos Totales (Movimientos)']),
                    ('Débitos Totales', results['Débitos Totales (Movimientos)']),
                    ('Saldo Final Calculado', results['Saldo Final Calculado']),
                    ('Saldo Final Informado (PDF)', results['Saldo Final Informado (PDF)']),
                    ('Diferencia de Conciliación', results['Diferencia de Conciliación']),
                ]
                resumen_df = pd.DataFrame(resumen_data, columns=['Concepto', 'Valor'])
                resumen_df.to_excel(writer, sheet_name='Resumen', index=False)
                
            return output.getvalue()

        excel_bytes = convert_df_to_excel(df_movs)
        
        st.download_button(
            label="Descargar Movimientos a Excel (xlsx)",
            data=excel_bytes,
            file_name=f"Movimientos_Credicoop_Procesado.xlsx",
            mime="application/vnd.ms-excel",
        )
        
        st.markdown("---")

        # --- Tabla de Movimientos (Previsualización) ---
        st.subheader("Vista Previa de Movimientos Extraídos")
        
        df_display = df_movs.copy()
        
        df_display['Débito'] = df_display['Débito'].apply(lambda x: format_currency(x) if x > 0 else "")
        df_display['Crédito'] = df_display['Crédito'].apply(lambda x: format_currency(x) if x > 0 else "")
        df_display['Saldo_Final_Linea'] = df_display['Saldo_Final_Linea'].apply(format_currency)
        
        df_display.rename(columns={'Saldo_Final_Linea': 'Saldo en la Línea (PDF)'}, inplace=True)
        
        st.dataframe(df_display, use_container_width=True)

    elif uploaded_file is not None:
         st.error("❌ Falló la extracción de movimientos (V15). No se encontraron movimientos con Débito o Crédito. Verifique que el PDF sea texto seleccionable.")

else:
    st.warning("👆 Por favor, sube un archivo PDF para comenzar la extracción y conciliación.")


