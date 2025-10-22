import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import io
import re

# ----------------------------------------------------------------------
# 1. FUNCIÓN DE EXTRACCIÓN Y LIMPIEZA DE TEXTO (SOLUCIÓN ROBUSTA)
# ----------------------------------------------------------------------

def extract_tables_from_pdf(uploaded_file):
    """
    Extrae texto crudo del PDF y usa expresiones regulares 
    para parsear las filas de movimientos del Credicoop.
    """
    pdf_bytes = uploaded_file.read()
    all_rows = []

    # Patrón RegEx para identificar una línea de movimiento:
    # Captura la FECHA, ignora COMBTE, y captura los campos DESCRIPCION, DEBITO, CREDITO y SALDO.
    # El patrón se basa en el formato específico de tu PDF:
    # "FECHA\n", "COMBTE\n", "DESCRIPCION\n", "DEBITO\n", "CREDITO\n", "SALDO\n"
    # Los campos numéricos están al final de la línea.
    
    # Este patrón busca el inicio de una línea con una FECHA (DD/MM/AA) o la palabra SALDO
    # y luego captura los 4 campos numéricos (Débito, Crédito, Saldo, y un campo extra)
    
    # 26/06/25
    # 01/07/25
    # 01/07/25

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            # Líneas de corte para las columnas basadas en el PDF de Credicoop (aprox)
            # FECHA | COMBTE | DESCRIPCION              | DEBITO | CREDITO | SALDO
            # 0     | 70     | 180                      | 450    | 570     | 700
            
            x_settings = [
                0,    # FECHA
                70,   # COMBTE
                180,  # DESCRIPCION
                450,  # DEBITO
                570,  # CREDITO
                700   # SALDO
            ]

            for page in pdf.pages:
                
                # Usamos extract_table con una lista de líneas de corte (x_settings) para forzar la división.
                # Se eliminan los parámetros que causaron el error (como keep_blank_chars)
                settings = {
                    "vertical_strategy": "explicit",
                    "horizontal_strategy": "text", # Más flexible horizontalmente
                    "explicit_vertical_lines": x_settings,
                    "snap_tolerance": 3,
                    "text_only": False,
                }
                
                tables = page.extract_tables(table_settings=settings)
                
                if tables:
                    for table in tables:
                        if table:
                            all_rows.extend(table)
        
        if not all_rows:
            st.error("No se pudieron extraer datos del PDF. Intenta con un PDF de mejor calidad.")
            return None

        # 2. Convertir a DataFrame y Limpieza Estructural

        df = pd.DataFrame(all_rows)
        df.dropna(how='all', inplace=True)

        # Buscar el encabezado ("FECHA", "DESCRIPCION", etc.)
        header_index = None
        for i, row in df.iterrows():
            if any(col in str(row[2]).upper() for col in ['DESCRIPCION', 'SALDO']):
                header_index = i
                break
        
        if header_index is None:
             st.warning("No se pudo identificar la fila de encabezado. Asumiendo la primera fila.")
             df.columns = df.iloc[0]
             df = df.iloc[1:].reset_index(drop=True)
        else:
            df.columns = df.iloc[header_index]
            df = df.iloc[header_index + 1:].reset_index(drop=True)

        
        # Renombrar y seleccionar las columnas clave (asumiendo que el corte funcionó)
        df.columns = ['FECHA', 'COMBTE', 'DESCRIPCION', 'DEBITO', 'CREDITO', 'SALDO']
        df = df[['FECHA', 'DESCRIPCION', 'DEBITO', 'CREDITO', 'SALDO']].copy()
        
        # Limpiar valores nulos y strings 'None'
        df = df.replace('None', np.nan).replace('', np.nan)
        
        # 3. Consolidación de Filas Parciales (la parte más crítica)
        
        df['FECHA'] = df['FECHA'].fillna(method='ffill')
        df.dropna(subset=['FECHA'], inplace=True) # Elimina filas que no pudieron ser asignadas a una fecha

        df_processed = []
        current_row = {}
        for _, row in df.iterrows():
            
            # Condición de continuación: No tiene Débito, Crédito o Saldo, pero tiene Descripción.
            has_no_value = pd.isna(row['DEBITO']) and pd.isna(row['CREDITO']) and pd.isna(row['SALDO'])
            has_description = not pd.isna(row['DESCRIPCION'])
            
            is_continuation_row = has_no_value and has_description

            if not is_continuation_row:
                if current_row: 
                    df_processed.append(current_row)
                current_row = row.to_dict()
                
            elif current_row:
                 # Añadir la descripción a la fila principal
                current_row['DESCRIPCION'] = str(current_row.get('DESCRIPCION', '')) + " " + str(row['DESCRIPCION'])
            
            elif not current_row and row['FECHA']: # Caso de la primera fila o SALDO ANTERIOR
                 current_row = row.to_dict()

        if current_row:
            df_processed.append(current_row)

        df_final = pd.DataFrame(df_processed)
        df_final.dropna(subset=['DESCRIPCION'], inplace=True)
        
        return df_final.drop_duplicates().reset_index(drop=True)

    except Exception as e:
        st.error(f"Error en la extracción de PDF: {e}")
        return None

# ----------------------------------------------------------------------
# FUNCIONES DE LIMPIEZA Y ANÁLISIS (Se mantienen robustas)
# ----------------------------------------------------------------------

def limpiar_y_transformar_df(df):
    """Limpia los datos, convierte formatos y calcula el NETO."""

    required_cols = ['FECHA', 'DESCRIPCION', 'DEBITO', 'CREDITO', 'SALDO']
    df.columns = required_cols 

    def limpiar_y_convertir_numerico(series):
        # Limpia miles (punto) y convierte la coma en punto decimal.
        series = series.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        return pd.to_numeric(series, errors='coerce').fillna(0)

    df['DEBITO'] = limpiar_y_convertir_numerico(df['DEBITO'])
    df['CREDITO'] = limpiar_y_convertir_numerico(df['CREDITO'])
    df['SALDO'] = limpiar_y_convertir_numerico(df['SALDO'])

    try:
        df['FECHA'] = pd.to_datetime(df['FECHA'], dayfirst=True, errors='coerce')
        df.dropna(subset=['FECHA'], inplace=True) 
    except Exception:
        st.warning("No se pudo convertir la columna 'FECHA'. Asegúrate de que el formato sea 'DD/MM/AA'.")
        return None

    df['NETO'] = df['CREDITO'] - df['DEBITO']
    return df

def categorizar_movimiento(descripcion, debito, credito):
    """Asigna una categoría a cada movimiento basado en la descripción."""
    desc = str(descripcion).upper()
    
    # SALDOS
    if 'SALDO ANTERIOR' in desc:
        return 'SALDO INICIAL'

    # DÉBITOS / GASTOS / PAGOS
    if debito > 0:
        if 'IMPUESTO LEY 25.413' in desc:
            if 'S/CREDITOS' in desc:
                return 'Impuesto - Ley 25.413 (Débito, asociado a Crédito)' 
            return 'Impuesto - Ley 25.413 (Débito)'
        elif 'I.V.A.' in desc or 'PERCEPCION IVA' in desc:
            return 'Impuesto - IVA / Percepciones'
        elif 'PAGO DE CHEQUE' in desc or 'COMISION CHEQUE PAGADO' in desc:
            return 'Gasto - Cheques y Comisiones'
        elif 'TRANSFER.' in desc and ('DISTINTO TITULAR' in desc or 'O/BCO' in desc or 'PIANETTI' in desc):
            return 'Gasto - Transferencia Pagada a Terceros'
        elif 'TRANSF.' in desc and 'IGUAL TITULAR' in desc:
            return 'Transferencia Interna (Egreso)'
        elif 'COMISION POR TRANSFERENCIA' in desc or 'SERVICIO MODULO' in desc or 'ECHQ- COMIS' in desc:
            return 'Gasto - Comisiones Bancarias'
        elif 'DEBITO/CREDITO AUT SEGURCOOP' in desc:
            return 'Gasto - Seguros/Débito Automático'
        elif 'INTERESES POR SALDO DEUDOR' in desc:
            return 'Gasto - Intereses Deudores'
        elif 'PAGO DE OBLIGACIONES A ARCA' in desc:
            return 'Gasto - Pago de Impuestos ARCA'

    # CRÉDITOS / INGRESOS
    elif credito > 0:
        if 'PAGO A COMERCIOS CABAL' in desc:
            return 'Ingreso - Venta con Cabal'
        elif 'CREDITO INMEDIATO' in desc or 'ACREDITAC' in desc or 'ECHO' in desc:
            return 'Ingreso - Acreditación de Cheque/DEBIN'
        elif 'TRANSFER.' in desc and ('DISTINTO TITULAR' in desc or 'O/BCO' in desc):
            return 'Ingreso - Transferencia Recibida'
        elif 'TRANSFERENCIA INMEDIATA E/CTAS. PROPIAS' in desc:
            return 'Transferencia Interna (Ingreso)'
            
    return 'Otros/Movimiento no categorizado'

def analizar_movimientos(df):
    """Aplica la categorización y realiza el cálculo de saldo."""
    
    df['CATEGORIA'] = df.apply(lambda row: categorizar_movimiento(row['DESCRIPCION'], row['DEBITO'], row['CREDITO']), axis=1)
    
    saldo_inicial_row = df[df['CATEGORIA'] == 'SALDO INICIAL']
    saldo_inicial = saldo_inicial_row['SALDO'].iloc[0] if not saldo_inicial_row.empty else 0

    df_movimientos = df[df['CATEGORIA'] != 'SALDO INICIAL'].copy()
    
    neto_acumulado = df_movimientos['NETO'].cumsum()
    
    df['SALDO_RECALCULADO'] = saldo_inicial + neto_acumulado.reindex(df.index).fillna(method='ffill').fillna(saldo_inicial)

    return df, saldo_inicial

# ----------------------------------------------------------------------
# LÓGICA DE STREAMLIT
# ----------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("🏦 Analizador y Conciliador Bancario Credicoop (Solución Final Online)")
st.markdown("Sube tu resumen de cuenta en formato **PDF** para intentar extraer los movimientos mediante análisis de texto. Este es un intento forzado para la complejidad del PDF del Credicoop en un entorno 100% online.")
st.markdown("---")

uploaded_file = st.file_uploader("Sube tu archivo PDF de movimientos bancarios", type=['pdf'])

if uploaded_file is not None:
    
    with st.spinner('Extrayendo y parseando texto del PDF... Esto puede tardar unos segundos.'):
        df_original = extract_tables_from_pdf(uploaded_file)
    
    if df_original is not None and not df_original.empty:
        try:
            with st.spinner('Limpiando y analizando datos...'):
                df_limpio = limpiar_y_transformar_df(df_original.copy())

            if df_limpio is not None and not df_limpio.empty:
                df_analizado, saldo_inicial = analizar_movimientos(df_limpio)
                
                # ... (Resto de la lógica de presentación) ...
                saldo_final_calculado = df_analizado['SALDO_RECALCULADO'].iloc[-1]
                saldo_final_resumen = df_analizado['SALDO'].iloc[-1] 
                diferencia = saldo_final_calculado - saldo_final_resumen

                # --- PRESENTACIÓN DE RESULTADOS ---

                st.header("1. Conciliación de Saldo")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Saldo Inicial", f"${saldo_inicial:,.2f}")
                col2.metric("Saldo Final Calculado", f"${saldo_final_calculado:,.2f}")
                col3.metric("Diferencia (Calculado - Resumen)", f"${diferencia:,.2f}", delta_color='inverse')

                if abs(diferencia) < 0.01:
                    st.success("✅ **¡Conciliado!** El saldo final calculado coincide con el saldo final reportado en el resumen. (Diferencia: $0,00)")
                else:
                    st.error(f"🚨 **¡Atención!** Hay una diferencia de ${diferencia:,.2f}. **Revisa la tabla 3** para ver si la extracción del PDF fue imperfecta, ya que es una extracción forzada.")

                st.markdown("---")
                st.header("2. Resumen de Gastos y Créditos por Categoría")

                df_resumen = df_analizado[~df_analizado['CATEGORIA'].isin(['SALDO INICIAL', 'Transferencia Interna (Ingreso)', 'Transferencia Interna (Egreso)', 'Otros/Movimiento no categorizado'])]
                resumen_neto = df_resumen.groupby('CATEGORIA')['NETO'].sum().sort_values(ascending=False)

                ingresos = resumen_neto[resumen_neto > 0]
                gastos = resumen_neto[resumen_neto < 0].abs()
                
                col4, col5 = st.columns(2)

                with col4:
                    st.subheader("Ingresos Totales (Créditos)")
                    st.table(ingresos.apply(lambda x: f"${x:,.2f}").reset_index(name='Total Neto'))
                    st.markdown(f"**Total Ingresos Netos: ${ingresos.sum():,.2f}**")

                with col5:
                    st.subheader("Gastos y Débitos (Débitos)")
                    st.table(gastos.apply(lambda x: f"${x:,.2f}").reset_index(name='Total Neto')) 
                    st.markdown(f"**Total Gastos Netos: ${gastos.sum():,.2f}**")

                st.markdown("---")
                st.header("3. Detalle Completo de Movimientos y Saldo Recalculado")
                
                df_final_display = df_analizado[['FECHA', 'DESCRIPCION', 'DEBITO', 'CREDITO', 'NETO', 'CATEGORIA', 'SALDO_RECALCULADO']].copy()
                df_final_display['FECHA'] = df_final_display['FECHA'].dt.strftime('%d/%m/%Y')
                
                for col in ['DEBITO', 'CREDITO', 'NETO', 'SALDO_RECALCULADO']:
                    df_final_display[col] = df_final_display[col].apply(lambda x: f"${x:,.2f}")

                st.dataframe(df_final_display, use_container_width=True, hide_index=True)

            else:
                st.error("El DataFrame de movimientos está vacío.")

        except Exception as e:
            st.error(f"Error crítico durante el análisis: {e}")
            
    else:
        st.error("La extracción del PDF falló. La estructura del documento del Banco Credicoop es demasiado compleja para ser procesada consistentemente en un entorno 100% Python/Streamlit Cloud.")


else:
    st.info("Sube tu archivo PDF de movimientos bancarios para comenzar el análisis.")
