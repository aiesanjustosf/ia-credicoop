import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import io
import re
import os # Aunque no se usa para temp files, lo mantenemos por si acaso

# ----------------------------------------------------------------------
# 1. FUNCIÓN DE EXTRACCIÓN Y LIMPIEZA CON PDFPLUMBER (ONLINE COMPATIBLE)
# ----------------------------------------------------------------------

def extract_tables_from_pdf(uploaded_file):
    """
    Extrae tablas de movimientos del PDF usando pdfplumber.
    Esta función es puramente Python y funciona en Streamlit Cloud.
    """
    
    pdf_bytes = uploaded_file.read()
    all_data = []

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                # Usar extract_tables() con la configuración predeterminada (lattice mode)
                tables = page.extract_tables()
                
                if tables:
                    for table in tables:
                        if table:
                            all_data.extend(table)
        
        if not all_data:
            st.error("No se pudieron extraer datos de tablas del PDF. Verifica el formato.")
            return None

        df = pd.DataFrame(all_data)

        # 1. Buscar la fila de encabezado
        header_index = None
        target_cols = ['FECHA', 'DESCRIPCION', 'DEBITO', 'CREDITO', 'SALDO']

        for i, row in df.iterrows():
            row_str = " ".join(str(x).upper() for x in row.dropna().astype(str))
            if any(col in row_str for col in ['DESCRIPCION', 'DEBITO', 'CREDITO', 'SALDO']):
                header_index = i
                break

        if header_index is None:
            st.error("No se pudo identificar la fila de encabezado (FECHA, DESCRIPCION, etc.).")
            return None

        # 2. Asignar la fila de encabezado y limpiar
        df.columns = df.iloc[header_index]
        df = df.iloc[header_index + 1:].reset_index(drop=True)

        # 3. Normalizar columnas: Se espera (FECHA, COMBTE, DESCRIPCION, DEBITO, CREDITO, SALDO)
        # Se renombra para mantener solo las 5 columnas clave:
        
        # Primero, eliminamos las columnas que sean todas nulas o vacías.
        df.dropna(axis=1, how='all', inplace=True) 

        # Si hay 6 columnas (FECHA, COMBTE, DESCRIPCION, DEBITO, CREDITO, SALDO)
        if df.shape[1] == 6:
            df.columns = ['FECHA', 'COMBTE', 'DESCRIPCION', 'DEBITO', 'CREDITO', 'SALDO']
            df = df[['FECHA', 'DESCRIPCION', 'DEBITO', 'CREDITO', 'SALDO']] # Elimina COMBTE
        # Si hay 5 columnas
        elif df.shape[1] == 5:
            df.columns = target_cols
        else:
            st.warning(f"La tabla extraída tiene {df.shape[1]} columnas, se esperaban 5 o 6. Se intenta forzar el formato.")
            df = df.iloc[:, :5]
            df.columns = target_cols
            
        # 4. Consolidar filas parciales (para descripciones multi-línea)
        df['FECHA'] = df['FECHA'].fillna(method='ffill')
        # Limpiamos los movimientos que son solo parte de una descripción
        df.dropna(subset=['FECHA'], inplace=True)
        
        df_final = []
        current_row = {}
        for _, row in df.iterrows():
            
            # Condición para identificar una fila de continuación (no tiene valores numéricos)
            is_continuation_row = (pd.isna(row['DEBITO']) or str(row['DEBITO']).strip() in ['', 'None']) and \
                                  (pd.isna(row['CREDITO']) or str(row['CREDITO']).strip() in ['', 'None']) and \
                                  (pd.isna(row['SALDO']) or str(row['SALDO']).strip() in ['', 'None']) and \
                                  (row['DESCRIPCION'] is not None)

            if not is_continuation_row:
                if current_row: 
                    df_final.append(current_row)
                current_row = row.to_dict()
                
            elif current_row:
                 # Añadir la descripción a la fila principal
                current_row['DESCRIPCION'] = str(current_row.get('DESCRIPCION', '')) + " " + str(row['DESCRIPCION'])
            
            elif not current_row and row['FECHA']: # Caso de la primera fila (SALDO ANTERIOR)
                 current_row = row.to_dict()

        if current_row:
            df_final.append(current_row)

        df_processed = pd.DataFrame(df_final)
        df_processed.dropna(subset=['DESCRIPCION'], inplace=True)
        
        return df_processed.drop_duplicates().reset_index(drop=True)

    except Exception as e:
        st.error(f"Error en la extracción de PDF: {e}")
        return None

# ----------------------------------------------------------------------
# FUNCIONES DE LIMPIEZA Y ANÁLISIS
# ----------------------------------------------------------------------

def limpiar_y_transformar_df(df):
    """Limpia los datos, convierte formatos y calcula el NETO."""

    required_cols = ['FECHA', 'DESCRIPCION', 'DEBITO', 'CREDITO', 'SALDO']
    df.columns = required_cols # Las forzamos a los nombres estándar

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
        elif 'TRANSFER.' in desc and ('DISTINTO TITULAR' in desc or 'O/BCO' in desc or '27326211740-1 M PIANETTI' in desc):
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
st.title("🏦 Analizador y Conciliador Bancario Credicoop (Versión 100% Online)")
st.markdown("Sube tu resumen de cuenta en formato **PDF** para extraer y categorizar automáticamente los movimientos. **No requiere Java ni instalación local.**")
st.markdown("---")

uploaded_file = st.file_uploader("Sube tu archivo PDF de movimientos bancarios", type=['pdf'])

if uploaded_file is not None:
    
    with st.spinner('Extrayendo tablas del PDF... Esto puede tardar unos segundos.'):
        df_original = extract_tables_from_pdf(uploaded_file)
    
    if df_original is not None and not df_original.empty:
        try:
            with st.spinner('Limpiando y analizando datos...'):
                df_limpio = limpiar_y_transformar_df(df_original.copy())

            if df_limpio is not None and not df_limpio.empty:
                df_analizado, saldo_inicial = analizar_movimientos(df_limpio)
                
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
                    st.error(f"🚨 **¡Atención!** Hay una diferencia. Diferencia: ${diferencia:,.2f}")

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

                with col5:
                    st.subheader("Gastos y Débitos (Débitos)")
                    # LÍNEA CORREGIDA (SOLUCIÓN AL SYNTAX ERROR)
                    st.table(gastos.apply(lambda x: f"${x:,.2f}").reset_index(name='Total Neto')) 
                    
                st.markdown(f"**Total Ingresos Netos: ${ingresos.sum():,.2f}**")
                st.markdown(f"**Total Gastos Netos: ${gastos.sum():,.2f}**") # Mantenemos el total aquí para mejor visualización

                st.markdown("---")
                st.header("3. Detalle Completo de Movimientos y Saldo Recalculado")
                
                df_final_display = df_analizado[['FECHA', 'DESCRIPCION', 'DEBITO', 'CREDITO', 'NETO', 'CATEGORIA', 'SALDO_RECALCULADO']].copy()
                df_final_display['FECHA'] = df_final_display['FECHA'].dt.strftime('%d/%m/%Y')
                
                for col in ['DEBITO', 'CREDITO', 'NETO', 'SALDO_RECALCULADO']:
                    df_final_display[col] = df_final_display[col].apply(lambda x: f"${x:,.2f}")

                st.dataframe(df_final_display, use_container_width=True, hide_index=True)

            else:
                st.error("El DataFrame de movimientos está vacío. Posiblemente la limpieza eliminó todos los datos.")

        except Exception as e:
            st.error(f"Error crítico durante el análisis: {e}")
            st.warning("Verifica el log de Streamlit Cloud para más detalles. Si hay un error en la columna `FECHA` o `SALDO`, puede ser un problema de extracción.")
            
    else:
        st.error("La extracción con PDFPlumber falló o no encontró datos tabulares. Intenta subir un archivo PDF con buena calidad de tabla.")


else:
    st.info("Sube tu archivo PDF de movimientos bancarios para comenzar el análisis.")
