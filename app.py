import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import io
import re

# ----------------------------------------------------------------------
# 1. FUNCIÓN DE EXTRACCIÓN Y LIMPIEZA CON PDFPLUMBER (MODO STREAM)
# ----------------------------------------------------------------------

def extract_tables_from_pdf(uploaded_file):
    """
    Extrae tablas de movimientos del PDF usando pdfplumber en modo 'stream' 
    para mejor manejo de formatos complejos (como los del Credicoop).
    """
    pdf_bytes = uploaded_file.read()
    all_data = []

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                # Usamos el modo 'stream' en lugar de 'lattice' para una detección más flexible.
                settings = {
                    "vertical_strategy": "lines", # Líneas verticales rígidas
                    "horizontal_strategy": "lines", # Líneas horizontales rígidas
                    "snap_tolerance": 3,
                    "min_words_vertical": 2, # Ayuda a que DESCRIPCION no se divida.
                    "text_only": False,
                    "keep_blank_chars": True
                }
                tables = page.extract_tables(table_settings=settings)
                
                if tables:
                    for table in tables:
                        if table:
                            all_data.extend(table)
        
        if not all_data:
            st.error("No se pudieron extraer datos de tablas del PDF. Asegúrate de que contenga tablas de movimientos.")
            return None

        # Consolidar los datos extraídos en un DataFrame
        df = pd.DataFrame(all_data)
        
        # Filtrar filas completamente vacías
        df.dropna(how='all', inplace=True)
        
        # Eliminar las filas que probablemente son encabezados repetidos
        df = df[~df.iloc[:, 0].astype(str).str.contains(r'FECHA|SALDO|TOTALES', na=False)]

        # 2. Reestructuración de Columnas para el formato Credicoop
        # El PDF de Credicoop a menudo tiene 6 columnas (FECHA, COMBTE, DESCRIPCION, DEBITO, CREDITO, SALDO)
        # o más si las descripciones se dividen.
        
        # Tomamos solo las primeras 6 columnas si existen, o rellenamos con NaN si no.
        max_cols = 6
        if df.shape[1] < max_cols:
            df = df.reindex(columns=range(max_cols))
        elif df.shape[1] > max_cols:
             df = df.iloc[:, :max_cols] # Cortamos columnas excedentes
             
        df.columns = ['FECHA', 'COMBTE', 'DESCRIPCION', 'DEBITO', 'CREDITO', 'SALDO']
        
        # La limpieza de continuación de filas es la más crítica:
        df_processed = []
        current_row = None
        for _, row in df.iterrows():
            
            # Limpiamos NaN a strings vacíos para facilitar las condiciones
            row_dict = row.astype(str).replace({'nan': '', 'None': ''}).to_dict()
            
            # Condición para identificar un nuevo movimiento (debe tener FECHA o ser SALDO ANTERIOR)
            is_new_movement = (row_dict['FECHA'] != '') or ('SALDO ANTERIOR' in row_dict['DESCRIPCION'].upper())
            
            # Condición para identificar una fila de continuación (no tiene valores numéricos en Débito/Crédito/Saldo)
            is_continuation_row = (row_dict['DEBITO'].strip() == '') and \
                                  (row_dict['CREDITO'].strip() == '') and \
                                  (row_dict['SALDO'].strip() == '') and \
                                  (row_dict['DESCRIPCION'].strip() != '')

            if is_new_movement and not is_continuation_row:
                # Iniciar un nuevo movimiento
                if current_row is not None:
                    df_processed.append(current_row)
                current_row = row_dict
                
            elif is_continuation_row and current_row is not None:
                # Añadir la descripción a la fila principal
                current_row['DESCRIPCION'] = current_row['DESCRIPCION'] + " " + row_dict['DESCRIPCION']
            
            elif is_new_movement and is_continuation_row:
                 # Caso de fila inicial mal formateada o SALDO ANTERIOR.
                 if current_row is not None:
                    df_processed.append(current_row)
                 current_row = row_dict
                 
        if current_row is not None:
            df_processed.append(current_row)

        df_final = pd.DataFrame(df_processed)
        df_final.dropna(subset=['DESCRIPCION'], inplace=True)
        
        # Finalmente, eliminamos la columna COMBTE y forzamos las 5 columnas requeridas
        if 'COMBTE' in df_final.columns:
            df_final = df_final.drop(columns=['COMBTE'])
            
        return df_final[['FECHA', 'DESCRIPCION', 'DEBITO', 'CREDITO', 'SALDO']].drop_duplicates().reset_index(drop=True)

    except Exception as e:
        st.error(f"Error en la extracción de PDF con pdfplumber: {e}")
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
        # Esto maneja el formato del Credicoop: 1.000,00 -> 1000.00
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
st.title("🏦 Analizador y Conciliador Bancario Credicoop (Versión 100% Online)")
st.markdown("Sube tu resumen de cuenta en formato **PDF** para extraer y categorizar automáticamente los movimientos.")
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
                st.error("El DataFrame de movimientos está vacío. Posiblemente la limpieza eliminó todos los datos.")

        except Exception as e:
            st.error(f"Error crítico durante el análisis: {e}")
            st.warning("Revisa el log de Streamlit Cloud. El error puede estar en la conversión de `FECHA` o en las columnas numéricas.")
            
    else:
        st.error("La extracción con PDFPlumber falló. Intenta subir un archivo PDF con buena calidad de tabla. Si el problema persiste, la estructura del PDF requiere una herramienta más avanzada (como Tabula/Java).")


else:
    st.info("Sube tu archivo PDF de movimientos bancarios para comenzar el análisis.")
