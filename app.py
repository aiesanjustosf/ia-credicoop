import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import io
import re

# ----------------------------------------------------------------------
# 1. FUNCIÓN DE EXTRACCIÓN Y LIMPIEZA DE TEXTO (CORRECCIÓN DE PDFPLUMBER)
# ----------------------------------------------------------------------

def extract_tables_from_pdf(uploaded_file):
    """
    Extrae texto crudo del PDF usando pdfplumber con configuración forzada 
    para evitar errores de incompatibilidad y manejar el formato Credicoop.
    """
    pdf_bytes = uploaded_file.read()
    all_rows = []

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            # Líneas de corte para las columnas basadas en el PDF de Credicoop (aprox)
            # FECHA | COMBTE | DESCRIPCION              | DEBITO | CREDITO | SALDO
            x_settings = [
                0,    # FECHA
                70,   # COMBTE
                180,  # DESCRIPCION
                450,  # DEBITO
                570,  # CREDITO
                700   # SALDO
            ]

            for page in pdf.pages:
                
                # Configuración explícita (versión simplificada y compatible)
                settings = {
                    "vertical_strategy": "explicit",
                    "horizontal_strategy": "text", 
                    "explicit_vertical_lines": x_settings,
                    "snap_tolerance": 3,
                }
                
                # NO SE USAN ARGUMENTOS QUE PUEDEN SER INCOMPATIBLES (ej: text_only, keep_blank_chars)
                tables = page.extract_tables(table_settings=settings)
                
                if tables:
                    for table in tables:
                        if table:
                            all_rows.extend(table)
        
        if not all_rows:
            st.error("No se pudieron extraer datos del PDF.")
            return None

        # 2. Convertir a DataFrame y Limpieza Estructural

        df = pd.DataFrame(all_rows)
        df.dropna(how='all', inplace=True)

        # ... (Mantener la lógica de limpieza y consolidación de filas) ...
        
        # Buscar el encabezado ("FECHA", "DESCRIPCION", etc.)
        header_index = None
        for i, row in df.iterrows():
            if any(col in str(row[2]).upper() for col in ['DESCRIPCION', 'SALDO']):
                header_index = i
                break
        
        if header_index is None:
             # Asumir la primera fila después de la limpieza de nulos
             if not df.empty:
                df.columns = df.iloc[0]
                df = df.iloc[1:].reset_index(drop=True)
             else:
                 return None
        else:
            df.columns = df.iloc[header_index]
            df = df.iloc[header_index + 1:].reset_index(drop=True)

        
        # Renombrar y seleccionar las columnas clave (asumiendo que el corte funcionó)
        # Limita a 6 columnas y las renombra
        df = df.iloc[:, :6] 
        df.columns = ['FECHA', 'COMBTE', 'DESCRIPCION', 'DEBITO', 'CREDITO', 'SALDO']
        df = df[['FECHA', 'DESCRIPCION', 'DEBITO', 'CREDITO', 'SALDO']].copy()
        
        # Limpiar valores nulos y strings 'None'
        df = df.replace('None', np.nan).replace('', np.nan)
        
        # 3. Consolidación de Filas Parciales (la parte más crítica)
        
        df['FECHA'] = df['FECHA'].fillna(method='ffill')
        df.dropna(subset=['FECHA'], inplace=True) 

        df_processed = []
        current_row = {}
        for _, row in df.iterrows():
            
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
            
            elif not current_row and row['FECHA']:
                 current_row = row.to_dict()

        if current_row:
            df_processed.append(current_row)

        df_final = pd.DataFrame(df_processed)
        df_final.dropna(subset=['DESCRIPCION'], inplace=True)
        
        return df_final.drop_duplicates().reset_index(drop=True)

    except Exception as e:
        # Aquí capturaremos si aún falla por la alineación del PDF, pero no por el argumento 'only'.
        st.error(f"Error en la extracción de PDF: {e}")
        return None

# ... (Mantener las funciones limpiar_y_transformar_df, categorizar_movimiento, analizar_movimientos y la LÓGICA DE STREAMLIT) ...

# ----------------------------------------------------------------------
# LÓGICA DE STREAMLIT (SIN CAMBIOS RESPECTO A LA VERSIÓN ANTERIOR)
# ----------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("🏦 Analizador y Conciliador Bancario Credicoop (Intento Final Online)")
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
                
                saldo_final_calculado = df_analizado['SALDO_RECALCULADO'].iloc[-1]
                saldo_final_resumen = df_analizado['SALDO'].iloc[-1] 
                diferencia = saldo_final_calculado - saldo_final_resumen

                st.header("1. Conciliación de Saldo")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Saldo Inicial", f"${saldo_inicial:,.2f}")
                col2.metric("Saldo Final Calculado", f"${saldo_final_calculado:,.2f}")
                col3.metric("Diferencia (Calculado - Resumen)", f"${diferencia:,.2f}", delta_color='inverse')

                if abs(diferencia) < 0.01:
                    st.success("✅ **¡Conciliado!** El saldo final calculado coincide con el saldo final reportado en el resumen. (Diferencia: $0,00)")
                else:
                    st.error(f"🚨 **¡Atención!** Hay una diferencia de ${diferencia:,.2f}. **Revisa la Tabla 3** para ver si la extracción del PDF fue imperfecta.")

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
        st.error("La extracción del PDF falló. Este formato es incompatible con las herramientas 100% Python en Streamlit.")


else:
    st.info("Sube tu archivo PDF de movimientos bancarios para comenzar el análisis.")
