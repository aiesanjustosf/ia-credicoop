import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import re

# --- 1. FUNCIÓN PARA OBTENER LOS DATOS DEL PDF (PRE-PROCESADOS) ---

def get_pdf_data():
    """
    Retorna los movimientos de cuenta extraídos y limpiados
    directamente de las tablas del CREDICOOP.pdf.
    """
    # Datos extraídos y consolidados de las tablas de movimientos del PDF (Páginas 1 y 2)
    # Se utiliza un formato CSV para facilitar la carga con Pandas.
    return """
FECHA,DESCRIPCION,DEBITO,CREDITO,SALDO
01/07/25,SALDO ANTERIOR,,,284365.38
01/07/25,Pago a Comercios Cabal CABAL-008703902009,,91901.14,
01/07/25,Pago de Cheque de Camara,823700.00,,
01/07/25,Comision Cheque Pagado por Clearing,500.00,,
01/07/25,I.V.A. Debito Fiscal 21%,105.00,,
01/07/25,Debito/Credito Aut Segurcoop Comercio SEGUR.SOCIO INT.COM.-2529077220000002,12864.31,,
01/07/25,Impuesto Ley 25.413 Ali Gral s/Creditos,551.41,,
01/07/25,Impuesto Ley 25.413 Alic Gral s/Debitos,5023.02,,-466477.22
02/07/25,Transferencia Inmediata e/Ctas. Propias 20228760057-VAR-BERGA, FABRICIO ROLANDO,,500000.00,
02/07/25,Pago a Comercios Cabal CABAL-008703902009,,84294.96,
02/07/25,Impuesto Ley 25.413 Ali Gral s/Creditos,505.77,,117311.97
03/07/25,Pago a Comercios Cabal CABAL-008703902009,,127203.58,
03/07/25,Impuesto Ley 25.413 Ali Gral s/Creditos,763.22,,243752.33
07/07/25,Pago de Cheque de Camara,807600.00,,
07/07/25,Comision Cheque Pagado por Clearing,500.00,,
07/07/25,I.V.A. Debito Fiscal 21%,105.00,,
07/07/25,Pago a Comercios Cabal CABAL-008703902009,,94816.01,
07/07/25,Credito Inmediato (DEBIN) 20228760057-VAR-FABRICIO ROLANDO BERGA,568.90,600000.00,
07/07/25,Impuesto Ley 25.413 Alic Gral s/Debitos,4849.23,,124945.21
08/07/25,Pago a Comercios Cabal CABAL-008703902009,,91901.14,
08/07/25,Impuesto Ley 25.413 Ali Gral s/Creditos,551.41,,216294.94
10/07/25,Pago a Comercios Cabal CABAL-008703902009,,109491.89,
10/07/25,Impuesto Ley 25.413 Ali Gral s/Creditos,656.95,,325129.88
14/07/25,Transf. Interbanking Distinto Titular Ord.: 30685376349-TARJETA NARANJA S A,,263149.97,927906.59
14/07/25,Impuesto Ley 25.413 Ali Gral s/Creditos,3638.49,,
15/07/25,Pago a Comercios Cabal CABAL-008703902009,,233974.92,
15/07/25,Transf. Interbanking Distinto Titular Ord.:30707736107-VIVI TRANQUILO SA,2982.75,,1422048.73
15/07/25,Impuesto Ley 25.413 Ali Gral s/Creditos,2982.75,,
16/07/25,Pago a Comercios Cabal CABAL-008703902009,,98581.46,
16/07/25,Transf. Inmediata e/Ctas. Dist. Titular 27217764144-VAR-IMPERIALE, ALEJANDRA L,150000.00,,
16/07/25,Impuesto Ley 25.413 Ali Gral s/Creditos,1491.49,,1669138.70
17/07/25,Pago a Comercios Cabal CABAL-008703902009,,40365.80,
17/07/25,Pago de Obligaciones a ARCA Tipo de Pago: ARCA VEP PENDIENTES,31009.98,,
17/07/25,Impuesto Ley 25.413 Ali Gral s/Creditos,242.19,,
17/07/25,Impuesto Ley 25.413 Alic Gral s/Debitos,186.06,,1678066.27
21/07/25,Pago a Comercios Cabal CABAL-008703902009,,145950.22,
21/07/25,ECHO Acreditac de Valores Camara Dep:3349905762-BCO GALICIA-Ch:00042705,,244398.00,
21/07/25,ECHQ- Comis acred Camara con Filial Bco,1588.59,,
21/07/25,I.V.A. Debito Fiscal 21%,333.60,,
21/07/25,ECHQ Acreditac de Valores Camara Dep: 3349905762-BCO STA FE-Ch:00878290,,1000000.00,
21/07/25,Impuesto Ley 25.413 Ali Gral s/Creditos,8342.09,,
21/07/25,Impuesto Ley 25.413 Alic Gral s/Debitos,11.53,,3058138.68
22/07/25,Pago de Obligaciones a ARCA Tipo de Pago: ARCA VEP PENDIENTES,1368345.86,,
22/07/25,Pago de Obligaciones a ARCA Tipo de Pago: ARCA VEP PENDIENTES,76629.50,,
22/07/25,Impuesto Ley 25.413 Alic Gral s/Debitos,8669.86,,1604493.46
23/07/25,Pago a Comercios Cabal CABAL-008703902009,,155598.08,
23/07/25,Transf.Inmediata e/Ctas. Dist Tit.0/Bco 30708225300-FAC-CMS SA,586652.44,,
23/07/25,Transf. Inmediata e/Ctas. Dist. Titular 20297203143-FAC-ESTRUBIA, ANDR S VICENT,,645609.61,
23/07/25,Impuesto Ley 25.413 Ali Gral s/Creditos,4807.25,,
23/07/25,Impuesto Ley 25.413 Alic Gral s/Debitos,3519.91,,1810721.55
24/07/25,Pago de Cheque de Camara,1000000.00,,
24/07/25,Comision Cheque Pagado por Clearing.,500.00,,
24/07/25,I.V.A. Debito Fiscal 21%,105.00,,
24/07/25,Comision por Transferencia B. INTERNET COM. USO-000470688,300.00,,
24/07/25,I.V.A. Debito Fiscal 21%,63.00,,
24/07/25,Pago a Comercios Cabal CABAL-008703902009,,137356.15,
24/07/25,Impuesto Ley 25.413 Ali Gral s/Creditos,824.14,,
24/07/25,Impuesto Ley 25.413 Alic Gral s/Debitos,6005.81,,940279.75
25/07/25,Pago a Comercios Cabal CABAL-008703902009,,152189.76,
25/07/25,Servicio Modulo NyP,37500.00,,
25/07/25,Percepcion IVA RG 2408 s/Comis-Gastos,1125.00,,
25/07/25,I.V.A. Debito Fiscal 21%,7875.00,,
25/07/25,Impuesto Ley 25.413 Ali Gral s/Creditos,913.14,,
25/07/25,Impuesto Ley 25.413 Alic Gral s/Debitos,279.00,,1044777.37
28/07/25,Transf. Inmediata e/Ctas.Igual Tit.O/Bco 20228760057-VAR-BERGA FABRICIO ROLANDO,1000000.00,,44777.37
29/07/25,Comision por Transferencia B. INTERNET COM. USO-000470688,300.00,,
29/07/25,I.V.A. Debito Fiscal 21%,63.00,,
29/07/25,Impuesto Ley 25.413 Alic Gral s/Debitos,2.18,,44412.19
30/07/25,Pago a Comercios Cabal CABAL-008703902009,,109499.19,
30/07/25,Impuesto Ley 25.413 Ali Gral s/Creditos,657.00,,153254.38
31/07/25,Transfer. e/Cuentas de Distinto Titular 27326211740-1 M PIANETTI,180000.00,,
31/07/25,Transfer. e/Cuentas de Distinto Titular 27326211740-1 M PIANETTI,5000.00,,
31/07/25,Transferencia Inmediata e/Ctas. Propias 20228760057-VAR-BERGA, FABRICIO ROLANDO,,50000.00,
31/07/25,Intereses por Saldo Deudor,741.25,,
31/07/25,I.V.A. Debito Fiscal 10,5%,77.83,,
31/07/25,Impuesto Ley 25.413 Alic Gral s/Debitos,1114.92,,16320.38
"""

# --- 2. FUNCIONES DE LIMPIEZA Y ANÁLISIS ---

def limpiar_y_transformar_df(df):
    """Limpia los datos, convierte formatos y calcula el NETO."""

    required_cols = ['FECHA', 'DESCRIPCION', 'DEBITO', 'CREDITO', 'SALDO']
    
    # Asegura que las columnas estén en mayúsculas para la coincidencia
    df.columns = df.columns.map(lambda x: x.upper().strip())

    if not all(col in df.columns for col in required_cols):
        st.error(f"El archivo debe contener las columnas: {', '.join(required_cols)}")
        return None

    # Función de limpieza: permite puntos como separadores decimales
    def limpiar_y_convertir_numerico(series):
        # La función get_pdf_data ya usa puntos, pero la dejo para uploads
        series = series.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        return pd.to_numeric(series, errors='coerce').fillna(0)

    df['DEBITO'] = limpiar_y_convertir_numerico(df['DEBITO'])
    df['CREDITO'] = limpiar_y_convertir_numerico(df['CREDITO'])
    df['SALDO'] = limpiar_y_convertir_numerico(df['SALDO'])

    try:
        df['FECHA'] = pd.to_datetime(df['FECHA'], dayfirst=True, errors='coerce')
        df.dropna(subset=['FECHA'], inplace=True) # Elimina filas donde la fecha no se pudo parsear
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
            return 'Impuesto - Ley 25.413 (Débito)'
        elif 'I.V.A.' in desc or 'PERCEPCION IVA' in desc:
            return 'Impuesto - IVA / Percepciones'
        elif 'PAGO DE CHEQUE' in desc or 'COMISION CHEQUE PAGADO' in desc:
            return 'Gasto - Cheques y Comisiones'
        elif 'TRANSFER.' in desc and 'DISTINTO TITULAR' in desc:
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
        elif 'IMPUESTO LEY 25.413 ALI GRAL S/CREDITOS' in desc:
             return 'Impuesto - Ley 25.413 (Débito, pero asociado a Crédito)'
        
    # CRÉDITOS / INGRESOS
    elif credito > 0:
        if 'PAGO A COMERCIOS CABAL' in desc:
            return 'Ingreso - Venta con Cabal'
        elif 'CREDITO INMEDIATO' in desc or 'ACREDITAC' in desc:
            return 'Ingreso - Acreditación de Cheque/DEBIN'
        elif 'TRANSFER.' in desc and 'DISTINTO TITULAR' in desc:
            return 'Ingreso - Transferencia Recibida'
        elif 'TRANSFERENCIA INMEDIATA E/CTAS. PROPIAS' in desc:
            return 'Transferencia Interna (Ingreso)'
            
    return 'Otros/Movimiento no categorizado'

def analizar_movimientos(df):
    """Aplica la categorización y realiza el cálculo de saldo."""
    
    df['CATEGORIA'] = df.apply(lambda row: categorizar_movimiento(row['DESCRIPCION'], row['DEBITO'], row['CREDITO']), axis=1)
    
    # Obtener el Saldo Inicial
    saldo_inicial_row = df[df['CATEGORIA'] == 'SALDO INICIAL']
    saldo_inicial = saldo_inicial_row['SALDO'].iloc[0] if not saldo_inicial_row.empty else 0

    # Calcular el Saldo Recalculado
    df_movimientos = df[df['CATEGORIA'] != 'SALDO INICIAL'].copy()
    
    neto_acumulado = df_movimientos['NETO'].cumsum()
    
    # Alinear el resultado del cumsum al DataFrame original y sumar el saldo inicial
    df['SALDO_RECALCULADO'] = saldo_inicial + neto_acumulado.reindex(df.index).fillna(method='ffill').fillna(saldo_inicial)

    return df, saldo_inicial

# --- 3. CONFIGURACIÓN DE STREAMLIT ---

st.set_page_config(layout="wide")
st.title("🏦 Analizador y Conciliador Bancario Credicoop")
st.markdown("---")

# Opción para usar el PDF (datos pre-cargados) o subir un archivo
st.sidebar.header("Selección de Fuente de Datos")
use_default_data = st.sidebar.checkbox("Analizar Resumen de CREDICOOP.pdf (Julio 2025)", value=True)
uploaded_file = None

if not use_default_data:
    st.sidebar.info("La aplicación esperará a que subas un archivo.")
    uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV o Excel de movimientos bancarios", type=['csv', 'xlsx', 'txt'])

df_original = None

if use_default_data:
    # Cargar los datos extraídos del PDF
    datos = get_pdf_data()
    df_original = pd.read_csv(StringIO(datos))
    st.info("Cargando y analizando datos extraídos del PDF (01/07/2025 al 31/07/2025).")
    
elif uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.txt'):
            df_original = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8')
        elif uploaded_file.name.endswith('.xlsx'):
            df_original = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}. Intenta usar otro separador si es CSV.")
        df_original = None

# --- 4. EJECUCIÓN DEL ANÁLISIS ---

if df_original is not None:
    with st.spinner('Procesando datos y realizando conciliación...'):
        df_limpio = limpiar_y_transformar_df(df_original.copy())

        if df_limpio is not None:
            df_analizado, saldo_inicial = analizar_movimientos(df_limpio)
            
            # El saldo final del resumen se toma de la última fila del archivo
            saldo_final_calculado = df_analizado['SALDO_RECALCULADO'].iloc[-1]
            saldo_final_resumen = df_analizado['SALDO'].iloc[-1] 
            diferencia = saldo_final_calculado - saldo_final_resumen

            # --- PRESENTACIÓN DE RESULTADOS ---

            st.header("1. Conciliación de Saldo")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Saldo Inicial (01/07/25)", f"${saldo_inicial:,.2f}")
            col2.metric("Saldo Final Calculado (31/07/25)", f"${saldo_final_calculado:,.2f}")
            col3.metric("Diferencia (Calculado - Resumen)", f"${diferencia:,.2f}", delta_color='inverse')

            if abs(diferencia) < 0.01:
                st.success("✅ **¡Conciliado!** El saldo final calculado coincide con el saldo final reportado en el resumen. (Diferencia: $0,00)")
            else:
                st.error("🚨 **¡Atención!** Hay una diferencia en el saldo. Revisar la extracción de datos o la fórmula de saldo final.")
                st.info(f"Saldo final del resumen (Última fila de tu archivo): ${saldo_final_resumen:,.2f}")

            st.markdown("---")

            st.header("2. Resumen de Gastos y Créditos por Categoría")

            # Filtrar y agrupar para el resumen
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
            
            # Formatear el DataFrame para la visualización final
            df_final_display = df_analizado[['FECHA', 'DESCRIPCION', 'DEBITO', 'CREDITO', 'NETO', 'CATEGORIA', 'SALDO_RECALCULADO']].copy()
            df_final_display['FECHA'] = df_final_display['FECHA'].dt.strftime('%d/%m/%Y')
            
            for col in ['DEBITO', 'CREDITO', 'NETO', 'SALDO_RECALCULADO']:
                 df_final_display[col] = df_final_display[col].apply(lambda x: f"${x:,.2f}")

            st.dataframe(df_final_display, use_container_width=True, hide_index=True)

else:
    st.info("Para comenzar, selecciona la opción para analizar los datos pre-cargados del PDF o sube tu propio archivo CSV/Excel de movimientos.")
