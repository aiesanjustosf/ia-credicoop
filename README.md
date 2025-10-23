
# Extractor Credicoop Online — v3.4 (fecha obligatoria, saldo omitido)

Reglas:
- Cada **movimiento** es una fila con **FECHA**.  
- Líneas **sin fecha** = continuación de la descripción anterior (no crean movimiento).  
- **Saldo** (columna derecha) se **omite siempre**; no hay saldo por fila ni por día.  
- **SALDO ANTERIOR** y **SALDO AL** sólo van al resumen (no como movimientos).

Export: columnas numéricas, texto exacto y centavos (int) para montos grandes.
