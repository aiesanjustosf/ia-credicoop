
# Extractor Credicoop Online (Streamlit + OCR)

Subí un PDF del Banco Credicoop y obtené los movimientos en tabla + conciliación, con **exportación a Excel**.
Funciona con PDFs de **texto** y también con **escaneados** (OCR vía Tesseract).

## Despliegue 100% online

### Opción A: Streamlit Community Cloud
1. Subí estos archivos a un repositorio público (GitHub).
2. En https://share.streamlit.io/ creá una nueva app apuntando a `app_credicoop.py`.
3. Streamlit instalará automáticamente:
   - paquetes Python desde **requirements.txt**
   - paquetes del sistema desde **packages.txt** (Tesseract + Poppler)
4. Abrí la URL y listo: subida de PDF y descarga de Excel, **todo online**.

### Opción B: Hugging Face Spaces (Streamlit)
1. Creá un Space tipo **Streamlit**.
2. Subí `app_credicoop.py`, `requirements.txt`, `packages.txt`.
3. HF instalará dependencias: funciona igual que en Streamlit Cloud.

> Si tu PDF es escaneado: activá **Forzar OCR** en la interfaz.

## Desarrollo local (opcional)
```bash
pip install -r requirements.txt
streamlit run app_credicoop.py
```
Para OCR local, instalá **Tesseract** y **Poppler** del sistema.

## Notas
- Columnas detectadas: **DEBITO**, **CREDITO**, **SALDO**; agrupa descripciones multilínea.
- Conciliación: `saldo_anterior - débitos + créditos = saldo_final` (muestra diferencia).
- Exporta **Excel** con hojas **Movimientos** y **Resumen**.
