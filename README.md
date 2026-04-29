# Comparador técnico de fórmulas | Rebranding

Aplicación Streamlit para comparar piensos del portfolio actual contra una gama estándar de destino a partir de exportaciones Excel tipo **Multi-Mix** y **Single-Mix**.

## Qué hace la app

La app reconstruye cada fórmula desde el Excel y calcula una **similitud heurística** entre productos origen y productos destino. El ranking combina cuatro componentes:

- **Nutrientes** seleccionados por el usuario
- **Ingredientes** y porcentajes de inclusión
- **Límites** de ingredientes y compatibilidades
- **Precio**

Un **score más bajo** indica mayor parecido relativo dentro de la comparación.

## Modos de análisis

### 1) Comparativa individual

Sirve para revisar un solo pienso origen frente a la gama destino.

Flujo:
1. Cargar archivo origen y archivo destino.
2. Elegir especie origen y especie destino.
3. Seleccionar el pienso origen.
4. Elegir métricas críticas y ajustar sus pesos.
5. Revisar el **top N** de candidatos destino.
6. Exportar el ranking y el informe en Excel o TXT.

### 2) Comparativa múltiple origen

Sirve para tratar varios piensos origen a la vez y acelerar la propuesta de equivalencias.

Flujo:
1. Cargar archivo origen y archivo destino.
2. Elegir especie origen y especie destino.
3. Seleccionar varios piensos origen en el multiselect. La app admite **20 o más** referencias en la misma ejecución.
4. Reutilizar los mismos selectores de nutrientes y pesos.
5. La app calcula un ranking independiente para cada pienso origen.
6. En la pestaña **Selección final** aparece una **matriz rápida origen → destino**:
   - cada fila es un pienso origen
   - las columnas `Opción 1`, `Opción 2`, `Opción 3`... muestran los candidatos del top N
   - para cada opción se muestran también **score**, **precio** y **% de diferencia de precio**
   - la columna **Opción elegida** permite decidir rápidamente qué candidato se selecciona para cada pienso origen
7. Debajo de la matriz, la app genera automáticamente:
   - tabla consolidada de selección final
   - comparativa de **precio origen vs destino**
   - comparativa de **nutrientes elegidos** con valores origen, destino y diferencia
8. Exportar toda la selección múltiple en Excel.

## Cómo interpretar el análisis heurístico

La herramienta **no aprueba automáticamente** un cambio de referencia. Prioriza candidatos para reducir revisión manual.

Conviene revisar siempre:
- nutrientes críticos de la especie
- materias primas diferenciales
- límites mínimos y máximos
- diferencia de precio
- viabilidad técnica y comercial

## Exportaciones disponibles

### Comparativa individual
- Ranking en CSV
- Informe comparativo en TXT
- Resultados completos en Excel

### Comparativa múltiple origen
- **Matriz de selección** en Excel
- Selección final consolidada
- Comparativa de nutrientes elegidos
- Ranking consolidado de todos los piensos origen
- Informe resumen de la selección múltiple
- Hojas largas de métricas e ingredientes

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run main.py
```

## Recomendación de uso

1. Ajustar primero bien las métricas y sus pesos por especie.
2. Usar la comparativa múltiple para una primera criba de SKUs.
3. En la matriz, elegir una opción por fila para avanzar rápido cuando trabajes con 20-30 referencias.
4. Revisar en detalle los candidatos elegidos antes de decidir la migración definitiva.
