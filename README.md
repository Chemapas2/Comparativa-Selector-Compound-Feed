# Comparador técnico de fórmulas para Rebranding

Aplicación en Streamlit para comparar un pienso actual contra una gama estándar y devolver los candidatos destino más parecidos.

## Qué hace

- Lee ficheros Excel en el formato de exportación tipo **Multi-Mix** como el ejemplo adjunto.
- Reconstruye cada bloque de fórmula a partir de la línea `Specification:`.
- Extrae:
  - producto
  - precio por tonelada
  - ingredientes incluidos y sus límites
  - analíticas/nutrientes
- Permite filtrar por especie.
- Calcula un ranking de similitud entre un pienso origen y los productos destino.
- Devuelve el **top N** de equivalencias propuestas.
- Muestra detalle por candidato:
  - diferencias en nutrientes
  - diferencias en ingredientes
  - compatibilidad de límites
  - diferencia de precio

## Lógica del score

El score total combina 4 componentes:

1. **Nutrientes**
2. **Ingredientes**
3. **Límites de ingredientes**
4. **Precio**

Un score más bajo significa mayor parecido.

Los pesos globales y los pesos por métrica se pueden editar en la interfaz.

## Instalación local

```bash
pip install -r requirements.txt
streamlit run main.py
```

## Despliegue en GitHub + Streamlit Community Cloud

1. Crear un repositorio en GitHub.
2. Subir al repositorio:
   - `main.py`
   - `requirements.txt`
   - `README.md`
3. Entrar en Streamlit Community Cloud.
4. Conectar el repositorio.
5. Seleccionar `main.py` como archivo de entrada.
6. Desplegar.

## Supuestos del parser

La app está pensada para archivos con una estructura equivalente al Excel de ejemplo:

- línea de cabecera con `Specification:`
- bloque `INCLUDED RAW MATERIALS`
- bloque `ANALYSIS`
- opcionalmente bloque `RAW MATERIAL SENSITIVITY`

Si algún repositorio usa otra maqueta, habrá que ajustar el parser.

## Recomendaciones siguientes

Para una versión 2 convendría añadir:

- reglas específicas por especie/subespecie
- equivalencias manuales aprobadas por el equipo técnico
- exportación a Excel con informe comparativo
- histórico de decisiones y validación final
- clasificación más precisa de especies y subespecies
