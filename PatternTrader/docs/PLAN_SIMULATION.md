# Estado del Plan: `simulate_pipeline.py`

Fecha de actualización: 2026-08-14

## Tareas Completadas

1. **Diseño y Requisitos**:
   - Replay por bloques de velas configurable (`--step`, `--warmup`, `--max-candles`).
   - Persistencia en PostgreSQL por defecto, con opción `--memory` para omitir base de datos.
   - Reporte limpio en consola (sin JSON), resumiendo eventos, patrones detectados, estadísticas del pipeline y señales enviadas.

2. **Implementación**:
   - Creación de `simulate_pipeline.py` en la raíz (CLI con `argparse`, carga robusta de archivos tab/csv, integración con bus de eventos y `PatternPipeline`).
   - Adición del método público `active_models()` y `ensure_models()` en `ScoringEngine` (`app/scoring/engine.py`) para registrar y rehidratar modelos ML per-symbol en el reporte.
   - Propiedad `scoring` expuesta en `PatternPipeline`.
   - Corrección robusta del `EventBus` (`app/core/events/bus.py`) para evitar acoplamientos de bucle de eventos (`asyncio.Queue`) entre ejecuciones aisladas y tests.
   - Ajustes en `settings.py` (`extra="ignore"` y prefijo `DB_`) para compatibilidad robusta con el archivo `.env` local.

3. **Pruebas y Verificación**:
   - `tests/unit/test_simulate_pipeline.py`: tests unitarios para carga de CSVs tab/comma, simulación en memoria y validación de carga de modelos ML per-symbol.
   - Suite de tests unitarios pasa exitosamente (62 pasados).
   - Verificación de funcionamiento con archivos OHLCV reales y bases de datos PostgreSQL de prueba.

4. **Documentación**:
   - Creado `docs/SIMULATION.md` con guía de uso, opciones de CLI y ejemplos.
