# Simulación del Pipeline (simulate_pipeline.py)

Script oficial para reproducir el flujo completo de detección, ciclo de vida, confirmación, scoring con ML per-symbol, estrategia y envío de señales (con soporte para Telegram y persistencia en PostgreSQL o memoria).

## Uso Básico

Ejecución en modo memoria (sin base de datos) con configuración por defecto:
```bash
python simulate_pipeline.py app/datos_test/USDCAD_H1_201005311000_202606010000.txt --memory
```

## Opciones CLI

- `--symbol SYMBOL`: Símbolo del par (por defecto se deriva del nombre del archivo, ej: `USDCAD`).
- `--timeframe TF`: Timeframe (por defecto se deriva del nombre del archivo, ej: `H1`).
- `--warmup N`: Velas iniciales para el primer análisis (por defecto: `200`).
- `--step N`: Velas nuevas añadidas en cada tick del replay (por defecto: `50`).
- `--max-candles N`: Ventana máxima de velas por tick enviada al pipeline (por defecto: `500`).
- `--speed SECS`: Pausa opcional en segundos entre ticks (por defecto: `0`).
- `--memory`: Ejecuta en memoria (repositorios `None`), sin requerir PostgreSQL.
- `--learning`: Activa el `LearningService` para aprendizaje continuo a partir de operaciones cerradas.
- `--strategy a,b,c`: Sobrescribe las estrategias activas (por defecto usa `settings.strategies.enabled`).
- `--model-dir PATH`: Directorio con los artefactos ML entrenados con `train_and_compare.py` (por defecto: `./models/`).
- `--telegram`: Fuerza el envío de notificaciones por Telegram (requiere token y chat id configurados).
- `--quiet`: Suprime logs detallados de componentes y muestra únicamente el resumen final.

## Ejemplos Avanzados

1. **Con persistencia en PostgreSQL y modelo ML pre-entrenado**:
   ```bash
   python simulate_pipeline.py app/datos_test/USDCAD_H1_201005311000_202606010000.txt \
     --model-dir ./models \
     --warmup 200 --step 50
   ```

2. **Con aprendizaje continuo y Telegram forzado**:
   ```bash
   python simulate_pipeline.py app/datos_test/USDCAD_H1_201005311000_202606010000.txt \
     --memory --learning --telegram
   ```
