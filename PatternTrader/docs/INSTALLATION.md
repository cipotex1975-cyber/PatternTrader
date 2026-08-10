# Guía de Instalación

## Requisitos Previos

### Software Mínimo

| Componente | Versión Mínima | Recomendada |
|------------|----------------|-------------|
| Python | 3.11+ | 3.12 |
| pip | 22.0+ | Última versión |
| PostgreSQL | 14+ | 16 |
| Git | 2.30+ | Última versión |

### Hardware Recomendado

- **RAM**: 8GB mínimo, 16GB recomendado
- **CPU**: 4 cores mínimo
- **Disco**: 10GB libres
- **Red**: Conexión estable a internet

---

## Instalación en Linux (Ubuntu/Debian)

### 1. Instalar Dependencias del Sistema

```bash
# Actualizar paquetes
sudo apt update && sudo apt upgrade -y

# Instalar Python 3.11+
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Instalar PostgreSQL
sudo apt install -y postgresql postgresql-contrib

# Instalar dependencias de compilación
sudo apt install -y build-essential libpq-dev
```

### 2. Clonar el Repositorio

```bash
# Clonar
git clone https://github.com/tu-usuario/pattern-trader.git
cd pattern-trader
```

### 3. Crear Entorno Virtual

```bash
# Crear entorno virtual
python3.11 -m venv venv

# Activar entorno virtual
source venv/bin/activate

# Verificar Python
python --version  # Debe mostrar Python 3.11+
```

### 4. Instalar Dependencias

```bash
# Instalar dependencias base
pip install --upgrade pip
pip install -e .

# Instalar dependencias de desarrollo (opcional)
pip install -e ".[dev]"
```

#### Dependencias Opcionales por Proveedor

Los proveedores de datos requieren dependencias adicionales solo si los vas a usar:

| Proveedor | Paquete | Comando |
|-----------|---------|---------|
| Binance / Bybit | `ccxt` | incluido |
| Yahoo Finance | `yfinance` | incluido |
| Polygon / AlphaVantage | `httpx` | incluido |
| MetaTrader 5 | `MetaTrader5` | `pip install MetaTrader5` |
| Interactive Brokers | `ib_async` | `pip install ib_async` |

> **Nota**: MetaTrader 5 requiere además un terminal MT5 corriendo localmente. Interactive Brokers requiere IB Gateway o TWS con acceso API habilitado. La plataforma funciona sin estos paquetes; los proveedores correspondientes solo fallarán si intentas usarlos sin instalar su dependencia.

### 5. Configurar PostgreSQL

```bash
# Crear base de datos
sudo -u postgres createdb pattern_trader

# Crear usuario
sudo -u postgres psql -c "CREATE USER pattern_user WITH PASSWORD 'tu_password';"

# Dar permisos
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE pattern_trader TO pattern_user;"

# Configurar tablas (migraciones Alembic)
alembic upgrade head

# Alternativa (solo desarrollo, sin historial de migraciones):
# python -c "from app.database.base import init_db; import asyncio; asyncio.run(init_db())"
```

### 6. Configurar Variables de Entorno

```bash
# Crear archivo .env
cat > .env << EOF
DB_HOST=localhost
DB_PORT=5432
DB_NAME=pattern_trader
DB_USER=pattern_user
DB_PASSWORD=tu_password
# Opcional: URL completa de conexión (tiene prioridad sobre los campos discretos)
# DATABASE_URL=postgresql+asyncpg://pattern_user:tu_password@localhost:5432/pattern_trader
TELEGRAM_BOT_TOKEN=tu_token
TELEGRAM_CHAT_ID=tu_chat_id
EOF
```

### 7. Verificar Instalación

```bash
# Ejecutar pruebas
pytest tests/unit/ -v

# Iniciar servidor
python -m app.main

# Verificar en otro terminal
curl http://localhost:8000/api/v1/health
```

---

## Instalación en macOS

### 1. Instalar Homebrew (si no está instalado)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Instalar Dependencias

```bash
# Instalar Python
brew install python@3.11

# Instalar PostgreSQL
brew install postgresql@16

# Iniciar PostgreSQL
brew services start postgresql@16
```

### 3. Seguir Pasos 2-7 de Linux

---

## Instalación con Docker

### 1. Instalar Docker

```bash
# macOS/Windows
Descargar Docker Desktop desde https://www.docker.com/products/docker-desktop

# Linux
sudo apt install -y docker.io docker-compose
```

### 2. Crear Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de dependencias
COPY pyproject.toml .

# Instalar dependencias
RUN pip install --no-cache-dir -e ".[dev]"

# Copiar código fuente
COPY . .

# Puerto expuesto
EXPOSE 8000

# Comando de inicio
CMD ["python", "-m", "app.main"]
```

### 3. Crear docker-compose.yml

```yaml
version: '3.8'

services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: pattern_trader
      POSTGRES_USER: pattern_user
      POSTGRES_PASSWORD: tu_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      DB_HOST: db
      DB_PORT: 5432
      DB_NAME: pattern_trader
      DB_USER: pattern_user
      DB_PASSWORD: tu_password
    depends_on:
      - db

volumes:
  postgres_data:
```

### 4. Ejecutar

```bash
# Construir y ejecutar
docker-compose up -d

# Ver logs
docker-compose logs -f app

# Detener
docker-compose down
```

---

## Instalación con Poetry

### 1. Instalar Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Configurar Proyecto

```bash
# Instalar dependencias
poetry install

# Activar entorno virtual
poetry shell

# Ejecutar
poetry run python -m app.main
```

---

## Verificación de Instalación

### Script de Verificación

Crear archivo `verify_installation.py`:

```python
#!/usr/bin/env python3
"""Script de verificación de instalación."""

import sys
import importlib

def check_python_version():
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requiere 3.11+)")
        return False

def check_dependency(package_name, display_name=None):
    try:
        importlib.import_module(package_name)
        print(f"✅ {display_name or package_name}")
        return True
    except ImportError:
        print(f"❌ {display_name or package_name}")
        return False

def main():
    print("Verificación de Instalación - PatternTrader")
    print("=" * 50)
    
    checks = [
        check_python_version(),
        check_dependency("fastapi", "FastAPI"),
        check_dependency("sqlalchemy", "SQLAlchemy"),
        check_dependency("pydantic", "Pydantic"),
        check_dependency("loguru", "Loguru"),
        check_dependency("pandas", "Pandas"),
        check_dependency("numpy", "NumPy"),
        check_dependency("sklearn", "Scikit-learn"),
        check_dependency("ccxt", "CCXT"),
    ]
    
    print("=" * 50)
    if all(checks):
        print("✅ Todas las verificaciones pasaron")
        return 0
    else:
        print("❌ Algunas verificaciones fallaron")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

Ejecutar:

```bash
python verify_installation.py
```

---

## Solución de Problemas

### Problema: `pip install` falla con errores de compilación

**Solución**:

```bash
# Ubuntu/Debian
sudo apt install -y build-essential python3-dev libpq-dev

# macOS
xcode-select --install
```

### Problema: No se puede conectar a PostgreSQL

**Solución**:

```bash
# Verificar que PostgreSQL está corriendo
sudo systemctl status postgresql

# Reiniciar si es necesario
sudo systemctl restart postgresql

# Verificar configuración de pg_hba.conf
sudo nano /etc/postgresql/16/main/pg_hba.conf

# Asegurar que tiene:
# local   all             all                                     md5
# host    all             all             127.0.0.1/32            md5
```

### Problema: `ModuleNotFoundError: No module named 'app'`

**Solución**:

```bash
# Asegurar que estás en el directorio correcto
cd pattern-trader

# Reinstalar en modo desarrollo
pip install -e .
```

### Problema: Errores de tipos con mypy

**Solución**:

```bash
# Instalar stubs
pip install types-PyYAML types-requests

# Ejecutar mypy ignorando errores opcionales
mypy app/ --ignore-missing-imports
```

---

## Actualización

```bash
# Actualizar código
git pull origin main

# Actualizar dependencias
pip install -e ".[dev]"

# Ejecutar migraciones de base de datos (si aplica)
alembic upgrade head

# Reiniciar servicios
python -m app.main
```

---

## Desinstalación

```bash
# Detener servicios
pkill -f "python -m app.main"

# Eliminar entorno virtual
deactivate
rm -rf venv/

# Eliminar base de datos
sudo -u postgres dropdb pattern_trader
sudo -u postgres psql -c "DROP USER pattern_user;"

# Eliminar archivos generados
rm -rf __pycache__ .pytest_cache .mypy_cache logs/
rm -rf *.egg-info dist build/
```
