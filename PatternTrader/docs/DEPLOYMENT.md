# Guía de Despliegue

## Visión General

Esta guía cubre las opciones para desplegar PatternTrader en diferentes entornos.

---

## Opciones de Despliegue

| Opción | Complejidad | Costo | Escalabilidad |
|--------|-------------|-------|---------------|
| Docker Local | Baja | Gratis | Baja |
| Docker Compose | Media | Gratis | Media |
| Docker + AWS ECS | Alta | Medio | Alta |
| Kubernetes | Muy Alta | Alto | Muy Alta |

---

## Despliegue con Docker

### 1. Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias
COPY pyproject.toml .

# Instalar dependencias
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e ".[dev]"

# Copiar código fuente
COPY . .

# Crear directorios necesarios
RUN mkdir -p logs models data/training

# Puerto expuesto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Comando de inicio
CMD ["python", "-m", "app.main"]
```

### 2. docker-compose.yml

```yaml
version: '3.8'

services:
  # Base de datos
  db:
    image: postgres:16-alpine
    container_name: pattern_trader_db
    environment:
      POSTGRES_DB: pattern_trader
      POSTGRES_USER: pattern_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U pattern_user -d pattern_trader"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # Aplicación
  app:
    build: .
    container_name: pattern_trader_app
    ports:
      - "8000:8000"
    environment:
      - DB_HOST=db
      - DB_PORT=5432
      - DB_NAME=pattern_trader
      - DB_USER=pattern_user
      - DB_PASSWORD=${DB_PASSWORD}
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
      - BINANCE_API_KEY=${BINANCE_API_KEY}
      - BINANCE_API_SECRET=${BINANCE_API_SECRET}
    depends_on:
      db:
        condition: service_healthy
    volumes:
      - ./logs:/app/logs
      - ./models:/app/models
    restart: unless-stopped

  # Redis (opcional, para caché)
  redis:
    image: redis:7-alpine
    container_name: pattern_trader_redis
    ports:
      - "6379:6379"
    restart: unless-stopped

volumes:
  postgres_data:
```

### 3. Variables de Entorno

Crear archivo `.env`:

```env
# Base de datos
DB_PASSWORD=tu_password_seguro
# Opcional: URL completa (prioridad sobre campos discretos)
# DATABASE_URL=postgresql+asyncpg://pattern_user:tu_password_seguro@db:5432/pattern_trader

# Telegram
TELEGRAM_BOT_TOKEN=123456:ABC-DEF
TELEGRAM_CHAT_ID=-1001234567890

# Binance
BINANCE_API_KEY=tu_api_key
BINANCE_API_SECRET=tu_api_secret
```

### 4. Ejecutar

```bash
# Construir y ejecutar
docker-compose up -d

# Aplicar migraciones Alembic (una vez la BD esté healthy)
docker-compose exec app alembic upgrade head

# Ver logs
docker-compose logs -f app

# Detener
docker-compose down
```

> La API ejecuta `init_db()` (crea el esquema si no existe) al arrancar, pero
> en producción se recomienda correr `alembic upgrade head` explícitamente
> para llevar la BD al esquema esperado.

---

## Despliegue en AWS

### Opción 1: EC2 + Docker

```bash
# 1. Crear instancia EC2 (Ubuntu 22.04)

# 2. Conectar por SSH
ssh -i tu-key.pem ubuntu@tu-ip

# 3. Instalar Docker
sudo apt update
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker ubuntu

# 4. Clonar repositorio
git clone https://github.com/tu-usuario/pattern-trader.git
cd pattern-trader

# 5. Configurar variables
cp .env.example .env
nano .env

# 6. Ejecutar
docker-compose up -d
```

### Opción 2: ECS Fargate

```yaml
# ecs-task-definition.json
{
  "family": "pattern-trader",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "pattern-trader",
      "image": "tu-ecr-repo:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "DB_HOST", "value": "tu-rds-endpoint"},
        {"name": "DB_NAME", "value": "pattern_trader"}
      ],
      "secrets": [
        {"name": "DB_PASSWORD", "valueFrom": "arn:aws:secretsmanager:..."}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/pattern-trader",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Opción 3: ECS + RDS

```bash
# 1. Crear RDS PostgreSQL
aws rds create-db-instance \
    --db-instance-identifier pattern-trader-db \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --engine-version 16 \
    --master-username pattern_user \
    --master-user-password tu_password \
    --allocated-storage 20 \
    --vpc-security-group-ids sg-xxxxx

# 2. Crear base de datos
psql -h tu-rds-endpoint -U pattern_user -d postgres
CREATE DATABASE pattern_trader;

# 2b. Aplicar migraciones Alembic (configurar DATABASE_URL en el entorno)
# DATABASE_URL=postgresql+asyncpg://pattern_user:tu_password@tu-rds-endpoint:5432/pattern_trader
alembic upgrade head

# 3. Desplegar ECS
aws ecs create-service \
    --cluster pattern-trader \
    --service-name pattern-trader \
    --task-definition pattern-trader \
    --desired-count 2 \
    --launch-type FARGATE
```

---

## Despliegue con Kubernetes

### 1. Manifests de Kubernetes

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pattern-trader
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pattern-trader
  template:
    metadata:
      labels:
        app: pattern-trader
    spec:
      containers:
      - name: pattern-trader
        image: tu-registry/pattern-trader:latest
        ports:
        - containerPort: 8000
        env:
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: pattern-trader-secrets
              key: db-host
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: pattern-trader-secrets
              key: db-password
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: pattern-trader-service
spec:
  selector:
    app: pattern-trader
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 2. Secrets

```bash
# Crear secrets
kubectl create secret generic pattern-trader-secrets \
    --from-literal=db-host=tu-rds-endpoint \
    --from-literal=db-password=tu_password \
    --from-literal=telegram-bot-token=tu_token
```

### 3. Desplegar

```bash
kubectl apply -f k8s/
kubectl get pods
kubectl logs -f deployment/pattern-trader
```

---

## Monitoreo

### Health Check

```bash
# Verificar salud
curl http://localhost:8000/api/v1/health

# Respuesta esperada
{
  "status": "healthy",
  "application": "PatternTrader",
  "version": "0.1.0",
  "environment": "development"
}
```

### Métricas

```python
# Endpoint de métricas (opcional)
@app.get("/metrics")
async def metrics():
    return {
        "active_patterns": len(lifecycle_engine.get_active()),
        "total_signals": len(signals),
        "uptime": monitor.get_uptime(),
    }
```

### Logs

```bash
# Ver logs con Docker
docker-compose logs -f app

# Ver logs con Kubernetes
kubectl logs -f deployment/pattern-trader

# Logs en archivo
tail -f logs/app.log
```

---

## Backup

### Backup de Base de Datos

```bash
# Backup manual
pg_dump -h localhost -U pattern_user pattern_trader > backup_$(date +%Y%m%d).sql

# Backup automático (cron)
0 2 * * * pg_dump -h localhost -U pattern_user pattern_trader | gzip > /backups/pattern_trader_$(date +\%Y\%m\%d).sql.gz
```

### Backup de Modelos

```bash
# Backup de modelos
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/
```

---

## Seguridad

### Recomendaciones

1. **No exponer puertos innecesarios**
2. **Usar variables de entorno para secretos**
3. **Habilitar HTTPS en producción**
4. **Implementar rate limiting**
5. **Usar firewall**
6. **Actualizar dependencias regularmente**

### nginx.conf (Reverse Proxy)

```nginx
server {
    listen 80;
    server_name tu-dominio.com;
    
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name tu-dominio.com;
    
    ssl_certificate /etc/letsencrypt/live/tu-dominio.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/tu-dominio.com/privkey.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## Escalamiento

### Horizontal

```yaml
# docker-compose.yml con replicas
services:
  app:
    deploy:
      replicas: 3
```

### Vertical

```yaml
# Aumentar recursos
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

---

## Troubleshooting

### Problema: La app no inicia

```bash
# Verificar logs
docker-compose logs app

# Verificar variables de entorno
docker-compose exec app env

# Verificar conexión a BD
docker-compose exec app python -c "from app.database.base import get_engine; print('OK')"
```

### Problema: No conecta a PostgreSQL

```bash
# Verificar que PostgreSQL está corriendo
docker-compose exec db pg_isready

# Verificar credenciales
docker-compose exec db psql -U pattern_user -d pattern_trader
```

### Problema: Memoria insuficiente

```bash
# Verificar uso de memoria
docker stats

# Aumentar límite en docker-compose.yml
services:
  app:
    deploy:
      resources:
        limits:
          memory: 4G
```

---

## Checklist de Despliegue

- [ ] Variables de entorno configuradas (incluye `DATABASE_URL` opcional)
- [ ] Base de datos creada y migrada (`alembic upgrade head`)
- [ ] Modelos ML entrenados
- [ ] Telegram configurado (opcional)
- [ ] SSL habilitado
- [ ] Firewall configurado
- [ ] Backup programado
- [ ] Monitoreo activo
- [ ] Logs configurados
- [ ] Health checks funcionando
