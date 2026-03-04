# BotAcademia Engine 🎓

**Motor de Tutoría IA basado en RAG** para la plataforma UTEL.  
Procesa consultas académicas desde WhatsApp/BOT LUA usando Gemini 2.5 Flash + ChromaDB.

---

## Arquitectura

```
BOT LUA  ──POST /api/v1/query──►  FastAPI (Orquestador)
                                        │
                          ┌─────────────┼─────────────────┐
                          ▼             ▼                   ▼
                    Gemini LLM     ChromaDB           PostgreSQL
                  (Pre-process   (Vector Search       (Query Logs
                   + Response)    por materia)         Histórico)
                          │
                   ┌──────┤  PRODUCCIÓN (USE_KAFKA=true)
                   │  Kafka Topics:
                   │    botacademia.incoming_messages  ──► Worker 1 (Preprocessor)
                   │    botacademia.processed_queries  ──► Worker 2 (RAG + LLM)
                   │
                   └── Redis (Caché de sesión, USE_REDIS=true)
```

### Pipeline de cada consulta

```
1. raw_message  ──►  Pre-procesador IA (Gemini)
                     → intent: academico | saludo | queja | fuera_de_tema
                     → sentiment: estresado | molesto | neutral | positivo
                     → clean_query: pregunta optimizada para búsqueda semántica

2. clean_query  ──►  ChromaDB (colección de la materia exacta)
                     → Top-5 chunks más relevantes

3. chunks + query ──►  Gemini (Generación RAG)
                        → Respuesta empática, basada 100% en el contexto

4. response  ──►  BOT LUA  +  PostgreSQL (log)
```

---

## Estructura del Proyecto

```
botAcademIA/
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── api/v1/routes/
│   │   ├── messages.py          # POST /query, GET /health
│   │   └── ingest.py            # POST /ingest, GET /ingest/list
│   ├── core/
│   │   ├── config.py            # Configuración centralizada (Pydantic Settings)
│   │   ├── database.py          # Conexión async PostgreSQL
│   │   └── logging.py           # Logging estructurado
│   ├── models/
│   │   ├── schemas.py           # Modelos Pydantic (request/response)
│   │   └── db_models.py         # Modelos SQLAlchemy (PostgreSQL)
│   ├── services/
│   │   ├── pipeline.py          # Orquestador del pipeline POC (síncrono)
│   │   ├── llm_service.py       # Wrapper Gemini (pre-process + RAG response)
│   │   ├── vector_store.py      # ChromaDB operations
│   │   ├── ingest_service.py    # Lecturas y chunking de archivos fuente
│   │   └── redis_cache.py       # Caché de sesión (Redis, producción)
│   └── workers/
│       ├── kafka_producer.py    # Producer Kafka (producción)
│       └── kafka_consumer.py    # Workers 1 y 2 Kafka (producción)
├── data/
│   └── materias/                # Archivos fuente por materia
│       ├── 258_Criminologia_B/
│       │   ├── contenido.json
│       │   ├── contenido.txt
│       │   └── descripcion.json
│       └── ...  (5 materias POC)
├── scripts/
│   └── ingest_materia.py        # CLI para vectorizar materias
├── docker-compose.yml           # POC: API + ChromaDB + PostgreSQL
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Setup Rápido (POC Local)

### 1. Pre-requisitos
- Docker Desktop instalado y corriendo
- Python 3.11+ (para correr el script de ingesta localmente si se desea)
- API Key de Google Gemini: [aistudio.google.com](https://aistudio.google.com)

### 2. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env y agregar tu GEMINI_API_KEY
```

### 3. Levantar servicios

```bash
# POC: FastAPI + ChromaDB + PostgreSQL
docker compose up -d

# Ver logs en tiempo real
docker compose logs -f botacademia_api
```

### 4. Vectorizar las materias (OBLIGATORIO antes de consultar)

```bash
# Dentro del contenedor
docker compose exec botacademia_api python scripts/ingest_materia.py --all

# O via API (mientras el contenedor corre)
curl -X POST http://localhost:8080/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"materia_id": "258_Criminologia_B"}'
```

### 5. Probar una consulta

```bash
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "interaction_id": "test_001",
    "materia_id": "258_Criminologia_B",
    "message": "no entiendo que es la criminologia y para que sirve??"
  }'
```

### 6. Swagger UI

Abre [http://localhost:8080/docs](http://localhost:8080/docs) para explorar todos los endpoints.

---

## Producción (Kafka + Redis)

```bash
# Activar todos los servicios del perfil 'production'
docker compose --profile production up -d

# En .env, cambiar:
USE_KAFKA=true
USE_REDIS=true
```

Con el perfil de producción se agregan:
- **Kafka** (3 particiones, retención 7 días, idempotencia activada)
- **Zookeeper** (coordinación Kafka)
- **Redis** (caché de sesión con TTL de 1 hora)
- **Kafka UI** en [http://localhost:8090](http://localhost:8090)

---

## API Reference

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/api/v1/query` | Consulta académica desde BOT LUA |
| `GET` | `/api/v1/health` | Estado de todos los servicios |
| `POST` | `/api/v1/ingest` | Vectorizar una materia |
| `GET` | `/api/v1/ingest/list` | Listar materias disponibles |
| `GET` | `/docs` | Swagger UI |

### Payload de consulta (BOT LUA → BotAcademia)

```json
{
  "interaction_id": "lua_conv_abc123",
  "materia_id": "258_Criminologia_B",
  "message": "no entiendo la diferencia entre crimen y delito"
}
```

### Respuesta (BotAcademia → BOT LUA)

```json
{
  "interaction_id": "lua_conv_abc123",
  "materia_id": "258_Criminologia_B",
  "response": "¡Hola! Entiendo tu duda. Según el material del curso...",
  "intent": "academico",
  "sentiment": "neutral",
  "processing_time_ms": 2340.5,
  "status": "completed"
}
```

---

## Escalabilidad

- **Nuevas materias:** subir los 3 archivos a `data/materias/<materia_id>/` y llamar `POST /ingest`
- **Frontend futuro:** reemplazar el script CLI por un endpoint `POST /upload` con `multipart/form-data`
- **800+ materias:** cada materia tiene su propia colección en ChromaDB aislada
- **Alta carga:** activar perfil `production`, escalar workers Kafka horizontalmente

---

## Stack Tecnológico

| Componente | Tecnología |
|------------|-----------|
| API | FastAPI 0.115 |
| LLM | Gemini 2.5 Flash |
| Base vectorial | ChromaDB 0.5 |
| Base relacional | PostgreSQL 16 |
| Mensajería | Apache Kafka 7.6 (Confluent) |
| Caché | Redis 7 |
| Contenedores | Docker + Docker Compose |
| Lenguaje | Python 3.11+ |


---

## Levantar los servicios

```bash
# 1. Primera vez o al cambiar Dockerfile / requirements.txt
docker compose build botacademia_api

# 2. Iniciar los 3 servicios (FastAPI + ChromaDB + PostgreSQL)
docker compose up -d

# 3. Verificar que los 3 están healthy
docker compose ps

# 4. Ver logs en tiempo real
docker compose logs -f botacademia_api
```

### Vectorizar las materias (solo la primera vez o al actualizar contenido)

```bash
# Cada materia en una llamada (forzar re-ingesta con force_reingest true si ya existe)
curl -X POST http://localhost:8080/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"materia_id":"258_Criminologia_B","force_reingest":false}'

curl -X POST http://localhost:8080/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"materia_id":"58_Estadistica_y_probabilidad","force_reingest":false}'

curl -X POST http://localhost:8080/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"materia_id":"152_Introduccion_a_la_administracion_publica_C","force_reingest":false}'

curl -X POST http://localhost:8080/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"materia_id":"155_Principios_y_perspectivas_de_la_administracion_C","force_reingest":false}'

curl -X POST http://localhost:8080/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"materia_id":"156_Sociologia_Rural_C","force_reingest":false}'

# Verificar materias indexadas
curl http://localhost:8080/api/v1/ingest/list

# Estado general del sistema
curl http://localhost:8080/api/v1/health
```

### Apagar el stack

```bash
docker compose down          # detener (volúmenes intactos)
docker compose down -v       # detener + borrar volúmenes (ChromaDB y PG se reinician desde cero)
```

---

## 10 Pruebas desde Postman / curl

> Importar en Postman: **New Request → POST → URL → Body → raw → JSON**

---

### 1 — Pregunta conceptual básica (Criminología)

```
POST http://localhost:8080/api/v1/query
Content-Type: application/json

{
  "interaction_id": "test_001",
  "materia_id": "258_Criminologia_B",
  "message": "que es la criminologia y para que sirve?"
}
```

---

### 2 — Pregunta con ortografía informal (prueba del pre-procesador LLM)

```
POST http://localhost:8080/api/v1/query
Content-Type: application/json

{
  "interaction_id": "test_002",
  "materia_id": "258_Criminologia_B",
  "message": "no entiendo la diferencia entre crimen y delito, me parece re confuso"
}
```

---

### 3 — Estudiante estresado antes de examen (prueba de sentimiento)

```
POST http://localhost:8080/api/v1/query
Content-Type: application/json

{
  "interaction_id": "test_003",
  "materia_id": "258_Criminologia_B",
  "message": "no entiendo NADA de las teorias criminologicas y el examen es manana, ayuda!!"
}
```

---

### 4 — Saludo puro (cortocircuito: no llama a RAG ni a Gemini)

```
POST http://localhost:8080/api/v1/query
Content-Type: application/json

{
  "interaction_id": "test_004",
  "materia_id": "258_Criminologia_B",
  "message": "hola buenas tardes, como estas?"
}
```

---

### 5 — Pregunta fuera de tema (debe responder que solo atiende dudas académicas)

```
POST http://localhost:8080/api/v1/query
Content-Type: application/json

{
  "interaction_id": "test_005",
  "materia_id": "258_Criminologia_B",
  "message": "cuanto cuesta un iphone 16 pro max?"
}
```

---

### 6 — Estadística: pregunta técnica

```
POST http://localhost:8080/api/v1/query
Content-Type: application/json

{
  "interaction_id": "test_006",
  "materia_id": "58_Estadistica_y_probabilidad",
  "message": "como se interpreta la desviacion estandar en un conjunto de datos?"
}
```

---

### 7 — Estadística: estudiante motivado (sentimiento positivo)

```
POST http://localhost:8080/api/v1/query
Content-Type: application/json

{
  "interaction_id": "test_007",
  "materia_id": "58_Estadistica_y_probabilidad",
  "message": "me encanta la estadistica! puedes explicarme que es la distribucion normal?"
}
```

---

### 8 — Sociología Rural: pregunta de contenido

```
POST http://localhost:8080/api/v1/query
Content-Type: application/json

{
  "interaction_id": "test_008",
  "materia_id": "156_Sociologia_Rural_C",
  "message": "cuales son las principales caracteristicas de las comunidades rurales en Mexico?"
}
```

---

### 9 — Administración Pública: comparación conceptual

```
POST http://localhost:8080/api/v1/query
Content-Type: application/json

{
  "interaction_id": "test_009",
  "materia_id": "152_Introduccion_a_la_administracion_publica_C",
  "message": "que diferencia hay entre administracion publica y administracion privada?"
}
```

---

### 10 — Principios de Administración: estudiante frustrado

```
POST http://localhost:8080/api/v1/query
Content-Type: application/json

{
  "interaction_id": "test_010",
  "materia_id": "155_Principios_y_perspectivas_de_la_administracion_C",
  "message": "llevo semanas sin entender nada de la nueva gestion publica, esto es demasiado dificil"
}
```

---

## Ver los logs de cada etapa

Cada prueba genera entradas en los siguientes archivos dentro de `logs/`:

| Archivo | Qué contiene |
|---|---|
| `pipeline.log` | Query original, query mejorada por LLM, fragmentos recuperados, respuesta final, timings |
| `ingest.log` | Chunks extraídos de cada archivo, batches subidos a ChromaDB |
| `rag.log` | Llamadas a Gemini (pre-proceso y generación), errores de LLM |
| `api.log` | Peticiones HTTP entrantes, lifespan events |
| `db.log` | Queries SQL ejecutadas en PostgreSQL |

```bash
# En tiempo real desde contenedor
docker compose exec botacademia_api tail -f /app/logs/pipeline.log

# Últimas 50 líneas desde el host (Windows)
Get-Content logs\pipeline.log -Tail 50
Get-Content logs\rag.log -Tail 50
Get-Content logs\ingest.log -Tail 50
```
docker compose build botacademia_api 2>&1
docker compose up -d --force-recreate botacademia_api
docker compose up -d --force-recreate botacademia_api
docker compose up -d --force-recreate botacademia_api
Start-Sleep -Seconds 8; docker compose exec botacademia_api env | Select-String "GEMINI_MAX|GEMINI_PRE"

Start-Sleep -Seconds 5; docker inspect botacademia_api --format "{{.State.Health.Status}}"; docker exec botacademia_api pip show flashrank 2>&1 | Select-String "Name|Version"