# Voice OAN API

An AI-powered agricultural assistant API that provides real-time, streaming responses to agricultural queries. Supports multiple languages (Gujarati, English) and offers a wide range of agricultural services including weather forecasts, market prices, scheme information, and more.

## Technology Stack

- **Framework**: FastAPI, Uvicorn, Gunicorn
- **AI/LLM**: Pydantic AI, OpenAI/Azure OpenAI/vLLM
- **Cache & Search**: Redis, Marqo (Vector Search)
- **Database**: PostgreSQL (SQLAlchemy)
- **Authentication**: JWT (PyJWT)
- **Monitoring**: Logfire
- **Infrastructure**: Docker, Supervisor
- **Language**: Python 3.10+

### Prerequisites

- Docker and Docker Compose
- Python 3.8+ (for local development)

### Docker Setup

#### 1. Create a Docker Network
```bash
docker network create svanetwork
```

#### 2. Run Redis Stack
```bash
docker run -d --name redis-stack --network svanetwork -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

#### 3. Run Marqo (Vector Search)
```bash
docker run --name marqo -p 8882:8882 \
    -e MARQO_MAX_CONCURRENT_SEARCH=50 \
    -e VESPA_POOL_SIZE=50 \
    marqoai/marqo:latest
```

#### 4. Start the Application
```bash
docker compose up --build --force-recreate --detach
```

#### 5. View Logs
```bash
docker logs -f sva_app
```

### Stop the Application
```bash
docker compose down --remove-orphans
```

## Development

### Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Locally

Run the application locally using the setup instructions above.

## API Contract

Primary endpoint: `GET /api/voice`

Query parameters currently expected by the backend:
- `query`: latest user utterance text
- `session_id`: stable conversation identifier; callers must reuse it across reconnects/retries for the same call
- `source_lang`: current input language, currently `gu` or `en`
- `target_lang`: response language
- `user_id`: caller identifier; phone number is used to resolve farmer context
- `provider`: optional provider marker, currently `RAYA`
- `process_id`: provider-side request/call fragment identifier used for hold/nudge correlation

Response contract:
- `text/plain`
- streamed assistant text chunks
- output is optimized for phone/TTS delivery
- no feedback question is appended at the tail of the stream

Authentication contract:
- bearer JWT required outside development
- JWT subject and claims are treated as caller identity context

## FE / Telephony Contract

This service is built for an interruptible phone flow where the frontend/provider may reconnect repeatedly during one logical call.

Caller requirements:
- reuse the same `session_id` for all retries/reconnects of the same call
- send a fresh request when audio capture or network resumes
- preserve `process_id` semantics expected by the provider integration

Backend behavior:
- Redis stores message history per `session_id`
- Redis also stores active request ownership per `session_id`
- if a newer request for the same `session_id` arrives, older in-flight streams stop before writing stale history or sending extra nudges
- client disconnects are treated as termination for the current in-flight request, not for the whole conversation

Current voice behavior:
- the assistant should stay brief and conversational
- output should avoid markdown, bracketed repeats, and punctuation-heavy formatting
- Gujarati responses should remain respectful and gender-neutral for the caller
- Sarlaben self-reference should stay feminine in Gujarati
- generic wait nudges should be short and neutral, for example "I'm getting back to you, please wait"

## Translation Pipeline

When `ENABLE_TRANSLATION_PIPELINE=true`:
- Gujarati input is pretranslated to English before agent execution
- agent reasoning runs in English
- output is translated back to the requested target language in streamed batches
- OpenAI pretranslation is attempted first
- TranslateGemma is the fallback path if OpenAI returns empty output or times out

Relevant tuning env vars:
- `OPENAI_PRETRANSLATION_MODEL` default: `gpt-5-mini`
- `OPENAI_PRETRANSLATION_TIMEOUT_SECONDS` default: `4.0`
- `NUDGE_TIMEOUT_SECONDS` default: `3.0`
- `SESSION_OWNER_TTL_SECONDS` default: `120`
- `SESSION_OWNER_REFRESH_INTERVAL_SECONDS` default: `15`

## Testing

Fast local regression suite:
```bash
pytest tests/test_voice_fixes.py tests/test_translation_vocabulary.py tests/test_voice_regressions_apr11_12.py -q
```

Integration regressions with real model calls and streamed voice behavior:
```bash
VOICE_PIPELINE_INTEGRATION=1 pytest tests/test_voice_regressions_apr11_12_integration.py -q
```

For the real model-backed checks in that file, set the relevant endpoints/keys first:
- `OPENAI_API_KEY`
- `TRANSLATEGEMMA_27B_BASE_ENDPOINT` or `TRANSLATEGEMMA_27B_BASE_ENDPOINTS`

## Maintenance

### Clean Up Docker Volumes
```bash
docker system prune -a --volumes
```

### Restart Services
```bash
docker compose down --remove-orphans
docker compose up --build --force-recreate --detach
```
