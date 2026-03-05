"""
Google Gemini service wrapper.
Handles both chat completion (tutoring responses) and
structured JSON extraction (pre-processor).

MOCK MODE (USE_MOCK_LLM=true in .env):
  - preprocess_query: devuelve valores por defecto sin llamar a Gemini.
  - generate_rag_response: devuelve respuesta de stub con los fragmentos recuperados,
    confirmando que el flujo completo (API → vector search → LLM) funcionó correctamente.
"""
import json
import re
import google.generativeai as genai
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Initialise SDK once (only when NOT in mock mode) ─────────────────────────────────────────────
if not settings.USE_MOCK_LLM:
    genai.configure(api_key=settings.GEMINI_API_KEY)

_model: genai.GenerativeModel | None = None
_preprocess_model: genai.GenerativeModel | None = None


def _get_model() -> genai.GenerativeModel:
    global _model
    if _model is None:
        _model = genai.GenerativeModel(
            model_name=settings.GEMINI_MODEL,
            generation_config=genai.types.GenerationConfig(
                temperature=settings.GEMINI_TEMPERATURE,
                max_output_tokens=settings.GEMINI_MAX_TOKENS,
            ),
        )
        logger.info("Gemini RAG model '%s' initialised (max_tokens=%d)",
                    settings.GEMINI_MODEL, settings.GEMINI_MAX_TOKENS)
    return _model


def _get_preprocess_model() -> genai.GenerativeModel:
    """Lightweight model instance for pre-processing: low token cap → faster JSON output."""
    global _preprocess_model
    if _preprocess_model is None:
        _preprocess_model = genai.GenerativeModel(
            model_name=settings.GEMINI_MODEL,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,                                  # deterministic JSON
                max_output_tokens=settings.GEMINI_PREPROCESS_MAX_TOKENS,
            ),
        )
        logger.info("Gemini preprocess model '%s' initialised (max_tokens=%d)",
                    settings.GEMINI_MODEL, settings.GEMINI_PREPROCESS_MAX_TOKENS)
    return _preprocess_model


# ── Tutor identity (injected before every LLM call) ─────────────────────────

def _materia_display_name(materia_id: str) -> str:
    """
    Convert a materia_id folder name to a human-readable course name.
    Examples:
      '258_Criminologia_B'                               → 'Criminologia'
      '58_Estadistica_y_probabilidad'                    → 'Estadistica Y Probabilidad'
      '152_Introduccion_a_la_administracion_publica_C'   → 'Introduccion A La Administracion Publica'
    """
    parts = materia_id.split("_")
    # Skip leading numeric prefix
    start = 1 if parts[0].isdigit() else 0
    # Skip trailing single-uppercase-letter suffix (e.g. '_B', '_C')
    end = len(parts) - 1 if (len(parts) > 1 and len(parts[-1]) == 1 and parts[-1].isupper()) else len(parts)
    return " ".join(parts[start:end]).title()


TUTOR_IDENTITY_TEMPLATE = (
    "Eres Bot Academia, tutor del curso {materia_name} en UTEL (español latinoamericano). "
    "Responde único sobre contenidos de {materia_name}."
)


# ── Pre-processor ─────────────────────────────────────────────────────────────

PREPROCESSOR_SYSTEM = """Eres un asistente de análisis de consultas académicas.
Tu tarea es analizar el mensaje de un estudiante y devolver ÚNICAMENTE un objeto JSON válido con los campos:
- intent: clasificación del mensaje. Valores posibles:
    "academico"       - pregunta sobre contenido de la materia
    "saludo"          - greeting sin contenido académico
    "queja"           - queja o insatisfacción
    "fuera_de_tema"   - tema que no tiene relación con la materia
    "despedida"       - el estudiante indica claramente que ya no tiene más dudas o se despide
                        (ejemplos: "gracias, hasta luego", "ya entendí todo, bye", "chao", "eso era todo", "muchas gracias ya no tengo dudas")
- sentiment: estado emocional. Valores posibles: "estresado", "molesto", "neutral", "positivo"
- confidence: número entre 0 y 1

NO incluyas texto adicional, solo el JSON."""


async def preprocess_query(raw_message: str, materia_id: str) -> dict:
    """
    Worker 1: Clean and classify the student's message.
    Returns a dict matching PreprocessResult schema.
    The dict always includes '_tokens_in' and '_tokens_out' keys for cost tracking.
    """    # ── MOCK MODE ──────────────────────────────────────────────────────────────────────────────
    if settings.USE_MOCK_LLM:
        logger.info("[MOCK] preprocess_query | materia=%s | msg=%s", materia_id, raw_message[:60])
        return {
            "intent": "academico",
            "sentiment": "neutral",
            "confidence": 1.0,
            "_tokens_in": 0,
            "_tokens_out": 0,
        }
    materia_name = _materia_display_name(materia_id)
    logger.debug("Tutor identity | materia=%s | name=%s", materia_id, materia_name)

    prompt = f"""Materia: {materia_name}
Mensaje del estudiante: {raw_message}

Responde ÚNICAMENTE con el JSON."""

    model = _get_preprocess_model()
    try:
        chat = model.start_chat(history=[
            {"role": "user", "parts": [PREPROCESSOR_SYSTEM]},
            {"role": "model", "parts": ['{"acknowledged": true}']},
        ])
        response = await chat.send_message_async(prompt)
        raw_text = response.text.strip()

        # ── Token usage from preprocess call ────────────────────────
        usage = response.usage_metadata
        tokens_in  = getattr(usage, "prompt_token_count",     0) or 0
        tokens_out = getattr(usage, "candidates_token_count", 0) or 0
        logger.info(
            "TOKENS preprocess | materia=%s | in=%d out=%d total=%d",
            materia_id, tokens_in, tokens_out, tokens_in + tokens_out,
        )

        # Extract JSON even if model adds markdown code fences
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            result["_tokens_in"]  = tokens_in
            result["_tokens_out"] = tokens_out
            return result

        logger.warning("Pre-processor returned non-JSON: %s", raw_text[:200])

    except Exception as exc:
        logger.error("Pre-processor LLM error: %s", exc)

    # Fallback: safe defaults so the pipeline doesn't break
    return {
        "intent": "academico",
        "sentiment": "neutral",
        "confidence": 0.5,
        "_tokens_in": 0,
        "_tokens_out": 0,
    }


# ── RAG response generator ────────────────────────────────────────────────────

RAG_SYSTEM = """MODO: ensamblaje directo. NO razonamiento interno. NO cadena de pensamiento. NO explicar el proceso. Escribe la respuesta final INMEDIATAMENTE.

Eres OmnIA, tutor oficial de UTEL. Responde SOLO con información del contexto proporcionado. Idioma: español latinoamericano. Tono: cercano y alentador.

FORMATO (obligatorio):
- Abre con emoji motivador + frase corta. Nunca "Entiendo que..." ni "Basado en el contexto...".
- Lista PLANA con guión (-) y *negrita* en el título de cada opción. Sin sub-bullets anidados.
- Si el contexto tiene N opciones o pasos, inclúyelos TODOS sin omitir ninguno.
- Máx. 2 líneas por punto. Doble salto entre secciones.
- Sin datos en contexto: ⚠️ No tengo ese dato, consúltalo con tu profesor.
- PROHIBIDO terminar con frases de cierre como "¿Te puedo ayudar con algo más?", "Espero haberte ayudado", "¡Aquí estoy para apoyarte!" o similares. Termina directamente después del último punto de contenido.
"""


async def generate_rag_response(  # noqa: PLR0913
    original_message: str,
    sentiment: str,
    intent: str,
    context_chunks: list[dict],
    materia_id: str,
    interaction_id: str,
    conversation_history: list[dict] | None = None,
) -> str:
    """
    Final LLM step: combine RAG context + session history → tutoring response.
    conversation_history is a list of {"role": "user"|"assistant", "content": str} dicts
    loaded from Redis; gives the LLM context of prior turns in this session.
    """
    if not context_chunks:
        context_text = "No se encontró información relevante en el material de esta materia."
    else:
        context_text = "\n---\n".join(
            f"[Fuente: {c['source']}]\n{c['content']}" for c in context_chunks
        )

    # Tone advisory based on sentiment
    tone_hint = {
        "estresado": "El estudiante parece estresado. Sé especialmente alentador y paciente.",
        "molesto": "El estudiante parece molesto. Comienza reconociendo su frustración antes de responder.",
        "positivo": "El estudiante está motivado. Mantén esa energía positiva.",
        "neutral": "",
    }.get(sentiment, "")

    materia_name = _materia_display_name(materia_id)
    identity = TUTOR_IDENTITY_TEMPLATE.format(materia_name=materia_name)

    # ── Build conversation history block (previous turns from Redis) ─────────────────────────
    history_block = ""
    if conversation_history:
        lines = []
        for msg in conversation_history:
            role_label = "Estudiante" if msg["role"] == "user" else "OmnIA"
            lines.append(f"{role_label}: {msg['content']}")
        history_block = "\n[Historial de la conversación (turnos anteriores)]\n" + "\n".join(lines) + "\n"

    prompt = f"""[Identidad del tutor]
{identity}

[Sistema de respuesta]
{RAG_SYSTEM}
{tone_hint}
{history_block}
[Contexto de la materia: {materia_name}]
{context_text}

[Pregunta del estudiante]
{original_message}

[ID de interacción: {interaction_id}]

Responde al estudiante:"""

    # ── MOCK MODE ──────────────────────────────────────────────────────────────────────────────
    if settings.USE_MOCK_LLM:
        chunks_found = len(context_chunks)
        sources = list({c["source"] for c in context_chunks}) if context_chunks else []
        logger.info(
            "[MOCK] generate_rag_response | materia=%s | chunks=%d | interaction=%s",
            materia_id, chunks_found, interaction_id,
        )
        mock_detail = (
            f" (flujo completo: pre-procesado ✓ → {chunks_found} fragmentos recuperados "
            f"de ChromaDB [{', '.join(sources) or 'sin fuentes'}] → LLM stub)"
        )
        return (
            f"[MODO DEMO] Es un curso de UTEL, aún no conectado al LLM pero sí "
            f"siguiendo el flujo completo como lo diseñamos.{mock_detail}",
            0, 0,
        )

    # ── Log full prompt sent to LLM (DEBUG) ─────────────────────────────────────────────────
    history_turns = len(conversation_history) // 2 if conversation_history else 0
    logger.debug(
        "PROMPT → LLM | interaction=%s | historial_turns=%d | prompt_chars=%d\n--- PROMPT START ---\n%s\n--- PROMPT END ---",
        interaction_id, history_turns, len(prompt), prompt,
    )

    model = _get_model()
    try:
        response = await model.generate_content_async(prompt)
        result = response.text.strip()

        # ── Extract token usage from Gemini metadata ──────────────────────────────────────
        usage = response.usage_metadata
        tokens_in  = getattr(usage, "prompt_token_count",     0) or 0
        tokens_out = getattr(usage, "candidates_token_count", 0) or 0
        logger.info(
            "TOKENS | interaction=%s | in=%d out=%d total=%d",
            interaction_id, tokens_in, tokens_out, tokens_in + tokens_out,
        )
        logger.debug(
            "RESPUESTA ← LLM | interaction=%s | response_chars=%d tokens_in=%d tokens_out=%d\n--- RESPONSE START ---\n%s\n--- RESPONSE END ---",
            interaction_id, len(result), tokens_in, tokens_out, result,
        )
        return result, tokens_in, tokens_out
    except Exception as exc:
        logger.error("RAG response generation error: %s", exc)
        return (
            "Disculpa, en este momento no puedo procesar tu consulta. "
            "Por favor intenta nuevamente en unos momentos.",
            0, 0,
        )


# ── Embedding (for future custom embedding pipeline) ─────────────────────────

async def get_embedding(text: str) -> list[float]:
    """Generate a Gemini text embedding vector."""
    try:
        result = genai.embed_content(
            model=settings.EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document",
        )
        return result["embedding"]
    except Exception as exc:
        logger.error("Embedding error: %s", exc)
        return []
