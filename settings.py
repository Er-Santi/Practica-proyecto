from pathlib import Path
import os

API_BASE_URL = "https://api.agentmail.to/v0"


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


REQUEST_TIMEOUT_SECONDS = 20


MAX_LLM_CALLS = 0


OPENROUTER_DEFAULT_MODEL = "openai/gpt-4o-mini"


DEFAULT_RECONNECT_INTERVAL_SECONDS = 5


HEARTBEAT_INTERVAL_SECONDS = 60


AUDIT_PATH = Path("auditoria.jsonl")


STATE_PATH = Path("estado_agente.json")


ASSIGNMENTS_PATH = Path("asignacion.csv")


PM_PROFILES_PATH = Path("pms.csv")


DEFAULT_SIMULATION_SEED = 42


PM_PROFILE_HOUR_RANGES = {
    "rapido": (1, 2),
    "normal": (3, 6),
    "lento": (7, 12),
}


REQUEST_MARKERS = (
    "solicitud de personal",
    "solicitud de puesto",
    "solicitud de posicion",
    "solicitud de posición",
    "solicitud de recurso",
    "solicitamos",
    "solicito",
    "requerimos",
    "necesitamos cubrir",
    "vacante",
    "staffing request",
    "position request",
    "resource request",
    "headcount request",
    "we need",
    "we are looking for",
    "we're looking for",
    "se busca",
    "buscamos",
    "estamos buscando",
    "necesitamos",
    "se requieren",
    "requerimos",
    "incorporar",
    "hiring",
    "need to hire",
    "looking to hire",
)

HIRING_ROLE_MARKERS = (
    "engineer",
    "developer",
    "designer",
    "analyst",
    "architect",
    "consultant",
    "tester",
    "quality engineer",
    "data engineer",
    "mainframe",
    "full stack",
    "frontend",
    "front end",
    "backend",
    "back end",
    "ingeniero",
    "ingeniera",
    "desarrollador",
    "desarrolladora",
    "analista",
    "arquitecto",
    "arquitecta",
    "tester",
    "qa",
    "data",
    "as400",
    "qe",
)


def configured_max_llm_calls() -> int:
    raw_value = os.environ.get("MAX_LLM_CALLS", str(MAX_LLM_CALLS)).strip()
    try:
        limit = int(raw_value)
    except ValueError as exc:
        raise ValueError("MAX_LLM_CALLS debe ser un entero mayor o igual que cero.") from exc
    if limit < 0:
        raise ValueError("MAX_LLM_CALLS debe ser un entero mayor o igual que cero.")
    return limit
