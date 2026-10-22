from typing import Any, TypedDict

class AgentState(TypedDict):
    hora_actual: int
    solicitudes: list[dict[str, Any]]
    correos_procesados: list[str]
    pendientes_aprobacion: list[dict[str, Any]]
    eventos: list[dict[str, Any]]
    llamadas_llm: int
    correos_nuevos: list[dict[str, Any]]
    correos_evento: list[dict[str, Any]]
    correos_clasificados: list[dict[str, Any]]
    clasificacion_actual: list[dict[str, Any]]
    firmas_solicitudes: list[str]
    ids_solicitudes: list[str]


def initial_agent_state() -> AgentState:
    return {
        "hora_actual": 0,
        "solicitudes": [],
        "correos_procesados": [],
        "pendientes_aprobacion": [],
        "eventos": [],
        "llamadas_llm": 0,
        "correos_nuevos": [],
        "correos_evento": [],
        "correos_clasificados": [],
        "clasificacion_actual": [],
        "firmas_solicitudes": [],
        "ids_solicitudes": [],
    }
