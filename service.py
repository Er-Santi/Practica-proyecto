from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from agent_config import AgentConfig, load_agent_config
from agent_state import AgentState, initial_agent_state
from agentmail_client import AgentMailClient
from audit_log import JsonlAuditLog, audit_error
from email_utils import (
    _ordered_by_timestamp, clarification_questions, classify_by_rules,
    confirmation_question,
    extract_request_details, get_message_id, get_sender, infer_request_status,
    load_assignments, load_director_assignments,
    normalize_text, request_fingerprint, simulated_confirmation_delay,
)
from exceptions import AgentMailError, OpenRouterError
from openrouter_client import OpenRouterClient
from pm_assignment import count_open_projects
from settings import (
    AUDIT_PATH,
    DEFAULT_RECONNECT_INTERVAL_SECONDS, DEFAULT_SIMULATION_SEED,
    HEARTBEAT_INTERVAL_SECONDS, configured_max_llm_calls,
)
from reporting import build_sla_report
from sla import (
    SLA_STOP_STATES,
    build_director_approval_graph,
    process_due_sla_actions,
)
from state_store import load_agent_state, save_agent_state
from workflow import build_graph

def apply_human_confirmation(
    state: AgentState,
    *,
    identifier: str,
    confirmed: bool,
    audit: JsonlAuditLog,
) -> AgentState:
    request = next(
        (
            item
            for item in state["solicitudes"]
            if item["id"] == identifier
        ),
        None,
    )
    if request is None:
        raise KeyError(f"No existe la solicitud {identifier}.")
    if not request.get("pm_responsable"):
        raise ValueError("Asigna un PM antes de confirmar la solicitud.")
    if request.get("estado") == "Cerrada":
        raise ValueError("No se puede confirmar una solicitud ya finalizada.")

    request["confirmacion_usuario"] = confirmed
    request["hora_confirmacion"] = state["hora_actual"] if confirmed else None
    request["confirmacion_simulada"] = False
    if confirmed:
        request["estado"] = "Confirmada"
        request["fecha_confirmacion"] = datetime.now(timezone.utc).isoformat()
        request["hora_confirmacion"] = state["hora_actual"]
    elif request["estado"] == "Confirmada":
        request["estado"] = "Notificada"
        request.pop("fecha_confirmacion", None)
        request["hora_confirmacion"] = None
    event = {
        "hora_simulada": state["hora_actual"],
        "solicitud": identifier,
        "nodo": "confirmacion_humana",
        "decision": "pm_confirmado_por_usuario" if confirmed else "pm_no_confirmado_por_usuario",
        "motivo": (
            "El usuario confirmó en la terminal que el PM se hará cargo."
            if confirmed
            else "El usuario indicó en la terminal que el PM no se hará cargo."
        ),
        "datos": {
            "estado": request["estado"],
            "pm_responsable": request["pm_responsable"],
            "trabajo_finalizado": False,
        },
    }
    audit.append(event)
    state["eventos"].append(event)
    answer = (
        "El PM acepta hacerse cargo"
        if confirmed
        else "El PM todavía no acepta hacerse cargo"
    )
    print(f"[VALIDACIÓN] {request['asunto']}: {answer}, según tu respuesta.")
    return state


def mark_request_opened(
    state: AgentState,
    *,
    identifier: str,
    audit: JsonlAuditLog,
) -> AgentState:
    request = next(
        (item for item in state["solicitudes"] if item["id"] == identifier),
        None,
    )
    if request is None or request["estado"] != "Nueva":
        return state
    request["estado"] = "Notificada"
    request["fecha_apertura"] = datetime.now(timezone.utc).isoformat()
    event = {
        "hora_simulada": state["hora_actual"],
        "solicitud": identifier,
        "nodo": "abrir_correo_terminal",
        "decision": "solicitud_abierta",
        "motivo": "El usuario abrió el correo desde la terminal.",
        "datos": {"estado": request["estado"]},
    }
    audit.append(event)
    state["eventos"].append(event)
    return state


def close_request(
    state: AgentState,
    *,
    request_number: int,
    audit: JsonlAuditLog,
) -> AgentState:
    if not 1 <= request_number <= len(state["solicitudes"]):
        raise IndexError("El número de solicitud está fuera de rango.")
    request = state["solicitudes"][request_number - 1]
    if request["estado"] == "Cerrada":
        raise ValueError("La solicitud ya está finalizada.")
    request["estado"] = "Cerrada"
    request["fecha_cierre"] = datetime.now(timezone.utc).isoformat()
    event = {
        "hora_simulada": state["hora_actual"],
        "solicitud": request["id"],
        "nodo": "cerrar_solicitud_terminal",
        "decision": "solicitud_cerrada",
        "motivo": "El usuario cerró la solicitud desde la terminal.",
    }
    audit.append(event)
    state["eventos"].append(event)
    return state


def complete_request(
    state: AgentState,
    *,
    request_number: int,
    audit: JsonlAuditLog,
) -> AgentState:
    if not 1 <= request_number <= len(state["solicitudes"]):
        raise IndexError("El número de solicitud está fuera de rango.")
    request = state["solicitudes"][request_number - 1]
    if request["estado"] == "Cerrada":
        raise ValueError("La solicitud ya está finalizada.")
    completed_at = datetime.now(timezone.utc).isoformat()
    request["estado"] = "Cerrada"
    request["fecha_cierre"] = completed_at
    request["fecha_completada"] = completed_at
    event = {
        "hora_simulada": state["hora_actual"],
        "solicitud": request["id"],
        "nodo": "completar_solicitud_terminal",
        "decision": "solicitud_completada",
        "motivo": "El usuario indicó que el trabajo de la solicitud terminó.",
        "datos": {
            "pm_responsable": request.get("pm_responsable"),
            "fecha_completada": completed_at,
        },
    }
    audit.append(event)
    state["eventos"].append(event)
    print(f"[COMPLETADA] {request['asunto']}.")
    return state


def assign_pm_to_request(
    state: AgentState,
    *,
    request_number: int,
    pm_name: str,
    audit: JsonlAuditLog,
    pm_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    simulation_seed: int = DEFAULT_SIMULATION_SEED,
    config: AgentConfig | None = None,
) -> AgentState:
    if not pm_name.strip():
        raise ValueError("El nombre del PM no puede estar vacío.")
    if not 1 <= request_number <= len(state["solicitudes"]):
        raise IndexError("El número de solicitud está fuera de rango.")

    request = state["solicitudes"][request_number - 1]
    pm_name = pm_name.strip()
    agent_config = config or load_agent_config()
    current_load = count_open_projects(
        state["solicitudes"],
        pm_name,
        exclude_request_id=request["id"],
    )
    if current_load >= agent_config.max_open_projects_per_pm:
        raise ValueError(
            f"El PM {pm_name} ya tiene {current_load} proyectos abiertos; "
            f"el límite es {agent_config.max_open_projects_per_pm}."
        )
    request["pm_responsable"] = pm_name
    request["hora_notificacion"] = state["hora_actual"]
    request["fecha_notificacion"] = datetime.now(timezone.utc).isoformat()
    request["recordatorio_enviado"] = False
    request["escalado_director"] = False
    request["nivel_maximo_escalamiento"] = "Ninguno"
    request.pop("hora_recordatorio", None)
    request.pop("fecha_recordatorio", None)
    request.pop("hora_escalamiento_director", None)
    request["hora_confirmacion"] = None
    request.pop("confirmacion_usuario", None)
    request["confirmacion_simulada"] = False
    request["pregunta_confirmacion"] = confirmation_question(request)
    request["datos_faltantes"] = [
        field
        for field, value in (
            ("cliente", request.get("cliente")),
            ("rol", request.get("rol")),
            ("cantidad_posiciones", request.get("cantidad_posiciones")),
            ("pm_responsable", request.get("pm_responsable")),
        )
        if value is None
    ]
    request["preguntas_aclaracion"] = clarification_questions(request)
    profile_map = {
        normalize_text(str(name)): profile
        for name, profile in (pm_profiles or {}).items()
    }
    profile = profile_map.get(
        normalize_text(pm_name),
        {"perfil": "normal", "horas_confirmacion": None},
    )
    request["perfil_pm"] = profile.get("perfil", "normal")
    request["horas_esperadas_confirmacion"] = simulated_confirmation_delay(
        request,
        profile,
        seed=simulation_seed,
    )

    assignment_event = {
        "hora_simulada": state["hora_actual"],
        "solicitud": request["id"],
        "nodo": "asignar_pm_terminal",
        "decision": "pm_asignado_por_usuario",
        "motivo": f"El usuario asignó la solicitud al PM {pm_name}.",
        "datos": {"pm_responsable": pm_name},
    }
    notification_event = {
        "hora_simulada": state["hora_actual"],
        "solicitud": request["id"],
        "nodo": "notificar_pm_simulado",
        "decision": "notificacion_simulada",
        "motivo": (
            f"Notificación simulada al PM {pm_name}; no se envió ningún correo real."
        ),
        "datos": {
            "pm_responsable": pm_name,
            "proyectos_abiertos": current_load + 1,
            "limite_proyectos_abiertos": agent_config.max_open_projects_per_pm,
            "hora_notificacion": state["hora_actual"],
            "estado": request["estado"],
            "datos_faltantes": request["datos_faltantes"],
            "preguntas_aclaracion": request["preguntas_aclaracion"],
        },
    }
    audit.append(assignment_event)
    audit.append(notification_event)
    state["eventos"].extend((assignment_event, notification_event))
    print(
        f"[ASIGNACIÓN] {request['asunto']} asignada a {pm_name}. "
        "Notificación simulada; no se envió correo real."
    )
    return state


def apply_pm_update(
    state: AgentState,
    *,
    request_number: int,
    response_text: str,
    audit: JsonlAuditLog,
    llm: OpenRouterClient | None,
    assignments: Mapping[str, str] | None = None,
    config: AgentConfig | None = None,
) -> AgentState:
    if not 1 <= request_number <= len(state["solicitudes"]):
        raise IndexError("El número de solicitud está fuera de rango.")
    if not response_text.strip():
        raise ValueError("La respuesta del PM no puede estar vacía.")

    request = state["solicitudes"][request_number - 1]
    if not request.get("pm_responsable"):
        raise ValueError("Asigna un PM antes de registrar su respuesta.")

    agent_config = config or load_agent_config()
    if llm is not None:
        limit = configured_max_llm_calls(agent_config.max_llm_calls)
        if limit and state["llamadas_llm"] >= limit:
            error = ValueError(
                f"Se alcanzó MAX_LLM_CALLS={limit}; no se pudo interpretar "
                "la respuesta del PM."
            )
            _record_update_error(state, request, audit, error)
            raise error
        state["llamadas_llm"] += 1
        try:
            updates = llm.extract_pm_update(response_text, request)
        except OpenRouterError as exc:
            _record_update_error(state, request, audit, exc)
            raise
    else:
        parsed = extract_request_details(
            {"text": response_text},
            assignments={},
        )
        updates = {
            key: parsed[key]
            for key in ("cliente", "rol", "cantidad_posiciones", "pm_responsable")
            if parsed[key] is not None
        }
        request_status = infer_request_status(response_text)
        if request_status:
            updates["estado_solicitud"] = request_status
        if not updates:
            error = ValueError(
                "Sin OpenRouter, escribe los datos como Cliente: ..., Rol: ..., "
                "Cantidad: ... o activa USE_LLM para responder en lenguaje natural."
            )
            _record_update_error(state, request, audit, error)
            raise error

    if not updates:
        error = ValueError(
            "No pude extraer datos nuevos de la respuesta del PM. "
            "Prueba con más contexto."
        )
        _record_update_error(state, request, audit, error)
        raise error

    before = {
        key: request.get(key)
        for key in (
            "cliente",
            "rol",
            "cantidad_posiciones",
            "posiciones",
            "pm_responsable",
            "estado",
        )
    }
    proposed_pm = updates.get("pm_responsable")
    if proposed_pm and normalize_text(str(proposed_pm)) != normalize_text(
        str(request.get("pm_responsable") or "")
    ):
        current_load = count_open_projects(
            state["solicitudes"],
            str(proposed_pm),
            exclude_request_id=request["id"],
        )
        if current_load >= agent_config.max_open_projects_per_pm:
            error = ValueError(
                f"El PM {proposed_pm} ya tiene {current_load} proyectos abiertos; "
                f"el límite es {agent_config.max_open_projects_per_pm}."
            )
            _record_update_error(state, request, audit, error)
            raise error
    for key in ("cliente", "rol", "cantidad_posiciones", "pm_responsable"):
        value = updates.get(key)
        if value is not None:
            request[key] = value
    reported_status = updates.get("estado_solicitud")
    if reported_status is not None:
        request_statuses = {
            "escalada": "Escalada",
            "confirmada": "Confirmada",
            "cerrada": "Cerrada",
        }
        status = request_statuses.get(str(reported_status))
        if status is None:
            error = ValueError(
                f"Estado de solicitud no reconocido: {reported_status!r}."
            )
            _record_update_error(state, request, audit, error)
            raise error
        request["estado"] = status
        request["fecha_cambio_estado"] = datetime.now(timezone.utc).isoformat()
        if status == "Escalada" and request.get(
            "nivel_maximo_escalamiento", "Ninguno"
        ) == "Ninguno":
            request["nivel_maximo_escalamiento"] = "PM"
        if status == "Confirmada":
            request["fecha_confirmacion"] = request["fecha_cambio_estado"]
            request["hora_confirmacion"] = state["hora_actual"]
        elif status == "Cerrada":
            request["fecha_cierre"] = request["fecha_cambio_estado"]
    if updates.get("posiciones"):
        request["posiciones"] = updates["posiciones"]
        if updates.get("cantidad_posiciones") is None:
            request["cantidad_posiciones"] = sum(
                item["cantidad_posiciones"] for item in updates["posiciones"]
            )
        if not updates.get("rol"):
            request["rol"] = " / ".join(
                dict.fromkeys(item["rol"] for item in updates["posiciones"])
            )

    pm_by_client = {
        normalize_text(str(client)): str(pm)
        for client, pm in (
            assignments if assignments is not None else load_assignments()
        ).items()
    }
    if not request.get("pm_responsable") and request.get("cliente"):
        request["pm_responsable"] = pm_by_client.get(
            normalize_text(request["cliente"])
        )

    request["datos_faltantes"] = [
        field
        for field, value in (
            ("cliente", request.get("cliente")),
            ("rol", request.get("rol")),
            ("cantidad_posiciones", request.get("cantidad_posiciones")),
            ("pm_responsable", request.get("pm_responsable")),
        )
        if value is None
    ]
    request["preguntas_aclaracion"] = clarification_questions(request)
    changed_fields = {
        key: request.get(key)
        for key, previous in before.items()
        if request.get(key) != previous
    }
    update_event = {
        "hora_simulada": state["hora_actual"],
        "solicitud": request["id"],
        "nodo": "respuesta_pm_terminal",
        "decision": "solicitud_actualizada_por_pm",
        "motivo": (
            updates.get("motivo")
            or "Se interpretó la respuesta del PM y se actualizó la solicitud."
        ),
        "datos": {
            "origen": "terminal",
            "campos_actualizados": changed_fields,
            "datos_faltantes": request["datos_faltantes"],
            "estado": request["estado"],
            "llamadas_llm": state["llamadas_llm"],
        },
    }
    request.setdefault("actualizaciones", []).append(
        {"origen": "terminal", "campos": changed_fields}
    )
    audit.append(update_event)
    state["eventos"].append(update_event)
    print(
        f"[RESPUESTA DEL PM] Se actualizaron {request['asunto']}. "
        + (
            "Falta completar: " + ", ".join(request["datos_faltantes"])
            if request["datos_faltantes"]
            else "La información requerida ya está completa."
        )
    )
    return state


def process_due_reminders(
    state: AgentState,
    *,
    audit: JsonlAuditLog,
    now: datetime | None = None,
    config: AgentConfig | None = None,
    directors: Mapping[str, str] | None = None,
    approval_handler: (
        Callable[[dict[str, Any]], dict[str, Any] | None] | None
    ) = None,
    approval_graph: Any | None = None,
) -> int:
    return process_due_sla_actions(
        state,
        audit=audit,
        config=config or load_agent_config(),
        directors=(
            directors
            if directors is not None
            else load_director_assignments()
        ),
        simulated=False,
        now=now,
        approval_handler=approval_handler,
        approval_graph=approval_graph,
    )


def _record_update_error(
    state: AgentState,
    request: dict[str, Any],
    audit: JsonlAuditLog,
    error: Exception,
) -> None:
    event = audit_error(
        audit,
        error,
        node="respuesta_pm_terminal",
        hora_simulada=state["hora_actual"],
        solicitud=request["id"],
    )
    state["eventos"].append(event)


def run_phase_one(
    client: AgentMailClient,
    *,
    audit_path: Path = AUDIT_PATH,
    state: AgentState | None = None,
    assignments: Mapping[str, str] | None = None,
    llm: OpenRouterClient | None = None,
    pm_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    simulation_seed: int = DEFAULT_SIMULATION_SEED,
    directors: Mapping[str, str] | None = None,
    config: AgentConfig | None = None,
) -> AgentState:
    """Ejecuta un sondeo completo del StateGraph; el servicio vuelve a invocarlo."""
    agent_config = config or load_agent_config()
    agent_config = replace(
        agent_config,
        max_llm_calls=configured_max_llm_calls(agent_config.max_llm_calls),
    )
    graph = build_graph(
        client,
        JsonlAuditLog(audit_path),
        assignments=assignments,
        directors=directors,
        llm=llm,
        pm_profiles=pm_profiles,
        simulation_seed=simulation_seed,
        config=agent_config,
    )
    return graph.invoke(
        state if state is not None else initial_agent_state(),
        config={"recursion_limit": agent_config.recursion_limit},
    )


def run_simulation(
    client: AgentMailClient,
    *,
    audit_path: Path = AUDIT_PATH,
    assignments: Mapping[str, str] | None = None,
    directors: Mapping[str, str] | None = None,
    llm: OpenRouterClient | None = None,
    pm_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    simulation_seed: int = DEFAULT_SIMULATION_SEED,
    config: AgentConfig | None = None,
) -> dict[str, Any]:
    agent_config = config or load_agent_config()
    agent_config = replace(
        agent_config,
        max_llm_calls=configured_max_llm_calls(agent_config.max_llm_calls),
    )
    audit = JsonlAuditLog(audit_path)
    graph = build_graph(
        client,
        audit,
        assignments=assignments,
        directors=directors,
        llm=llm,
        pm_profiles=pm_profiles,
        simulation_seed=simulation_seed,
        config=agent_config,
    )
    state = initial_agent_state()
    state["simular_confirmacion_automatica"] = True
    for tick in range(agent_config.max_simulated_hours):
        state["simulation_tick_only"] = tick > 0
        state = graph.invoke(
            state,
            config={"recursion_limit": agent_config.recursion_limit},
        )
        if all(
            request.get("estado") in SLA_STOP_STATES
            for request in state["solicitudes"]
        ):
            break

    requests_acknowledged = bool(state["solicitudes"]) and all(
        request.get("estado") in SLA_STOP_STATES
        for request in state["solicitudes"]
    )
    if requests_acknowledged:
        hours_simulated = max(0, state["hora_actual"] - 1)
    elif not state["solicitudes"]:
        hours_simulated = 0
    else:
        hours_simulated = state["hora_actual"]
    report = build_sla_report(
        state,
        config=agent_config,
        seed=simulation_seed,
        llm_enabled=llm is not None,
        hours_simulated=hours_simulated,
    )
    event = {
        "hora_simulada": report["horas_simuladas"],
        "solicitud": None,
        "nodo": "reporte_final_sla",
        "decision": "reporte_sla_generado",
        "motivo": (
            "La simulación terminó cuando cada PM confirmó que se hará cargo "
            "o la solicitud se cerró."
            if report["motivo_finalizacion"] == "todas_confirmadas_o_cerradas"
            else (
                "La simulación no encontró solicitudes clasificadas."
                if report["motivo_finalizacion"] == "sin_solicitudes"
                else "La simulación alcanzó el máximo de horas configurado."
            )
        ),
        "datos": {
            "motivo_finalizacion": report["motivo_finalizacion"],
            "horas_simuladas": report["horas_simuladas"],
            "total_solicitudes": report["resumen_sla"]["total_solicitudes"],
            "cumplieron_sla": report["resumen_sla"]["cumplieron"],
            "no_cumplieron_sla": report["resumen_sla"]["no_cumplieron"],
            "llamadas_llm": state["llamadas_llm"],
        },
    }
    audit.append(event)
    state["eventos"].append(event)
    return report


def run_agent_forever(
    client: AgentMailClient,
    *,
    audit_path: Path,
    stop_event: threading.Event,
    on_state: Callable[[AgentState], None],
    reconnect_interval: float = DEFAULT_RECONNECT_INTERVAL_SECONDS,
    heartbeat_interval: float = HEARTBEAT_INTERVAL_SECONDS,
    assignments: Mapping[str, str] | None = None,
    llm: OpenRouterClient | None = None,
    pm_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    simulation_seed: int = DEFAULT_SIMULATION_SEED,
    state_lock: Any | None = None,
    state_path: Path | None = None,
    initial_state: AgentState | None = None,
    directors: Mapping[str, str] | None = None,
    config: AgentConfig | None = None,
    approval_handler: (
        Callable[[dict[str, Any]], dict[str, Any] | None] | None
    ) = None,
) -> AgentState:
    """Load the existing inbox once, then process WebSocket events and deadlines."""
    if reconnect_interval <= 0:
        raise ValueError("reconnect_interval debe ser mayor que cero.")
    if heartbeat_interval <= 0:
        raise ValueError("heartbeat_interval debe ser mayor que cero.")

    agent_config = config or load_agent_config()
    agent_config = replace(
        agent_config,
        max_llm_calls=configured_max_llm_calls(agent_config.max_llm_calls),
    )
    graph = build_graph(
        client,
        JsonlAuditLog(audit_path),
        assignments=assignments,
        directors=directors,
        llm=llm,
        pm_profiles=pm_profiles,
        simulation_seed=simulation_seed,
        config=agent_config,
    )
    approval_graph = build_director_approval_graph()
    audit = JsonlAuditLog(audit_path)
    state = (
        initial_state
        if initial_state is not None
        else load_agent_state(state_path)
        if state_path is not None
        else initial_agent_state()
    )
    state["llamadas_llm"] = 0
    state["simular_confirmacion_automatica"] = False
    for request in state["solicitudes"]:
        if request.get("aprobacion_director_pendiente"):
            request["aprobacion_director_pendiente"] = False
            request["aprobacion_director_checkpoint_pendiente"] = False
            request.pop("aprobacion_director_thread_id", None)
    lock = state_lock or threading.RLock()
    next_heartbeat = time.monotonic() + heartbeat_interval
    reminder_interval = (
        agent_config.sla_check_interval_seconds
        if agent_config.testing_enabled
        else 60
    )
    next_reminder_check = time.monotonic() + reminder_interval

    async def receive_events(
        event_queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        async for event in client.message_events():
            await event_queue.put(event)

    def save_state_locked() -> None:
        if state_path is not None:
            save_agent_state(state_path, state)

    async def process_message(
        message: dict[str, Any],
        *,
        historical: bool,
        event_id: str | None = None,
    ) -> bool:
        nonlocal state
        identifier = get_message_id(message)
        if not identifier:
            audit_error(
                audit,
                "Se descartó un mensaje sin message_id.",
                node="leer_bandeja_agentmail",
                hora_simulada=state["hora_actual"],
                details={"event_id": event_id},
            )
            return False
        with lock:
            if identifier in state["correos_procesados"]:
                return False

        prefix = "CORREO ANTIGUO" if historical else "NUEVO CORREO"
        print(
            f"\n[{prefix}] De: {get_sender(message)} | "
            f"Asunto: {message.get('subject') or '(sin asunto)'}"
        )

        def invoke_graph() -> None:
            nonlocal state
            with lock:
                state = graph.invoke(
                    {
                        **state,
                        "correos_evento": [message],
                        "simular_confirmacion_automatica": False,
                    },
                    config={"recursion_limit": agent_config.recursion_limit},
                )
                state["simular_confirmacion_automatica"] = False
                save_state_locked()
                on_state(state)

        try:
            await asyncio.to_thread(invoke_graph)
            return True
        except Exception as exc:
            audit_error(
                audit,
                exc,
                node="procesar_correo_langgraph",
                hora_simulada=state["hora_actual"],
                solicitud=identifier,
                details={"event_id": event_id, "historico": historical},
            )
            print(
                f"\n[ERROR LangGraph] No se pudo procesar el correo "
                f"{identifier}: {type(exc).__name__}: {exc}"
            )
            return False

    def find_reclassifiable_messages() -> list[dict[str, Any]]:
        with lock:
            previous_classifications = list(state["correos_clasificados"])
            known_fingerprints = set(state["firmas_solicitudes"])
            known_request_ids = set(state["ids_solicitudes"])
        candidates: list[dict[str, Any]] = []
        for message in previous_classifications:
            if message.get("categoria") != "no_relacionado":
                continue
            category, _ = classify_by_rules(
                message,
                known_fingerprints=known_fingerprints,
                known_request_ids=known_request_ids,
            )
            identifier = get_message_id(message)
            if category != "solicitud_posicion" or not identifier:
                continue
            candidates.append(message)
            known_fingerprints.add(request_fingerprint(message))
            known_request_ids.add(identifier)
        return candidates

    async def reprocess_previously_missed_messages() -> None:
        for message in find_reclassifiable_messages():
            identifier = get_message_id(message)
            with lock:
                previous_classifications = [
                    item
                    for item in state["correos_clasificados"]
                    if get_message_id(item) == identifier
                ]
                was_processed = identifier in state["correos_procesados"]
                state["correos_clasificados"] = [
                    item
                    for item in state["correos_clasificados"]
                    if get_message_id(item) != identifier
                ]
                state["correos_procesados"] = [
                    item
                    for item in state["correos_procesados"]
                    if item != identifier
                ]
                migration_event = {
                    "hora_simulada": state["hora_actual"],
                    "solicitud": identifier,
                    "nodo": "recuperar_clasificacion_anterior",
                    "decision": "correo_no_relacionado_reevaluado",
                    "motivo": (
                        "Se reevaluó un correo antes descartado; la versión "
                        "actual de las reglas detecta una solicitud de personal."
                    ),
                    "datos": {
                        "categoria_anterior": "no_relacionado",
                        "categoria_nueva": "solicitud_posicion",
                    },
                }
                audit.append(migration_event)
                state["eventos"].append(migration_event)
            if await process_message(message, historical=True):
                continue
            with lock:
                if not any(
                    get_message_id(item) == identifier
                    for item in state["correos_clasificados"]
                ):
                    state["correos_clasificados"].extend(
                        previous_classifications
                    )
                if was_processed and identifier not in state["correos_procesados"]:
                    state["correos_procesados"].append(identifier)
                state["eventos"] = [
                    event
                    for event in state["eventos"]
                    if event is not migration_event
                ]

    async def synchronize_existing_messages() -> None:
        summaries = await asyncio.to_thread(client.list_messages)
        unique_summaries: dict[str, dict[str, Any]] = {}
        for summary in summaries:
            identifier = get_message_id(summary)
            if not identifier:
                audit_error(
                    audit,
                    "AgentMail devolvió un correo antiguo sin message_id.",
                    node="leer_bandeja_agentmail",
                    hora_simulada=state["hora_actual"],
                )
                continue
            with lock:
                already_processed = identifier in state["correos_procesados"]
            if not already_processed:
                unique_summaries[identifier] = summary

        pending: list[dict[str, Any]] = []
        for identifier in unique_summaries:
            try:
                message = await asyncio.to_thread(client.get_message, identifier)
            except AgentMailError as exc:
                audit_error(
                    audit,
                    exc,
                    node="leer_correo_historico",
                    hora_simulada=state["hora_actual"],
                    solicitud=identifier,
                )
                print(
                    f"\n[ERROR AgentMail] No se pudo recuperar el correo "
                    f"histórico {identifier}: {exc}"
                )
                continue
            message.setdefault("message_id", identifier)
            pending.append(message)

        for message in _ordered_by_timestamp(pending):
            if stop_event.is_set():
                return
            await process_message(message, historical=True)

    def check_reminders() -> int:
        with lock:
            def wait_for_approval(
                payload: dict[str, Any],
            ) -> dict[str, Any] | None:
                if approval_handler is None:
                    raise RuntimeError(
                        "No se configuró el canal de aprobación humana."
                    )
                lock.release()
                try:
                    return approval_handler(payload)
                finally:
                    lock.acquire()

            previous_event_count = len(state["eventos"])
            sent = process_due_reminders(
                state,
                audit=audit,
                config=agent_config,
                directors=directors,
                approval_handler=(
                    wait_for_approval if approval_handler is not None else None
                ),
                approval_graph=approval_graph,
            )
            if sent or len(state["eventos"]) != previous_event_count:
                save_state_locked()
                on_state(state)
            return sent

    async def run_listener() -> AgentState:
        nonlocal state, next_heartbeat, next_reminder_check
        last_error: str | None = None

        while not stop_event.is_set():
            event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            listener = asyncio.create_task(receive_events(event_queue))
            try:
                while not stop_event.is_set():
                    timeout = min(
                        1.0,
                        max(0.1, next_heartbeat - time.monotonic()),
                    )
                    try:
                        event = await asyncio.wait_for(
                            event_queue.get(),
                            timeout=timeout,
                        )
                    except asyncio.TimeoutError:
                        if listener.done() and event_queue.empty():
                            listener.result()
                            raise AgentMailError(
                                "La conexión WebSocket de AgentMail se cerró."
                            )
                        now = time.monotonic()
                        if now >= next_heartbeat:
                            print(".", end="", flush=True)
                            next_heartbeat = now + heartbeat_interval
                        if now >= next_reminder_check:
                            await asyncio.to_thread(check_reminders)
                            next_reminder_check = now + reminder_interval
                        continue

                    if event.get("type") == "subscribed":
                        print(
                            f"\n[AgentMail] Escuchando nuevos correos de "
                            f"{client.inbox_id} por WebSocket."
                        )
                        last_error = None
                        try:
                            await reprocess_previously_missed_messages()
                            await synchronize_existing_messages()
                        except AgentMailError as exc:
                            audit_error(
                                audit,
                                exc,
                                node="sincronizar_bandeja_historica",
                                hora_simulada=state["hora_actual"],
                            )
                            print(
                                f"\n[ERROR AgentMail] No se pudo sincronizar "
                                f"la bandeja histórica: {exc}"
                            )
                        await asyncio.to_thread(check_reminders)
                        next_reminder_check = time.monotonic() + reminder_interval
                        continue
                    if event.get("type") != "message_received":
                        print(
                            "\n[AgentMail] Se ignoró un evento WebSocket "
                            "con formato inesperado."
                        )
                        continue

                    message = event.get("message")
                    if not isinstance(message, dict):
                        audit_error(
                            audit,
                            "Se recibió un evento sin mensaje válido.",
                            node="leer_evento_agentmail",
                            hora_simulada=state["hora_actual"],
                            details={"event_id": event.get("event_id")},
                        )
                        print("\n[ERROR AgentMail] Evento sin mensaje válido.")
                        continue
                    await process_message(
                        message,
                        historical=False,
                        event_id=str(event.get("event_id") or ""),
                    )

                    now = time.monotonic()
                    if now >= next_heartbeat:
                        print(".", end="", flush=True)
                        next_heartbeat = now + heartbeat_interval
                    if now >= next_reminder_check:
                        await asyncio.to_thread(check_reminders)
                        next_reminder_check = now + reminder_interval
            except AgentMailError as exc:
                if stop_event.is_set():
                    break
                error = str(exc)
                if error != last_error:
                    audit_error(
                        audit,
                        exc,
                        node="escuchar_agentmail",
                        hora_simulada=state["hora_actual"],
                    )
                    print(f"\n[ERROR AgentMail] {error} Se reintentará.")
                    last_error = error
            except Exception as exc:
                if stop_event.is_set():
                    break
                audit_error(
                    audit,
                    exc,
                    node="escuchar_agentmail",
                    hora_simulada=state["hora_actual"],
                )
                error = f"{type(exc).__name__}: {exc}"
                if error != last_error:
                    print(f"\n[ERROR WebSocket] {error} Se reintentará.")
                    last_error = error
            finally:
                if not listener.done():
                    listener.cancel()
                try:
                    await listener
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    if not stop_event.is_set():
                        audit_error(
                            audit,
                            exc,
                            node="cerrar_escucha_agentmail",
                            hora_simulada=state["hora_actual"],
                        )
                        print(
                            f"\n[ERROR AgentMail] Falló el cierre de la escucha: "
                            f"{type(exc).__name__}: {exc}"
                        )

            if not stop_event.is_set():
                reconnect_deadline = time.monotonic() + reconnect_interval
                while (
                    not stop_event.is_set()
                    and time.monotonic() < reconnect_deadline
                ):
                    now = time.monotonic()
                    if now >= next_heartbeat:
                        print(".", end="", flush=True)
                        next_heartbeat = now + heartbeat_interval
                    await asyncio.sleep(
                        min(0.25, reconnect_deadline - time.monotonic())
                    )

        return state

    return asyncio.run(run_listener())
