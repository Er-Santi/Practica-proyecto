from __future__ import annotations

import asyncio
import re
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping

from agent_config import AgentConfig, load_agent_config
from agent_state import AgentState, initial_agent_state
from agentmail_client import AgentMailClient
from audit_log import JsonlAuditLog, audit_error
from email_utils import (
    _ordered_by_timestamp, associate_pm_email_addresses,
    clarification_questions, classify_by_rules, confirmation_question,
    detect_pm_unavailability, extract_request_details, get_message_id,
    get_sender, get_sender_address, infer_request_status,
    load_assignments, load_director_assignments, message_text,
    normalize_text, request_fingerprint, simulated_confirmation_delay,
)
from exceptions import AgentMailError, OpenRouterError
from email_notifications import (
    send_case_query_reply,
    send_lifecycle_notification,
)
from openrouter_client import OpenRouterClient
from pm_assignment import (
    build_pm_roster,
    canonical_pm_name,
    count_open_projects,
    is_pm_eligible,
    save_pm_eligibility,
    select_least_loaded_pm,
)
from settings import (
    AUDIT_PATH,
    DEFAULT_RECONNECT_INTERVAL_SECONDS, DEFAULT_SIMULATION_SEED,
    HEARTBEAT_INTERVAL_SECONDS, PM_PROFILES_PATH, configured_max_llm_calls,
)
from reporting import build_sla_report
from sla import (
    SLA_STOP_STATES,
    build_director_approval_graph,
    process_due_sla_actions,
)
from state_store import load_agent_state, save_agent_state
from workflow import build_graph


def _subject_identifies_request(subject: str, request_id: str) -> bool:
    normalized_subject = normalize_text(subject)
    normalized_id = normalize_text(request_id)
    return bool(
        normalized_id
        and re.search(
            rf"(?<!\w){re.escape(normalized_id)}(?!\w)",
            normalized_subject,
        )
    )


def _supersede_pm_assignment_emails(request: dict[str, Any]) -> None:
    request.pop("notificacion_pm_message_id", None)
    for notification in request.get("notificaciones_email", []):
        if (
            isinstance(notification, MutableMapping)
            and notification.get("tipo")
            in {"solicitud_asignada", "solicitud_reasignada"}
        ):
            notification["superseded"] = True


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
    reason: str,
    audit: JsonlAuditLog,
) -> AgentState:
    normalized_reason = reason.strip()
    if not normalized_reason:
        raise ValueError("Debes indicar la razón para cerrar la solicitud.")
    if not 1 <= request_number <= len(state["solicitudes"]):
        raise IndexError("El número de solicitud está fuera de rango.")
    request = state["solicitudes"][request_number - 1]
    if request["estado"] == "Cerrada":
        raise ValueError("La solicitud ya está finalizada.")
    request["estado"] = "Cerrada"
    request["fecha_cierre"] = datetime.now(timezone.utc).isoformat()
    request["razon_finalizacion"] = normalized_reason
    event = {
        "hora_simulada": state["hora_actual"],
        "solicitud": request["id"],
        "nodo": "cerrar_solicitud_terminal",
        "decision": "solicitud_cerrada",
        "motivo": normalized_reason,
        "datos": {"razon_finalizacion": normalized_reason},
    }
    audit.append(event)
    state["eventos"].append(event)
    return state


def complete_request(
    state: AgentState,
    *,
    request_number: int,
    reason: str,
    audit: JsonlAuditLog,
) -> AgentState:
    normalized_reason = reason.strip()
    if not normalized_reason:
        raise ValueError("Debes indicar la razón para completar la solicitud.")
    if not 1 <= request_number <= len(state["solicitudes"]):
        raise IndexError("El número de solicitud está fuera de rango.")
    request = state["solicitudes"][request_number - 1]
    if request["estado"] == "Cerrada":
        raise ValueError("La solicitud ya está finalizada.")
    completed_at = datetime.now(timezone.utc).isoformat()
    request["estado"] = "Cerrada"
    request["fecha_cierre"] = completed_at
    request["fecha_completada"] = completed_at
    request["razon_finalizacion"] = normalized_reason
    event = {
        "hora_simulada": state["hora_actual"],
        "solicitud": request["id"],
        "nodo": "completar_solicitud_terminal",
        "decision": "solicitud_completada",
        "motivo": normalized_reason,
        "datos": {
            "pm_responsable": request.get("pm_responsable"),
            "fecha_completada": completed_at,
            "razon_finalizacion": normalized_reason,
        },
    }
    audit.append(event)
    state["eventos"].append(event)
    print(f"[COMPLETADA] {request['asunto']}.")
    return state


def set_pm_eligibility(
    state: AgentState,
    *,
    pm_name: str,
    eligible: bool,
    pm_profiles: MutableMapping[str, dict[str, Any]],
    profiles_path: Path = PM_PROFILES_PATH,
    assignments: Mapping[str, str] | None = None,
    audit: JsonlAuditLog,
    simulation_seed: int = DEFAULT_SIMULATION_SEED,
    config: AgentConfig | None = None,
    reassign_open_requests: bool = True,
) -> AgentState:
    canonical_name = canonical_pm_name(pm_name, pm_profiles)
    normalized_pm = normalize_text(canonical_name)

    def profile_matches(candidate: str) -> bool:
        return normalize_text(canonical_pm_name(candidate, pm_profiles)) == normalized_pm

    exact_profile_keys = [
        key
        for key, profile in pm_profiles.items()
        if normalize_text(str(key)) == normalized_pm
        or normalize_text(str(profile.get("nombre") or "")) == normalized_pm
    ]
    profile_matches_by_alias = [
        key
        for key, profile in pm_profiles.items()
        if profile_matches(str(key))
        or profile_matches(str(profile.get("nombre") or ""))
    ]
    matching_keys = exact_profile_keys or list(dict.fromkeys(profile_matches_by_alias))
    if len(matching_keys) > 1:
        raise ValueError(
            f"El nombre {pm_name!r} coincide con varios PM; escribe el nombre completo."
        )
    profile_key = matching_keys[0] if matching_keys else None
    profile_data: dict[str, Any] | None = None
    if profile_key is None:
        configured_pm = next(
            (
                str(name).strip()
                for name in (assignments or {}).values()
                if profile_matches(str(name))
            ),
            None,
        )
        if configured_pm is None:
            raise ValueError(
                f"No existe el PM {pm_name!r} en el roster ni en asignacion.csv."
            )
        profile_key = normalized_pm
        profile_data = {
            "nombre": configured_pm,
            "perfil": "normal",
            "horas_confirmacion": None,
            "elegible": True,
        }
    profile = (
        profile_data
        if profile_data is not None
        else pm_profiles[profile_key]
    )
    current_eligibility = profile.get("elegible", True) is not False
    already_in_requested_state = current_eligibility == eligible
    if already_in_requested_state and eligible:
        print(
            f"[PM] {profile.get('nombre', pm_name)} ya está "
            f"{'elegible' if eligible else 'no elegible'}."
        )
        return state

    display_name = str(profile.get("nombre") or pm_name).strip()
    agent_config = config or load_agent_config()
    if not already_in_requested_state:
        save_pm_eligibility(
            profiles_path,
            pm_name=display_name,
            eligible=eligible,
            pm_aliases=profile.get("emails", []),
        )
        if profile_data is not None:
            pm_profiles[profile_key] = profile_data
        profile["elegible"] = eligible
        eligibility_event = {
            "hora_simulada": state["hora_actual"],
            "solicitud": None,
            "nodo": "cambiar_elegibilidad_pm",
            "decision": "pm_elegibilidad_actualizada",
            "motivo": (
                f"El PM {display_name} ahora está "
                f"{'elegible' if eligible else 'no elegible'} para recibir solicitudes."
            ),
            "datos": {
                "pm_responsable": display_name,
                "elegible": eligible,
                "archivo_perfiles": str(profiles_path),
            },
        }
        audit.append(eligibility_event)
        state["eventos"].append(eligibility_event)
    else:
        print(
            f"[PM] {display_name} ya está no elegible; "
            "se revisarán sus solicitudes activas."
        )

    reassigned = 0
    if not eligible and reassign_open_requests:
        pm_roster = build_pm_roster(assignments, pm_profiles)
        for request_index, request in enumerate(state["solicitudes"], start=1):
            if (
                request.get("estado") == "Cerrada"
                or not profile_matches(
                    str(request.get("pm_responsable") or "")
                )
            ):
                continue

            old_pm = str(request["pm_responsable"])
            replacement, loads = select_least_loaded_pm(
                state["solicitudes"],
                pm_roster,
                maximum_open_projects=agent_config.max_open_projects_per_pm,
                pm_profiles=pm_profiles,
            )
            if replacement is None:
                _supersede_pm_assignment_emails(request)
                request["pm_responsable"] = None
                request["estado"] = "Nueva"
                request["hora_notificacion"] = None
                request["fecha_notificacion"] = None
                request["confirmacion_usuario"] = None
                request["confirmacion_simulada"] = False
                request["hora_confirmacion"] = None
                request["escalado_director"] = False
                request["recordatorio_enviado"] = False
                request["segundo_recordatorio_enviado"] = False
                request["recordatorio_siguiente_nivel_enviado"] = False
                request["comercial_delivery_involucrados"] = False
                request["nivel_maximo_escalamiento"] = "Ninguno"
                request["aprobacion_director_pendiente"] = False
                request.pop("aprobacion_director_thread_id", None)
                request.pop("fecha_confirmacion", None)
                request["datos_faltantes"] = list(
                    dict.fromkeys(
                        [*(request.get("datos_faltantes") or []), "pm_responsable"]
                    )
                )
                request["preguntas_aclaracion"] = clarification_questions(request)
                event = {
                    "hora_simulada": state["hora_actual"],
                    "solicitud": request["id"],
                    "nodo": "reasignar_solicitudes_pm_no_elegible",
                    "decision": "solicitud_sin_pm_elegible",
                    "motivo": (
                        f"{old_pm} fue marcado no elegible y no hay otro PM "
                        "elegible con capacidad disponible."
                    ),
                    "datos": {
                        "pm_anterior": old_pm,
                        "cargas_abiertas": loads,
                        "limite_proyectos_abiertos": (
                            agent_config.max_open_projects_per_pm
                        ),
                    },
                }
                audit.append(event)
                state["eventos"].append(event)
                continue

            assign_pm_to_request(
                state,
                request_number=request_index,
                pm_name=replacement,
                audit=audit,
                pm_profiles=pm_profiles,
                simulation_seed=simulation_seed,
                config=agent_config,
            )
            request["estado"] = "Notificada"
            request["confirmacion_usuario"] = None
            request["confirmacion_simulada"] = False
            request["hora_confirmacion"] = None
            request.pop("fecha_confirmacion", None)
            for field in (
                "recordatorio_enviado",
                "segundo_recordatorio_enviado",
                "escalado_director",
                "recordatorio_siguiente_nivel_enviado",
                "comercial_delivery_involucrados",
                "aprobacion_director_pendiente",
                "aprobacion_director_checkpoint_pendiente",
            ):
                if field in request:
                    request[field] = False
            request.pop("aprobacion_director_thread_id", None)
            request["nivel_maximo_escalamiento"] = "Ninguno"
            event = {
                "hora_simulada": state["hora_actual"],
                "solicitud": request["id"],
                "nodo": "reasignar_solicitudes_pm_no_elegible",
                "decision": "solicitud_reasignada",
                "motivo": (
                    f"Se reasignó la solicitud de {old_pm} a {replacement} "
                    "porque el PM anterior dejó de ser elegible."
                ),
                "datos": {
                    "pm_anterior": old_pm,
                    "pm_nuevo": replacement,
                    "estado": request["estado"],
                },
            }
            audit.append(event)
            state["eventos"].append(event)
            reassigned += 1

    print(
        f"[PM] {display_name}: "
        f"{'elegible' if eligible else 'no elegible'}."
        + (
            f" Solicitudes activas reasignadas: {reassigned}."
            if not eligible
            else ""
        )
    )
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
    if not is_pm_eligible(pm_name, pm_profiles):
        raise ValueError(f"El PM {pm_name} está marcado como no elegible.")
    agent_config = config or load_agent_config()
    current_load = count_open_projects(
        state["solicitudes"],
        pm_name,
        exclude_request_id=request["id"],
        pm_profiles=pm_profiles,
    )
    if current_load >= agent_config.max_open_projects_per_pm:
        raise ValueError(
            f"El PM {pm_name} ya tiene {current_load} proyectos abiertos; "
            f"el límite es {agent_config.max_open_projects_per_pm}."
        )
    _supersede_pm_assignment_emails(request)
    request["pm_responsable"] = pm_name
    request["hora_notificacion"] = state["hora_actual"]
    request["fecha_notificacion"] = datetime.now(timezone.utc).isoformat()
    request["recordatorio_enviado"] = False
    request["segundo_recordatorio_enviado"] = False
    request["escalado_director"] = False
    request["recordatorio_siguiente_nivel_enviado"] = False
    request["comercial_delivery_involucrados"] = False
    request["aprobacion_director_pendiente"] = False
    request["aprobacion_director_checkpoint_pendiente"] = False
    request["nivel_maximo_escalamiento"] = "Ninguno"
    for field in (
        "hora_recordatorio",
        "fecha_recordatorio",
        "hora_segundo_recordatorio",
        "fecha_segundo_recordatorio",
        "hora_escalamiento_director",
        "fecha_escalamiento_director",
        "hora_recordatorio_siguiente_nivel",
        "fecha_recordatorio_siguiente_nivel",
        "hora_ultimo_recordatorio_sla",
        "fecha_ultimo_recordatorio_sla",
        "aprobacion_director_thread_id",
    ):
        request.pop(field, None)
    request["hora_confirmacion"] = None
    request["confirmacion_usuario"] = None
    request["confirmacion_simulada"] = False
    request.pop("fecha_confirmacion", None)
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
    review_fields = [
        field
        for field in ("rol", "cantidad_posiciones")
        if field in request["datos_faltantes"]
    ]
    was_under_human_review = request.get("revision_humana", False)
    request["revision_humana"] = bool(review_fields)
    request["campos_revision_humana"] = review_fields
    if was_under_human_review and not review_fields:
        resolved_at = datetime.now(timezone.utc).isoformat()
        request["fecha_revision_humana_resuelta"] = resolved_at
        if request.get("estado") == "Nueva":
            request["estado"] = "Notificada"
            request["hora_notificacion"] = state["hora_actual"]
            request["fecha_notificacion"] = resolved_at
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
    pm_profiles: Mapping[str, Mapping[str, Any]] | None = None,
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
    if llm is not None and agent_config.llm_enabled:
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
            pm_profiles=pm_profiles,
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
            "completada": "Cerrada",
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
            request["aprobacion_director_pendiente"] = False
            request["aprobacion_director_checkpoint_pendiente"] = False
            request["escalamiento_ausencia_pendiente"] = False
            request.pop("aprobacion_director_thread_id", None)
        elif status == "Cerrada":
            request["fecha_cierre"] = request["fecha_cambio_estado"]
            request["razon_finalizacion"] = response_text.strip()
            if reported_status == "completada":
                request["fecha_completada"] = request["fecha_cambio_estado"]
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
    review_fields = [
        field
        for field in ("rol", "cantidad_posiciones")
        if field in request["datos_faltantes"]
    ]
    was_under_human_review = request.get("revision_humana", False)
    request["revision_humana"] = bool(review_fields)
    request["campos_revision_humana"] = review_fields
    if was_under_human_review and not review_fields:
        resolved_at = datetime.now(timezone.utc).isoformat()
        request["fecha_revision_humana_resuelta"] = resolved_at
        if request.get("estado") == "Nueva":
            request["estado"] = "Notificada"
            request["hora_notificacion"] = state["hora_actual"]
            request["fecha_notificacion"] = resolved_at
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
    pm_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    mail_client: AgentMailClient | None = None,
    llm: OpenRouterClient | None = None,
    approval_handler: (
        Callable[[dict[str, Any]], dict[str, Any] | None] | None
    ) = None,
    approval_graph: Any | None = None,
) -> int:
    agent_config = config or load_agent_config()
    loaded_profiles = pm_profiles or {}

    def profile_email(pm_name: str) -> str | None:
        profiles = loaded_profiles or {}
        normalized_pm = normalize_text(pm_name)
        profile = next(
            (
                item
                for key, item in profiles.items()
                if normalized_pm
                in {
                    normalize_text(str(key)),
                    normalize_text(str(item.get("nombre") or "")),
                    *(
                        normalize_text(str(address))
                        for address in item.get("emails", [])
                    ),
                }
            ),
            None,
        )
        if profile is None:
            return None
        addresses = profile.get("emails") or []
        if isinstance(addresses, str):
            addresses = [addresses]
        return next(
            (
                str(address).strip().casefold()
                for address in addresses
                if isinstance(address, str)
                and "@" in address
            ),
            None,
        )

    def send_sla_notification(
        request: dict[str, Any],
        notification_type: str,
        reason: str,
        details: dict[str, Any],
        recipient: str,
    ) -> dict[str, Any]:
        if mail_client is None:
            raise AgentMailError(
                "No se configuró el cliente para enviar notificaciones por correo."
            )
        if notification_type == "Escalamiento aprobado al director":
            recipient_email = str(request.get("correo_solicitante") or "").strip()
            event_type = "solicitud_escalada"
        elif (
            notification_type == "Recordatorio al director"
            or "escalamiento" in normalize_text(notification_type)
            or request.get("escalada_a_jefe")
        ):
            recipient_email = agent_config.manager_email
            event_type = (
                "escalamiento_jefe"
                if "escalamiento" in normalize_text(notification_type)
                else "recordatorio"
            )
        else:
            recipient_email = profile_email(str(request.get("pm_responsable") or ""))
            event_type = "recordatorio"
        if not recipient_email:
            raise ValueError(
                f"No hay correo configurado para el destinatario {recipient!r}."
            )
        email_details = send_lifecycle_notification(
            mail_client,
            recipient_email,
            event_type,
            request,
            context={
                "razon": reason,
                "tipo_notificacion": notification_type,
                **details,
            },
            state=state,
            audit=audit,
            config=agent_config,
            llm=llm,
        )
        if notification_type == "Primer recordatorio al PM":
            request["correo_recordatorio_real_enviado"] = True
        email_details["destinatario_logico"] = recipient
        return email_details

    def send_approval_email(payload: dict[str, Any]) -> dict[str, Any]:
        if mail_client is None:
            raise AgentMailError(
                "No se configuró el cliente para enviar el correo de escalamiento."
            )
        if not agent_config.manager_email:
            raise ValueError("No hay correo configurado para el jefe.")
        request = next(
            (
                item
                for item in state["solicitudes"]
                if item.get("id") == payload.get("solicitud")
            ),
            None,
        )
        if request is None:
            raise ValueError(
                f"No existe la solicitud {payload.get('solicitud')}."
            )
        return send_lifecycle_notification(
            mail_client,
            agent_config.manager_email,
            "escalamiento_jefe",
            request,
            context={
                "razon": request.get("motivo_escalamiento")
                or "El PM no confirmó dentro del plazo SLA.",
                "jefe": agent_config.manager_name,
                "pregunta": payload.get("mensaje"),
                "intento": payload.get("intento"),
            },
            state=state,
            audit=audit,
            config=agent_config,
            llm=llm,
        )

    return process_due_sla_actions(
        state,
        audit=audit,
        config=agent_config,
        directors=(
            directors
            if directors is not None
            else load_director_assignments()
        ),
        simulated=False,
        now=now,
        approval_handler=approval_handler,
        approval_graph=approval_graph,
        notification_email_handler=(
            send_sla_notification
            if mail_client is not None and agent_config.reminder_email_enabled
            else None
        ),
        approval_email_handler=(
            send_approval_email
            if mail_client is not None and agent_config.reminder_email_enabled
            else None
        ),
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
    pm_profiles: MutableMapping[str, dict[str, Any]] | None = None,
    simulation_seed: int = DEFAULT_SIMULATION_SEED,
    state_lock: Any | None = None,
    state_path: Path | None = None,
    initial_state: AgentState | None = None,
    directors: Mapping[str, str] | None = None,
    config: AgentConfig | None = None,
    profiles_path: Path = PM_PROFILES_PATH,
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
    if pm_profiles is not None:
        associate_pm_email_addresses(
            pm_profiles,
            (assignments or {}).values(),
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

    def migrate_active_requests_to_roster() -> bool:
        roster = build_pm_roster(assignments, pm_profiles)
        if not roster:
            return False
        changed = False
        for request in state["solicitudes"]:
            if (
                request.get("estado") == "Cerrada"
                or request.get("escalada_a_jefe")
                or request.get("jefe_responsable")
            ):
                continue
            previous_pm = str(request.get("pm_responsable") or "")
            if not previous_pm:
                continue
            canonical_previous = normalize_text(
                canonical_pm_name(previous_pm, pm_profiles)
            )
            roster_names = {
                normalize_text(canonical_pm_name(name, pm_profiles))
                for name in roster
            }
            if canonical_previous in roster_names:
                continue
            replacement, loads = select_least_loaded_pm(
                state["solicitudes"],
                roster,
                maximum_open_projects=agent_config.max_open_projects_per_pm,
                pm_profiles=pm_profiles,
            )
            request["pm_anterior"] = previous_pm
            if replacement is None:
                _supersede_pm_assignment_emails(request)
                request["pm_responsable"] = None
                request["responsable_actual"] = None
                request["estado"] = "Nueva"
                if "pm_responsable" not in request.get("datos_faltantes", []):
                    request.setdefault("datos_faltantes", []).append(
                        "pm_responsable"
                    )
                decision = "solicitud_sin_pm_despues_de_migracion"
                reason = (
                    "El PM anterior ya no forma parte del roster y no hay PM "
                    "elegible con capacidad."
                )
            else:
                _supersede_pm_assignment_emails(request)
                request["pm_responsable"] = replacement
                request["responsable_actual"] = replacement
                request["jefe_responsable"] = None
                request["escalada_a_jefe"] = False
                request["estado"] = "Notificada"
                request["hora_notificacion"] = state["hora_actual"]
                request["fecha_notificacion"] = datetime.now(
                    timezone.utc
                ).isoformat()
                for field in (
                    "recordatorio_enviado",
                    "segundo_recordatorio_enviado",
                    "primer_recordatorio_enviado",
                    "escalado_director",
                    "recordatorio_siguiente_nivel_enviado",
                    "comercial_delivery_involucrados",
                ):
                    request[field] = False
                request["nivel_maximo_escalamiento"] = "Ninguno"
                decision = "solicitud_reasignada_por_cambio_de_roster"
                reason = (
                    f"Se reasignó de {previous_pm} a {replacement} porque "
                    "el PM anterior salió del roster."
                )
            event = {
                "hora_simulada": state["hora_actual"],
                "solicitud": request["id"],
                "nodo": "migrar_roster_pm",
                "decision": decision,
                "motivo": reason,
                "datos": {
                    "pm_anterior": previous_pm,
                    "pm_nuevo": replacement,
                    "cargas_abiertas": loads,
                },
            }
            audit.append(event)
            state["eventos"].append(event)
            changed = True
            if replacement:
                profile = next(
                    (
                        item
                        for item in (pm_profiles or {}).values()
                        if normalize_text(str(item.get("nombre") or ""))
                        == normalize_text(replacement)
                    ),
                    {},
                )
                addresses = profile.get("emails") or []
                if isinstance(addresses, str):
                    addresses = [addresses]
                recipient = next(
                    (
                        str(address).strip().casefold()
                        for address in addresses
                        if isinstance(address, str) and "@" in address
                    ),
                    None,
                )
                if recipient:
                    try:
                        details = send_lifecycle_notification(
                            client,
                            recipient,
                            "solicitud_reasignada",
                            request,
                            context={"razon": reason},
                            state=state,
                            audit=audit,
                            config=agent_config,
                            llm=llm,
                        )
                    except (AgentMailError, ValueError) as exc:
                        if isinstance(exc, ValueError):
                            error = audit_error(
                                audit,
                                exc,
                                node="notificar_migracion_roster",
                                hora_simulada=state["hora_actual"],
                                solicitud=request["id"],
                            )
                            state["eventos"].append(error)
                        print(
                            f"\n[ERROR CORREO MIGRACIÓN] {exc}"
                        )
                    else:
                        request["notificacion_pm_message_id"] = (
                            details.get("message_id")
                        )
                        request.setdefault("notificaciones_email", []).append(
                            {"tipo": "solicitud_reasignada", **details}
                        )
        return changed

    def absence_already_recorded(identifier: str) -> bool:
        return any(
            event.get("decision") == "pm_marcado_no_elegible_por_ausencia"
            and (event.get("datos") or {}).get("message_id") == identifier
            for event in state["eventos"]
        )

    def process_pm_absence(
        message: dict[str, Any],
        availability: dict[str, str],
    ) -> bool:
        identifier = get_message_id(message)
        try:
            with lock:
                if absence_already_recorded(identifier):
                    return True
                absent_email = availability["correo_pm"].casefold()
                matching_pm_names = [
                    str(profile.get("nombre") or key)
                    for key, profile in (pm_profiles or {}).items()
                    if absent_email
                    in {
                        str(address).strip().casefold()
                        for address in (profile.get("emails") or [])
                    }
                ] or [availability["pm_responsable"]]
                references = message_references(message)
                linked_requests = [
                    request
                    for request in state["solicitudes"]
                    if request.get("estado") != "Cerrada"
                    and (
                        bool(
                            references
                            & {
                                str(
                                    request.get("notificacion_pm_message_id")
                                    or ""
                                )
                                .strip()
                                .strip("<>"),
                                str(request.get("id") or "")
                                .strip()
                                .strip("<>"),
                                *(
                                    str(item.get("message_id") or "")
                                    .strip()
                                    .strip("<>")
                                    for item in request.get(
                                        "notificaciones_email", []
                                    )
                                    if isinstance(item, Mapping)
                                    and not item.get("superseded")
                                    and item.get("tipo")
                                    in {
                                        "solicitud_asignada",
                                        "solicitud_reasignada",
                                    }
                                ),
                            }
                            - {""}
                        )
                        or (
                            bool(request.get("id"))
                            and _subject_identifies_request(
                                str(message.get("subject") or ""),
                                str(request["id"]),
                            )
                        )
                    )
                ]
                linked_pm_names = list(
                    dict.fromkeys(
                        str(request.get("pm_responsable") or "").strip()
                        for request in linked_requests
                        if request.get("pm_responsable")
                    )
                )
                if linked_pm_names:
                    absent_pm_names = linked_pm_names
                elif len(matching_pm_names) == 1:
                    absent_pm_names = matching_pm_names
                else:
                    event = {
                        "hora_simulada": state["hora_actual"],
                        "solicitud": None,
                        "nodo": "detectar_ausencia_pm",
                        "decision": "respuesta_ausencia_pm_ambigua",
                        "motivo": (
                            "El correo indica ausencia, pero el remitente "
                            "comparte dirección con varios PM y no se pudo "
                            "identificar la solicitud relacionada."
                        ),
                        "datos": {
                            "pm_candidatos": matching_pm_names,
                            "correo_pm": absent_email,
                            "message_id": identifier,
                        },
                    }
                    audit.append(event)
                    state["eventos"].append(event)
                    state["correos_procesados"].append(identifier)
                    state["correos_clasificados"].append(
                        {
                            **message,
                            "categoria": "respuesta_ausencia_pm_ambigua",
                        }
                    )
                    save_state_locked()
                    on_state(state)
                    print(
                        "[REVISIÓN HUMANA] Se detectó una ausencia, pero no "
                        "se deshabilitó ningún PM porque el correo es compartido "
                        "y no se identificó la solicitud asociada."
                    )
                    return True
                absent_pm_keys = {
                    normalize_text(canonical_pm_name(name, pm_profiles))
                    for name in absent_pm_names
                }
                affected_requests = [
                    request
                    for request in state["solicitudes"]
                    if request.get("estado") != "Cerrada"
                    and normalize_text(
                        canonical_pm_name(
                            str(request.get("pm_responsable") or ""),
                            pm_profiles,
                        )
                    )
                    in absent_pm_keys
                ]
                for pm_name in absent_pm_names:
                    set_pm_eligibility(
                        state,
                        pm_name=pm_name,
                        eligible=False,
                        pm_profiles=pm_profiles or {},
                        profiles_path=profiles_path,
                        assignments=assignments,
                        audit=audit,
                        simulation_seed=simulation_seed,
                        config=agent_config,
                        reassign_open_requests=False,
                    )
                event = {
                    "hora_simulada": state["hora_actual"],
                    "solicitud": None,
                    "nodo": "detectar_ausencia_pm",
                    "decision": "pm_marcado_no_elegible_por_ausencia",
                    "motivo": availability["motivo"],
                    "datos": {
                        "pm_responsables": absent_pm_names,
                        "correo_pm": availability["correo_pm"],
                        "message_id": identifier,
                        "solicitudes_escaladas": [
                            request["id"] for request in affected_requests
                        ],
                    },
                }
                audit.append(event)
                state["eventos"].append(event)
                for request in affected_requests:
                    old_pm = str(request.get("pm_responsable") or "")
                    request["pm_anterior_por_ausencia"] = old_pm
                    request["jefe_responsable"] = agent_config.manager_name
                    request["responsable_actual"] = agent_config.manager_name
                    request["escalada_a_jefe"] = True
                    request["escalado_director"] = True
                    request["escalamiento_ausencia_pendiente"] = True
                    request["nivel_maximo_escalamiento"] = "Director"
                    request["estado"] = "Escalada"
                    request["motivo_escalamiento"] = availability["motivo"]
                    request["fecha_escalamiento_director"] = (
                        datetime.now(timezone.utc).isoformat()
                    )
                    if agent_config.manager_email:
                        try:
                            email_details = send_lifecycle_notification(
                                client,
                                agent_config.manager_email,
                                "escalamiento_jefe",
                                request,
                                context={
                                    "jefe": agent_config.manager_name,
                                    "razon": availability["motivo"],
                                    "pm_ausentes": absent_pm_names,
                                    "pregunta": (
                                        "Confirma si aceptarás hacerte cargo "
                                        "respondiendo a este correo."
                                    ),
                                },
                                state=state,
                                audit=audit,
                                config=agent_config,
                                llm=llm,
                            )
                        except (AgentMailError, ValueError) as exc:
                            if isinstance(exc, ValueError):
                                error_event = audit_error(
                                    audit,
                                    exc,
                                    node="escalar_ausencia_pm_por_correo",
                                    hora_simulada=state["hora_actual"],
                                    solicitud=request["id"],
                                )
                                state["eventos"].append(error_event)
                            print(
                                f"\n[ERROR CORREO AUSENCIA PM] {exc}"
                            )
                        else:
                            request["aprobacion_director_message_id"] = (
                                email_details.get("message_id")
                            )
                            request.setdefault(
                                "notificaciones_email", []
                            ).append(
                                {
                                    "tipo": "escalamiento_jefe",
                                    **email_details,
                                }
                            )
                    else:
                        error_event = audit_error(
                            audit,
                            "No hay correo configurado para el jefe.",
                            node="escalar_ausencia_pm_por_correo",
                            hora_simulada=state["hora_actual"],
                            solicitud=request["id"],
                        )
                        state["eventos"].append(error_event)
                classified_message = {
                    **message,
                    "categoria": "respuesta_automatica_pm_no_disponible",
                    "pm_responsable": availability["pm_responsable"],
                }
                existing_classification = next(
                    (
                        item
                        for item in state["correos_clasificados"]
                        if get_message_id(item) == identifier
                    ),
                    None,
                )
                if existing_classification is None:
                    state["correos_clasificados"].append(classified_message)
                else:
                    existing_classification.update(classified_message)
                if identifier not in state["correos_procesados"]:
                    state["correos_procesados"].append(identifier)
                save_state_locked()
                on_state(state)
            print(
                f"\n[PM NO DISPONIBLE] {availability['pm_responsable']} "
                f"({availability['correo_pm']}); dejó de recibir nuevas "
                "solicitudes y sus casos activos se escalaron al jefe."
            )
            return True
        except (OSError, ValueError) as exc:
            audit_error(
                audit,
                exc,
                node="procesar_ausencia_pm",
                hora_simulada=state["hora_actual"],
                solicitud=identifier,
            )
            print(
                f"\n[ERROR PM NO DISPONIBLE] No se pudo deshabilitar "
                f"{availability['pm_responsable']}: {exc}"
            )
            return False

    def reprocess_previous_pm_absences() -> None:
        if pm_profiles is None:
            return
        with lock:
            previous_classifications = list(state["correos_clasificados"])
        for message in previous_classifications:
            identifier = get_message_id(message)
            if not identifier:
                continue
            with lock:
                if absence_already_recorded(identifier):
                    continue
            availability = detect_pm_unavailability(message, pm_profiles)
            if availability is not None:
                print(
                    f"\n[REVISIÓN HISTÓRICA] Se detectó una respuesta de ausencia "
                    f"de {availability['pm_responsable']} previamente procesada."
                )
                process_pm_absence(message, availability)

    def message_references(message: Mapping[str, Any]) -> set[str]:
        values: list[str] = []
        for key in ("in_reply_to", "references"):
            raw_value = message.get(key)
            if isinstance(raw_value, str):
                values.extend(raw_value.split())
            elif isinstance(raw_value, list):
                values.extend(str(value) for value in raw_value)
        headers = message.get("headers")
        if isinstance(headers, Mapping):
            for name, value in headers.items():
                if normalize_text(str(name)) in {"in reply to", "references"}:
                    values.extend(
                        re.findall(r"<[^>]+>|[^\s,]+", str(value))
                    )
        return {
            value.strip().strip("<>").strip()
            for value in values
            if value.strip().strip("<>").strip()
        }

    def sender_roles(sender_address: str) -> list[str]:
        roles: list[str] = []
        for profile in (pm_profiles or {}).values():
            addresses = profile.get("emails") or []
            if isinstance(addresses, str):
                addresses = [addresses]
            if sender_address in {
                str(address).strip().casefold()
                for address in addresses
            }:
                roles.append(str(profile.get("nombre") or ""))
        if (
            agent_config.manager_email
            and sender_address == agent_config.manager_email.casefold()
        ):
            roles.append(agent_config.manager_name)
        return list(dict.fromkeys(role for role in roles if role))

    def handle_manager_approval(message: dict[str, Any]) -> bool:
        sender_address = get_sender_address(message)
        if (
            not sender_address
            or not agent_config.manager_email
            or sender_address != agent_config.manager_email.casefold()
        ):
            return False
        references = message_references(message)
        body = "\n".join(
            str(message.get(key) or "")
            for key in ("text", "extracted_text", "preview")
        )
        body = "\n".join(
            line for line in body.splitlines() if not line.lstrip().startswith(">")
        )
        normalized_body = normalize_text(body)
        decision: bool | None = None
        if re.search(
            r"\b(no|rechazo|declino|no acepto|no puedo|"
            r"i decline|i cannot|can't take)\b",
            normalized_body,
        ):
            decision = False
        elif re.search(
            r"\b(si|yes|acepto|apruebo|aprobado|me hago cargo|"
            r"me encargo|lo tomo|i accept|i approve|i can handle)\b",
            normalized_body,
        ):
            decision = True
        if decision is None:
            return False

        with lock:
            def matches_pending_approval(
                candidate: Mapping[str, Any],
            ) -> bool:
                approval_ids = {
                    str(
                        candidate.get("aprobacion_director_message_id") or ""
                    ).strip().strip("<>"),
                    *(
                        str(item.get("message_id") or "")
                        .strip()
                        .strip("<>")
                        for item in candidate.get("notificaciones_email", [])
                        if isinstance(item, Mapping)
                        and not item.get("superseded")
                        and item.get("tipo")
                        in {"escalamiento_jefe", "aprobacion_director"}
                    ),
                }
                approval_ids.discard("")
                has_pending_approval = bool(
                    candidate.get("aprobacion_director_pendiente")
                    or candidate.get("escalamiento_ausencia_pendiente")
                )
                subject_identifies_request = (
                    bool(candidate.get("id"))
                    and _subject_identifies_request(
                        str(message.get("subject") or ""),
                        str(candidate["id"]),
                    )
                )
                return has_pending_approval and (
                    bool(references.intersection(approval_ids))
                    or (not references and subject_identifies_request)
                )

            request = next(
                (
                    candidate
                    for candidate in state["solicitudes"]
                    if matches_pending_approval(candidate)
                    and candidate.get("estado") != "Cerrada"
                ),
                None,
            )
            if request is None:
                return False
            reason = body.strip() or (
                "Erick aceptó el escalamiento."
                if decision
                else "Erick rechazó el escalamiento."
            )
            request["respuesta_aprobacion_director"] = {
                "approved": decision,
                "reason": reason,
            }
            if request.get("escalamiento_ausencia_pendiente"):
                request["escalamiento_ausencia_pendiente"] = False
                request["aprobacion_director_pendiente"] = False
                if decision:
                    request["jefe_responsable"] = agent_config.manager_name
                    request["responsable_actual"] = agent_config.manager_name
                    request["escalada_a_jefe"] = True
                    request["escalado_director"] = True
                    request["estado"] = "Confirmada"
                    request["fecha_confirmacion"] = datetime.now(
                        timezone.utc
                    ).isoformat()
                    request["hora_confirmacion"] = state["hora_actual"]
                    requester_email = str(
                        request.get("correo_solicitante") or ""
                    ).strip()
                    if requester_email:
                        try:
                            details = send_lifecycle_notification(
                                client,
                                requester_email,
                                "solicitud_escalada",
                                request,
                                context={
                                    "jefe": agent_config.manager_name,
                                    "razon": reason,
                                },
                                state=state,
                                audit=audit,
                                config=agent_config,
                                llm=llm,
                            )
                        except (AgentMailError, ValueError) as exc:
                            if isinstance(exc, ValueError):
                                error_event = audit_error(
                                    audit,
                                    exc,
                                    node="notificar_aprobacion_jefe",
                                    hora_simulada=state["hora_actual"],
                                    solicitud=request["id"],
                                    details={
                                        "destinatario": requester_email
                                    },
                                )
                                state["eventos"].append(error_event)
                            print(
                                f"\n[ERROR CORREO ESCALAMIENTO] {exc}"
                            )
                        else:
                            request.setdefault(
                                "notificaciones_email", []
                            ).append(
                                {
                                    "tipo": "solicitud_escalada",
                                    **details,
                                }
                            )
                else:
                    replacement, loads = select_least_loaded_pm(
                        state["solicitudes"],
                        build_pm_roster(assignments, pm_profiles),
                        maximum_open_projects=(
                            agent_config.max_open_projects_per_pm
                        ),
                        pm_profiles=pm_profiles,
                    )
                    if replacement:
                        _supersede_pm_assignment_emails(request)
                        request["pm_responsable"] = replacement
                        request["responsable_actual"] = replacement
                        request["jefe_responsable"] = None
                        request["escalada_a_jefe"] = False
                        request["escalado_director"] = False
                        request["nivel_maximo_escalamiento"] = "Ninguno"
                        request["estado"] = "Notificada"
                        request["hora_notificacion"] = state["hora_actual"]
                        request["fecha_notificacion"] = datetime.now(
                            timezone.utc
                        ).isoformat()
                        request["motivo_reasignacion"] = reason
                        request["respuesta_aprobacion_director"] = None
                        profile = next(
                            (
                                candidate
                                for candidate in (pm_profiles or {}).values()
                                if normalize_text(
                                    str(candidate.get("nombre") or "")
                                )
                                == normalize_text(replacement)
                            ),
                            {},
                        )
                        addresses = profile.get("emails") or []
                        if isinstance(addresses, str):
                            addresses = [addresses]
                        replacement_email = next(
                            (
                                str(address).strip().casefold()
                                for address in addresses
                                if isinstance(address, str)
                                and "@" in address
                            ),
                            None,
                        )
                        if replacement_email:
                            try:
                                details = send_lifecycle_notification(
                                    client,
                                    replacement_email,
                                    "solicitud_reasignada",
                                    request,
                                    context={
                                        "razon": reason,
                                        "cargas_abiertas": loads,
                                    },
                                    state=state,
                                    audit=audit,
                                    config=agent_config,
                                    llm=llm,
                                )
                                request["notificacion_pm_message_id"] = (
                                    details.get("message_id")
                                )
                                request.setdefault(
                                    "notificaciones_email", []
                                ).append(
                                    {
                                        "tipo": "solicitud_reasignada",
                                        **details,
                                    }
                                )
                            except (AgentMailError, ValueError) as exc:
                                if isinstance(exc, ValueError):
                                    error_event = audit_error(
                                        audit,
                                        exc,
                                        node="notificar_reasignacion_por_rechazo",
                                        hora_simulada=state["hora_actual"],
                                        solicitud=request["id"],
                                        details={
                                            "destinatario": replacement_email
                                        },
                                    )
                                    state["eventos"].append(error_event)
                                print(
                                    f"\n[ERROR CORREO REASIGNACIÓN] {exc}"
                                )
            event = {
                "hora_simulada": state["hora_actual"],
                "solicitud": request["id"],
                "nodo": "respuesta_escalamiento_jefe_por_correo",
                "decision": (
                    "jefe_acepto_escalamiento"
                    if decision
                    else "jefe_rechazo_escalamiento"
                ),
                "motivo": reason,
                "datos": {
                    "jefe": agent_config.manager_name,
                    "message_id": get_message_id(message),
                    "escalada_a_jefe": request.get("escalada_a_jefe"),
                    "responsable_actual": request.get("responsable_actual"),
                },
            }
            audit.append(event)
            state["eventos"].append(event)
            state["correos_procesados"].append(get_message_id(message))
            state["correos_clasificados"].append(
                {
                    **message,
                    "categoria": "respuesta_escalamiento_jefe",
                    "solicitud_id": request["id"],
                }
            )
            save_state_locked()
            on_state(state)
        return True

    def fallback_case_query(message: Mapping[str, Any]) -> bool:
        text = normalize_text(
            " ".join(
                (
                    str(message.get("subject") or ""),
                    message_text(dict(message)),
                )
            )
        )
        return any(
            phrase in text
            for phrase in (
                "mis casos",
                "mis solicitudes",
                "casos pendientes",
                "solicitudes pendientes",
                "lista de solicitudes",
                "lista de mis casos",
                "estado de mis solicitudes",
                "estado de mis casos",
                "que solicitudes tengo",
                "que casos tengo",
                "que tengo pendiente",
                "que debo atender",
                "status",
                "my pending cases",
                "my pending requests",
                "what cases are assigned to me",
                "what do i need to handle",
                "list my cases",
                "show my requests",
            )
        )

    def handle_case_query(message: dict[str, Any]) -> bool:
        sender_address = get_sender_address(message)
        if not sender_address:
            return False
        roles = sender_roles(sender_address)
        if not roles:
            return False
        query_result: bool | None = None
        classifier = getattr(llm, "classify_case_query", None) if llm else None
        with lock:
            llm_calls = state["llamadas_llm"]
        if llm is not None and agent_config.llm_enabled:
            limit = configured_max_llm_calls(agent_config.max_llm_calls)
            for attempt in range(2):
                if limit and llm_calls >= limit:
                    break
                with lock:
                    state["llamadas_llm"] += 1
                    llm_calls = state["llamadas_llm"]
                try:
                    if not callable(classifier):
                        raise OpenRouterError(
                            "El cliente LLM no puede interpretar consultas de casos."
                        )
                    query_result = classifier(message)
                    break
                except OpenRouterError as exc:
                    with lock:
                        error = audit_error(
                            audit,
                            exc,
                            node="clasificar_consulta_casos_llm",
                            hora_simulada=state["hora_actual"],
                            solicitud=get_message_id(message),
                            details={
                                "intento": attempt + 1,
                                "respaldo_por_reglas": True,
                            },
                        )
                        state["eventos"].append(error)
        if query_result is None:
            query_result = fallback_case_query(message)
        if not query_result:
            return False

        role_keys = {
            normalize_text(canonical_pm_name(role, pm_profiles))
            for role in roles
        }
        with lock:
            pending_requests = [
                request
                for request in state["solicitudes"]
                if request.get("estado") != "Cerrada"
                and any(
                    request.get(field)
                    and normalize_text(
                        canonical_pm_name(str(request[field]), pm_profiles)
                    )
                    in role_keys
                    for field in (
                        "pm_responsable",
                        "jefe_responsable",
                        "responsable_actual",
                    )
                )
            ]
        identifier = get_message_id(message)
        reply_sent = False
        try:
            send_case_query_reply(
                client,
                recipient=sender_address,
                message_id=identifier,
                requests=pending_requests,
                state=state,
                audit=audit,
            )
        except AgentMailError as exc:
            print(f"\n[ERROR CORREO CONSULTA] No se pudo enviar respuesta: {exc}")
        else:
            reply_sent = True
        with lock:
            if reply_sent:
                state["correos_procesados"].append(identifier)
            state["correos_clasificados"].append(
                {
                    **message,
                    "categoria": (
                        "consulta_casos"
                        if reply_sent
                        else "consulta_casos_pendiente"
                    ),
                }
            )
            save_state_locked()
            on_state(state)
        return True

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

        prefix = "CORREO ANTIGUO" if historical else "NUEVO CORREO"
        print(
            f"\n[{prefix}] De: {get_sender(message)} | "
            f"Asunto: {message.get('subject') or '(sin asunto)'}"
        )

        availability = detect_pm_unavailability(
            message,
            pm_profiles or {},
        )
        with lock:
            already_processed = identifier in state["correos_procesados"]
            already_recorded_absence = absence_already_recorded(identifier)
        if already_processed and (
            availability is None
            or already_recorded_absence
        ):
            return False
        if availability is not None and pm_profiles is not None:
            return process_pm_absence(message, availability)
        if handle_manager_approval(message):
            return True
        if handle_case_query(message):
            return True

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
                pm_profiles=pm_profiles,
                mail_client=client,
                llm=llm,
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
        with lock:
            if migrate_active_requests_to_roster():
                save_state_locked()
                on_state(state)
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
                            reprocess_previous_pm_absences()
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
