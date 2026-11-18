from __future__ import annotations

import re
from typing import Any

from agent_config import AgentConfig
from agent_state import AgentState
from agentmail_client import AgentMailClient
from audit_log import JsonlAuditLog, audit_error
from exceptions import AgentMailError, OpenRouterError
from openrouter_client import OpenRouterClient
from settings import configured_max_llm_calls


_EVENT_LABELS = {
    "solicitud_asignada": "Nueva solicitud asignada",
    "recordatorio": "Recordatorio de solicitud",
    "confirmacion_pm": "Confirmación de solicitud",
    "escalamiento_jefe": "Aprobación de escalamiento solicitada",
    "solicitud_escalada": "Solicitud aceptada por el jefe",
    "solicitud_completada": "Solicitud completada",
    "solicitud_cerrada": "Solicitud cerrada",
    "solicitud_reasignada": "Solicitud reasignada",
}


def _fallback_body(
    event_type: str,
    request: dict[str, Any],
    context: dict[str, Any],
) -> str:
    greeting = str(context.get("saludo") or "Hola,")
    identifier = str(request.get("id") or "sin identificador")
    subject = str(request.get("asunto") or identifier)
    lines = [greeting, ""]
    opening = {
        "solicitud_asignada": (
            "Se te asignó una nueva solicitud. Confirma si puedes atenderla "
            "respondiendo a este correo."
        ),
        "recordatorio": (
            "Te recordamos que esta solicitud sigue pendiente de confirmación "
            "o actualización."
        ),
        "confirmacion_pm": (
            "El PM confirmó que atenderá la solicitud."
        ),
        "escalamiento_jefe": (
            "El PM no confirmó a tiempo o indicó que no está disponible. "
            "Revisa la solicitud y responde a este correo indicando si aceptas "
            "hacerte cargo."
        ),
        "solicitud_escalada": (
            "El jefe aceptó hacerse cargo de esta solicitud."
        ),
        "solicitud_completada": (
            "Se registró que el trabajo de esta solicitud fue completado."
        ),
        "solicitud_cerrada": (
            "Se registró el cierre o cancelación de esta solicitud."
        ),
        "solicitud_reasignada": (
            "Se te reasignó esta solicitud. Confirma si puedes atenderla "
            "respondiendo a este correo."
        ),
    }
    lines.extend(
        [
            opening[event_type],
            "",
            f"Solicitud: {identifier}",
            f"Asunto: {subject}",
        ]
    )
    for label, key in (
        ("Cliente", "cliente"),
        ("Rol", "rol"),
        ("Cantidad", "cantidad_posiciones"),
        ("PM responsable", "pm_responsable"),
        ("Jefe responsable", "jefe_responsable"),
        ("Estado", "estado"),
    ):
        value = request.get(key)
        if value is not None and str(value).strip():
            lines.append(f"{label}: {value}")
    reason = context.get("razon") or request.get("razon_finalizacion")
    if reason:
        lines.append(f"Motivo: {reason}")
    for key, label in (
        ("umbral", "Umbral SLA"),
        ("unidad_umbral", "Unidad"),
        ("numero_notificacion", "Número de recordatorio"),
    ):
        value = context.get(key)
        if value is not None:
            lines.append(f"{label}: {value}")
    lines.extend(["", "Mensaje automático de AgentMail."])
    return "\n".join(lines)


def send_lifecycle_notification(
    client: AgentMailClient,
    recipient: str,
    event_type: str,
    request: dict[str, Any],
    *,
    context: dict[str, Any],
    state: AgentState,
    audit: JsonlAuditLog,
    config: AgentConfig,
    llm: OpenRouterClient | None,
) -> dict[str, Any]:
    if event_type not in _EVENT_LABELS:
        raise ValueError(f"Tipo de notificación desconocido: {event_type}.")
    normalized_recipient = recipient.strip().casefold()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", normalized_recipient):
        raise ValueError(
            f"No hay una dirección de correo válida para {recipient!r}."
        )

    body: str | None = None
    method = "plantilla"
    limit = configured_max_llm_calls(config.max_llm_calls)
    writer = getattr(llm, "write_lifecycle_email", None) if llm else None
    if llm is not None and config.llm_enabled:
        for attempt in range(2):
            if limit and state["llamadas_llm"] >= limit:
                break
            state["llamadas_llm"] += 1
            try:
                if not callable(writer):
                    raise OpenRouterError(
                        "El cliente LLM no puede redactar correos de ciclo de vida."
                    )
                body = writer(event_type, request, context)
                method = "llm"
                break
            except OpenRouterError as exc:
                event = audit_error(
                    audit,
                    exc,
                    node="redactar_correo_ciclo_vida_llm",
                    hora_simulada=state["hora_actual"],
                    solicitud=str(request.get("id") or "") or None,
                    details={
                        "tipo_evento": event_type,
                        "intento": attempt + 1,
                        "respaldo_por_plantilla": True,
                    },
                )
                state["eventos"].append(event)

        if body is None and limit and state["llamadas_llm"] >= limit:
            event = audit_error(
                audit,
                "Se alcanzó el límite de llamadas a OpenRouter.",
                node="redactar_correo_ciclo_vida_llm",
                hora_simulada=state["hora_actual"],
                solicitud=str(request.get("id") or "") or None,
                details={
                    "tipo_evento": event_type,
                    "limite": limit,
                    "respaldo_por_plantilla": True,
                },
            )
            state["eventos"].append(event)

    if body is None:
        body = _fallback_body(event_type, request, context)

    subject_text = " ".join(
        str(request.get("asunto") or request.get("id") or "").split()
    )[:120]
    email_subject = (
        f"[{_EVENT_LABELS[event_type]}] {request.get('id')}: {subject_text}"
    )
    try:
        message_id = client.send(normalized_recipient, email_subject, body)
    except AgentMailError as exc:
        event = audit_error(
            audit,
            exc,
            node="enviar_correo_ciclo_vida",
            hora_simulada=state["hora_actual"],
            solicitud=str(request.get("id") or "") or None,
            details={
                "tipo_evento": event_type,
                "destinatario": normalized_recipient,
                "metodo_redaccion": method,
            },
        )
        state["eventos"].append(event)
        raise

    event = {
        "hora_simulada": state["hora_actual"],
        "solicitud": str(request.get("id") or "") or None,
        "nodo": "enviar_correo_ciclo_vida",
        "decision": "correo_ciclo_vida_enviado",
        "motivo": f"Se envió correo de tipo {event_type}.",
        "datos": {
            "tipo_evento": event_type,
            "destinatario": normalized_recipient,
            "asunto": email_subject,
            "message_id": message_id,
            "metodo_redaccion": method,
            **context,
        },
    }
    audit.append(event)
    state["eventos"].append(event)
    return {
        "destinatario": normalized_recipient,
        "asunto": email_subject,
        "message_id": message_id,
        "metodo_redaccion": method,
    }


def send_case_query_reply(
    client: AgentMailClient,
    *,
    recipient: str,
    message_id: str,
    requests: list[dict[str, Any]],
    state: AgentState,
    audit: JsonlAuditLog,
) -> None:
    active_requests = [
        request
        for request in requests
        if request.get("estado") != "Cerrada"
    ]
    lines = ["Hola,", "", "Estas son las solicitudes que siguen pendientes:"]
    if active_requests:
        for request in active_requests:
            lines.extend(
                [
                    "",
                    f"- {request.get('id')}: "
                    f"{request.get('asunto') or '(sin asunto)'}",
                    f"  Cliente: {request.get('cliente') or 'sin identificar'}",
                    f"  Rol: {request.get('rol') or 'sin identificar'}",
                    "  Cantidad: "
                    f"{request.get('cantidad_posiciones') or 'sin identificar'}",
                    f"  PM: {request.get('pm_responsable') or 'sin asignar'}",
                    f"  Jefe: {request.get('jefe_responsable') or 'sin asignar'}",
                    f"  Estado: {request.get('estado') or 'sin estado'}",
                ]
            )
    else:
        lines.append("\nNo hay solicitudes pendientes en este momento.")
    body = "\n".join([*lines, "", "Mensaje automático de AgentMail."])
    try:
        client.send(
            recipient,
            "Solicitudes pendientes asignadas",
            body,
        )
    except AgentMailError as exc:
        event = audit_error(
            audit,
            exc,
            node="responder_consulta_casos_por_correo",
            hora_simulada=state["hora_actual"],
            solicitud=message_id,
            details={"destinatario": recipient},
        )
        state["eventos"].append(event)
        raise
    event = {
        "hora_simulada": state["hora_actual"],
        "solicitud": message_id,
        "nodo": "responder_consulta_casos_por_correo",
        "decision": "consulta_casos_respondida",
        "motivo": "Se envió al remitente la lista de solicitudes pendientes.",
        "datos": {
            "destinatario": recipient,
            "cantidad_solicitudes": len(active_requests),
        },
    }
    audit.append(event)
    state["eventos"].append(event)
