from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Mapping, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agent_config import AgentConfig
from agent_state import AgentState
from audit_log import JsonlAuditLog, audit_error
from email_utils import normalize_text

SLA_STOP_STATES = frozenset({"Confirmada", "Cerrada"})


class DirectorApprovalState(TypedDict):
    approval_request: dict[str, Any]
    decision: dict[str, Any]


def build_director_approval_graph() -> Any:
    def request_approval(state: DirectorApprovalState) -> dict[str, Any]:
        decision = interrupt(state["approval_request"])
        if not isinstance(decision, dict):
            raise ValueError("La aprobación humana debe ser un objeto.")
        return {"decision": decision}

    graph = StateGraph(DirectorApprovalState)
    graph.add_node("esperar_aprobacion_humana", request_approval)
    graph.add_edge(START, "esperar_aprobacion_humana")
    graph.add_edge("esperar_aprobacion_humana", END)
    return graph.compile(checkpointer=MemorySaver())


def request_director_approval(
    graph: Any,
    *,
    thread_id: str,
    payload: dict[str, Any],
    approval_handler: Callable[[dict[str, Any]], dict[str, Any] | None],
    checkpoint_pending: bool = False,
) -> dict[str, Any] | None:
    graph_config = {"configurable": {"thread_id": thread_id}}
    if not checkpoint_pending:
        paused = graph.invoke(
            {"approval_request": payload},
            config=graph_config,
        )
        if not paused.get("__interrupt__"):
            raise RuntimeError(
                "LangGraph no pausó para solicitar aprobación humana."
            )

    response = approval_handler(payload)
    if response is None:
        return None
    if not isinstance(response, dict) or not isinstance(
        response.get("approved"),
        bool,
    ):
        raise ValueError(
            "La respuesta humana debe incluir approved como valor booleano."
        )
    if not response["approved"] and not str(response.get("reason", "")).strip():
        raise ValueError("El rechazo debe incluir un motivo.")

    resumed = graph.invoke(
        Command(resume=response),
        config=graph_config,
    )
    decision = resumed.get("decision")
    if not isinstance(decision, dict):
        raise RuntimeError("LangGraph reanudó sin devolver la decisión humana.")
    return decision


def _elapsed_hours(
    state: AgentState,
    request: dict[str, Any],
    *,
    simulated: bool,
    now: datetime | None,
) -> float | None:
    if simulated:
        notified_at = request.get("hora_notificacion")
        if not isinstance(notified_at, (int, float)):
            return None
        return max(0.0, state["hora_actual"] - notified_at)

    notified_at_text = request.get("fecha_notificacion")
    if not isinstance(notified_at_text, str):
        if request.get("hora_notificacion") is not None:
            raise ValueError(
                f"Falta fecha_notificacion para la solicitud {request['id']}."
            )
        return None
    try:
        notified_at = datetime.fromisoformat(
            notified_at_text.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError(
            f"fecha_notificacion inválida para solicitud {request['id']}: "
            f"{notified_at_text!r}"
        ) from exc
    if notified_at.tzinfo is None:
        notified_at = notified_at.replace(tzinfo=timezone.utc)
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    return max(0.0, (current_time - notified_at).total_seconds() / 3600)


def _approval_repeat_due(
    request: dict[str, Any],
    *,
    elapsed_for_thresholds: float,
    repeat_after: int,
    simulated: bool,
    now: datetime,
    testing_minutes: bool,
) -> bool:
    if simulated:
        previous = request.get("hora_ultima_pregunta_aprobacion_director")
        return (
            isinstance(previous, (int, float))
            and elapsed_for_thresholds - previous >= repeat_after
        )

    previous_text = request.get("fecha_ultima_pregunta_aprobacion_director")
    if not isinstance(previous_text, str):
        return True
    try:
        previous_time = datetime.fromisoformat(
            previous_text.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError(
            "fecha_ultima_pregunta_aprobacion_director inválida para "
            f"solicitud {request['id']}: {previous_text!r}"
        ) from exc
    if previous_time.tzinfo is None:
        previous_time = previous_time.replace(tzinfo=timezone.utc)
    elapsed = (now - previous_time).total_seconds() / 3600
    if testing_minutes:
        elapsed *= 60
    return elapsed >= repeat_after


def process_due_sla_actions(
    state: AgentState,
    *,
    audit: JsonlAuditLog,
    config: AgentConfig,
    directors: Mapping[str, str] | None = None,
    simulated: bool,
    now: datetime | None = None,
    approval_handler: (
        Callable[[dict[str, Any]], dict[str, Any] | None] | None
    ) = None,
    approval_graph: Any | None = None,
) -> int:
    director_by_client = {
        normalize_text(str(client)): str(director)
        for client, director in (directors or {}).items()
    }
    actions = 0
    testing_minutes = not simulated and config.testing_enabled
    time_unit = "minutos" if testing_minutes else "horas"
    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    if testing_minutes:
        thresholds = {
            "first_reminder": config.test_reminder_after_minutes,
            "second_reminder": config.test_second_reminder_after_minutes,
            "director_escalation": config.test_escalate_after_minutes,
            "next_level_reminder": config.test_next_level_reminder_after_minutes,
            "commercial_delivery": config.test_commercial_delivery_after_minutes,
            "repeat": config.test_repeat_after_minutes,
        }
    else:
        thresholds = {
            "first_reminder": config.reminder_after_hours,
            "second_reminder": config.second_reminder_after_hours,
            "director_escalation": config.escalate_after_hours,
            "next_level_reminder": config.next_level_reminder_after_hours,
            "commercial_delivery": config.commercial_delivery_after_hours,
            "repeat": config.repeat_after_hours,
        }

    def record_action(
        request: dict[str, Any],
        *,
        decision: str,
        reason: str,
        details: dict[str, Any],
        output: str,
        notification_type: str | None = None,
        recipient: str | None = None,
    ) -> None:
        nonlocal actions
        if notification_type is not None:
            if not recipient:
                raise ValueError(
                    "Cada notificación de seguimiento debe tener destinatario."
                )
            notification_number = int(
                request.get("numero_notificacion_seguimiento", 0)
            ) + 1
            request["numero_notificacion_seguimiento"] = notification_number
            details = {
                **details,
                "numero_notificacion": notification_number,
                "tipo_notificacion": notification_type,
                "destinatario": recipient,
                "pm_responsable": request.get("pm_responsable"),
            }
            output = (
                f"[NOTIFICACIÓN DE SEGUIMIENTO SIMULADA #{notification_number}] "
                f"Tipo: {notification_type} | Para: {recipient} | "
                f"PM asignado: {request.get('pm_responsable')} | {output}"
            )
        event = {
            "hora_simulada": state["hora_actual"],
            "solicitud": request["id"],
            "nodo": "procesar_sla",
            "decision": decision,
            "motivo": reason,
            "datos": details,
        }
        audit.append(event)
        state["eventos"].append(event)
        print(output)
        actions += 1

    def elapsed_threshold_value(elapsed_hours: float) -> float:
        return elapsed_hours * 60 if testing_minutes else elapsed_hours

    for request_index, request in enumerate(state["solicitudes"], start=1):
        if (
            request.get("estado") in SLA_STOP_STATES
            or not request.get("pm_responsable")
        ):
            continue
        try:
            elapsed = _elapsed_hours(
                state,
                request,
                simulated=simulated,
                now=now,
            )
        except ValueError as exc:
            if not request.get("error_fecha_notificacion_sla_auditado"):
                event = audit_error(
                    audit,
                    exc,
                    node="calcular_umbral_sla",
                    hora_simulada=state["hora_actual"],
                    solicitud=request["id"],
                )
                state["eventos"].append(event)
                request["error_fecha_notificacion_sla_auditado"] = True
            continue
        if elapsed is None:
            continue
        elapsed_for_thresholds = elapsed_threshold_value(elapsed)
        common_details = {
            "horas_transcurridas": elapsed,
            "tiempo_transcurrido_umbral": elapsed_for_thresholds,
            "unidad_umbral": time_unit,
            "pm_responsable": request.get("pm_responsable"),
            "reloj_simulado": simulated,
            "notificacion_simulada": True,
        }

        if (
            elapsed_for_thresholds >= thresholds["first_reminder"]
            and not request.get(
                "primer_recordatorio_enviado",
                request.get("recordatorio_enviado", False),
            )
        ):
            request["primer_recordatorio_enviado"] = True
            request["recordatorio_enviado"] = True
            request["hora_recordatorio"] = state["hora_actual"] if simulated else None
            request["fecha_recordatorio"] = (
                None if simulated else now_utc.isoformat()
            )
            request["nivel_maximo_escalamiento"] = "PM"
            if request.get("estado") not in {"Escalada", *SLA_STOP_STATES}:
                request["estado"] = "Recordatorio enviado"
            record_action(
                request,
                decision="recordatorio_enviado",
                reason=(
                    f"Primer recordatorio a las "
                    f"{thresholds['first_reminder']} {time_unit} sin "
                    "confirmación."
                ),
                details={
                    **common_details,
                    "umbral": thresholds["first_reminder"],
                    "nivel": "PM",
                },
                notification_type="Primer recordatorio al PM",
                recipient=str(request["pm_responsable"]),
                output=(
                    f"{request.get('asunto', request['id'])}"
                ),
            )

        if (
            elapsed_for_thresholds >= thresholds["second_reminder"]
            and not request.get("segundo_recordatorio_enviado", False)
        ):
            request["segundo_recordatorio_enviado"] = True
            request["hora_segundo_recordatorio"] = (
                state["hora_actual"] if simulated else None
            )
            request["fecha_segundo_recordatorio"] = (
                None if simulated else now_utc.isoformat()
            )
            record_action(
                request,
                decision="segundo_recordatorio_enviado",
                reason=(
                    f"Segundo recordatorio al PM a las "
                    f"{thresholds['second_reminder']} {time_unit}."
                ),
                details={
                    **common_details,
                    "umbral": thresholds["second_reminder"],
                    "nivel": "PM",
                },
                notification_type="Segundo recordatorio al PM",
                recipient=str(request["pm_responsable"]),
                output=(
                    f"{request.get('asunto', request['id'])}"
                ),
            )

        if (
            elapsed_for_thresholds >= thresholds["director_escalation"]
            and not request.get("escalado_director", False)
        ):
            director = director_by_client.get(
                normalize_text(str(request.get("cliente") or ""))
            )
            approval_reason = ""
            approved = False
            approval_decision: dict[str, Any] | None = None
            if simulated:
                approved = True
                approval_reason = (
                    "Aprobación automática exclusiva del modo de simulación."
                )
                approval_decision = {
                    "approved": True,
                    "reason": approval_reason,
                }
            elif approval_handler is not None:
                approval_graph_instance = (
                    approval_graph or build_director_approval_graph()
                )
                approval_pending = request.get(
                    "aprobacion_director_pendiente",
                    False,
                )

                def make_approval_payload(attempt: int) -> dict[str, Any]:
                    return {
                        "solicitud": request["id"],
                        "numero_solicitud": request_index,
                        "asunto": request.get("asunto", request["id"]),
                        "cliente": request.get("cliente"),
                        "pm_responsable": request.get("pm_responsable"),
                        "director": director,
                        "horas_transcurridas": elapsed,
                        "umbral": thresholds["director_escalation"],
                        "unidad_umbral": time_unit,
                        "intento": attempt,
                        "mensaje": (
                            f"¿Apruebas escalar al director "
                            f"{director or '(sin asignar)'} la solicitud "
                            f"«{request.get('asunto', request['id'])}»?"
                        ),
                    }

                def record_approval_request(
                    payload: dict[str, Any],
                ) -> None:
                    approval_event = {
                        "hora_simulada": state["hora_actual"],
                        "solicitud": request["id"],
                        "nodo": "aprobacion_humana_director",
                        "decision": "aprobacion_director_solicitada",
                        "motivo": (
                            "LangGraph pausó el flujo para solicitar aprobación "
                            "humana antes de escalar."
                        ),
                        "datos": payload,
                    }
                    audit.append(approval_event)
                    state["eventos"].append(approval_event)

                def record_no_approval_response(
                    payload: dict[str, Any],
                ) -> None:
                    event = {
                        "hora_simulada": state["hora_actual"],
                        "solicitud": request["id"],
                        "nodo": "aprobacion_humana_director",
                        "decision": "aprobacion_director_sin_respuesta",
                        "motivo": (
                            "La aprobación sigue pendiente; se volverá a "
                            "preguntar con la próxima notificación de seguimiento."
                        ),
                        "datos": {
                            "intento": payload["intento"],
                            "director": director,
                            "umbral_repeticion": thresholds["repeat"],
                            "unidad_umbral": time_unit,
                        },
                    }
                    audit.append(event)
                    state["eventos"].append(event)

                if approval_pending:
                    attempt = int(
                        request.get("intentos_aprobacion_director", 0)
                    )
                    thread_id = str(
                        request.get("aprobacion_director_thread_id")
                        or f"{request['id']}:director-approval:{attempt}"
                    )
                    approval_payload = make_approval_payload(attempt)
                    approval_decision = request_director_approval(
                        approval_graph_instance,
                        thread_id=thread_id,
                        payload=approval_payload,
                        approval_handler=approval_handler,
                        checkpoint_pending=request.get(
                            "aprobacion_director_checkpoint_pendiente",
                            False,
                        ),
                    )
                    if approval_decision is None and _approval_repeat_due(
                        request,
                        elapsed_for_thresholds=elapsed_for_thresholds,
                        repeat_after=thresholds["repeat"],
                        simulated=simulated,
                        now=now_utc,
                        testing_minutes=testing_minutes,
                    ):
                        record_action(
                            request,
                            decision="repetir_solicitud_aprobacion_director",
                            reason=(
                                "La aprobación para escalar sigue sin respuesta; "
                                "se reitera la notificación y la pregunta."
                            ),
                            details={
                                "director": director,
                                "umbral_repeticion": thresholds["repeat"],
                                "unidad_umbral": time_unit,
                            },
                            notification_type=(
                                "Seguimiento de aprobación para escalar"
                            ),
                            recipient=director or "Director (sin asignar)",
                            output=(
                                f"{request.get('asunto', request['id'])}"
                            ),
                        )
                        attempt += 1
                        request["intentos_aprobacion_director"] = attempt
                        approval_payload = make_approval_payload(attempt)
                        record_approval_request(approval_payload)
                        request["hora_ultima_pregunta_aprobacion_director"] = (
                            state["hora_actual"] if simulated else None
                        )
                        request["fecha_ultima_pregunta_aprobacion_director"] = (
                            None if simulated else now_utc.isoformat()
                        )
                        approval_decision = request_director_approval(
                            approval_graph_instance,
                            thread_id=thread_id,
                            payload=approval_payload,
                            approval_handler=approval_handler,
                            checkpoint_pending=True,
                        )
                        if approval_decision is None:
                            record_no_approval_response(approval_payload)
                else:
                    attempt = int(
                        request.get("intentos_aprobacion_director", 0)
                    ) + 1
                    request["intentos_aprobacion_director"] = attempt
                    thread_id = (
                        f"{request['id']}:director-approval:{attempt}"
                    )
                    approval_payload = make_approval_payload(attempt)
                    record_approval_request(approval_payload)
                    request["hora_ultima_pregunta_aprobacion_director"] = (
                        state["hora_actual"] if simulated else None
                    )
                    request["fecha_ultima_pregunta_aprobacion_director"] = (
                        None if simulated else now_utc.isoformat()
                    )
                    approval_decision = request_director_approval(
                        approval_graph_instance,
                        thread_id=thread_id,
                        payload=approval_payload,
                        approval_handler=approval_handler,
                    )
                    if approval_decision is None:
                        request["aprobacion_director_pendiente"] = True
                        request["aprobacion_director_checkpoint_pendiente"] = (
                            True
                        )
                        request["aprobacion_director_thread_id"] = thread_id
                        record_no_approval_response(approval_payload)

            elif not request.get("error_aprobacion_director_auditado"):
                event = audit_error(
                    audit,
                    "No se puede escalar al director sin un canal de aprobación humana.",
                    node="aprobacion_humana_director",
                    hora_simulada=state["hora_actual"],
                    solicitud=request["id"],
                )
                state["eventos"].append(event)
                request["error_aprobacion_director_auditado"] = True

            if request.get("estado") in SLA_STOP_STATES:
                request["aprobacion_director_pendiente"] = False
                request["aprobacion_director_checkpoint_pendiente"] = False
                request.pop("aprobacion_director_thread_id", None)
                record_action(
                    request,
                    decision="escalamiento_cancelado_solicitud_finalizada",
                    reason=(
                        "La solicitud se confirmó o cerró mientras esperaba "
                        "la aprobación humana; no se escaló al director."
                    ),
                    details={
                        "estado": request["estado"],
                        "director": director,
                    },
                    output=(
                        f"[ESCALAMIENTO CANCELADO] {request.get('asunto', request['id'])} "
                        f"| Estado: {request['estado']}."
                    ),
                )
                continue

            if approval_decision is None:
                continue
            request["aprobacion_director_pendiente"] = False
            request["aprobacion_director_checkpoint_pendiente"] = False
            request.pop("aprobacion_director_thread_id", None)
            approved = approval_decision["approved"]
            approval_reason = str(
                approval_decision.get("reason")
                or (
                    "Aprobado por el usuario."
                    if approved
                    else "Rechazado por el usuario."
                )
            )

            if approved:
                request["escalado_director"] = True
                request["hora_escalamiento_director"] = (
                    state["hora_actual"] if simulated else None
                )
                request["fecha_escalamiento_director"] = (
                    None if simulated else now_utc.isoformat()
                )
                request["nivel_maximo_escalamiento"] = "Director"
                request["estado"] = "Escalada"
                record_action(
                    request,
                    decision=(
                        "escalamiento_director_simulado"
                        if simulated
                        else "escalamiento_director_aprobado"
                    ),
                    reason=approval_reason,
                    details={
                        **common_details,
                        "director": director,
                        "cliente": request.get("cliente"),
                        "umbral": thresholds["director_escalation"],
                        "nivel_maximo_escalamiento": "Director",
                    },
                    notification_type="Escalamiento aprobado al director",
                    recipient=director or "Director (sin asignar)",
                    output=(
                        f"{request.get('asunto', request['id'])}"
                    ),
                )
            elif approval_handler is not None:
                request["aprobacion_director_pendiente"] = False
                request["ultima_aprobacion_director_rechazada"] = (
                    now_utc.isoformat()
                )
                record_action(
                    request,
                    decision="escalamiento_director_rechazado",
                    reason=approval_reason,
                    details={
                        **common_details,
                        "director": director,
                        "umbral": thresholds["director_escalation"],
                        "intento": request["intentos_aprobacion_director"],
                    },
                    output=(
                        f"[ESCALAMIENTO RECHAZADO] {request.get('asunto', request['id'])} "
                        f"| Motivo: {approval_reason}. Se reevaluará en la siguiente vuelta."
                    ),
                )

        if (
            request.get("escalado_director", False)
            and elapsed_for_thresholds >= thresholds["next_level_reminder"]
            and not request.get("recordatorio_siguiente_nivel_enviado", False)
        ):
            request["recordatorio_siguiente_nivel_enviado"] = True
            request["hora_recordatorio_siguiente_nivel"] = (
                state["hora_actual"] if simulated else None
            )
            request["fecha_recordatorio_siguiente_nivel"] = (
                None if simulated else now_utc.isoformat()
            )
            record_action(
                request,
                decision="recordatorio_siguiente_nivel_enviado",
                reason=(
                    f"Recordatorio al director a las "
                    f"{thresholds['next_level_reminder']} {time_unit}."
                ),
                details={
                    **common_details,
                    "director": director_by_client.get(
                        normalize_text(str(request.get("cliente") or ""))
                    ),
                    "umbral": thresholds["next_level_reminder"],
                    "nivel": "Director",
                },
                notification_type="Recordatorio al director",
                recipient=(
                    director_by_client.get(
                        normalize_text(str(request.get("cliente") or ""))
                    )
                    or "Director (sin asignar)"
                ),
                output=(
                    f"{request.get('asunto', request['id'])}"
                ),
            )

        if (
            request.get("escalado_director", False)
            and elapsed_for_thresholds >= thresholds["commercial_delivery"]
            and not request.get("comercial_delivery_involucrados", False)
        ):
            request["comercial_delivery_involucrados"] = True
            request["nivel_maximo_escalamiento"] = "Comercial/Delivery"
            request["hora_ultimo_recordatorio_sla"] = (
                elapsed_for_thresholds if simulated else None
            )
            request["fecha_ultimo_recordatorio_sla"] = (
                None if simulated else now_utc.isoformat()
            )
            record_action(
                request,
                decision="involucrar_comercial_y_delivery",
                reason=(
                    f"Se involucró al equipo comercial y al responsable de "
                    f"delivery a las {thresholds['commercial_delivery']} "
                    f"{time_unit}."
                ),
                details={
                    **common_details,
                    "umbral": thresholds["commercial_delivery"],
                    "nivel_maximo_escalamiento": "Comercial/Delivery",
                    "equipos": [
                        "Equipo Comercial",
                        "Responsable de Delivery",
                    ],
                },
                notification_type="Involucrar Comercial y Delivery",
                recipient="Equipo Comercial; Responsable de Delivery",
                output=(
                    f"{request.get('asunto', request['id'])}"
                ),
            )
        elif (
            request.get("comercial_delivery_involucrados", False)
            and elapsed_for_thresholds >= thresholds["commercial_delivery"]
        ):
            repeat_due = False
            if simulated:
                previous = request.get("hora_ultimo_recordatorio_sla")
                repeat_due = (
                    isinstance(previous, (int, float))
                    and elapsed_for_thresholds - previous >= thresholds["repeat"]
                )
            else:
                previous_text = request.get("fecha_ultimo_recordatorio_sla")
                if isinstance(previous_text, str):
                    try:
                        previous_time = datetime.fromisoformat(
                            previous_text.replace("Z", "+00:00")
                        )
                    except ValueError as exc:
                        raise ValueError(
                            "fecha_ultimo_recordatorio_sla inválida para "
                            f"solicitud {request['id']}: {previous_text!r}"
                        ) from exc
                    if previous_time.tzinfo is None:
                        previous_time = previous_time.replace(tzinfo=timezone.utc)
                    repeat_elapsed = (
                        now_utc - previous_time
                    ).total_seconds() / 3600
                    if testing_minutes:
                        repeat_elapsed *= 60
                    repeat_due = repeat_elapsed >= thresholds["repeat"]

            if repeat_due:
                request["hora_ultimo_recordatorio_sla"] = (
                    elapsed_for_thresholds if simulated else None
                )
                request["fecha_ultimo_recordatorio_sla"] = (
                    None if simulated else now_utc.isoformat()
                )
                record_action(
                    request,
                    decision="repetir_recordatorio_escalamiento",
                    reason=(
                        f"Seguimiento repetido cada {thresholds['repeat']} "
                        f"{time_unit} hasta confirmar recepción."
                    ),
                    details={
                        **common_details,
                        "umbral_repeticion": thresholds["repeat"],
                        "equipos": [
                            "Director",
                            "Equipo Comercial",
                            "Responsable de Delivery",
                        ],
                    },
                    notification_type="Seguimiento repetido",
                    recipient=(
                        "Director; Equipo Comercial; Responsable de Delivery"
                    ),
                    output=(
                        f"{request.get('asunto', request['id'])}"
                    ),
                )

    return actions
