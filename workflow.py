from datetime import datetime, timezone
from typing import Any, Mapping

from langgraph.graph import END, START, StateGraph

from agent_state import AgentState
from agentmail_client import AgentMailClient
from audit_log import JsonlAuditLog, audit_error
from email_utils import (
    _ordered_by_timestamp, classify_by_rules, clarification_questions,
    confirmation_question, extract_request_details, get_message_id, get_sender,
    infer_request_status, load_assignments, load_pm_profiles, message_text,
    normalize_text, request_fingerprint,
    simulated_confirmation_delay,
)
from exceptions import AgentMailError, OpenRouterError
from openrouter_client import OpenRouterClient
from settings import DEFAULT_SIMULATION_SEED, configured_max_llm_calls

def build_graph(
    client: AgentMailClient,
    audit: JsonlAuditLog,
    *,
    assignments: Mapping[str, str] | None = None,
    llm: OpenRouterClient | None = None,
    pm_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    simulation_seed: int = DEFAULT_SIMULATION_SEED,
):
    llm_call_limit = configured_max_llm_calls()
    pm_by_client = (
        {normalize_text(str(client)): str(pm) for client, pm in assignments.items()}
        if assignments is not None
        else load_assignments()
    )
    profiles_by_pm = (
        {
            normalize_text(str(pm)): dict(profile)
            for pm, profile in pm_profiles.items()
        }
        if pm_profiles is not None
        else load_pm_profiles()
    )

    def record(
        state: AgentState,
        *,
        identifier: str | None,
        node: str,
        decision: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = {
            "hora_simulada": state["hora_actual"],
            "solicitud": identifier,
            "nodo": node,
            "decision": decision,
            "motivo": reason,
        }
        if details:
            event["datos"] = details
        audit.append(event)
        return event

    def read_inbox(state: AgentState) -> dict[str, Any]:
        event_messages = state.get("correos_evento", [])
        if event_messages:
            processed = set(state["correos_procesados"])
            emails: list[dict[str, Any]] = []
            for message in event_messages:
                identifier = get_message_id(message)
                if not identifier:
                    raise AgentMailError(
                        "Un evento de AgentMail no contiene message_id."
                    )
                if identifier not in processed:
                    emails.append({**message, "message_id": identifier})
            return {
                "correos_nuevos": _ordered_by_timestamp(emails),
                "correos_evento": [],
                "clasificacion_actual": [],
            }

        summaries = client.list_messages()
        processed = set(state["correos_procesados"])
        summaries_by_id: dict[str, dict[str, Any]] = {}
        for summary in summaries:
            identifier = get_message_id(summary)
            if not identifier:
                raise AgentMailError("Un correo de AgentMail no tiene message_id.")
            if identifier not in processed:
                summaries_by_id[identifier] = summary

        emails: list[dict[str, Any]] = []
        for identifier, summary in summaries_by_id.items():
            full_message = client.get_message(identifier)
            full_message.setdefault("message_id", identifier)
            emails.append(full_message)
        return {
            "correos_nuevos": _ordered_by_timestamp(emails),
            "correos_evento": [],
            "clasificacion_actual": [],
        }

    def classify_emails(state: AgentState) -> dict[str, Any]:
        known_fingerprints = set(state["firmas_solicitudes"])
        known_request_ids = set(state["ids_solicitudes"])
        requests_found = list(state["solicitudes"])
        fingerprints = list(state["firmas_solicitudes"])
        request_ids = list(state["ids_solicitudes"])
        processed_ids = list(state["correos_procesados"])
        classified: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        llm_calls = state["llamadas_llm"]
        known_requests = [
            {
                key: request.get(key)
                for key in (
                    "id",
                    "cliente",
                    "rol",
                    "cantidad_posiciones",
                    "posiciones",
                    "pm_responsable",
                    "estado",
                    "datos_faltantes",
                    "asunto",
                )
            }
            for request in state["solicitudes"]
        ]

        for email in state["correos_nuevos"]:
            identifier = get_message_id(email)
            rule_category, rule_reason = classify_by_rules(
                email,
                known_fingerprints=known_fingerprints,
                known_request_ids=known_request_ids,
            )
            llm_result: dict[str, Any] | None = None
            llm_error: str | None = None
            attempt_errors: list[OpenRouterError] = []
            if llm is not None:
                for attempt in range(2):
                    if llm_call_limit and llm_calls >= llm_call_limit:
                        break
                    llm_calls += 1
                    try:
                        llm_result = llm.classify_message(email, known_requests)
                        break
                    except OpenRouterError as exc:
                        llm_error = str(exc)
                        attempt_errors.append(exc)
                        if attempt == 1 or (
                            llm_call_limit and llm_calls >= llm_call_limit
                        ):
                            break

            if llm_result is not None:
                category = llm_result["categoria"]
                reason = (
                    llm_result.get("motivo")
                    or f"Clasificado por el modelo OpenRouter {llm.model}."
                )
                classification_method = "openrouter"
            else:
                category = rule_category
                reason = rule_reason
                classification_method = "reglas"
                if llm is not None:
                    if llm_error:
                        reason = f"Respaldo por reglas tras fallo de OpenRouter: {llm_error}"
                    elif llm_call_limit and llm_calls >= llm_call_limit:
                        reason = (
                            f"{rule_reason} Se alcanzó el límite de "
                            f"{llm_call_limit} llamadas a OpenRouter."
                        )

            for error in attempt_errors:
                events.append(
                    audit_error(
                        audit,
                        error,
                        node="clasificar_con_openrouter",
                        hora_simulada=state["hora_actual"],
                        solicitud=identifier,
                        details={
                            "intentos": len(attempt_errors),
                            "respaldo_por_reglas": llm_result is None,
                        },
                    )
                )

            classified_email = {**email, "categoria": category}
            classified.append(classified_email)
            processed_ids.append(identifier)

            if category == "actualizacion_solicitud":
                references = email.get("references") or []
                if isinstance(references, str):
                    references = references.split()
                related_ids = {str(item) for item in references}
                reply_to = email.get("in_reply_to")
                if reply_to:
                    related_ids.add(str(reply_to))
                related_id = (
                    llm_result.get("solicitud_relacionada_id")
                    if llm_result is not None
                    else None
                )
                if related_id not in {request["id"] for request in requests_found}:
                    related_id = next(
                        (
                            request["id"]
                            for request in requests_found
                            if request["id"] in related_ids
                        ),
                        None,
                    )

                target_request = next(
                    (
                        request
                        for request in requests_found
                        if request["id"] == related_id
                    ),
                    None,
                )
                details = {
                    "solicitud_relacionada_id": related_id,
                    "campos_actualizados": [],
                }
                if target_request is None:
                    update_reason = (
                        "OpenRouter interpretó el correo como una actualización, "
                        "pero no se pudo relacionar con una solicitud conocida."
                    )
                    events.append(
                        record(
                            state,
                            identifier=identifier,
                            node="actualizar_solicitud_desde_correo",
                            decision="error_actualizacion_sin_referencia",
                            reason=update_reason,
                            details={
                                "referencias": sorted(related_ids),
                                "solicitud_sugerida_por_modelo": related_id,
                            },
                        )
                    )
                else:
                    before = {
                        key: target_request.get(key)
                        for key in (
                            "cliente",
                            "rol",
                            "cantidad_posiciones",
                            "posiciones",
                            "pm_responsable",
                            "estado",
                        )
                    }
                    for key in ("cliente", "rol", "cantidad_posiciones", "pm_responsable"):
                        value = llm_result.get(key) if llm_result else None
                        if value is not None:
                            target_request[key] = value
                    if llm_result and llm_result.get("posiciones"):
                        target_request["posiciones"] = llm_result["posiciones"]
                        if target_request.get("cantidad_posiciones") is None:
                            target_request["cantidad_posiciones"] = sum(
                                item["cantidad_posiciones"]
                                for item in llm_result["posiciones"]
                            )
                        if not target_request.get("rol"):
                            target_request["rol"] = " / ".join(
                                dict.fromkeys(
                                    item["rol"] for item in llm_result["posiciones"]
                                )
                            )
                    if (
                        not target_request.get("pm_responsable")
                        and target_request.get("cliente")
                    ):
                        target_request["pm_responsable"] = pm_by_client.get(
                            normalize_text(target_request["cliente"])
                        )
                    target_request["datos_faltantes"] = [
                        field
                        for field, value in (
                            ("cliente", target_request.get("cliente")),
                            ("rol", target_request.get("rol")),
                            ("cantidad_posiciones", target_request.get("cantidad_posiciones")),
                            ("pm_responsable", target_request.get("pm_responsable")),
                        )
                        if value is None
                    ]
                    target_request["preguntas_aclaracion"] = (
                        llm_result.get("preguntas_aclaracion", [])
                        if llm_result
                        else []
                    ) or clarification_questions(target_request)
                    reported_status = (
                        llm_result.get("estado_solicitud")
                        if llm_result
                        else infer_request_status(message_text(email))
                    )
                    if reported_status:
                        state_labels = {
                            "escalada": "Escalada",
                            "confirmada": "Confirmada",
                            "cerrada": "Cerrada",
                        }
                        target_request["estado"] = state_labels[reported_status]
                        target_request["fecha_cambio_estado"] = datetime.now(
                            timezone.utc
                        ).isoformat()
                        if reported_status == "confirmada":
                            target_request["fecha_confirmacion"] = (
                                target_request["fecha_cambio_estado"]
                            )
                        elif reported_status == "cerrada":
                            target_request["fecha_cierre"] = (
                                target_request["fecha_cambio_estado"]
                            )
                    updates = {
                        key: target_request.get(key)
                        for key in before
                        if target_request.get(key) != before[key]
                    }
                    target_request.setdefault("actualizaciones", []).append(
                        {"message_id": identifier, "campos": updates}
                    )
                    details = {
                        "solicitud_relacionada_id": related_id,
                        "campos_actualizados": updates,
                        "datos_faltantes": target_request["datos_faltantes"],
                        "pm_responsable": target_request.get("pm_responsable"),
                    }
                    classified_email["solicitud_id"] = related_id
                    update_reason = (
                        llm_result.get("motivo")
                        if llm_result
                        else "El PM aportó información para actualizar la solicitud."
                    )
                    events.append(
                        record(
                            state,
                            identifier=identifier,
                            node="actualizar_solicitud_desde_correo",
                            decision="solicitud_actualizada_por_correo",
                            reason=update_reason,
                            details=details,
                        )
                    )
                    known_requests = [
                        {
                            key: request.get(key)
                            for key in (
                                "id",
                                "cliente",
                                "rol",
                                "cantidad_posiciones",
                                "posiciones",
                                "pm_responsable",
                                "datos_faltantes",
                                "estado",
                                "asunto",
                            )
                        }
                        for request in requests_found
                    ]
            elif category == "solicitud_posicion":
                details = extract_request_details(email, assignments=pm_by_client)
                if llm_result is not None:
                    for key in (
                        "cliente",
                        "rol",
                        "cantidad_posiciones",
                        "pm_responsable",
                    ):
                        llm_key = "rol" if key == "rol" else key
                        if llm_result.get(llm_key) is not None:
                            details[key] = llm_result[llm_key]
                    positions = llm_result.get("posiciones") or []
                    if positions:
                        details["posiciones"] = positions
                        if details["cantidad_posiciones"] is None:
                            details["cantidad_posiciones"] = sum(
                                item["cantidad_posiciones"]
                                for item in positions
                            )
                        if not details["rol"]:
                            details["rol"] = " / ".join(
                                dict.fromkeys(item["rol"] for item in positions)
                            )
                    if (
                        not details["pm_responsable"]
                        and details["cliente"]
                    ):
                        details["pm_responsable"] = pm_by_client.get(
                            normalize_text(details["cliente"])
                        )
                    details["datos_faltantes"] = [
                        key
                        for key, value in (
                            ("cliente", details["cliente"]),
                            ("rol", details["rol"]),
                            ("cantidad_posiciones", details["cantidad_posiciones"]),
                            ("pm_responsable", details["pm_responsable"]),
                        )
                        if value is None
                    ]
                ai_confirmation_question = (
                    llm_result.get("pregunta_confirmacion")
                    if llm_result is not None
                    else None
                )
                ai_clarifications = (
                    llm_result.get("preguntas_aclaracion", [])
                    if llm_result is not None
                    else []
                )
                fingerprint = request_fingerprint(email)
                known_fingerprints.add(fingerprint)
                fingerprints.append(fingerprint)
                known_request_ids.add(identifier)
                request_ids.append(identifier)
                requests_found.append(
                    {
                        "id": identifier,
                        "remitente": get_sender(email),
                        "asunto": str(email.get("subject") or "(sin asunto)"),
                        "cliente": details["cliente"],
                        "rol": details["rol"],
                        "cantidad_posiciones": details["cantidad_posiciones"],
                        "pm_responsable": details["pm_responsable"],
                        "posiciones": details.get("posiciones", []),
                        "datos_faltantes": details["datos_faltantes"],
                        "metodo_clasificacion": classification_method,
                        "pregunta_confirmacion": (
                            ai_confirmation_question
                            or confirmation_question(
                                {
                                    **details,
                                    "asunto": email.get("subject"),
                                }
                            )
                        ),
                        "preguntas_aclaracion": (
                            ai_clarifications
                            or clarification_questions(
                                {
                                    **details,
                                    "datos_faltantes": details["datos_faltantes"],
                                }
                            )
                        ),
                        "perfil_pm": None,
                        "horas_esperadas_confirmacion": None,
                        "hora_confirmacion": None,
                        "confirmacion_usuario": None,
                        "estado": "Nueva",
                        "hora_notificacion": None,
                        "fecha_recepcion": (
                            str(
                                email.get("timestamp")
                                or email.get("received_at")
                                or email.get("created_at")
                                or datetime.now(timezone.utc).isoformat()
                            )
                        ),
                        "fecha_recordatorio": None,
                        "recordatorio_enviado": False,
                        "actualizaciones": [],
                    }
                )
                known_requests.append(
                    {
                        "id": identifier,
                        "cliente": details["cliente"],
                        "rol": details["rol"],
                        "cantidad_posiciones": details["cantidad_posiciones"],
                        "posiciones": details.get("posiciones", []),
                        "pm_responsable": details["pm_responsable"],
                        "datos_faltantes": details["datos_faltantes"],
                        "asunto": str(email.get("subject") or "(sin asunto)"),
                    }
                )
                classified_email["solicitud"] = {
                    key: value
                    for key, value in details.items()
                    if key != "datos_faltantes"
                }
            else:
                details = {}

            audit_details = {
                **details,
                "metodo_clasificacion": classification_method,
                "llamadas_llm": llm_calls,
            }
            if category == "solicitud_posicion":
                audit_details["pregunta_confirmacion"] = requests_found[-1][
                    "pregunta_confirmacion"
                ]
                audit_details["preguntas_aclaracion"] = requests_found[-1][
                    "preguntas_aclaracion"
                ]
            elif category == "actualizacion_solicitud":
                audit_details.update(details)
            events.append(
                record(
                    state,
                    identifier=identifier,
                    node="clasificar_por_reglas",
                    decision=category,
                    reason=reason,
                    details=audit_details,
                )
            )

        return {
            "correos_clasificados": [*state["correos_clasificados"], *classified],
            "clasificacion_actual": classified,
            "solicitudes": requests_found,
            "correos_procesados": processed_ids,
            "firmas_solicitudes": fingerprints,
            "ids_solicitudes": request_ids,
            "llamadas_llm": llm_calls,
            "eventos": [*state["eventos"], *events],
        }

    def notify_pm_simulated(state: AgentState) -> dict[str, Any]:
        requests_by_id = {
            request["id"]: {**request}
            for request in state["solicitudes"]
        }
        events: list[dict[str, Any]] = []
        for email in state["clasificacion_actual"]:
            identifier = get_message_id(email)
            if email["categoria"] == "solicitud_posicion":
                request = requests_by_id[identifier]
                recipient = request["pm_responsable"]
                questions = request.get("preguntas_aclaracion", [])
                if recipient:
                    request["hora_notificacion"] = state["hora_actual"]
                    request["fecha_notificacion"] = datetime.now(
                        timezone.utc
                    ).isoformat()
                    if state["simular_confirmacion_automatica"]:
                        request["estado"] = "Notificada"
                    missing = request.get("datos_faltantes", [])
                    print(
                        "[NOTIFICACIÓN SIMULADA] "
                        f"Para: {recipient} | Cliente: {request['cliente'] or 'No identificado'} | "
                        f"Rol: {request['rol'] or 'No identificado'} | "
                        f"Cantidad: {request['cantidad_posiciones'] or 'No identificada'} | "
                        f"{request['asunto']} | correo {identifier}"
                    )
                    if missing:
                        print(
                            "  Información por completar: "
                            + ", ".join(missing)
                        )
                        for question in questions:
                            print(f"  El PM puede responder: {question}")
                        print(
                            "  Escribe `pm "
                            f"{next(index for index, item in enumerate(state['solicitudes'], 1) if item['id'] == identifier)}"
                            "` para aportar la respuesta del PM."
                        )
                    decision = "notificacion_simulada"
                    reason = (
                        f"Notificación simulada al PM {recipient} en la hora "
                        f"{state['hora_actual']}; no se envió ningún correo real."
                        + (
                            " Quedan datos pendientes: " + ", ".join(missing) + "."
                            if missing
                            else ""
                        )
                    )
                else:
                    decision = "notificacion_pendiente"
                    reason = (
                        "No se pudo notificar porque no se encontró un PM responsable "
                        "en el correo ni en la tabla de asignación."
                    )
                    print(
                        "[NOTIFICACIÓN PENDIENTE] "
                        f"Sin PM asignado | Cliente: {request['cliente'] or 'No identificado'} | "
                        f"{request['asunto']} | correo {identifier}"
                    )
            else:
                decision = "notificacion_omitida"
                reason = f"No se notifica porque la clasificación fue {email['categoria']}."
            events.append(
                record(
                    state,
                    identifier=identifier,
                    node="notificar_pm_simulado",
                    decision=decision,
                    reason=reason,
                    details=(
                        {
                            "cliente": request["cliente"],
                            "pm_responsable": request["pm_responsable"],
                            "hora_notificacion": request["hora_notificacion"],
                            "estado": request["estado"],
                            "datos_faltantes": request.get("datos_faltantes", []),
                            "preguntas_aclaracion": questions,
                        }
                        if email["categoria"] == "solicitud_posicion"
                        else None
                    ),
                )
            )
        return {
            "solicitudes": list(requests_by_id.values()),
            "eventos": [*state["eventos"], *events],
        }

    def advance_clock(state: AgentState) -> dict[str, Any]:
        return {"hora_actual": state["hora_actual"] + 1}

    def check_pm_confirmation(state: AgentState) -> dict[str, Any]:
        if not state["simular_confirmacion_automatica"]:
            return {}
        requests_by_id = {
            request["id"]: {**request}
            for request in state["solicitudes"]
        }
        events: list[dict[str, Any]] = []

        for request in requests_by_id.values():
            if (
                not request["estado"].startswith("Notificada")
                or request.get("confirmacion_usuario") is not None
                or request.get("datos_faltantes")
            ):
                continue

            profile = profiles_by_pm.get(
                normalize_text(str(request.get("pm_responsable") or "")),
                {"perfil": "normal", "horas_confirmacion": None},
            )
            delay = simulated_confirmation_delay(
                request,
                profile,
                seed=simulation_seed,
            )
            request["perfil_pm"] = profile.get("perfil", "normal")
            request["horas_esperadas_confirmacion"] = delay

            if delay is None:
                continue
            elapsed_hours = state["hora_actual"] - request["hora_notificacion"]
            if elapsed_hours < delay:
                continue

            request["estado"] = "Confirmada"
            request["hora_confirmacion"] = state["hora_actual"]
            request["confirmacion_simulada"] = True
            event = record(
                state,
                identifier=request["id"],
                node="consultar_confirmacion_pm",
                decision="pm_confirmo_simulado",
                reason=(
                    f"El perfil {request['perfil_pm']} confirma tras "
                    f"{elapsed_hours} horas."
                ),
                details={
                    "pm_responsable": request["pm_responsable"],
                    "perfil_pm": request["perfil_pm"],
                    "hora_notificacion": request["hora_notificacion"],
                    "hora_confirmacion": state["hora_actual"],
                },
            )
            events.append(event)
            print(
                f"[CONFIRMACIÓN SIMULADA] {request['pm_responsable']} confirmó "
                f"la recepción de: {request['asunto']}."
            )

        return {
            "solicitudes": list(requests_by_id.values()),
            "eventos": [*state["eventos"], *events],
        }

    workflow = StateGraph(AgentState)
    workflow.add_node("leer_bandeja", read_inbox)
    workflow.add_node("clasificar_por_reglas", classify_emails)
    workflow.add_node("notificar_pm_simulado", notify_pm_simulated)
    workflow.add_node("consultar_confirmacion_pm", check_pm_confirmation)
    workflow.add_node("avanzar_reloj", advance_clock)
    workflow.add_edge(START, "leer_bandeja")
    workflow.add_edge("leer_bandeja", "clasificar_por_reglas")
    workflow.add_edge("clasificar_por_reglas", "notificar_pm_simulado")
    workflow.add_edge("notificar_pm_simulado", "consultar_confirmacion_pm")
    workflow.add_edge("consultar_confirmacion_pm", "avanzar_reloj")
    workflow.add_edge("avanzar_reloj", END)
    return workflow.compile()
