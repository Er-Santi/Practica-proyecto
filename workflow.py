import re
from datetime import datetime, timezone
from typing import Any, Mapping

from langgraph.graph import END, START, StateGraph

from agent_config import AgentConfig, load_agent_config
from agent_state import AgentState
from agentmail_client import AgentMailClient
from audit_log import JsonlAuditLog, audit_error
from email_utils import (
    _ordered_by_timestamp, classify_by_rules, clarification_questions,
    confirmation_question, extract_request_details, get_message_id, get_sender,
    get_sender_address,
    associate_pm_email_addresses, infer_request_status, load_assignments,
    load_director_assignments, load_pm_profiles, message_text,
    normalize_text, request_fingerprint,
    simulated_confirmation_delay,
)
from exceptions import AgentMailError, OpenRouterError
from openrouter_client import OpenRouterClient
from settings import (
    ASSIGNMENT_REPORT_PATH,
    DEFAULT_SIMULATION_SEED,
    configured_max_llm_calls,
)
from pm_assignment import (
    build_pm_roster,
    canonical_pm_name,
    count_open_projects,
    is_pm_eligible,
    select_least_loaded_pm,
)
from sla import process_due_sla_actions

def build_graph(
    client: AgentMailClient,
    audit: JsonlAuditLog,
    *,
    assignments: Mapping[str, str] | None = None,
    directors: Mapping[str, str] | None = None,
    llm: OpenRouterClient | None = None,
    pm_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    simulation_seed: int = DEFAULT_SIMULATION_SEED,
    config: AgentConfig | None = None,
):
    agent_config = config or load_agent_config()
    llm_call_limit = configured_max_llm_calls(agent_config.max_llm_calls)
    pm_by_client = (
        {normalize_text(str(client)): str(pm) for client, pm in assignments.items()}
        if assignments is not None
        else load_assignments()
    )
    director_by_client = (
        {normalize_text(str(client)): str(director) for client, director in directors.items()}
        if directors is not None
        else load_director_assignments()
    )
    assignment_issue_report = JsonlAuditLog(
        audit.path.with_name(ASSIGNMENT_REPORT_PATH.name)
    )
    profiles_by_pm = (
        {
            normalize_text(str(pm)): {
                **dict(profile),
                "nombre": str(profile.get("nombre") or pm).strip(),
            }
            for pm, profile in pm_profiles.items()
        }
        if pm_profiles is not None
        else load_pm_profiles(pm_addresses=pm_by_client.values())
    )
    associate_pm_email_addresses(profiles_by_pm, pm_by_client.values())

    def available_pm_roster() -> tuple[str, ...]:
        return build_pm_roster(pm_by_client, profiles_by_pm)

    def sender_pm_reply_name(email: Mapping[str, Any]) -> str | None:
        sender_address = get_sender_address(email)
        if not sender_address:
            return None

        def addresses_for(profile: Mapping[str, Any]) -> list[str]:
            addresses = profile.get("emails") or []
            if isinstance(addresses, str):
                addresses = [addresses]
            return [
                str(address).strip().casefold()
                for address in addresses
            ]

        profile = next(
            (
                profile
                for profile in profiles_by_pm.values()
                if sender_address in addresses_for(profile)
            ),
            None,
        )
        if profile is None:
            return None
        headers = email.get("headers")
        header_values = (
            {
                normalize_text(str(key)): str(value)
                for key, value in headers.items()
            }
            if isinstance(headers, Mapping)
            else {}
        )
        has_thread_metadata = bool(
            email.get("in_reply_to")
            or email.get("references")
            or header_values.get("in reply to")
            or header_values.get("references")
        )
        has_reply_subject = bool(
            re.match(
                r"^\s*(?:(?:re|fw|fwd)\s*:\s*)+",
                str(email.get("subject") or ""),
                flags=re.IGNORECASE,
            )
        )
        if not has_thread_metadata and not has_reply_subject:
            return None
        return str(profile.get("nombre") or "")

    def related_request_for_pm_reply(
        email: Mapping[str, Any],
        pm_name: str,
        requests: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        canonical_sender = normalize_text(
            canonical_pm_name(pm_name, profiles_by_pm)
        )
        assigned_requests = [
            request
            for request in requests
            if request.get("pm_responsable")
            and request.get("estado") not in {"Cerrada", "Completada"}
            and normalize_text(
                canonical_pm_name(
                    str(request["pm_responsable"]),
                    profiles_by_pm,
                )
            )
            == canonical_sender
        ]
        reply_subject = normalize_text(str(email.get("subject") or ""))
        reply_subject = re.sub(
            r"^(?:(?:re|fw|fwd)\s*:\s*)+",
            "",
            reply_subject,
        )
        reply_subject = re.sub(
            r"(?:recordatorio de seguimiento:\s*)+",
            "",
            reply_subject,
        )
        reply_subject = re.sub(r"\basunto:\s*", "", reply_subject)
        subject_matches = [
            request
            for request in assigned_requests
            if normalize_text(str(request.get("asunto") or ""))
            .removeprefix("asunto: ")
            in reply_subject
        ]
        if len(subject_matches) == 1:
            return subject_matches[0]
        if len(assigned_requests) == 1:
            return assigned_requests[0]
        return None

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
        if state.get("simulation_tick_only"):
            return {
                "correos_nuevos": [],
                "simulation_tick_only": False,
                "correos_evento": [],
                "clasificacion_actual": [],
            }
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
            sender_address = get_sender_address(email)
            configured_inbox = str(client.inbox_id).strip().casefold()
            if sender_address and sender_address == configured_inbox:
                classified_email = {
                    **email,
                    "categoria": "correo_saliente_ignorado",
                }
                classified.append(classified_email)
                processed_ids.append(identifier)
                events.append(
                    record(
                        state,
                        identifier=identifier,
                        node="clasificar_por_reglas",
                        decision="correo_saliente_ignorado",
                        reason=(
                            "Se ignoró un mensaje enviado desde el inbox del "
                            "agente; no es una solicitud entrante."
                        ),
                        details={"remitente": sender_address},
                    )
                )
                continue
            pm_reply_name = sender_pm_reply_name(email)
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

            if pm_reply_name is not None:
                category = "actualizacion_solicitud"
                reason = (
                    f"Respuesta en hilo de {pm_reply_name}; se procesa como "
                    "actualización para no volver a crear una solicitud."
                )
                classification_method = "reglas_respuesta_pm"
            elif llm_result is not None:
                if (
                    rule_category == "solicitud_posicion"
                    and llm_result["categoria"] == "no_relacionado"
                ):
                    category = rule_category
                    reason = (
                        f"Las reglas encontraron una necesidad explícita de personal "
                        f"({rule_reason}), aunque OpenRouter la marcó como "
                        "no relacionada."
                    )
                    classification_method = "reglas_validacion"
                else:
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
                headers = email.get("headers")
                if isinstance(headers, Mapping):
                    for header_name, header_value in headers.items():
                        if normalize_text(str(header_name)) in {
                            "in reply to",
                            "references",
                        }:
                            related_ids.update(
                                str(value)
                                for value in re.findall(
                                    r"<[^>]+>|[^\s,]+",
                                    str(header_value),
                                )
                            )
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
                if related_id is None and pm_reply_name is not None:
                    related_request = related_request_for_pm_reply(
                        email,
                        pm_reply_name,
                        requests_found,
                    )
                    if related_request is not None:
                        related_id = related_request["id"]

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
                    for key in ("cliente", "rol", "cantidad_posiciones"):
                        value = llm_result.get(key) if llm_result else None
                        if value is not None:
                            target_request[key] = value
                    proposed_pm = (
                        llm_result.get("pm_responsable")
                        if llm_result
                        else None
                    )
                    if proposed_pm and normalize_text(str(proposed_pm)) != normalize_text(
                        str(target_request.get("pm_responsable") or "")
                    ):
                        pm_load = count_open_projects(
                            requests_found,
                            str(proposed_pm),
                            exclude_request_id=target_request["id"],
                            pm_profiles=profiles_by_pm,
                        )
                        proposed_pm_eligible = is_pm_eligible(
                            str(proposed_pm),
                            profiles_by_pm,
                        )
                        if (
                            pm_load < agent_config.max_open_projects_per_pm
                            and proposed_pm_eligible
                        ):
                            target_request["pm_responsable"] = str(proposed_pm)
                        else:
                            events.append(
                                record(
                                    state,
                                    identifier=identifier,
                                    node="actualizar_solicitud_desde_correo",
                                    decision="asignacion_pm_rechazada_por_limite",
                                    reason=(
                                        f"Se rechazó asignar al PM {proposed_pm}: "
                                        + (
                                            f"tiene {pm_load} proyectos abiertos."
                                            if proposed_pm_eligible
                                            else "está marcado como no elegible."
                                        )
                                    ),
                                    details={
                                        "pm_responsable": proposed_pm,
                                        "proyectos_abiertos": pm_load,
                                        "elegible": proposed_pm_eligible,
                                        "limite_proyectos_abiertos": (
                                            agent_config.max_open_projects_per_pm
                                        ),
                                    },
                                )
                            )
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
                        selected_pm, _ = select_least_loaded_pm(
                            requests_found,
                            available_pm_roster(),
                            maximum_open_projects=(
                                agent_config.max_open_projects_per_pm
                            ),
                            pm_profiles=profiles_by_pm,
                        )
                        target_request["pm_responsable"] = selected_pm
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
                    human_review_fields = [
                        field
                        for field in ("rol", "cantidad_posiciones")
                        if field in target_request["datos_faltantes"]
                    ]
                    was_under_human_review = target_request.get(
                        "revision_humana",
                        False,
                    )
                    target_request["revision_humana"] = bool(
                        human_review_fields
                    )
                    target_request["campos_revision_humana"] = (
                        human_review_fields
                    )
                    if was_under_human_review and not human_review_fields:
                        resolved_at = datetime.now(timezone.utc).isoformat()
                        target_request["fecha_revision_humana_resuelta"] = (
                            resolved_at
                        )
                        if target_request.get("estado") == "Nueva":
                            target_request["estado"] = "Notificada"
                            target_request["hora_notificacion"] = (
                                state["hora_actual"]
                            )
                            target_request["fecha_notificacion"] = resolved_at
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
                        if (
                            reported_status == "escalada"
                            and target_request.get(
                                "nivel_maximo_escalamiento",
                                "Ninguno",
                            )
                            == "Ninguno"
                        ):
                            target_request["nivel_maximo_escalamiento"] = "PM"
                        target_request["fecha_cambio_estado"] = datetime.now(
                            timezone.utc
                        ).isoformat()
                        if reported_status == "confirmada":
                            target_request["hora_confirmacion"] = state["hora_actual"]
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
                selected_pm, pm_loads = select_least_loaded_pm(
                    requests_found,
                    available_pm_roster(),
                    maximum_open_projects=agent_config.max_open_projects_per_pm,
                    pm_profiles=profiles_by_pm,
                )
                details["pm_responsable"] = selected_pm
                details["datos_faltantes"] = [
                    key
                    for key, value in (
                        ("cliente", details["cliente"]),
                        ("rol", details["rol"]),
                        ("cantidad_posiciones", details["cantidad_posiciones"]),
                        ("pm_responsable", selected_pm),
                    )
                    if value is None
                ]
                details["asignacion_automatica"] = {
                    "pm_asignado": selected_pm,
                    "cargas_abiertas": pm_loads,
                    "limite_proyectos_abiertos": (
                        agent_config.max_open_projects_per_pm
                    ),
                }
                if selected_pm is None:
                    issue_event = record(
                        state,
                        identifier=identifier,
                        node="asignacion_automatica_pm",
                        decision="sin_pm_disponible",
                        reason=(
                            "No hay PM configurado con menos de "
                            f"{agent_config.max_open_projects_per_pm} proyectos "
                            "abiertos; la solicitud queda pendiente de asignación."
                        ),
                        details={
                            "cliente": details["cliente"],
                            "cargas_abiertas": pm_loads,
                            "limite_proyectos_abiertos": (
                                agent_config.max_open_projects_per_pm
                            ),
                            "asignacion_reportada_en": str(
                                assignment_issue_report.path
                            ),
                        },
                    )
                    assignment_issue_report.append(issue_event)
                    events.append(issue_event)
                    print(
                        "[SIN PM DISPONIBLE] "
                        f"{email.get('subject') or identifier}. "
                        "Se escribió un reporte en "
                        f"{assignment_issue_report.path}."
                    )
                else:
                    events.append(
                        record(
                            state,
                            identifier=identifier,
                            node="asignacion_automatica_pm",
                            decision="pm_asignado_automaticamente",
                            reason=(
                                f"Se asignó al PM con menor carga abierta "
                                f"({pm_loads[selected_pm]} de "
                                f"{agent_config.max_open_projects_per_pm})."
                            ),
                            details=details["asignacion_automatica"],
                        )
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
                human_review_fields = [
                    field
                    for field in ("rol", "cantidad_posiciones")
                    if field in details["datos_faltantes"]
                ]
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
                        "revision_humana": bool(human_review_fields),
                        "campos_revision_humana": human_review_fields,
                        "correo_revision_humana_enviado": False,
                        "perfil_pm": None,
                        "horas_esperadas_confirmacion": None,
                        "hora_confirmacion": None,
                        "confirmacion_usuario": None,
                        "estado": "Nueva",
                        "nivel_maximo_escalamiento": "Ninguno",
                        "escalado_director": False,
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
        llm_calls = state["llamadas_llm"]
        for email in state["clasificacion_actual"]:
            identifier = get_message_id(email)
            human_review_fields: list[str] = []
            writing_method: str | None = None
            if email["categoria"] == "solicitud_posicion":
                request = requests_by_id[identifier]
                recipient = request["pm_responsable"]
                questions = request.get("preguntas_aclaracion", [])
                human_review_fields = [
                    field
                    for field in ("rol", "cantidad_posiciones")
                    if field in (request.get("datos_faltantes") or [])
                ]
                if human_review_fields:
                    request["revision_humana"] = True
                    request["campos_revision_humana"] = human_review_fields
                    profile = profiles_by_pm.get(
                        normalize_text(str(recipient or "")),
                        {},
                    )
                    profile_emails = profile.get("emails") or []
                    if isinstance(profile_emails, str):
                        profile_emails = [profile_emails]
                    raw_addresses = [
                        *profile_emails,
                        profile.get("correo"),
                        recipient,
                    ]
                    recipient_email = next(
                        (
                            str(address).strip().casefold()
                            for address in raw_addresses
                            if isinstance(address, str)
                            and re.fullmatch(
                                r"[^@\s]+@[^@\s]+\.[^@\s]+",
                                address.strip(),
                            )
                        ),
                        None,
                    )
                    if not recipient:
                        decision = "revision_humana_sin_pm"
                        reason = (
                            "La solicitud requiere revisión humana porque falta "
                            "el rol o la cantidad, pero no hay un PM asignado."
                        )
                        request["revision_humana_motivo"] = reason
                    elif recipient_email is None:
                        decision = "revision_humana_sin_correo_pm"
                        reason = (
                            f"La solicitud requiere revisión humana y se asignó "
                            f"a {recipient}, pero no hay un correo válido "
                            "configurado para ese PM."
                        )
                        request["revision_humana_motivo"] = reason
                    elif state["simular_confirmacion_automatica"]:
                        decision = "revision_humana_pendiente"
                        reason = (
                            "La solicitud requiere revisión humana; en modo "
                            "simulación no se envió un correo real al PM."
                        )
                        request["revision_humana_motivo"] = reason
                        print(
                            "[REVISIÓN HUMANA SIMULADA] "
                            f"Para: {recipient} ({recipient_email}) | "
                            "Falta completar: "
                            + ", ".join(human_review_fields)
                        )
                    else:
                        body: str | None = None
                        writing_method = "plantilla"
                        if llm is not None:
                            writer = getattr(
                                llm,
                                "write_clarification_email",
                                None,
                            )
                            for attempt in range(2):
                                if (
                                    llm_call_limit
                                    and llm_calls >= llm_call_limit
                                ):
                                    break
                                llm_calls += 1
                                try:
                                    if not callable(writer):
                                        raise OpenRouterError(
                                            "El cliente LLM no puede redactar "
                                            "correos de aclaración."
                                        )
                                    body = writer(email, request)
                                    writing_method = "llm"
                                    break
                                except OpenRouterError as exc:
                                    error_event = audit_error(
                                        audit,
                                        exc,
                                        node="redactar_aclaracion_llm",
                                        hora_simulada=state["hora_actual"],
                                        solicitud=identifier,
                                        details={
                                            "intento": attempt + 1,
                                            "respaldo_por_plantilla": True,
                                        },
                                    )
                                    events.append(error_event)

                            if (
                                body is None
                                and llm_call_limit
                                and llm_calls >= llm_call_limit
                            ):
                                events.append(
                                    audit_error(
                                        audit,
                                        "Se alcanzó el límite de llamadas a OpenRouter.",
                                        node="redactar_aclaracion_llm",
                                        hora_simulada=state["hora_actual"],
                                        solicitud=identifier,
                                        details={
                                            "limite": llm_call_limit,
                                            "respaldo_por_plantilla": True,
                                        },
                                    )
                                )

                        if body is None:
                            lines = [
                                f"Hola {recipient},",
                                "",
                                "Esta solicitud requiere revisión humana porque "
                                "falta información clave:",
                                f"ID: {identifier}",
                                f"Asunto: {request['asunto']}",
                                f"Remitente original: {get_sender(email)}",
                            ]
                            for label, key in (
                                ("Cliente", "cliente"),
                                ("Rol", "rol"),
                                ("Cantidad de posiciones", "cantidad_posiciones"),
                            ):
                                value = request.get(key)
                                if value is not None and str(value).strip():
                                    lines.append(f"{label}: {value}")
                            lines.extend(
                                [
                                    "",
                                    "Datos por confirmar: "
                                    + ", ".join(human_review_fields),
                                    "Por favor revisa el correo original y confirma "
                                    "el dato faltante o indica qué debemos preguntar "
                                    "al remitente.",
                                ]
                            )
                            if questions:
                                lines.extend(
                                    ["", "Preguntas sugeridas:", *questions]
                                )
                            original_body = str(
                                email.get("text")
                                or email.get("extracted_text")
                                or email.get("preview")
                                or ""
                            ).strip()
                            if original_body:
                                if len(original_body) > 2000:
                                    original_body = original_body[:1997] + "..."
                                lines.extend(
                                    ["", "Mensaje original:", original_body]
                                )
                            body = "\n".join(lines)

                        subject_text = " ".join(
                            str(request.get("asunto") or identifier).split()
                        )[:120]
                        subject = (
                            f"Revisión humana [Solicitud {identifier}]: "
                            f"{subject_text}"
                        )
                        client.send(recipient_email, subject, body)
                        request["correo_revision_humana_enviado"] = True
                        request["fecha_correo_revision_humana"] = (
                            datetime.now(timezone.utc).isoformat()
                        )
                        decision = "correo_revision_humana_enviado"
                        reason = (
                            f"Se envió al PM {recipient} un correo para revisar "
                            "la solicitud ambigua."
                        )
                        request["revision_humana_motivo"] = reason
                        print(
                            f"[REVISIÓN HUMANA] Correo enviado a {recipient} "
                            f"({recipient_email}); falta confirmar: "
                            f"{', '.join(human_review_fields)}."
                        )
                elif recipient:
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
                    node=(
                        "notificar_revision_humana_pm"
                        if human_review_fields
                        else "notificar_pm_simulado"
                    ),
                    decision=decision,
                    reason=reason,
                    details=(
                        {
                            "cliente": request["cliente"],
                            "pm_responsable": request["pm_responsable"],
                            "hora_notificacion": request.get("hora_notificacion"),
                            "estado": request["estado"],
                            "datos_faltantes": request.get("datos_faltantes", []),
                            "preguntas_aclaracion": questions,
                            "revision_humana": request.get(
                                "revision_humana",
                                False,
                            ),
                            "campos_revision_humana": human_review_fields,
                            "metodo_redaccion": writing_method,
                        }
                        if email["categoria"] == "solicitud_posicion"
                        else None
                    ),
                )
            )
        return {
            "solicitudes": list(requests_by_id.values()),
            "llamadas_llm": llm_calls,
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
                (
                    request["estado"]
                    not in {"Notificada", "Recordatorio enviado"}
                    and not (
                        request["estado"] == "Escalada"
                        and request.get("escalado_director", False)
                    )
                )
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
            notification_hour = request.get("hora_notificacion")
            if not isinstance(notification_hour, (int, float)):
                continue
            elapsed_hours = state["hora_actual"] - notification_hour
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
                f"[CONFIRMACIÓN SIMULADA] {request['pm_responsable']} "
                f"aceptó hacerse cargo de: {request['asunto']}."
            )

        return {
            "solicitudes": list(requests_by_id.values()),
            "eventos": [*state["eventos"], *events],
        }

    def apply_sla_thresholds(state: AgentState) -> dict[str, Any]:
        if not state["simular_confirmacion_automatica"]:
            return {}
        process_due_sla_actions(
            state,
            audit=audit,
            config=agent_config,
            directors=director_by_client,
            simulated=True,
        )
        return {}

    workflow = StateGraph(AgentState)
    workflow.add_node("leer_bandeja", read_inbox)
    workflow.add_node("clasificar_por_reglas", classify_emails)
    workflow.add_node("notificar_pm_simulado", notify_pm_simulated)
    workflow.add_node("consultar_confirmacion_pm", check_pm_confirmation)
    workflow.add_node("aplicar_umbrales_sla", apply_sla_thresholds)
    workflow.add_node("avanzar_reloj", advance_clock)
    workflow.add_edge(START, "leer_bandeja")
    workflow.add_edge("leer_bandeja", "clasificar_por_reglas")
    workflow.add_edge("clasificar_por_reglas", "notificar_pm_simulado")
    workflow.add_edge("notificar_pm_simulado", "consultar_confirmacion_pm")
    workflow.add_edge("consultar_confirmacion_pm", "aplicar_umbrales_sla")
    workflow.add_edge("aplicar_umbrales_sla", "avanzar_reloj")
    workflow.add_edge("avanzar_reloj", END)
    return workflow.compile()
