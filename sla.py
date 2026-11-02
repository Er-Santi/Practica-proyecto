from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from agent_config import AgentConfig
from agent_state import AgentState
from audit_log import JsonlAuditLog, audit_error
from email_utils import normalize_text

FINAL_STATES = {"Confirmada", "Cerrada"}


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


def process_due_sla_actions(
    state: AgentState,
    *,
    audit: JsonlAuditLog,
    config: AgentConfig,
    directors: Mapping[str, str] | None = None,
    simulated: bool,
    now: datetime | None = None,
) -> int:
    director_by_client = {
        normalize_text(str(client)): str(director)
        for client, director in (directors or {}).items()
    }
    actions = 0

    for request in state["solicitudes"]:
        if (
            request.get("estado") in FINAL_STATES
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
        testing_minutes = not simulated and config.testing_enabled
        time_unit = "minutos" if testing_minutes else "horas"
        elapsed_for_thresholds = elapsed * 60 if testing_minutes else elapsed
        reminder_threshold = (
            config.test_reminder_after_minutes
            if testing_minutes
            else config.reminder_after_hours
        )
        escalation_threshold = (
            config.test_escalate_after_minutes
            if testing_minutes
            else config.escalate_after_hours
        )
        common_details = {
            "horas_transcurridas": elapsed,
            "tiempo_transcurrido_umbral": elapsed_for_thresholds,
            "unidad_umbral": time_unit,
            "pm_responsable": request.get("pm_responsable"),
            "reloj_simulado": simulated,
            "notificacion_simulada": True,
        }

        if (
            request.get("pm_responsable")
            and elapsed_for_thresholds >= reminder_threshold
            and not request.get("recordatorio_enviado", False)
        ):
            request["recordatorio_enviado"] = True
            request["hora_recordatorio"] = state["hora_actual"] if simulated else None
            request["fecha_recordatorio"] = (
                None if simulated else (now or datetime.now(timezone.utc)).isoformat()
            )
            request["nivel_maximo_escalamiento"] = "PM"
            if request.get("estado") not in {"Escalada", *FINAL_STATES}:
                request["estado"] = "Recordatorio enviado"
            event = {
                "hora_simulada": state["hora_actual"],
                "solicitud": request["id"],
                "nodo": "recordatorio_sla_pm",
                "decision": "recordatorio_enviado",
                "motivo": (
                    f"Se alcanzó el umbral de {reminder_threshold} {time_unit} "
                    "sin confirmación; recordatorio simulado al PM."
                ),
                "datos": {
                    **common_details,
                    "umbral": reminder_threshold,
                    "nivel_maximo_escalamiento": "PM",
                },
            }
            audit.append(event)
            state["eventos"].append(event)
            print(
                f"[RECORDATORIO SIMULADO] PM {request['pm_responsable']} | "
                f"{request.get('asunto', request['id'])}"
            )
            actions += 1

        if (
            elapsed_for_thresholds >= escalation_threshold
            and not request.get("escalado_director", False)
        ):
            director = director_by_client.get(
                normalize_text(str(request.get("cliente") or ""))
            )
            request["escalado_director"] = True
            request["hora_escalamiento_director"] = (
                state["hora_actual"] if simulated else None
            )
            request["fecha_escalamiento_director"] = (
                None if simulated else (now or datetime.now(timezone.utc)).isoformat()
            )
            request["nivel_maximo_escalamiento"] = "Director"
            request["estado"] = "Escalada"
            event = {
                "hora_simulada": state["hora_actual"],
                "solicitud": request["id"],
                "nodo": "escalar_sla_director",
                "decision": (
                    "escalamiento_director_simulado"
                    if director
                    else "escalamiento_director_sin_destinatario"
                ),
                "motivo": (
                    f"Se alcanzó el umbral de {escalation_threshold} {time_unit} "
                    "sin confirmación; se escaló al director de forma simulada."
                    if director
                    else (
                        f"Se alcanzó el umbral de {escalation_threshold} {time_unit} "
                        "sin confirmación, pero no hay director asignado para el cliente."
                    )
                ),
                "datos": {
                    **common_details,
                    "director": director,
                    "cliente": request.get("cliente"),
                    "umbral": escalation_threshold,
                    "nivel_maximo_escalamiento": "Director",
                },
            }
            audit.append(event)
            state["eventos"].append(event)
            print(
                f"[ESCALAMIENTO SIMULADO] Director "
                f"{director or '(sin asignar)'} | "
                f"{request.get('asunto', request['id'])}"
            )
            actions += 1

    return actions
