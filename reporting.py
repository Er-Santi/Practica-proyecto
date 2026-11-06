from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agent_config import AgentConfig
from agent_state import AgentState
from sla import SLA_STOP_STATES


def build_sla_report(
    state: AgentState,
    *,
    config: AgentConfig,
    seed: int,
    llm_enabled: bool,
    hours_simulated: int | None = None,
) -> dict[str, Any]:
    requests: list[dict[str, Any]] = []
    for request in state["solicitudes"]:
        confirmation_hour = request.get("hora_confirmacion")
        if not isinstance(confirmation_hour, int):
            confirmation_hour = None
        complied = (
            request.get("estado") in SLA_STOP_STATES
            and confirmation_hour is not None
            and confirmation_hour <= config.sla_target_hours
        )
        requests.append(
            {
                "id": request["id"],
                "asunto": request["asunto"],
                "estado_final": request["estado"],
                "pm_confirmo": (
                    request.get("estado") in SLA_STOP_STATES
                    and confirmation_hour is not None
                ),
                "trabajo_finalizado": bool(request.get("fecha_completada")),
                "hora_confirmacion": confirmation_hour,
                "nivel_maximo_escalamiento": request.get(
                    "nivel_maximo_escalamiento",
                    "Ninguno",
                ),
                "cumplio_sla": complied,
            }
        )

    terminal = all(
        item["estado_final"] in SLA_STOP_STATES for item in requests
    )
    if not requests:
        stop_reason = "sin_solicitudes"
    elif terminal:
        stop_reason = "todas_confirmadas_o_cerradas"
    else:
        stop_reason = "maximo_horas_simuladas"

    compliant_count = sum(item["cumplio_sla"] for item in requests)
    return {
        "semilla": seed,
        "modo": "con_llm" if llm_enabled else "sin_llm",
        "horas_simuladas": (
            state["hora_actual"]
            if hours_simulated is None
            else hours_simulated
        ),
        "motivo_finalizacion": stop_reason,
        "limites": {
            "max_horas_simuladas": config.max_simulated_hours,
            "recursion_limit": config.recursion_limit,
            "max_llamadas_llm": config.max_llm_calls,
            "llamadas_llm_realizadas": state["llamadas_llm"],
        },
        "umbrales_sla": {
            "primer_recordatorio_pm_horas": config.reminder_after_hours,
            "segundo_recordatorio_pm_horas": config.second_reminder_after_hours,
            "escalamiento_director_horas": config.escalate_after_hours,
            "recordatorio_siguiente_nivel_horas": (
                config.next_level_reminder_after_hours
            ),
            "involucrar_comercial_delivery_horas": (
                config.commercial_delivery_after_hours
            ),
            "repetir_cada_horas": config.repeat_after_hours,
            "objetivo_sla_horas": config.sla_target_hours,
        },
        "resumen_sla": {
            "total_solicitudes": len(requests),
            "cumplieron": compliant_count,
            "no_cumplieron": len(requests) - compliant_count,
        },
        "solicitudes": requests,
    }


def write_sla_report(path: Path, report: dict[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError) as exc:
        raise OSError(f"No se pudo escribir el reporte SLA en {path}: {exc}") from exc
