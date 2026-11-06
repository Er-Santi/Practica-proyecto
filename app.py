#!/usr/bin/env python3
"""CLI entry point and backwards-compatible public imports for the agent."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import requests
from dotenv import load_dotenv

from agentmail import AsyncAgentMail
from agent_config import AgentConfig, load_agent_config
from agent_state import AgentState, initial_agent_state
from agentmail_client import AgentMailClient, serialize_agentmail_message
from audit_log import JsonlAuditLog, audit_error
from email_utils import (
    _ordered_by_timestamp,
    _extract_labeled_value,
    classify_by_rules,
    clarification_questions,
    confirmation_question,
    extract_request_details,
    get_message_id,
    get_sender,
    load_assignments,
    load_director_assignments,
    load_pm_profiles,
    message_text,
    normalize_text,
    request_fingerprint,
    simulated_confirmation_delay,
)
from exceptions import AgentMailError, OpenRouterError
from openrouter_client import OpenRouterClient
from service import (
    apply_human_confirmation,
    apply_pm_update,
    assign_pm_to_request,
    complete_request,
    close_request,
    mark_request_opened,
    process_due_reminders,
    run_agent_forever,
    run_phase_one,
    run_simulation,
)
from settings import (
    API_BASE_URL,
    ASSIGNMENT_REPORT_PATH,
    ASSIGNMENTS_PATH,
    AUDIT_PATH,
    CONFIG_PATH,
    DEFAULT_RECONNECT_INTERVAL_SECONDS,
    DEFAULT_SIMULATION_SEED,
    HEARTBEAT_INTERVAL_SECONDS,
    MAX_LLM_CALLS,
    OPENROUTER_API_URL,
    OPENROUTER_DEFAULT_MODEL,
    PM_PROFILES_PATH,
    REQUEST_TIMEOUT_SECONDS,
    STATE_PATH,
    configured_max_llm_calls,
)
from reporting import write_sla_report
from state_store import AgentStateStoreError
from terminal import run_terminal_agent
from workflow import build_graph

__all__ = [
    "API_BASE_URL",
    "ASSIGNMENTS_PATH",
    "ASSIGNMENT_REPORT_PATH",
    "AUDIT_PATH",
    "CONFIG_PATH",
    "AgentMailClient",
    "AgentMailError",
    "AgentConfig",
    "AgentState",
    "AsyncAgentMail",
    "DEFAULT_RECONNECT_INTERVAL_SECONDS",
    "DEFAULT_SIMULATION_SEED",
    "HEARTBEAT_INTERVAL_SECONDS",
    "JsonlAuditLog",
    "MAX_LLM_CALLS",
    "OPENROUTER_API_URL",
    "OPENROUTER_DEFAULT_MODEL",
    "OpenRouterClient",
    "OpenRouterError",
    "PM_PROFILES_PATH",
    "REQUEST_TIMEOUT_SECONDS",
    "STATE_PATH",
    "_extract_labeled_value",
    "_ordered_by_timestamp",
    "apply_human_confirmation",
    "apply_pm_update",
    "assign_pm_to_request",
    "build_graph",
    "clarification_questions",
    "classify_by_rules",
    "complete_request",
    "close_request",
    "confirmation_question",
    "configured_max_llm_calls",
    "extract_request_details",
    "get_message_id",
    "get_sender",
    "initial_agent_state",
    "load_assignments",
    "load_agent_config",
    "load_director_assignments",
    "load_pm_profiles",
    "mark_request_opened",
    "process_due_reminders",
    "message_text",
    "normalize_text",
    "request_fingerprint",
    "requests",
    "run_agent_forever",
    "run_phase_one",
    "run_simulation",
    "run_terminal_agent",
    "serialize_agentmail_message",
    "simulated_confirmation_delay",
    "write_sla_report",
]


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=(
            "Agente LangGraph para leer y clasificar AgentMail, con modo "
            "de simulación SLA acotado."
        )
    )
    parser.add_argument(
        "--reconnect-seconds",
        type=float,
        default=float(
            os.environ.get(
                "RECONNECT_INTERVAL_SECONDS",
                str(DEFAULT_RECONNECT_INTERVAL_SECONDS),
            )
        ),
        help=(
            "segundos antes de reconectar el WebSocket si se corta "
            f"(por defecto {DEFAULT_RECONNECT_INTERVAL_SECONDS})"
        ),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=AUDIT_PATH,
        help="ruta del log JSON Lines de auditoría (por defecto auditoria.jsonl)",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=STATE_PATH,
        help="ruta del estado persistente de solicitudes (por defecto estado_agente.json)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="umbrales SLA y límites de la corrida (por defecto config.yaml)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="simula la bandeja hasta finalizar las solicitudes o alcanzar el límite",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reporte_sla.json"),
        help="ruta del reporte final JSON en modo --simulate",
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=ASSIGNMENTS_PATH,
        help="CSV de asignaciones cliente-PM (por defecto asignacion.csv)",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=PM_PROFILES_PATH,
        help="CSV de perfiles de respuesta de PM (por defecto pms.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("SIMULATION_SEED", DEFAULT_SIMULATION_SEED)),
        help="semilla reproducible para tiempos simulados de respuesta",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENROUTER_MODEL", OPENROUTER_DEFAULT_MODEL),
        help=f"modelo de OpenRouter (por defecto {OPENROUTER_DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    audit = JsonlAuditLog(args.audit)
    api_key = os.environ.get("AGENTMAIL_API_KEY", "").strip()
    inbox_id = os.environ.get("AGENTMAIL_INBOX_ID", "").strip()
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if args.reconnect_seconds <= 0:
        error = ValueError("--reconnect-seconds debe ser mayor que cero.")
        audit_error(audit, error, node="validar_configuracion")
        parser.error(str(error))

    try:
        agent_config = load_agent_config(args.config)
        use_llm_setting = os.environ.get("USE_LLM")
        if use_llm_setting is None:
            use_llm = agent_config.llm_enabled
        else:
            normalized_llm_setting = use_llm_setting.strip().casefold()
            if normalized_llm_setting not in {
                "true",
                "false",
                "1",
                "0",
                "yes",
                "no",
            }:
                raise ValueError("USE_LLM debe ser true o false.")
            use_llm = normalized_llm_setting in {"true", "1", "yes"}

        if not api_key or not inbox_id:
            raise ValueError(
                "Configura AGENTMAIL_API_KEY y AGENTMAIL_INBOX_ID en .env."
            )
        if use_llm and not openrouter_key:
            raise ValueError(
                "OpenRouter está activado; configura OPENROUTER_API_KEY en .env."
            )
        agent_config = replace(
            agent_config,
            max_llm_calls=configured_max_llm_calls(agent_config.max_llm_calls),
        )
        assignments = load_assignments(args.assignments)
        directors = load_director_assignments(args.assignments)
        pm_profiles = load_pm_profiles(args.profiles)
        client = AgentMailClient(api_key, inbox_id)
        llm = OpenRouterClient(openrouter_key, args.model) if use_llm else None
    except ValueError as exc:
        audit_error(audit, exc, node="validar_configuracion")
        print(f"Error de configuración: {exc}")
        return 1
    except AgentMailError as exc:
        audit_error(audit, exc, node="inicializar_agentmail")
        print(f"No se pudo conectar con AgentMail: {exc}")
        return 1

    if args.simulate:
        print(f"Usando la bandeja AgentMail para la simulación: {inbox_id}")
    else:
        print(f"Conectando por WebSocket al inbox de AgentMail: {inbox_id}")
    if llm is not None:
        print(
            f"Clasificación con OpenRouter activada: {args.model}. "
            "Cada correo nuevo se interpretará con el modelo; sus respuestas y "
            "contenido se envían al proveedor configurado."
        )
        print(
            "Límite de llamadas OpenRouter por corrida: "
            f"{agent_config.max_llm_calls}."
        )
    else:
        print("OpenRouter está desactivado; se utilizarán las reglas de respaldo.")
    if agent_config.testing_enabled:
        print(
            "Modo de prueba SLA activo: avisos al PM a los "
            f"{agent_config.test_reminder_after_minutes} y "
            f"{agent_config.test_second_reminder_after_minutes} min; "
            "aprobación para escalar al director a los "
            f"{agent_config.test_escalate_after_minutes} min; recordatorio al "
            "director a los "
            f"{agent_config.test_next_level_reminder_after_minutes} min; "
            "Comercial/Delivery a los "
            f"{agent_config.test_commercial_delivery_after_minutes} min y "
            f"repetición cada {agent_config.test_repeat_after_minutes} min; "
            f"objetivo SLA {agent_config.test_sla_target_minutes} min; "
            "comprobación cada "
            f"{agent_config.sla_check_interval_seconds} s."
        )
    else:
        print(
            "Modo SLA normal: avisos al PM a las "
            f"{agent_config.reminder_after_hours} y "
            f"{agent_config.second_reminder_after_hours} h; aprobación para "
            "escalar al director a las "
            f"{agent_config.escalate_after_hours} h; recordatorio al director "
            f"a las {agent_config.next_level_reminder_after_hours} h; "
            "Comercial/Delivery a las "
            f"{agent_config.commercial_delivery_after_hours} h y repetición "
            f"cada {agent_config.repeat_after_hours} h."
        )

    if args.simulate:
        try:
            report = run_simulation(
                client,
                audit_path=args.audit,
                assignments=assignments,
                directors=directors,
                llm=llm,
                pm_profiles=pm_profiles,
                simulation_seed=args.seed,
                config=agent_config,
            )
            write_sla_report(args.report, report)
        except (AgentMailError, OSError, ValueError) as exc:
            audit_error(audit, exc, node="ejecutar_simulacion_sla")
            print(f"No se pudo completar la simulación: {exc}")
            return 1
        print(
            f"Simulación terminada: {report['motivo_finalizacion']} "
            f"tras {report['horas_simuladas']} horas simuladas."
        )
        print(f"Reporte SLA: {args.report}")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print(
        "Confirmación de PM simulada por perfil; puedes validar o corregir "
        "si el PM se hará cargo con `confirm <n> sí|no`; marca el trabajo "
        "terminado con `complete <n>`."
    )
    try:
        run_terminal_agent(
            client,
            audit_path=args.audit,
            reconnect_interval=args.reconnect_seconds,
            assignments=assignments,
            llm=llm,
            pm_profiles=pm_profiles,
            simulation_seed=args.seed,
            state_path=args.state,
            config=agent_config,
            directors=directors,
        )
    except KeyboardInterrupt:
        print("\nAgente detenido.")
    except AgentStateStoreError as exc:
        audit_error(audit, exc, node="cargar_estado_agente")
        print(f"No se pudo cargar el estado guardado: {exc}")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
