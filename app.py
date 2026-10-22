#!/usr/bin/env python3
"""CLI entry point and backwards-compatible public imports for the agent."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from agentmail import AsyncAgentMail
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
    run_agent_forever,
    run_phase_one,
)
from settings import (
    API_BASE_URL,
    ASSIGNMENTS_PATH,
    AUDIT_PATH,
    DEFAULT_RECONNECT_INTERVAL_SECONDS,
    DEFAULT_SIMULATION_SEED,
    HEARTBEAT_INTERVAL_SECONDS,
    MAX_LLM_CALLS,
    OPENROUTER_API_URL,
    OPENROUTER_DEFAULT_MODEL,
    PM_PROFILES_PATH,
    REQUEST_TIMEOUT_SECONDS,
    configured_max_llm_calls,
)
from terminal import run_terminal_agent
from workflow import build_graph

__all__ = [
    "API_BASE_URL",
    "ASSIGNMENTS_PATH",
    "AUDIT_PATH",
    "AgentMailClient",
    "AgentMailError",
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
    "_extract_labeled_value",
    "_ordered_by_timestamp",
    "apply_human_confirmation",
    "apply_pm_update",
    "assign_pm_to_request",
    "build_graph",
    "clarification_questions",
    "classify_by_rules",
    "confirmation_question",
    "configured_max_llm_calls",
    "extract_request_details",
    "get_message_id",
    "get_sender",
    "initial_agent_state",
    "load_assignments",
    "load_pm_profiles",
    "message_text",
    "normalize_text",
    "request_fingerprint",
    "requests",
    "run_agent_forever",
    "run_phase_one",
    "run_terminal_agent",
    "serialize_agentmail_message",
    "simulated_confirmation_delay",
]


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Agente LangGraph continuo para leer y clasificar correos de AgentMail."
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
    use_llm_setting = os.environ.get("USE_LLM", "true").strip().casefold()
    if use_llm_setting not in {"true", "false", "1", "0", "yes", "no"}:
        error = ValueError("USE_LLM debe ser true o false.")
        audit_error(audit, error, node="validar_configuracion")
        parser.error(str(error))
    use_llm = use_llm_setting in {"true", "1", "yes"}
    if args.reconnect_seconds <= 0:
        error = ValueError("--reconnect-seconds debe ser mayor que cero.")
        audit_error(audit, error, node="validar_configuracion")
        parser.error(str(error))

    try:
        if not api_key or not inbox_id:
            raise ValueError(
                "Configura AGENTMAIL_API_KEY y AGENTMAIL_INBOX_ID en .env."
            )
        if use_llm and not openrouter_key:
            raise ValueError(
                "OpenRouter está activado; configura OPENROUTER_API_KEY en .env."
            )
        llm_call_limit = configured_max_llm_calls()
        assignments = load_assignments(args.assignments)
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

    print(f"Conectando por WebSocket al inbox de AgentMail: {inbox_id}")
    if llm is not None:
        print(
            f"Clasificación con OpenRouter activada: {args.model}. "
            "Cada correo nuevo se interpretará con el modelo; sus respuestas y "
            "contenido se envían al proveedor configurado."
        )
        if llm_call_limit:
            print(f"Límite configurado de llamadas OpenRouter: {llm_call_limit}.")
        else:
            print(
                "Llamadas OpenRouter sin límite predeterminado; define "
                "MAX_LLM_CALLS en .env para limitar el consumo."
            )
    else:
        print("OpenRouter está desactivado; se utilizarán las reglas de respaldo.")
    print(
        "Confirmación de PM simulada por perfil; puedes validar o corregir "
        "cada solicitud con `confirm <n> sí|no`."
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
        )
    except KeyboardInterrupt:
        print("\nAgente detenido.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
