import threading
from queue import Empty, Full, Queue
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from agent_config import AgentConfig, load_agent_config
from agent_state import AgentState
from agentmail_client import AgentMailClient
from audit_log import JsonlAuditLog, audit_error
from email_utils import (
    get_message_id,
    get_sender,
    load_director_assignments,
    load_pm_profiles,
    normalize_text,
)
from email_notifications import send_lifecycle_notification
from exceptions import AgentMailError, OpenRouterError
from openrouter_client import OpenRouterClient
from pm_assignment import count_open_projects
from pm_assignment import build_pm_roster, is_pm_eligible
from service import (
    apply_human_confirmation,
    apply_pm_update,
    assign_pm_to_request,
    complete_request,
    close_request,
    mark_request_opened,
    run_agent_forever,
    set_pm_eligibility,
)
from settings import (
    DEFAULT_SIMULATION_SEED,
    PM_PROFILES_PATH,
    REQUEST_TIMEOUT_SECONDS,
    STATE_PATH,
    configured_max_llm_calls,
)
from state_store import AgentStateStoreError, load_agent_state, save_agent_state

COMPLETED_REQUEST_STATES = frozenset({"Cerrada"})


def completed_requests(state: AgentState) -> list[tuple[int, dict[str, Any]]]:
    return [
        (index, request)
        for index, request in enumerate(state["solicitudes"], start=1)
        if request.get("estado") in COMPLETED_REQUEST_STATES
    ]


def _audit_terminal_error(
    audit_path: Path,
    node: str,
    error: Exception,
    *,
    request_number: int | None = None,
) -> None:
    audit_error(
        JsonlAuditLog(audit_path),
        error,
        node=node,
        details={"numero_solicitud": request_number} if request_number else None,
    )


def display_message(message: dict[str, Any], index: int) -> None:
    print(
        f"\n[{index}] {message.get('subject') or '(sin asunto)'}"
        f"\n    Clasificación: {message.get('categoria', 'Pendiente')}"
        f"\n    De: {get_sender(message)}"
        f"\n    Hora: {message.get('timestamp') or message.get('received_at') or 'Desconocida'}"
        f"\n    ID: {get_message_id(message) or 'No disponible'}"
    )


def display_full_message(message: dict[str, Any], index: int) -> None:
    display_message(message, index)
    print("\n--- Cuerpo ---")
    print(
        message.get("text")
        or message.get("extracted_text")
        or message.get("html")
        or message.get("extracted_html")
        or message.get("preview")
        or "(sin contenido)"
    )
    print("-" * 60)


def read_multiline(session: PromptSession[str]) -> str | None:
    print("Escribe el mensaje. Línea vacía para terminar; /cancelar para abortar.")
    lines: list[str] = []
    while True:
        line = session.prompt("> ")
        if line.strip().casefold() == "/cancelar":
            print("Cancelado.")
            return None
        if not line:
            return "\n".join(lines)
        lines.append(line)


def selected_message(
    command: str,
    messages: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]] | None:
    parts = command.split(maxsplit=1)
    if len(parts) != 2:
        print("Indica el número del correo, por ejemplo: read 1.")
        return None
    try:
        index = int(parts[1]) - 1
    except ValueError:
        print("El número de correo debe ser un entero.")
        return None
    if not 0 <= index < len(messages):
        print("Número fuera de rango. Usa `list` para ver los correos.")
        return None
    return index + 1, messages[index]


def run_terminal_agent(
    client: AgentMailClient,
    *,
    audit_path: Path,
    reconnect_interval: float,
    assignments: Mapping[str, str] | None = None,
    llm: OpenRouterClient | None = None,
    pm_profiles: MutableMapping[str, dict[str, Any]] | None = None,
    simulation_seed: int = DEFAULT_SIMULATION_SEED,
    state_path: Path = STATE_PATH,
    config: AgentConfig | None = None,
    directors: Mapping[str, str] | None = None,
    profiles_path: Path = PM_PROFILES_PATH,
) -> None:
    agent_config = config or load_agent_config()
    director_contacts = (
        directors if directors is not None else load_director_assignments()
    )
    stop_event = threading.Event()
    state_lock = threading.RLock()
    approval_lock = threading.Lock()
    pending_approvals: dict[str, tuple[dict[str, Any], Queue[dict[str, Any]]]] = {}
    state_snapshot = load_agent_state(state_path)
    announced_request_ids = {
        request["id"] for request in state_snapshot["solicitudes"]
    }
    session: PromptSession[str] = PromptSession()

    def wait_for_director_approval(
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        identifier = str(payload["solicitud"])
        attempt = int(payload.get("intento", 1))
        should_prompt = False
        with approval_lock:
            existing = pending_approvals.get(identifier)
            if existing is None:
                response_queue: Queue[dict[str, Any]] = Queue(maxsize=1)
                pending_approvals[identifier] = (payload, response_queue)
                should_prompt = True
            else:
                previous_payload, response_queue = existing
                should_prompt = attempt > int(previous_payload.get("intento", 1))
                pending_approvals[identifier] = (payload, response_queue)
        try:
            try:
                response = response_queue.get_nowait()
            except Empty:
                if not should_prompt:
                    return None
                print(
                    f"\n[APROBACIÓN REQUERIDA] "
                    f"Solicitud #{payload['numero_solicitud']} | "
                    f"{payload['asunto']} | "
                    f"Cliente: {payload.get('cliente') or 'No identificado'} | "
                    f"PM: {payload.get('pm_responsable') or 'sin asignar'} | "
                    f"Director: {payload.get('director') or 'sin asignar'}"
                )
                print(
                    f"{payload['mensaje']} Escribe `approve "
                    f"{payload['numero_solicitud']} sí` o `approve "
                    f"{payload['numero_solicitud']} no <motivo>`."
                )
                try:
                    response = response_queue.get(
                        timeout=agent_config.sla_check_interval_seconds
                    )
                except Empty:
                    if stop_event.is_set():
                        raise RuntimeError(
                            "El agente se detuvo antes de recibir la aprobación "
                            "del director."
                        )
                    return None
            with approval_lock:
                current = pending_approvals.get(identifier)
                if current is not None and current[1] is response_queue:
                    pending_approvals.pop(identifier, None)
            return response
        except RuntimeError:
            with approval_lock:
                pending_approvals.pop(identifier, None)
            raise

    def resolve_director_approval(
        *,
        request_number: int,
        approved: bool,
        reason: str,
    ) -> bool:
        with approval_lock:
            match = next(
                (
                    item
                    for item in pending_approvals.values()
                    if item[0].get("numero_solicitud") == request_number
                ),
                None,
            )
        if match is None:
            return False
        response_queue = match[1]
        try:
            response_queue.put_nowait(
                {
                    "approved": approved,
                    "reason": reason,
                }
            )
        except Full:
            return False
        return True

    def active_messages(state: AgentState) -> list[dict[str, Any]]:
        requests_by_id = {
            request["id"]: request
            for request in state["solicitudes"]
        }
        visible: list[dict[str, Any]] = []
        for message in state["correos_clasificados"]:
            related_id = message.get("solicitud_id") or get_message_id(message)
            request = requests_by_id.get(str(related_id))
            if (
                request is None
                or request.get("estado") not in COMPLETED_REQUEST_STATES
            ):
                visible.append(message)
        return visible

    def persist_state() -> None:
        try:
            with state_lock:
                save_agent_state(state_path, state_snapshot)
        except AgentStateStoreError as exc:
            audit_error(
                JsonlAuditLog(audit_path),
                exc,
                node="persistir_estado_agente",
            )
            print(f"\n[ERROR Estado] {exc}")

    def profile_email_for_pm(pm_name: str | None) -> str | None:
        if not pm_name:
            return None
        for key, profile in (pm_profiles or {}).items():
            if normalize_text(pm_name) not in {
                normalize_text(str(key)),
                normalize_text(str(profile.get("nombre") or "")),
            }:
                continue
            addresses = profile.get("emails") or []
            if isinstance(addresses, str):
                addresses = [addresses]
            return next(
                (
                    address.strip().casefold()
                    for address in addresses
                    if isinstance(address, str) and "@" in address
                ),
                None,
            )
        return None

    def notify_lifecycle(
        request: dict[str, Any],
        event_type: str,
        *,
        reason: str,
    ) -> None:
        recipients: list[str] = []
        if event_type in {"solicitud_asignada", "solicitud_reasignada"}:
            pm_email = profile_email_for_pm(
                str(request.get("pm_responsable") or "")
            )
            if pm_email:
                recipients.append(pm_email)
        else:
            requester_email = str(
                request.get("correo_solicitante") or ""
            ).strip()
            if requester_email:
                recipients.append(requester_email)
            pm_email = profile_email_for_pm(
                str(request.get("pm_responsable") or "")
            )
            if pm_email:
                recipients.append(pm_email)
            if (
                request.get("escalada_a_jefe")
                or request.get("jefe_responsable")
            ) and agent_config.manager_email:
                recipients.append(agent_config.manager_email)
            if event_type == "escalamiento_jefe":
                recipients = (
                    [agent_config.manager_email]
                    if agent_config.manager_email
                    else []
                )
        recipients = list(
            dict.fromkeys(
                address.strip().casefold()
                for address in recipients
                if address and "@" in address
            )
        )
        if not recipients:
            error = audit_error(
                JsonlAuditLog(audit_path),
                "No hay destinatarios configurados para la notificación.",
                node="notificar_cambio_terminal",
                hora_simulada=state_snapshot["hora_actual"],
                solicitud=str(request.get("id") or "") or None,
                details={"tipo_evento": event_type},
            )
            state_snapshot["eventos"].append(error)
            print(
                f"\n[ERROR CORREO] No hay destinatario configurado para "
                f"{event_type}."
            )
            return
        for recipient in recipients:
            try:
                details = send_lifecycle_notification(
                    client,
                    recipient,
                    event_type,
                    request,
                    context={"razon": reason},
                    state=state_snapshot,
                    audit=JsonlAuditLog(audit_path),
                    config=agent_config,
                    llm=llm,
                )
            except (AgentMailError, ValueError) as exc:
                if isinstance(exc, ValueError):
                    error = audit_error(
                        JsonlAuditLog(audit_path),
                        exc,
                        node="notificar_cambio_terminal",
                        hora_simulada=state_snapshot["hora_actual"],
                        solicitud=str(request.get("id") or "") or None,
                        details={
                            "tipo_evento": event_type,
                            "destinatario": recipient,
                        },
                    )
                    state_snapshot["eventos"].append(error)
                print(
                    f"\n[ERROR CORREO] No se pudo enviar "
                    f"{event_type} a {recipient}: {exc}"
                )
            else:
                request.setdefault("notificaciones_email", []).append(
                    {"tipo": event_type, **details}
                )
                if event_type in {
                    "solicitud_asignada",
                    "solicitud_reasignada",
                }:
                    request["notificacion_pm_message_id"] = (
                        details.get("message_id")
                    )
                    if request.get("estado") == "Nueva":
                        request["estado"] = "Notificada"

    def store_state(new_state: AgentState) -> None:
        nonlocal state_snapshot
        new_requests: list[tuple[int, dict[str, Any]]] = []
        with state_lock:
            state_snapshot = new_state
            for index, request in enumerate(new_state["solicitudes"], start=1):
                if request["id"] not in announced_request_ids:
                    announced_request_ids.add(request["id"])
                    new_requests.append((index, request))
        for index, request in new_requests:
            print(
                f"\n[VALIDACIÓN DE SOLICITUD #{index}] "
                f"{request['asunto']} | PM: "
                f"{request['pm_responsable'] or 'sin asignar'}"
            )
            if request.get("datos_faltantes"):
                print(
                    "[INFORMACIÓN PENDIENTE] "
                    + ", ".join(request["datos_faltantes"])
                )
            if request.get("revision_humana"):
                print(
                    "[REVISIÓN HUMANA PENDIENTE] Falta confirmar: "
                    + ", ".join(request.get("campos_revision_humana", []))
                )
                if request.get("correo_revision_humana_enviado"):
                    print(
                        "[REVISIÓN HUMANA] Se envió un correo al PM asignado."
                    )
                else:
                    print(
                        "[REVISIÓN HUMANA] "
                        + str(
                            request.get(
                                "revision_humana_motivo",
                                "No se envió un correo de revisión al PM.",
                            )
                        )
                    )
            print(f"[PREGUNTA DE IA] {request['pregunta_confirmacion']}")
            for question in request["preguntas_aclaracion"]:
                print(f"[ACLARACIÓN DE IA] {question}")
            print(
                f"¿Confirmas o corriges la respuesta del PM? "
                f"Escribe `confirm {index} sí` o `confirm {index} no`."
            )
            if request.get("perfil_pm") == "ausente":
                print("[PERFIL SIMULADO] Este PM no confirmará automáticamente.")
            elif request.get("horas_esperadas_confirmacion") is not None:
                print(
                    "[PERFIL SIMULADO] Confirmación esperada tras "
                    f"{request['horas_esperadas_confirmacion']} horas."
                )

    def run_worker() -> None:
        try:
            run_agent_forever(
                client,
                audit_path=audit_path,
                stop_event=stop_event,
                on_state=store_state,
                reconnect_interval=reconnect_interval,
                assignments=assignments,
                directors=director_contacts,
                llm=llm,
                pm_profiles=pm_profiles,
                simulation_seed=simulation_seed,
                state_lock=state_lock,
                state_path=state_path,
                initial_state=state_snapshot,
                config=agent_config,
                profiles_path=profiles_path,
                approval_handler=wait_for_director_approval,
            )
        except Exception as exc:
            _audit_terminal_error(audit_path, "ejecutar_agente", exc)
            print(
                f"\n[ERROR Agente] El proceso se detuvo: "
                f"{type(exc).__name__}: {exc}"
            )

    worker = threading.Thread(
        target=run_worker,
        name="langgraph-agentmail-loop",
        daemon=True,
    )
    print(
        "Agente LangGraph iniciado. Escucha nuevos correos por WebSocket; "
        "no consulta la bandeja periódicamente."
    )
    print(
        f"Si se pierde la conexión, se reintentará cada "
        f"{reconnect_interval:g} segundos."
    )
    print("Se imprimirá un punto cada minuto. Escribe `help` para comandos o `stop` para detener.")
    worker.start()

    try:
        with patch_stdout():
            while True:
                try:
                    command = session.prompt("agentmail> ").strip()
                except EOFError:
                    break
                except KeyboardInterrupt:
                    break
                if not command:
                    continue

                normalized = command.casefold()
                with state_lock:
                    state = state_snapshot
                messages = active_messages(state)

                if normalized in {"stop", "quit", "exit", "q"}:
                    break
                if normalized in {"help", "h", "?"}:
                    print(
                        "Comandos:\n"
                        "  list, l             Lista los correos procesados\n"
                        "  read <n>, r <n>     Lee un correo completo\n"
                        "  reply <n>, rep <n>  Responde al correo con AgentMail\n"
                        "  new, n              Envía un correo desde el inbox AgentMail\n"
                        "  history, historial  Consulta solicitudes cerradas o completadas\n"
                        "  status              Muestra hora, correos y solicitudes detectadas\n"
                        "  assign <n> <PM>     Asigna un PM a una solicitud de esta sesión\n"
                        "  pm-eligibility <PM> si|no  Habilita/deshabilita un PM; al deshabilitarlo reasigna solicitudes activas\n"
                        "  pm <n>              Escribe en lenguaje natural la respuesta del PM\n"
                        "  confirm <n> sí|no   Confirma si el PM se hará cargo de la solicitud\n"
                        "  approve <n> sí|no [motivo]  Aprueba o rechaza el escalamiento al director\n"
                        "  complete <n> <razón>  Marca como terminado el trabajo de una solicitud\n"
                        "  close <n> <razón>     Cierra una solicitud cancelada (número de status)\n"
                        "  help, h, ?          Ayuda\n"
                        "  stop, quit, q       Detiene el agente"
                    )
                elif normalized in {"status", "s"}:
                    llm_limit = configured_max_llm_calls()
                    llm_usage = (
                        f"{state['llamadas_llm']}/{llm_limit}"
                        if llm_limit
                        else f"{state['llamadas_llm']} (sin límite)"
                    )
                    print(
                        f"Hora simulada: {state['hora_actual']} | "
                        f"Correos procesados: {len(state['correos_procesados'])} | "
                        f"Solicitudes abiertas: {sum(request['estado'] != 'Cerrada' for request in state['solicitudes'])} | "
                        f"Finalizadas: {sum(request['estado'] == 'Cerrada' for request in state['solicitudes'])} | "
                        f"Llamadas OpenRouter: {llm_usage}"
                    )
                    pm_roster = build_pm_roster(
                        assignments,
                        pm_profiles,
                        eligible_only=False,
                    )
                    if pm_roster:
                        print("PMs y carga abierta:")
                        for pm_name in pm_roster:
                            open_projects = count_open_projects(
                                state["solicitudes"],
                                pm_name,
                                pm_profiles=pm_profiles,
                            )
                            print(
                                f"  {pm_name}: "
                                f"{'elegible' if is_pm_eligible(pm_name, pm_profiles) else 'NO ELEGIBLE'} | "
                                f"{open_projects}/"
                                f"{agent_config.max_open_projects_per_pm}"
                            )
                    else:
                        print(
                            "No hay PM configurados en asignacion.csv; "
                            f"el límite por PM es "
                            f"{agent_config.max_open_projects_per_pm}."
                        )
                    for index, request in enumerate(state["solicitudes"], start=1):
                        if request["estado"] == "Cerrada":
                            continue
                        print(
                            f"  #{index} {request['asunto']} | "
                            f"PM: {request['pm_responsable'] or 'sin asignar'} | "
                            f"Estado: {request['estado']} | "
                            f"Perfil: {request.get('perfil_pm') or 'pendiente'}"
                        )
                elif normalized in {
                    "history",
                    "historial",
                    "archivadas",
                    "archived",
                }:
                    completed = completed_requests(state)
                    if not completed:
                        print("No hay solicitudes cerradas o completadas.")
                        continue
                    print("Solicitudes cerradas/completadas (conservadas en el historial):")
                    for index, request in completed:
                        final_date = request.get("fecha_completada") or request.get(
                            "fecha_cierre"
                        )
                        print(
                            f"  #{index} {request['asunto']} | "
                            f"Estado: {request['estado']} | "
                            f"PM: {request.get('pm_responsable') or 'sin asignar'}"
                        )
                        if final_date:
                            print(f"    Fecha de finalización: {final_date}")
                        if request.get("razon_finalizacion"):
                            print(
                                f"    Razón: {request['razon_finalizacion']}"
                            )
                elif normalized.startswith(
                    (
                        "pm-eligibility ",
                        "pm-status ",
                        "elegibilidad_pm ",
                    )
                ):
                    parts = command.split()
                    if len(parts) < 3:
                        print(
                            "Uso: pm-eligibility <nombre del PM> si|no."
                        )
                        continue
                    pm_name = " ".join(parts[1:-1])
                    if not pm_name:
                        print(
                            "Indica el nombre del PM sin dejarlo vacío."
                        )
                        continue
                    eligibility_value = normalize_text(parts[-1])
                    if eligibility_value in {
                        "si",
                        "yes",
                        "eligible",
                        "elegible",
                        "available",
                        "on",
                    }:
                        eligible = True
                    elif eligibility_value in {
                        "no",
                        "ineligible",
                        "no elegible",
                        "inelegible",
                        "unavailable",
                        "off",
                    }:
                        eligible = False
                    else:
                        print("Indica `si` o `no` para la elegibilidad.")
                        continue
                    if pm_profiles is None:
                        print(
                            "No hay roster de PM cargado; configura el archivo "
                            "de perfiles antes de cambiar elegibilidad."
                        )
                        continue
                    try:
                        with state_lock:
                            latest_profiles = load_pm_profiles(
                                profiles_path,
                                pm_addresses=(assignments or {}).values(),
                            )
                            for profile_key, latest_profile in (
                                latest_profiles.items()
                            ):
                                matching_key = next(
                                    (
                                        key
                                        for key, current_profile in (
                                            pm_profiles.items()
                                        )
                                        if normalize_text(str(key))
                                        == normalize_text(profile_key)
                                        or normalize_text(
                                            str(
                                                current_profile.get("nombre")
                                                or ""
                                            )
                                        )
                                        == normalize_text(
                                            str(
                                                latest_profile.get("nombre")
                                                or ""
                                            )
                                        )
                                    ),
                                    None,
                                )
                                if matching_key is None:
                                    pm_profiles[profile_key] = latest_profile
                                else:
                                    current_profile = pm_profiles[matching_key]
                                    current_profile.clear()
                                    current_profile.update(latest_profile)
                            previous_assignments = {
                                request["id"]: request.get("pm_responsable")
                                for request in state_snapshot["solicitudes"]
                            }
                            set_pm_eligibility(
                                state_snapshot,
                                pm_name=pm_name,
                                eligible=eligible,
                                pm_profiles=pm_profiles,
                                profiles_path=profiles_path,
                                assignments=assignments,
                                audit=JsonlAuditLog(audit_path),
                                simulation_seed=simulation_seed,
                                config=agent_config,
                            )
                            reassigned_requests = [
                                request
                                for request in state_snapshot["solicitudes"]
                                if request.get("pm_responsable")
                                and request.get("pm_responsable")
                                != previous_assignments.get(request["id"])
                            ]
                            for request in reassigned_requests:
                                notify_lifecycle(
                                    request,
                                    "solicitud_reasignada",
                                    reason=str(
                                        request.get("motivo_reasignacion")
                                        or f"{pm_name} dejó de ser elegible."
                                    ),
                                )
                        persist_state()
                    except (OSError, ValueError) as exc:
                        print(f"No se pudo cambiar la elegibilidad del PM: {exc}")
                        _audit_terminal_error(
                            audit_path,
                            "cambiar_elegibilidad_pm_terminal",
                            exc,
                        )
                elif normalized.startswith(("confirm ", "confirmar ")):
                    parts = command.split()
                    if len(parts) != 3:
                        print("Uso: confirm <número de solicitud> sí|no.")
                        continue
                    try:
                        request_index = int(parts[1]) - 1
                    except ValueError:
                        print("El número de solicitud debe ser un entero.")
                        continue
                    requests = state["solicitudes"]
                    if not 0 <= request_index < len(requests):
                        print("Número fuera de rango. Usa `status`.")
                        continue
                    answer = normalize_text(parts[2])
                    if answer not in {"si", "no", "yes", "y", "n"}:
                        print("Responde `sí` o `no`.")
                        continue
                    try:
                        with state_lock:
                            state = state_snapshot
                            requests = state["solicitudes"]
                            apply_human_confirmation(
                                state,
                                identifier=requests[request_index]["id"],
                                confirmed=answer in {"si", "yes", "y"},
                                audit=JsonlAuditLog(audit_path),
                            )
                            if answer in {"si", "yes", "y"}:
                                request = requests[request_index]
                                notify_lifecycle(
                                    request,
                                    "confirmacion_pm",
                                    reason="Se confirmó que la solicitud será atendida.",
                                )
                        persist_state()
                    except (KeyError, ValueError) as exc:
                        print(f"No se pudo registrar la confirmación: {exc}")
                        _audit_terminal_error(
                            audit_path,
                            "confirmacion_pm_terminal",
                            exc,
                        )
                elif normalized.startswith(("approve ", "aprobar ")):
                    parts = command.split(maxsplit=3)
                    if len(parts) < 3:
                        print(
                            "Uso: approve <número de solicitud> sí|no [motivo]."
                        )
                        continue
                    try:
                        request_index = int(parts[1])
                    except ValueError:
                        print("El número de solicitud debe ser un entero.")
                        continue
                    answer = normalize_text(parts[2])
                    if answer not in {"si", "no", "yes", "y", "n"}:
                        print("Responde `sí` o `no`.")
                        continue
                    approved = answer in {"si", "yes", "y"}
                    reason = parts[3].strip() if len(parts) == 4 else ""
                    if not approved and not reason:
                        print("Indica el motivo del rechazo.")
                        continue
                    if approved and not reason:
                        reason = "Aprobado por el usuario en la terminal."
                    if resolve_director_approval(
                        request_number=request_index,
                        approved=approved,
                        reason=reason,
                    ):
                        print(
                            "Aprobación enviada al grafo."
                            if approved
                            else "Rechazo enviado; se reevaluará en la siguiente vuelta."
                        )
                    else:
                        print(
                            "No hay una aprobación pendiente para esa solicitud."
                        )
                elif normalized.startswith(("assign ", "asignar ")):
                    parts = command.split(maxsplit=2)
                    if len(parts) != 3:
                        print("Uso: assign <número de solicitud> <nombre del PM>.")
                        continue
                    try:
                        request_index = int(parts[1])
                    except ValueError:
                        print("El número de solicitud debe ser un entero.")
                        continue
                    try:
                        with state_lock:
                            assign_pm_to_request(
                                state_snapshot,
                                request_number=request_index,
                                pm_name=parts[2],
                                audit=JsonlAuditLog(audit_path),
                                pm_profiles=pm_profiles,
                                simulation_seed=simulation_seed,
                                config=agent_config,
                            )
                            request = state_snapshot["solicitudes"][
                                request_index - 1
                            ]
                            notify_lifecycle(
                                request,
                                "solicitud_asignada",
                                reason="Asignación realizada desde la terminal.",
                            )
                        persist_state()
                    except (IndexError, ValueError) as exc:
                        print(f"No se pudo asignar el PM: {exc}")
                        _audit_terminal_error(
                            audit_path,
                            "asignar_pm_terminal",
                            exc,
                            request_number=request_index,
                        )
                elif normalized.startswith(("pm ", "respuesta_pm ")):
                    parts = command.split(maxsplit=1)
                    if len(parts) != 2:
                        print("Uso: pm <número de solicitud>.")
                        continue
                    try:
                        request_index = int(parts[1])
                    except ValueError:
                        print("El número de solicitud debe ser un entero.")
                        continue
                    if not 1 <= request_index <= len(state["solicitudes"]):
                        print("Número fuera de rango. Usa `status`.")
                        continue
                    response_text = read_multiline(session)
                    if response_text is None or not response_text.strip():
                        if response_text == "":
                            print("La respuesta está vacía; no se aplicó.")
                        continue
                    try:
                        with state_lock:
                            request = state_snapshot["solicitudes"][
                                request_index - 1
                            ]
                            previous_status = request.get("estado")
                            previous_pm = request.get("pm_responsable")
                            previous_completion_date = request.get(
                                "fecha_completada"
                            )
                            apply_pm_update(
                                state_snapshot,
                                request_number=request_index,
                                response_text=response_text,
                                audit=JsonlAuditLog(audit_path),
                                llm=llm,
                                assignments=assignments,
                                pm_profiles=pm_profiles,
                                config=agent_config,
                            )
                            if (
                                request.get("estado") == "Confirmada"
                                and previous_status != "Confirmada"
                            ):
                                notify_lifecycle(
                                    request,
                                    "confirmacion_pm",
                                    reason="El PM confirmó que atenderá la solicitud.",
                                )
                            if (
                                request.get("estado") == "Cerrada"
                                and previous_status != "Cerrada"
                            ):
                                event_type = (
                                    "solicitud_completada"
                                    if request.get("fecha_completada")
                                    != previous_completion_date
                                    and request.get("fecha_completada")
                                    else "solicitud_cerrada"
                                )
                                notify_lifecycle(
                                    request,
                                    event_type,
                                    reason=str(
                                        request.get("razon_finalizacion")
                                        or response_text
                                    ),
                                )
                            if (
                                request.get("pm_responsable") != previous_pm
                                and request.get("pm_responsable")
                            ):
                                notify_lifecycle(
                                    request,
                                    "solicitud_reasignada",
                                    reason=str(
                                        request.get("motivo_reasignacion")
                                        or "La solicitud se reasignó tras la respuesta del PM."
                                    ),
                                )
                        persist_state()
                    except (IndexError, OpenRouterError, ValueError) as exc:
                        print(f"No se pudo aplicar la respuesta del PM: {exc}")
                elif normalized.startswith(
                    ("complete ", "completar ", "done ", "finalizar ")
                ):
                    parts = command.split(maxsplit=2)
                    if len(parts) != 3 or not parts[2].strip():
                        print(
                            "Uso: complete <número de solicitud> <razón en texto>."
                        )
                        continue
                    try:
                        request_index = int(parts[1].removeprefix("#"))
                    except ValueError:
                        print("El número de solicitud debe ser un entero.")
                        continue
                    try:
                        with state_lock:
                            complete_request(
                                state_snapshot,
                                request_number=request_index,
                                reason=parts[2],
                                audit=JsonlAuditLog(audit_path),
                            )
                            request = state_snapshot["solicitudes"][
                                request_index - 1
                            ]
                            notify_lifecycle(
                                request,
                                "solicitud_completada",
                                reason=parts[2],
                            )
                        persist_state()
                    except (IndexError, ValueError) as exc:
                        print(f"No se pudo completar la solicitud: {exc}")
                        _audit_terminal_error(
                            audit_path,
                            "completar_solicitud_terminal",
                            exc,
                            request_number=request_index,
                        )
                elif normalized.startswith(("close ", "cerrar ")):
                    parts = command.split(maxsplit=2)
                    if len(parts) != 3 or not parts[2].strip():
                        print(
                            "Uso: close <número de solicitud> <razón en texto> "
                            "(usa el número mostrado por status)."
                        )
                        continue
                    try:
                        request_index = int(parts[1].removeprefix("#"))
                    except ValueError:
                        print("El número de solicitud debe ser un entero.")
                        continue
                    try:
                        with state_lock:
                            close_request(
                                state_snapshot,
                                request_number=request_index,
                                reason=parts[2],
                                audit=JsonlAuditLog(audit_path),
                            )
                            request = state_snapshot["solicitudes"][
                                request_index - 1
                            ]
                            notify_lifecycle(
                                request,
                                "solicitud_cerrada",
                                reason=parts[2],
                            )
                        persist_state()
                        request = state_snapshot["solicitudes"][request_index - 1]
                        print(
                            f"[CERRADA] Solicitud #{request_index}: "
                            f"{request.get('asunto') or request.get('id')}."
                        )
                    except (IndexError, ValueError) as exc:
                        print(f"No se pudo cerrar la solicitud: {exc}")
                        _audit_terminal_error(
                            audit_path,
                            "cerrar_solicitud_terminal",
                            exc,
                            request_number=request_index,
                        )
                elif normalized in {"list", "l"}:
                    if not messages:
                        print("Todavía no se han procesado correos.")
                    else:
                        for index, message in enumerate(messages, start=1):
                            display_message(message, index)
                elif normalized.startswith(("read ", "r ")):
                    selected = selected_message(command, messages)
                    if selected is not None:
                        index, message = selected
                        display_full_message(message, index)
                        related_id = str(
                            message.get("solicitud_id")
                            or get_message_id(message)
                        )
                        with state_lock:
                            mark_request_opened(
                                state_snapshot,
                                identifier=related_id,
                                audit=JsonlAuditLog(audit_path),
                            )
                        persist_state()
                elif normalized.startswith(("reply ", "rep ")):
                    selected = selected_message(command, messages)
                    if selected is not None:
                        _, message = selected
                        identifier = get_message_id(message)
                        if not identifier:
                            print("No se puede responder: el correo no tiene message_id.")
                            _audit_terminal_error(
                                audit_path,
                                "responder_correo_terminal",
                                ValueError("El correo seleccionado no tiene message_id."),
                            )
                            continue
                        body = read_multiline(session)
                        if body:
                            try:
                                client.reply(identifier, body)
                                print("Respuesta enviada por AgentMail.")
                            except AgentMailError as exc:
                                print(f"No se pudo enviar la respuesta: {exc}")
                                _audit_terminal_error(
                                    audit_path,
                                    "responder_correo_terminal",
                                    exc,
                                )
                        elif body == "":
                            print("La respuesta está vacía; no se envió.")
                elif normalized in {"new", "n"}:
                    recipient = session.prompt("Para (correo): ").strip()
                    subject = session.prompt("Asunto: ").strip()
                    if not recipient:
                        print("El destinatario es obligatorio.")
                        continue
                    body = read_multiline(session)
                    if body:
                        try:
                            client.send(recipient, subject, body)
                            print("Correo enviado por AgentMail.")
                        except AgentMailError as exc:
                            print(f"No se pudo enviar el correo: {exc}")
                            _audit_terminal_error(
                                audit_path,
                                "enviar_correo_terminal",
                                exc,
                            )
                    elif body == "":
                        print("El cuerpo está vacío; no se envió.")
                else:
                    print("Comando desconocido. Escribe `help`.")
    finally:
        stop_event.set()
        worker.join(timeout=REQUEST_TIMEOUT_SECONDS + reconnect_interval + 2)
        if worker.is_alive():
            print("El agente sigue terminando una solicitud de AgentMail en curso.")
        else:
            print("\nAgente detenido.")
