import threading
from pathlib import Path
from typing import Any, Mapping

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from agent_state import AgentState, initial_agent_state
from agentmail_client import AgentMailClient
from audit_log import JsonlAuditLog, audit_error
from email_utils import get_message_id, get_sender, normalize_text
from exceptions import AgentMailError, OpenRouterError
from openrouter_client import OpenRouterClient
from service import (
    apply_human_confirmation,
    apply_pm_update,
    assign_pm_to_request,
    run_agent_forever,
)
from settings import (
    DEFAULT_SIMULATION_SEED,
    REQUEST_TIMEOUT_SECONDS,
    configured_max_llm_calls,
)


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
    pm_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    simulation_seed: int = DEFAULT_SIMULATION_SEED,
) -> None:
    stop_event = threading.Event()
    state_lock = threading.RLock()
    state_snapshot = initial_agent_state()
    announced_request_ids: set[str] = set()
    session: PromptSession[str] = PromptSession()

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
                llm=llm,
                pm_profiles=pm_profiles,
                simulation_seed=simulation_seed,
                state_lock=state_lock,
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
                messages = list(state["correos_clasificados"])

                if normalized in {"stop", "quit", "exit", "q"}:
                    break
                if normalized in {"help", "h", "?"}:
                    print(
                        "Comandos:\n"
                        "  list, l             Lista los correos procesados\n"
                        "  read <n>, r <n>     Lee un correo completo\n"
                        "  reply <n>, rep <n>  Responde al correo con AgentMail\n"
                        "  new, n              Envía un correo desde el inbox AgentMail\n"
                        "  status              Muestra hora, correos y solicitudes detectadas\n"
                        "  assign <n> <PM>     Asigna un PM a una solicitud de esta sesión\n"
                        "  pm <n>              Escribe en lenguaje natural la respuesta del PM\n"
                        "  confirm <n> sí|no   Confirma o corrige inmediatamente la recepción del PM\n"
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
                        f"Solicitudes: {len(state['solicitudes'])} | "
                        f"Llamadas OpenRouter: {llm_usage}"
                    )
                    for index, request in enumerate(state["solicitudes"], start=1):
                        print(
                            f"  #{index} {request['asunto']} | "
                            f"PM: {request['pm_responsable'] or 'sin asignar'} | "
                            f"Estado: {request['estado']} | "
                            f"Perfil: {request.get('perfil_pm') or 'pendiente'}"
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
                    except (KeyError, ValueError) as exc:
                        print(f"No se pudo registrar la confirmación: {exc}")
                        _audit_terminal_error(
                            audit_path,
                            "confirmacion_pm_terminal",
                            exc,
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
                            )
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
                            apply_pm_update(
                                state_snapshot,
                                request_number=request_index,
                                response_text=response_text,
                                audit=JsonlAuditLog(audit_path),
                                llm=llm,
                                assignments=assignments,
                            )
                    except (IndexError, OpenRouterError, ValueError) as exc:
                        print(f"No se pudo aplicar la respuesta del PM: {exc}")
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
