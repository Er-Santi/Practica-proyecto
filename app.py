#!/usr/bin/env python3
"""Fase 1 del agente de solicitudes: AgentMail, reglas, avisos y reloj simulado."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

API_BASE_URL = "https://api.agentmail.to/v0"
REQUEST_TIMEOUT_SECONDS = 20
DEFAULT_POLL_INTERVAL_SECONDS = 5
HEARTBEAT_INTERVAL_SECONDS = 60
AUDIT_PATH = Path("auditoria.jsonl")

REQUEST_MARKERS = (
    "solicitud de personal",
    "solicitud de puesto",
    "solicitud de posicion",
    "solicitud de posición",
    "solicitud de recurso",
    "solicitamos",
    "solicito",
    "requerimos",
    "necesitamos cubrir",
    "vacante",
    "staffing request",
    "position request",
    "resource request",
    "headcount request",
)


class AgentMailError(RuntimeError):
    """Error de comunicación o respuesta inválida de AgentMail."""


class AgentState(TypedDict):
    hora_actual: int
    solicitudes: list[dict[str, Any]]
    correos_procesados: list[str]
    pendientes_aprobacion: list[dict[str, Any]]
    eventos: list[dict[str, Any]]
    llamadas_llm: int
    correos_nuevos: list[dict[str, Any]]
    correos_clasificados: list[dict[str, Any]]
    clasificacion_actual: list[dict[str, Any]]
    firmas_solicitudes: list[str]
    ids_solicitudes: list[str]


class AgentMailClient:
    """Cliente mínimo de lectura de inbox para la Fase 1."""

    def __init__(self, api_key: str, inbox_id: str) -> None:
        self.inbox_id = inbox_id
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._inbox_path = quote(inbox_id, safe="")

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = requests.request(
                method,
                f"{API_BASE_URL}{path}",
                headers=self.headers,
                params=params,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            response = getattr(exc, "response", None)
            detail = response.text[:500].strip() if response is not None else ""
            message = f"Falló la conexión con AgentMail: {exc}"
            if detail:
                message += f"\nRespuesta: {detail}"
            raise AgentMailError(message) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise AgentMailError("AgentMail devolvió una respuesta que no es JSON.") from exc
        if not isinstance(data, dict):
            raise AgentMailError("AgentMail devolvió una respuesta con formato inválido.")
        return data

    def list_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        page_token: str | None = None
        seen_tokens: set[str] = set()

        while True:
            params: dict[str, Any] = {"limit": 100}
            if page_token:
                params["page_token"] = page_token
            data = self._request(
                "GET",
                f"/inboxes/{self._inbox_path}/messages",
                params=params,
            )
            page = data.get("messages")
            if not isinstance(page, list):
                raise AgentMailError("La respuesta de AgentMail no contiene la lista de mensajes.")
            messages.extend(item for item in page if isinstance(item, dict))

            next_token = data.get("next_page_token")
            if not isinstance(next_token, str) or not next_token:
                return messages
            if next_token in seen_tokens:
                raise AgentMailError("AgentMail repitió el token de página; se detuvo la lectura.")
            seen_tokens.add(next_token)
            page_token = next_token

    def get_message(self, identifier: str) -> dict[str, Any]:
        data = self._request(
            "GET",
            f"/inboxes/{self._inbox_path}/messages/{quote(identifier, safe='')}",
        )
        # Algunos endpoints devuelven el objeto directamente; otros lo envuelven.
        message = data.get("message", data)
        if not isinstance(message, dict):
            raise AgentMailError(f"AgentMail devolvió un mensaje inválido para {identifier}.")
        return message

    def reply(self, identifier: str, text: str) -> None:
        self._request(
            "POST",
            f"/inboxes/{self._inbox_path}/messages/{quote(identifier, safe='')}/reply",
            payload={"text": text},
        )

    def send(self, recipient: str, subject: str, text: str) -> None:
        self._request(
            "POST",
            f"/inboxes/{self._inbox_path}/messages/send",
            payload={"to": recipient, "subject": subject, "text": text},
        )


class JsonlAuditLog:
    """Escribe cada evento como una línea JSON independiente."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, Any]) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(event, ensure_ascii=False) + "\n")


def get_message_id(message: dict[str, Any]) -> str:
    identifier = message.get("message_id") or message.get("messageId")
    return identifier.strip() if isinstance(identifier, str) else ""


def get_sender(message: dict[str, Any]) -> str:
    sender = message.get("from", "Remitente desconocido")
    if isinstance(sender, dict):
        name = str(sender.get("name") or "").strip()
        address = str(sender.get("email") or "").strip()
        return f"{name} <{address}>" if name and address else address or name or "Remitente desconocido"
    return str(sender)


def message_text(message: dict[str, Any]) -> str:
    fields = (
        "subject",
        "text",
        "extracted_text",
        "preview",
        "html",
        "extracted_html",
    )
    return "\n".join(str(message.get(field) or "") for field in fields)


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", without_accents).strip()


def request_fingerprint(message: dict[str, Any]) -> str:
    subject = normalize_text(str(message.get("subject") or ""))
    subject = re.sub(r"^(?:(?:re|fw|fwd)\s*:\s*)+", "", subject)
    body = str(
        message.get("text")
        or message.get("extracted_text")
        or message.get("preview")
        or ""
    )
    body_lines = [line.strip() for line in body.splitlines() if not line.lstrip().startswith(">")]
    normalized_body = normalize_text("\n".join(body_lines))
    signature = f"{subject}\n{normalized_body}"
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def classify_by_rules(
    message: dict[str, Any],
    *,
    known_fingerprints: set[str],
    known_request_ids: set[str],
) -> tuple[str, str]:
    body = message_text(message)
    normalized_body = normalize_text(body)
    identifier = get_message_id(message)

    references = message.get("references") or []
    if isinstance(references, str):
        references = references.split()
    reply_to = message.get("in_reply_to")
    referenced_ids = {str(reference) for reference in references}
    if reply_to:
        referenced_ids.add(str(reply_to))
    if referenced_ids.intersection(known_request_ids):
        return "duplicado", "El correo hace referencia a una solicitud ya registrada."

    is_position_request = any(marker in normalized_body for marker in REQUEST_MARKERS)
    if not is_position_request:
        return "no_relacionado", "No contiene marcadores de solicitud de personal o posición."

    fingerprint = request_fingerprint(message)
    if fingerprint in known_fingerprints:
        return "duplicado", "El contenido coincide con una solicitud ya clasificada."
    if identifier and identifier in known_request_ids:
        return "duplicado", "El identificador ya pertenece a una solicitud registrada."
    return "solicitud_posicion", "Contiene una regla textual de solicitud de personal o posición."


def _ordered_by_timestamp(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def timestamp_key(message: dict[str, Any]) -> str:
        timestamp = message.get("timestamp") or message.get("received_at") or ""
        return str(timestamp)

    return sorted(messages, key=timestamp_key)


def build_graph(
    client: AgentMailClient,
    audit: JsonlAuditLog,
):
    def record(
        state: AgentState,
        *,
        identifier: str | None,
        node: str,
        decision: str,
        reason: str,
        hour: int | None = None,
    ) -> dict[str, Any]:
        event = {
            "hora_simulada": state["hora_actual"] if hour is None else hour,
            "solicitud": identifier,
            "nodo": node,
            "decision": decision,
            "motivo": reason,
        }
        audit.append(event)
        return event

    def read_inbox(state: AgentState) -> dict[str, Any]:
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
        events: list[dict[str, Any]] = []
        for identifier, summary in summaries_by_id.items():
            full_message = client.get_message(identifier)
            full_message.setdefault("message_id", identifier)
            emails.append(full_message)
            events.append(
                record(
                    state,
                    identifier=identifier,
                    node="leer_bandeja",
                    decision="correo_leido",
                    reason="Correo nuevo leído desde AgentMail.",
                )
            )

        poll_event = record(
            state,
            identifier=None,
            node="leer_bandeja",
            decision="sondeo_completado",
            reason=f"Sondeo completado; se encontraron {len(emails)} correos nuevos.",
        )
        return {
            "correos_nuevos": _ordered_by_timestamp(emails),
            "clasificacion_actual": [],
            "eventos": [*state["eventos"], poll_event, *events],
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

        for email in state["correos_nuevos"]:
            identifier = get_message_id(email)
            category, reason = classify_by_rules(
                email,
                known_fingerprints=known_fingerprints,
                known_request_ids=known_request_ids,
            )
            classified_email = {**email, "categoria": category}
            classified.append(classified_email)
            processed_ids.append(identifier)

            if category == "solicitud_posicion":
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
                        "estado": "Nueva",
                        "hora_notificacion": None,
                    }
                )

            events.append(
                record(
                    state,
                    identifier=identifier,
                    node="clasificar_por_reglas",
                    decision=category,
                    reason=reason,
                )
            )

        return {
            "correos_clasificados": [*state["correos_clasificados"], *classified],
            "clasificacion_actual": classified,
            "solicitudes": requests_found,
            "correos_procesados": processed_ids,
            "firmas_solicitudes": fingerprints,
            "ids_solicitudes": request_ids,
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
                request["hora_notificacion"] = state["hora_actual"]
                request["estado"] = "Notificada"
                print(
                    "[NOTIFICACIÓN SIMULADA] "
                    f"PM pendiente de asignación | {request['asunto']} | "
                    f"correo {identifier}"
                )
                decision = "notificacion_simulada"
                reason = "Se registró el aviso de nueva solicitud; no se envió ningún correo real."
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
                )
            )
        return {
            "solicitudes": list(requests_by_id.values()),
            "eventos": [*state["eventos"], *events],
        }

    def advance_clock(state: AgentState) -> dict[str, Any]:
        next_hour = state["hora_actual"] + 1
        event = record(
            state,
            identifier=None,
            node="avanzar_reloj",
            decision="hora_avanzada",
            reason=f"El reloj simulado avanzó a la hora {next_hour}.",
            hour=next_hour,
        )
        return {
            "hora_actual": next_hour,
            "eventos": [*state["eventos"], event],
        }

    workflow = StateGraph(AgentState)
    workflow.add_node("leer_bandeja", read_inbox)
    workflow.add_node("clasificar_por_reglas", classify_emails)
    workflow.add_node("notificar_pm_simulado", notify_pm_simulated)
    workflow.add_node("avanzar_reloj", advance_clock)
    workflow.add_edge(START, "leer_bandeja")
    workflow.add_edge("leer_bandeja", "clasificar_por_reglas")
    workflow.add_edge("clasificar_por_reglas", "notificar_pm_simulado")
    workflow.add_edge("notificar_pm_simulado", "avanzar_reloj")
    workflow.add_edge("avanzar_reloj", END)
    return workflow.compile()


def initial_agent_state() -> AgentState:
    return {
        "hora_actual": 0,
        "solicitudes": [],
        "correos_procesados": [],
        "pendientes_aprobacion": [],
        "eventos": [],
        "llamadas_llm": 0,
        "correos_nuevos": [],
        "correos_clasificados": [],
        "clasificacion_actual": [],
        "firmas_solicitudes": [],
        "ids_solicitudes": [],
    }


def run_phase_one(
    client: AgentMailClient,
    *,
    audit_path: Path = AUDIT_PATH,
    state: AgentState | None = None,
) -> AgentState:
    """Ejecuta un sondeo completo del StateGraph; el servicio vuelve a invocarlo."""
    graph = build_graph(client, JsonlAuditLog(audit_path))
    return graph.invoke(
        state if state is not None else initial_agent_state(),
        config={"recursion_limit": 16},
    )


def run_agent_forever(
    client: AgentMailClient,
    *,
    audit_path: Path,
    stop_event: threading.Event,
    on_state: Any,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    heartbeat_interval: float = HEARTBEAT_INTERVAL_SECONDS,
) -> AgentState:
    if poll_interval <= 0:
        raise ValueError("poll_interval debe ser mayor que cero.")
    if heartbeat_interval <= 0:
        raise ValueError("heartbeat_interval debe ser mayor que cero.")

    graph = build_graph(client, JsonlAuditLog(audit_path))
    state = initial_agent_state()
    last_error: str | None = None
    next_heartbeat = time.monotonic() + heartbeat_interval

    while not stop_event.is_set():
        try:
            state = graph.invoke(state, config={"recursion_limit": 16})
            on_state(state)
            if last_error is not None:
                print("\n[CONEXIÓN] AgentMail volvió a responder.")
                last_error = None
        except AgentMailError as exc:
            error = str(exc)
            audit = {
                "hora_simulada": state["hora_actual"],
                "solicitud": None,
                "nodo": "bucle_agente",
                "decision": "error_sondeo",
                "motivo": error,
            }
            JsonlAuditLog(audit_path).append(audit)
            if error != last_error:
                print(f"\n[ERROR AgentMail] {error}")
                last_error = error

        now = time.monotonic()
        if now >= next_heartbeat:
            print(".", end="", flush=True)
            next_heartbeat = now + heartbeat_interval
        stop_event.wait(poll_interval)

    return state


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
    poll_interval: float,
) -> None:
    stop_event = threading.Event()
    state_lock = threading.Lock()
    state_snapshot = initial_agent_state()
    session: PromptSession[str] = PromptSession()

    def store_state(new_state: AgentState) -> None:
        nonlocal state_snapshot
        with state_lock:
            state_snapshot = new_state

    worker = threading.Thread(
        target=run_agent_forever,
        kwargs={
            "client": client,
            "audit_path": audit_path,
            "stop_event": stop_event,
            "on_state": store_state,
            "poll_interval": poll_interval,
        },
        name="langgraph-agentmail-loop",
        daemon=True,
    )
    print(f"Agente LangGraph iniciado. Sondeo cada {poll_interval:g} segundos.")
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
                        "  help, h, ?          Ayuda\n"
                        "  stop, quit, q       Detiene el agente"
                    )
                elif normalized in {"status", "s"}:
                    print(
                        f"Hora simulada: {state['hora_actual']} | "
                        f"Correos procesados: {len(state['correos_procesados'])} | "
                        f"Solicitudes: {len(state['solicitudes'])}"
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
                elif normalized.startswith(("reply ", "rep ")):
                    selected = selected_message(command, messages)
                    if selected is not None:
                        _, message = selected
                        identifier = get_message_id(message)
                        if not identifier:
                            print("No se puede responder: el correo no tiene message_id.")
                            continue
                        body = read_multiline(session)
                        if body:
                            try:
                                client.reply(identifier, body)
                                print("Respuesta enviada por AgentMail.")
                            except AgentMailError as exc:
                                print(f"No se pudo enviar la respuesta: {exc}")
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
                    elif body == "":
                        print("El cuerpo está vacío; no se envió.")
                else:
                    print("Comando desconocido. Escribe `help`.")
    finally:
        stop_event.set()
        worker.join(timeout=REQUEST_TIMEOUT_SECONDS + poll_interval + 2)
        if worker.is_alive():
            print("El agente sigue terminando una solicitud de AgentMail en curso.")
        else:
            print("\nAgente detenido.")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Agente LangGraph continuo para leer y clasificar correos de AgentMail."
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=float(
            os.environ.get(
                "POLL_INTERVAL_SECONDS",
                str(DEFAULT_POLL_INTERVAL_SECONDS),
            )
        ),
        help=f"segundos entre sondeos (por defecto {DEFAULT_POLL_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=AUDIT_PATH,
        help="ruta del log JSON Lines de auditoría (por defecto auditoria.jsonl)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("AGENTMAIL_API_KEY", "").strip()
    inbox_id = os.environ.get("AGENTMAIL_INBOX_ID", "").strip()
    if not api_key or not inbox_id:
        print("Configura AGENTMAIL_API_KEY y AGENTMAIL_INBOX_ID en .env.")
        return 1
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds debe ser mayor que cero.")

    try:
        client = AgentMailClient(api_key, inbox_id)
        client.list_messages()
    except AgentMailError as exc:
        print(f"No se pudo conectar con AgentMail: {exc}")
        return 1

    print(f"Conexión confirmada con AgentMail: {inbox_id}")
    try:
        run_terminal_agent(
            client,
            audit_path=args.audit,
            poll_interval=args.poll_seconds,
        )
    except KeyboardInterrupt:
        print("\nAgente detenido.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
