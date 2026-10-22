from __future__ import annotations

from typing import Any, AsyncIterator
from urllib.parse import quote

from agentmail import AsyncAgentMail, MessageReceivedEvent, Subscribe, Subscribed
import requests

from exceptions import AgentMailError
from settings import API_BASE_URL, REQUEST_TIMEOUT_SECONDS

class AgentMailClient:
    """Cliente de AgentMail para API de correo y eventos WebSocket en tiempo real."""

    def __init__(self, api_key: str, inbox_id: str) -> None:
        self.api_key = api_key
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

    async def message_events(self) -> AsyncIterator[dict[str, Any]]:
        """Yield new inbox messages through AgentMail's WebSocket event stream."""
        sdk_client = AsyncAgentMail(api_key=self.api_key)
        try:
            async with sdk_client.websockets.connect() as socket:
                await socket.send_subscribe(
                    Subscribe(
                        inbox_ids=[self.inbox_id],
                        event_types=["message.received"],
                    )
                )
                async for event in socket:
                    if isinstance(event, Subscribed):
                        yield {"type": "subscribed"}
                    elif isinstance(event, MessageReceivedEvent):
                        yield {
                            "type": "message_received",
                            "event_type": event.event_type,
                            "event_id": event.event_id,
                            "message": serialize_agentmail_message(event.message),
                        }
                    else:
                        raise AgentMailError(
                            "AgentMail devolvió un evento WebSocket inesperado."
                        )
        except AgentMailError:
            raise
        except Exception as exc:
            raise AgentMailError(
                "Falló la conexión WebSocket con AgentMail "
                f"({type(exc).__name__})."
            ) from exc


def serialize_agentmail_message(message: Any) -> dict[str, Any]:
    """Convert the SDK email model into the JSON-compatible app representation."""
    if not hasattr(message, "model_dump"):
        raise AgentMailError("AgentMail devolvió un mensaje WebSocket inválido.")
    serialized = message.model_dump(
        mode="json",
        by_alias=True,
        exclude_none=True,
    )
    if not isinstance(serialized, dict):
        raise AgentMailError("AgentMail devolvió un mensaje WebSocket inválido.")
    return serialized
