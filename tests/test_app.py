import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from app import (
    AgentMailClient,
    JsonlAuditLog,
    classify_by_rules,
    get_message_id,
    get_sender,
    request_fingerprint,
    run_agent_forever,
    run_phase_one,
)


def make_message(
    identifier: str,
    subject: str,
    body: str,
    *,
    sender: str = "pm@example.test",
    **extra: object,
) -> dict[str, object]:
    return {
        "message_id": identifier,
        "from": sender,
        "subject": subject,
        "text": body,
        "timestamp": f"2026-10-09T10:{identifier[-1]}0:00Z",
        **extra,
    }


def test_classifies_position_request_in_spanish() -> None:
    category, _ = classify_by_rules(
        make_message("m1", "Solicitud de posición", "Solicitamos un ingeniero."),
        known_fingerprints=set(),
        known_request_ids=set(),
    )
    assert category == "solicitud_posicion"


def test_classifies_position_request_in_english() -> None:
    category, _ = classify_by_rules(
        make_message("m1", "Position request", "We have a position request."),
        known_fingerprints=set(),
        known_request_ids=set(),
    )
    assert category == "solicitud_posicion"


def test_classifies_unrelated_email() -> None:
    category, _ = classify_by_rules(
        make_message("m1", "Invitación", "Nos vemos en la reunión del martes."),
        known_fingerprints=set(),
        known_request_ids=set(),
    )
    assert category == "no_relacionado"


def test_classifies_duplicate_by_normalized_content() -> None:
    original = make_message("m1", "Solicitud de posición", "Solicitamos un ingeniero.")
    duplicate = make_message("m2", "RE: Solicitud de posición", "Solicitamos un ingeniero.")
    category, reason = classify_by_rules(
        duplicate,
        known_fingerprints={request_fingerprint(original)},
        known_request_ids={"m1"},
    )
    assert category == "duplicado"
    assert "coincide" in reason


def test_classifies_reply_as_duplicate_using_reference() -> None:
    category, _ = classify_by_rules(
        make_message(
            "m2",
            "Re: Solicitud",
            "Gracias, recibimos la solicitud.",
            in_reply_to="m1",
        ),
        known_fingerprints=set(),
        known_request_ids={"m1"},
    )
    assert category == "duplicado"


def test_message_id_accepts_agentmail_snake_case() -> None:
    assert get_message_id({"message_id": "m-1"}) == "m-1"


def test_message_id_accepts_legacy_camel_case() -> None:
    assert get_message_id({"messageId": "m-2"}) == "m-2"


def test_agentmail_client_fetches_all_pages(monkeypatch) -> None:
    first_response = Mock()
    first_response.raise_for_status.return_value = None
    first_response.json.return_value = {
        "messages": [{"message_id": "m1"}],
        "next_page_token": "page-2",
    }
    second_response = Mock()
    second_response.raise_for_status.return_value = None
    second_response.json.return_value = {"messages": [{"message_id": "m2"}]}
    request = Mock(side_effect=[first_response, second_response])
    monkeypatch.setattr("app.requests.request", request)
    client = AgentMailClient("unit-test-key", "inbox@example.test")

    messages = client.list_messages()

    assert [message["message_id"] for message in messages] == ["m1", "m2"]
    assert request.call_count == 2
    assert request.call_args_list[0].args[:2] == (
        "GET",
        "https://api.agentmail.to/v0/inboxes/inbox%40example.test/messages",
    )
    assert request.call_args_list[0].kwargs["headers"]["Authorization"] == (
        "Bearer unit-test-key"
    )
    assert request.call_args_list[1].kwargs["params"]["page_token"] == "page-2"


def test_sender_is_safe_when_agentmail_returns_a_string() -> None:
    assert get_sender({"from": "Nombre <pm@example.test>"}) == "Nombre <pm@example.test>"


def test_sender_accepts_structured_value() -> None:
    assert get_sender({"from": {"name": "PM", "email": "pm@example.test"}}) == (
        "PM <pm@example.test>"
    )


def test_audit_log_writes_one_json_object_per_line(tmp_path: Path) -> None:
    log_path = tmp_path / "audit" / "events.jsonl"
    audit = JsonlAuditLog(log_path)
    expected = {
        "hora_simulada": 0,
        "solicitud": "m1",
        "nodo": "clasificar_por_reglas",
        "decision": "solicitud_posicion",
        "motivo": "marcador detectado",
    }
    audit.append(expected)
    assert [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()] == [
        expected
    ]


def test_graph_processes_and_notifies_each_category(tmp_path: Path, capsys) -> None:
    request = make_message("m1", "Solicitud de posición", "Solicitamos un ingeniero.")
    duplicate = make_message("m2", "RE: Solicitud de posición", "Solicitamos un ingeniero.")
    unrelated = make_message("m3", "Invitación", "Reunión el martes.")
    client = FakeAgentMailClient([[request, duplicate, unrelated]])

    result = run_phase_one(client, audit_path=tmp_path / "audit.jsonl")

    assert result["correos_procesados"] == ["m1", "m2", "m3"]
    assert [item["categoria"] for item in result["correos_clasificados"]] == [
        "solicitud_posicion",
        "duplicado",
        "no_relacionado",
    ]
    assert len(result["solicitudes"]) == 1
    assert "NOTIFICACIÓN SIMULADA" in capsys.readouterr().out


def test_graph_processes_one_inbox_poll_per_invocation(tmp_path: Path) -> None:
    first = make_message("m1", "Solicitud de posición", "Solicitamos un ingeniero.")
    client = FakeAgentMailClient([[first]])

    result = run_phase_one(client, audit_path=tmp_path / "audit.jsonl")

    assert result["correos_procesados"] == ["m1"]
    assert client.list_calls == 1
    assert result["hora_actual"] == 1


def test_service_repeats_graph_until_stopped_and_emits_heartbeat(
    tmp_path: Path,
    capsys,
) -> None:
    first = make_message("m1", "Solicitud de posición", "Solicitamos un ingeniero.")
    second = make_message("m2", "Solicitud de puesto", "Requerimos una analista.")
    stop_event = threading.Event()
    client = FakeAgentMailClient(
        [[first], [first, second], [first, second]],
        stop_event=stop_event,
        stop_after_calls=3,
    )
    published_states = []

    result = run_agent_forever(
        client,
        audit_path=tmp_path / "audit.jsonl",
        stop_event=stop_event,
        on_state=published_states.append,
        poll_interval=0.001,
        heartbeat_interval=1e-9,
    )

    assert client.list_calls == 3
    assert result["hora_actual"] == 3
    assert result["correos_procesados"] == ["m1", "m2"]
    assert len(published_states) == 3
    assert "." in capsys.readouterr().out


def test_service_rejects_nonpositive_poll_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="poll_interval"):
        run_agent_forever(
            FakeAgentMailClient([[]]),
            audit_path=tmp_path / "audit.jsonl",
            stop_event=threading.Event(),
            on_state=lambda _state: None,
            poll_interval=0,
        )


def test_graph_audit_contains_required_fields(tmp_path: Path) -> None:
    message = make_message("m1", "Solicitud de posición", "Solicitamos un ingeniero.")
    audit_path = tmp_path / "audit.jsonl"
    run_phase_one(FakeAgentMailClient([[message]]), audit_path=audit_path)

    events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert events
    assert all(
        {"hora_simulada", "solicitud", "nodo", "decision", "motivo"} <= event.keys()
        for event in events
    )


def test_agentmail_client_reply_and_send_use_real_endpoints(monkeypatch) -> None:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"message_id": "sent-1"}
    request = Mock(return_value=response)
    monkeypatch.setattr("app.requests.request", request)
    client = AgentMailClient("unit-test-key", "inbox@example.test")

    client.reply("m-1", "Respuesta")
    client.send("to@example.test", "Asunto", "Cuerpo")

    assert request.call_args_list[0].args[:2] == (
        "POST",
        "https://api.agentmail.to/v0/inboxes/inbox%40example.test/messages/m-1/reply",
    )
    assert request.call_args_list[0].kwargs["json"] == {"text": "Respuesta"}
    assert request.call_args_list[1].args[1].endswith("/messages/send")
    assert request.call_args_list[1].kwargs["json"] == {
        "to": "to@example.test",
        "subject": "Asunto",
        "text": "Cuerpo",
    }


class FakeAgentMailClient(AgentMailClient):
    def __init__(
        self,
        inbox_responses: list[list[dict[str, Any]]],
        *,
        stop_event: threading.Event | None = None,
        stop_after_calls: int | None = None,
    ) -> None:
        self.inbox_id = "test@agentmail.to"
        self.inbox_responses = inbox_responses
        self.list_calls = 0
        self.stop_event = stop_event
        self.stop_after_calls = stop_after_calls
        self.message_by_id = {
            message["message_id"]: message
            for page in inbox_responses
            for message in page
        }

    def list_messages(self) -> list[dict[str, Any]]:
        index = min(self.list_calls, len(self.inbox_responses) - 1)
        self.list_calls += 1
        if (
            self.stop_event is not None
            and self.stop_after_calls is not None
            and self.list_calls >= self.stop_after_calls
        ):
            self.stop_event.set()
        return list(self.inbox_responses[index])

    def get_message(self, identifier: str) -> dict[str, Any]:
        return dict(self.message_by_id[identifier])
