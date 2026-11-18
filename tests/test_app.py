import asyncio
import builtins
import json
import sys
import threading
import time
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from agentmail import MessageReceivedEvent, Subscribed

from agent_config import AgentConfig, load_agent_config
from app import (
    ASSIGNMENT_REPORT_PATH,
    AgentMailClient,
    AgentStateStoreError,
    JsonlAuditLog,
    OpenRouterClient,
    OpenRouterError,
    apply_human_confirmation,
    apply_pm_update,
    assign_pm_to_request,
    classify_by_rules,
    complete_request,
    close_request,
    configured_max_llm_calls,
    get_message_id,
    get_sender,
    initial_agent_state,
    load_assignments,
    load_director_assignments,
    load_pm_profiles,
    main,
    mark_request_opened,
    request_fingerprint,
    run_agent_forever,
    run_phase_one,
    run_simulation,
    process_due_reminders,
    simulated_confirmation_delay,
    set_pm_eligibility,
    write_sla_report,
)
from state_store import load_agent_state, save_agent_state
from email_utils import detect_pm_unavailability, infer_request_status
from sla import build_director_approval_graph
from terminal import completed_requests, run_terminal_agent
from pm_assignment import count_open_projects


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


def test_completed_requests_remain_available_in_history() -> None:
    state = initial_agent_state()
    state["solicitudes"] = [
        {"id": "m1", "asunto": "Confirmada", "estado": "Confirmada"},
        {"id": "m2", "asunto": "Activa", "estado": "Notificada"},
        {"id": "m3", "asunto": "Cerrada", "estado": "Cerrada"},
    ]

    history = completed_requests(state)

    assert [index for index, _ in history] == [3]
    assert [request["id"] for _, request in history] == ["m3"]
    assert len(state["solicitudes"]) == 3


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


def test_rules_detect_client_requirement_without_formal_request_phrase() -> None:
    category, _ = classify_by_rules(
        make_message(
            "m1",
            "High priority client requirement",
            "The client has a high priority requirement for 4 full stack engineers.",
        ),
        known_fingerprints=set(),
        known_request_ids=set(),
    )

    assert category == "solicitud_posicion"


@pytest.mark.parametrize(
    ("subject", "body"),
    [
        (
            "Solicitud urgente de 4 desarrolladores Frontend",
            "Tenemos una nueva solicitud por parte de nuestro cliente para "
            "encontrar 4 desarrolladores frontend.",
        ),
        (
            "Requerimiento de 2 ingenieros DevOps",
            "Nuestro cliente está buscando 2 ingenieros DevOps para fortalecer "
            "su infraestructura.",
        ),
    ],
)
def test_rules_detect_natural_language_staffing_requests(
    subject: str,
    body: str,
) -> None:
    category, _ = classify_by_rules(
        make_message("natural-request", subject, body),
        known_fingerprints=set(),
        known_request_ids=set(),
    )

    assert category == "solicitud_posicion"


def test_rules_distinguish_pm_acceptance_from_completed_work() -> None:
    assert infer_request_status("Me hago cargo de esta solicitud.") == "confirmada"
    assert infer_request_status("I will take ownership of this.") == "confirmada"
    assert infer_request_status("The positions are filled.") == "completada"
    assert infer_request_status("Solicitud completada.") == "completada"


def test_positive_staffing_rules_override_llm_unrelated_result(
    tmp_path: Path,
) -> None:
    class RejectingClassifier:
        model = "test"

        def classify_message(self, message, known_requests):
            return {
                "categoria": "no_relacionado",
                "motivo": "El clasificador de prueba no detectó la solicitud.",
            }

    message = make_message(
        "llm-conflict",
        "Solicitud urgente de 4 desarrolladores Frontend",
        "Tenemos una nueva solicitud para encontrar 4 desarrolladores frontend.",
    )
    result = run_phase_one(
        FakeAgentMailClient([[message]]),
        audit_path=tmp_path / "audit.jsonl",
        assignments={"Acme": "PM Ana"},
        llm=RejectingClassifier(),
    )

    assert [request["id"] for request in result["solicitudes"]] == [
        "llm-conflict"
    ]
    classification = next(
        event
        for event in result["eventos"]
        if event.get("decision") == "solicitud_posicion"
    )
    assert classification["datos"]["metodo_clasificacion"] == "reglas_validacion"
    assert "OpenRouter la marcó como no relacionada" in classification["motivo"]


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


def test_agentmail_websocket_subscribes_to_inbox_and_yields_messages(
    monkeypatch,
) -> None:
    received = MessageReceivedEvent.model_construct(
        event_type="message.received",
        event_id="evt-1",
        message={
            "inbox_id": "inbox@example.test",
            "thread_id": "thread-1",
            "message_id": "m1",
            "labels": [],
            "timestamp": "2026-01-01T00:00:00Z",
            "from": "pm@example.test",
            "to": [],
            "subject": "Solicitud",
            "text": "Solicitamos una persona.",
            "size": 1,
            "updated_at": "2026-01-01T00:00:00Z",
            "created_at": "2026-01-01T00:00:00Z",
        },
        thread={},
    )
    subscribed = Subscribed.model_construct()

    class FakeSocket:
        def __init__(self) -> None:
            self.events = iter([subscribed, received])
            self.subscription = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return exc_type is None and exc is None and traceback is None

        async def send_subscribe(self, subscription) -> None:
            self.subscription = subscription

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.events)
            except StopIteration:
                raise StopAsyncIteration

    socket = FakeSocket()

    class FakeSdkClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "unit-test-key"
            self.websockets = Mock()
            self.websockets.connect.return_value = socket

    monkeypatch.setattr("agentmail_client.AsyncAgentMail", FakeSdkClient)
    client = AgentMailClient("unit-test-key", "inbox@example.test")

    async def collect_events():
        return [event async for event in client.message_events()]

    import asyncio

    events = asyncio.run(collect_events())

    assert events[0] == {"type": "subscribed"}
    assert events[1]["type"] == "message_received"
    assert events[1]["message"]["message_id"] == "m1"
    assert events[1]["message"]["from"] == "pm@example.test"
    assert socket.subscription.inbox_ids == ["inbox@example.test"]
    assert socket.subscription.event_types == ["message.received"]


def test_openrouter_client_sends_model_and_parses_json(monkeypatch) -> None:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "categoria": "solicitud_posicion",
                            "cliente": "Acme",
                            "rol": "Analista",
                            "cantidad_posiciones": 2,
                            "pm_responsable": None,
                            "motivo": "Solicita personal para un cliente.",
                            "pregunta_confirmacion": "¿Confirmó el PM la solicitud para Acme?",
                            "preguntas_aclaracion": [],
                        }
                    )
                }
            }
        ]
    }
    request = Mock(return_value=response)
    monkeypatch.setattr("app.requests.post", request)
    model = OpenRouterClient("unit-test-key", "provider/test-model")

    result = model.classify_message(
        {
            "message_id": "m1",
            "subject": "Vacante",
            "text": "Buscamos dos analistas para Acme.",
        },
        [],
    )

    assert result["categoria"] == "solicitud_posicion"
    assert result["cantidad_posiciones"] == 2
    assert "Confirmó" in result["pregunta_confirmacion"]
    assert request.call_args.args[0] == (
        "https://openrouter.ai/api/v1/chat/completions"
    )
    assert request.call_args.kwargs["headers"]["Authorization"] == (
        "Bearer unit-test-key"
    )
    assert request.call_args.kwargs["json"]["model"] == "provider/test-model"
    assert request.call_args.kwargs["headers"]["Authorization"].startswith(
        "Bearer "
    )
    assert request.call_args.kwargs["headers"]["Authorization"].endswith(
        "unit-test-key"
    )
    assert request.call_args.kwargs["json"]["response_format"] == {
        "type": "json_object"
    }
    prompt_data = json.loads(
        request.call_args.kwargs["json"]["messages"][1]["content"]
    )
    assert prompt_data["message_id"] == "m1"


def test_openrouter_client_writes_human_review_email(monkeypatch) -> None:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "cuerpo": (
                                "Hola PM Ana, ¿puedes confirmar el rol y la "
                                "cantidad de posiciones?"
                            )
                        }
                    )
                }
            }
        ]
    }
    request = Mock(return_value=response)
    monkeypatch.setattr("app.requests.post", request)
    model = OpenRouterClient("unit-test-key", "provider/test-model")

    body = model.write_clarification_email(
        {
            "subject": "Solicitud de personal",
            "from": "solicitante@example.test",
            "text": "Buscamos personal para Acme.",
        },
        {
            "id": "request-13",
            "cliente": "Acme",
            "rol": None,
            "cantidad_posiciones": None,
            "pm_responsable": "PM Ana",
            "datos_faltantes": ["rol", "cantidad_posiciones"],
        },
    )

    assert "confirmar el rol" in body
    prompt_data = json.loads(
        request.call_args.kwargs["json"]["messages"][1]["content"]
    )
    assert prompt_data["solicitud"]["id"] == "request-13"
    assert prompt_data["correo_original"]["remitente"] == (
        "solicitante@example.test"
    )


def test_ambiguous_request_emails_assigned_pm_for_human_review(
    tmp_path: Path,
) -> None:
    message = make_message(
        "request-13",
        "Solicitud de personal",
        "Cliente: Acme\nBuscamos personal.",
        sender="requester@example.test",
    )
    client = FakeAgentMailClient([[message]])
    sent_messages: list[tuple[str, str, str]] = []
    client.send = lambda recipient, subject, text: sent_messages.append(
        (recipient, subject, text)
    )

    class ClarifyingLLM:
        model = "test-model"

        def classify_message(self, message, known_requests):
            assert message["message_id"] == "request-13"
            assert known_requests == []
            return {
                "categoria": "solicitud_posicion",
                "solicitud_relacionada_id": None,
                "cliente": "Acme",
                "rol": None,
                "cantidad_posiciones": None,
                "pm_responsable": None,
                "posiciones": [],
                "estado_solicitud": None,
                "motivo": "Faltan datos para definir la solicitud.",
                "pregunta_confirmacion": None,
                "preguntas_aclaracion": [
                    "¿Qué rol se necesita?",
                    "¿Cuántas posiciones se solicitan?",
                ],
            }

        def write_clarification_email(self, message, request):
            assert message["subject"] == "Solicitud de personal"
            assert request["campos_revision_humana"] == [
                "rol",
                "cantidad_posiciones",
            ]
            return "Hola PM Ana, confirma el rol y la cantidad solicitados."

    result = run_phase_one(
        client,
        audit_path=tmp_path / "audit.jsonl",
        state={
            **initial_agent_state(),
            "simular_confirmacion_automatica": False,
        },
        assignments={"Acme": "PM Ana"},
        pm_profiles={
            "PM Ana": {
                "nombre": "PM Ana",
                "perfil": "normal",
                "elegible": True,
                "emails": ["ana@example.test"],
            }
        },
        llm=ClarifyingLLM(),
        config=AgentConfig(),
    )

    request = result["solicitudes"][0]
    assert request["revision_humana"] is True
    assert request["campos_revision_humana"] == [
        "rol",
        "cantidad_posiciones",
    ]
    assert request["correo_revision_humana_enviado"] is True
    assert request["hora_notificacion"] is None
    assert sent_messages == [
        (
            "ana@example.test",
            "Revisión humana [Solicitud request-13]: Solicitud de personal",
            "Hola PM Ana, confirma el rol y la cantidad solicitados.",
        )
    ]
    assert result["llamadas_llm"] == 2
    assert any(
        event["decision"] == "correo_revision_humana_enviado"
        for event in result["eventos"]
    )

    response = make_message(
        "request-13-reply",
        sent_messages[0][1],
        "Rol: Analista\nCantidad: 2",
        sender="ana@example.test",
        in_reply_to="clarification-message",
    )

    class ResolvingLLM:
        model = "test-model"

        def classify_message(self, message, known_requests):
            assert message["message_id"] == "request-13-reply"
            assert known_requests[0]["id"] == "request-13"
            return {
                "categoria": "actualizacion_solicitud",
                "solicitud_relacionada_id": "request-13",
                "cliente": None,
                "rol": "Analista",
                "cantidad_posiciones": 2,
                "pm_responsable": None,
                "posiciones": [],
                "estado_solicitud": None,
                "motivo": "El PM confirmó los datos pendientes.",
                "pregunta_confirmacion": None,
                "preguntas_aclaracion": [],
            }

    resolved = run_phase_one(
        FakeAgentMailClient([[response]]),
        audit_path=tmp_path / "audit.jsonl",
        state=result,
        assignments={"Acme": "PM Ana"},
        pm_profiles={
            "PM Ana": {
                "nombre": "PM Ana",
                "perfil": "normal",
                "elegible": True,
                "emails": ["ana@example.test"],
            }
        },
        llm=ResolvingLLM(),
        config=AgentConfig(),
    )

    resolved_request = resolved["solicitudes"][0]
    assert resolved_request["revision_humana"] is False
    assert resolved_request["campos_revision_humana"] == []
    assert resolved_request["estado"] == "Notificada"
    assert resolved_request["hora_notificacion"] is not None
    assert resolved_request["fecha_revision_humana_resuelta"]


def test_ambiguous_request_simulation_does_not_send_email(
    tmp_path: Path,
) -> None:
    message = make_message(
        "request-ambiguous",
        "Solicitud de personal",
        "Cliente: Acme\nBuscamos personal.",
    )
    client = FakeAgentMailClient([[message]])
    sent_messages: list[tuple[str, str, str]] = []
    client.send = lambda recipient, subject, text: sent_messages.append(
        (recipient, subject, text)
    )

    result = run_phase_one(
        client,
        audit_path=tmp_path / "audit.jsonl",
        assignments={"Acme": "PM Ana"},
        pm_profiles={
            "PM Ana": {
                "nombre": "PM Ana",
                "perfil": "normal",
                "elegible": True,
                "emails": ["ana@example.test"],
            }
        },
        config=AgentConfig(),
    )

    assert result["solicitudes"][0]["revision_humana"] is True
    assert result["solicitudes"][0]["correo_revision_humana_enviado"] is False
    assert sent_messages == []
    assert any(
        event["decision"] == "revision_humana_pendiente"
        for event in result["eventos"]
    )


def test_ambiguous_request_retries_llm_and_uses_email_template(
    tmp_path: Path,
) -> None:
    message = make_message(
        "request-template",
        "Solicitud de personal",
        "Cliente: Acme\nBuscamos personal.",
        sender="requester@example.test",
    )
    client = FakeAgentMailClient([[message]])
    sent_messages: list[tuple[str, str, str]] = []
    client.send = lambda recipient, subject, text: sent_messages.append(
        (recipient, subject, text)
    )

    class FailingClarificationWriter:
        model = "test-model"

        def classify_message(self, message, known_requests):
            assert message["message_id"] == "request-template"
            assert known_requests == []
            return {
                "categoria": "solicitud_posicion",
                "solicitud_relacionada_id": None,
                "cliente": "Acme",
                "rol": None,
                "cantidad_posiciones": None,
                "pm_responsable": None,
                "posiciones": [],
                "estado_solicitud": None,
                "motivo": "Faltan datos para definir la solicitud.",
                "pregunta_confirmacion": None,
                "preguntas_aclaracion": [],
            }

        def write_clarification_email(self, message, request):
            assert message["message_id"] == "request-template"
            assert request["revision_humana"] is True
            raise OpenRouterError("salida inválida")

    result = run_phase_one(
        client,
        audit_path=tmp_path / "audit.jsonl",
        state={
            **initial_agent_state(),
            "simular_confirmacion_automatica": False,
        },
        assignments={"Acme": "PM Ana"},
        pm_profiles={
            "PM Ana": {
                "nombre": "PM Ana",
                "perfil": "normal",
                "elegible": True,
                "emails": ["ana@example.test"],
            }
        },
        llm=FailingClarificationWriter(),
        config=AgentConfig(),
    )

    assert len(sent_messages) == 1
    assert "Cliente: Acme" in sent_messages[0][2]
    assert "Remitente original: requester@example.test" in sent_messages[0][2]
    assert "Buscamos personal." in sent_messages[0][2]
    assert result["llamadas_llm"] == 3
    assert sum(
        event["decision"] == "error" for event in result["eventos"]
    ) == 2
    sent_event = next(
        event
        for event in result["eventos"]
        if event["decision"] == "correo_revision_humana_enviado"
    )
    assert sent_event["datos"]["metodo_redaccion"] == "plantilla"


def test_openrouter_processes_prose_and_position_table(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    prose_positions = [
        {
            "rol": "Full Stack Java-React Engineer - Frontend",
            "cantidad_posiciones": 2,
            "ubicacion": "LATAM",
            "skills": "React, TypeScript",
        },
        {
            "rol": "Full Stack Java-React Engineer - Backend",
            "cantidad_posiciones": 2,
            "ubicacion": "LATAM",
            "skills": "Java, Spring Boot",
        },
    ]
    table_positions = [
        {"rol": "Mainframe Dev", "cantidad_posiciones": 8, "ubicacion": "US-Onshore", "skills": "AS400"},
        {"rol": "Full stack Dev", "cantidad_posiciones": 10, "ubicacion": "US-Onshore", "skills": "Java full-stack"},
        {"rol": "Full stack Dev", "cantidad_posiciones": 5, "ubicacion": "Offshore", "skills": "Java full-stack"},
        {"rol": "QE", "cantidad_posiciones": 3, "ubicacion": "US-Onshore", "skills": "Selenium, Playwright"},
        {"rol": "QE", "cantidad_posiciones": 3, "ubicacion": "Offshore", "skills": "Selenium, Playwright"},
        {"rol": "Data engineer", "cantidad_posiciones": 3, "ubicacion": "US-Onshore", "skills": "Python, SQL"},
        {"rol": "Data engineer", "cantidad_posiciones": 2, "ubicacion": "Offshore", "skills": "Python, SQL"},
    ]
    completions = {
        "m1": {
            "cliente": "Acme",
            "rol": "Full Stack Java-React Engineer",
            "cantidad_posiciones": 4,
            "posiciones": prose_positions,
        },
        "m2": {
            "cliente": "Acme",
            "rol": "Mainframe Dev / Full stack Dev / QE / Data engineer",
            "cantidad_posiciones": 34,
            "posiciones": table_positions,
        },
    }
    prompts: list[dict[str, object]] = []

    def fake_completion(*args, **kwargs):
        prompt = kwargs["json"]
        prompts.append(prompt)
        message_data = json.loads(prompt["messages"][1]["content"])
        result = {
            "categoria": "solicitud_posicion",
            **completions[message_data["message_id"]],
            "pm_responsable": None,
            "motivo": "El correo describe necesidades de personal.",
            "pregunta_confirmacion": "¿El PM recibió las posiciones solicitadas?",
            "preguntas_aclaracion": [],
        }
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(result)}}]
        }
        return response

    monkeypatch.setattr("app.requests.post", fake_completion)
    messages = [
        make_message(
            "m1",
            "High priority client requirement",
            "Good Morning Team. We have a high priority requirement for the client "
            "for 4 Full Stack Engineers. 2 should be front end focused and 2 more "
            "back end focused. Must-have: Java, Spring Boot, React, TypeScript, "
            "REST API, Kubernetes and Kafka.",
        ),
        make_message(
            "m2",
            "Latest project requirements",
            "Role | High-level Skillset | Location | Total positions\n"
            "Mainframe Dev | AS400 | US-Onshore | 8\n"
            "Full stack Dev | Java full-stack | US-Onshore | 10\n"
            "Full stack Dev | Java full-stack | Offshore | 5\n"
            "QE | Selenium, Playwright | US-Onshore | 3\n"
            "QE | Selenium, Playwright | Offshore | 3\n"
            "Data engineers | Python, SQL | US-Onshore | 3\n"
            "Data engineers | Python, SQL | Offshore | 2",
        ),
    ]
    llm = OpenRouterClient("unit-test-key", "provider/test-model")

    result = run_phase_one(
        FakeAgentMailClient([messages]),
        audit_path=tmp_path / "audit.jsonl",
        assignments={"Acme": "PM Ana"},
        pm_profiles={"PM Ana": {"perfil": "normal"}},
        llm=llm,
    )

    assert [request["cantidad_posiciones"] for request in result["solicitudes"]] == [
        4,
        34,
    ]
    assert result["solicitudes"][0]["posiciones"] == prose_positions
    assert result["solicitudes"][1]["posiciones"] == table_positions
    assert all(request["pm_responsable"] == "PM Ana" for request in result["solicitudes"])
    assert all(request["hora_notificacion"] == 0 for request in result["solicitudes"])
    assert len(prompts) == 2
    system_prompt = prompts[0]["messages"][0]["content"]
    assert "No exijas frases formales" in system_prompt
    assert "posiciones" in system_prompt
    assert len(capsys.readouterr().out.split("[NOTIFICACIÓN SIMULADA]")) == 3


def test_openrouter_extracts_pm_update_from_natural_language(monkeypatch) -> None:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "cliente": "Acme",
                            "rol": None,
                            "cantidad_posiciones": 4,
                            "pm_responsable": None,
                            "posiciones": [],
                            "motivo": "El PM precisó la cantidad.",
                        }
                    )
                }
            }
        ]
    }
    request = Mock(return_value=response)
    monkeypatch.setattr("app.requests.post", request)

    update = OpenRouterClient(
        "unit-test-key",
        "provider/test-model",
    ).extract_pm_update(
        "La solicitud es para Acme y son cuatro personas.",
        {"id": "m1", "datos_faltantes": ["cantidad_posiciones"]},
    )

    assert update == {"cliente": "Acme", "cantidad_posiciones": 4}
    prompt_data = json.loads(
        request.call_args.kwargs["json"]["messages"][1]["content"]
    )
    assert prompt_data["solicitud_actual"]["id"] == "m1"


def test_pm_natural_language_update_completes_missing_request_fields(
    tmp_path: Path,
) -> None:
    message = make_message(
        "m1",
        "Solicitud de personal",
        "Cliente: Acme",
    )
    audit_path = tmp_path / "audit.jsonl"
    state = run_phase_one(
        FakeAgentMailClient([[message]]),
        audit_path=audit_path,
        assignments={"Acme": "PM Ana"},
        pm_profiles={"PM Ana": {"perfil": "normal"}},
    )

    class FakeOpenRouter:
        def extract_pm_update(self, text, request):
            assert "four engineers" in text
            assert request["id"] == "m1"
            return {"rol": "Full Stack Engineer", "cantidad_posiciones": 4}

    apply_pm_update(
        state,
        request_number=1,
        response_text="We need four engineers.",
        audit=JsonlAuditLog(audit_path),
        llm=FakeOpenRouter(),
        assignments={"Acme": "PM Ana"},
    )

    request = state["solicitudes"][0]
    assert request["rol"] == "Full Stack Engineer"
    assert request["cantidad_posiciones"] == 4
    assert request["datos_faltantes"] == []
    assert request["estado"] == "Notificada"
    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["decision"] == "solicitud_actualizada_por_pm" for event in events)


def test_openrouter_update_failure_is_audited(tmp_path: Path) -> None:
    message = make_message(
        "m1",
        "Solicitud de personal",
        "Cliente: Acme\nRol: Analista\nCantidad: 2",
    )
    audit_path = tmp_path / "audit.jsonl"
    state = run_phase_one(
        FakeAgentMailClient([[message]]),
        audit_path=audit_path,
        assignments={"Acme": "PM Ana"},
    )

    class FailingOpenRouter:
        def extract_pm_update(self, text, request):
            raise OpenRouterError("No se pudo interpretar la respuesta.")

    with pytest.raises(OpenRouterError, match="interpretar"):
        apply_pm_update(
            state,
            request_number=1,
            response_text="Con una frase normal.",
            audit=JsonlAuditLog(audit_path),
            llm=FailingOpenRouter(),
        )

    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    error_event = events[-1]
    assert error_event["decision"] == "error"
    assert error_event["nodo"] == "respuesta_pm_terminal"
    assert error_event["solicitud"] == "m1"


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
    body = "Cliente: Acme\nRol solicitado: Ingeniero\nCantidad: 1"
    request = make_message("m1", "Solicitud de posición", body)
    duplicate = make_message("m2", "RE: Solicitud de posición", body)
    unrelated = make_message("m3", "Invitación", "Reunión el martes.")
    client = FakeAgentMailClient([[request, duplicate, unrelated]])

    result = run_phase_one(
        client,
        audit_path=tmp_path / "audit.jsonl",
        assignments={"Acme": "PM Ana"},
    )

    assert result["correos_procesados"] == ["m1", "m2", "m3"]
    assert [item["categoria"] for item in result["correos_clasificados"]] == [
        "solicitud_posicion",
        "duplicado",
        "no_relacionado",
    ]
    assert len(result["solicitudes"]) == 1
    assert result["llamadas_llm"] == 0
    assert "NOTIFICACIÓN SIMULADA" in capsys.readouterr().out
    events = [
        json.loads(line)
        for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(
        event["datos"]["metodo_clasificacion"] == "reglas"
        for event in events
        if event["nodo"] == "clasificar_por_reglas"
    )


def test_graph_uses_openrouter_for_natural_language_and_assignment(
    tmp_path: Path,
    capsys,
) -> None:
    class FakeOpenRouter:
        model = "provider/test-model"

        def classify_message(self, message, known_requests):
            assert message["message_id"] == "m1"
            assert isinstance(known_requests, list)
            return {
                "categoria": "solicitud_posicion",
                "cliente": "Acme",
                "rol": "Analista de datos",
                "cantidad_posiciones": 2,
                "pm_responsable": None,
                "motivo": "El correo pide dos perfiles para Acme.",
                "pregunta_confirmacion": (
                    "¿El PM confirmó recepción de dos analistas para Acme?"
                ),
                "preguntas_aclaracion": [],
            }

    message = make_message(
        "m1",
        "Necesitamos perfiles para el nuevo proyecto",
        "Buscamos dos analistas de datos para apoyar al equipo de Acme.",
    )
    audit_path = tmp_path / "audit.jsonl"

    result = run_phase_one(
        FakeAgentMailClient([[message]]),
        audit_path=audit_path,
        assignments={"Acme": "PM Santiago"},
        llm=FakeOpenRouter(),
    )

    assert result["correos_clasificados"][0]["categoria"] == "solicitud_posicion"
    request = result["solicitudes"][0]
    assert request["cliente"] == "Acme"
    assert request["rol"] == "Analista de datos"
    assert request["cantidad_posiciones"] == 2
    assert request["pm_responsable"] == "PM Allisson"
    assert "dos analistas" in request["pregunta_confirmacion"]
    assert request["preguntas_aclaracion"] == []
    assert request["estado"] == "Notificada"
    assert request["hora_notificacion"] == 0
    assert result["llamadas_llm"] == 1
    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        event["decision"] == "solicitud_posicion"
        and event["datos"]["metodo_clasificacion"] == "openrouter"
        for event in events
    )
    assert "Para: PM Allisson" in capsys.readouterr().out


def test_openrouter_classifies_duplicate_against_existing_requests(
    tmp_path: Path,
) -> None:
    class DuplicateAwareOpenRouter:
        model = "provider/test-model"

        def __init__(self) -> None:
            self.calls = 0

        def classify_message(self, message, known_requests):
            self.calls += 1
            if message["message_id"] == "m1":
                assert known_requests == []
                return {
                    "categoria": "solicitud_posicion",
                    "cliente": "Acme",
                    "rol": "Analista",
                    "cantidad_posiciones": 1,
                    "pm_responsable": "PM Ana",
                    "motivo": "Primera solicitud.",
                    "pregunta_confirmacion": "¿El PM se hará cargo de la solicitud para Acme?",
                    "preguntas_aclaracion": [],
                }
            assert known_requests[0]["id"] == "m1"
            return {
                "categoria": "duplicado",
                "cliente": None,
                "rol": None,
                "cantidad_posiciones": None,
                "pm_responsable": None,
                "motivo": "Reenvío de una solicitud existente.",
                "pregunta_confirmacion": None,
                "preguntas_aclaracion": [],
            }

    body = "Cliente: Acme\nRol: Analista\nCantidad: 1"
    llm = DuplicateAwareOpenRouter()

    result = run_phase_one(
        FakeAgentMailClient(
            [[
                make_message("m1", "Solicitud de personal", body),
                make_message("m2", "Reenvío de la solicitud", body),
            ]]
        ),
        audit_path=tmp_path / "audit.jsonl",
        llm=llm,
    )

    assert llm.calls == 2
    assert [email["categoria"] for email in result["correos_clasificados"]] == [
        "solicitud_posicion",
        "duplicado",
    ]
    assert len(result["solicitudes"]) == 1


def test_openrouter_retries_once_then_falls_back_to_rules(tmp_path: Path) -> None:
    class FailingOpenRouter:
        model = "provider/test-model"

        def __init__(self) -> None:
            self.calls = 0

        def classify_message(self, message, known_requests):
            self.calls += int(bool(message) and isinstance(known_requests, list))
            raise OpenRouterError("respuesta inválida")

    llm = FailingOpenRouter()
    message = make_message("m1", "Invitación", "Reunión de equipo el lunes.")
    audit_path = tmp_path / "audit.jsonl"

    result = run_phase_one(
        FakeAgentMailClient([[message]]),
        audit_path=audit_path,
        llm=llm,
    )

    assert llm.calls == 2
    assert result["llamadas_llm"] == 2
    assert result["correos_clasificados"][0]["categoria"] == "no_relacionado"
    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events[0]["decision"] == "error"
    assert events[0]["nodo"] == "clasificar_con_openrouter"
    classification = next(
        event
        for event in events
        if event["nodo"] == "clasificar_por_reglas"
    )
    assert "Respaldo por reglas" in classification["motivo"]
    assert classification["datos"]["metodo_clasificacion"] == "reglas"


def test_invalid_openrouter_output_retries_once_logs_and_falls_back(
    tmp_path: Path,
    monkeypatch,
) -> None:
    response_contents = iter(
        [
            "{not valid json",
            json.dumps({"categoria": "categoria_desconocida"}),
        ]
    )

    def fake_post(*args, **kwargs):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {"message": {"content": next(response_contents)}}
            ]
        }
        return response

    request = Mock(side_effect=fake_post)
    monkeypatch.setattr("app.requests.post", request)
    llm = OpenRouterClient("test-key", "provider/test-model")
    audit_path = tmp_path / "audit.jsonl"

    state = run_phase_one(
        FakeAgentMailClient(
            [[
                make_message(
                    "m-invalid-llm",
                    "Solicitud de personal",
                    "Cliente: Acme\nRol: Ingeniero\nCantidad: 2",
                )
            ]]
        ),
        audit_path=audit_path,
        assignments={"Acme": "PM Ana"},
        pm_profiles={"PM Ana": {"perfil": "normal"}},
        llm=llm,
    )

    assert request.call_count == 2
    assert state["llamadas_llm"] == 2
    assert state["solicitudes"][0]["cliente"] == "Acme"
    assert state["solicitudes"][0]["rol"] == "Ingeniero"
    assert state["solicitudes"][0]["cantidad_posiciones"] == 2
    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    errors = [
        event
        for event in events
        if event["nodo"] == "clasificar_con_openrouter"
    ]
    assert len(errors) == 2
    assert all(event["decision"] == "error" for event in errors)
    assert all(event["datos"]["respaldo_por_reglas"] is True for event in errors)
    classification = next(
        event for event in events if event["nodo"] == "clasificar_por_reglas"
    )
    assert classification["datos"]["metodo_clasificacion"] == "reglas"


def test_openrouter_budget_falls_back_to_rules(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MAX_LLM_CALLS", "60")

    class CountingOpenRouter:
        model = "provider/test-model"
        calls = 0

        def classify_message(self, message, known_requests):
            self.calls += int(bool(message) and isinstance(known_requests, list))
            return {}

    llm = CountingOpenRouter()
    state = initial_agent_state()
    state["llamadas_llm"] = 60
    message = make_message("m1", "Invitación", "Reunión de equipo el lunes.")
    audit_path = tmp_path / "audit.jsonl"

    result = run_phase_one(
        FakeAgentMailClient([[message]]),
        audit_path=audit_path,
        state=state,
        llm=llm,
    )

    assert llm.calls == 0
    assert result["llamadas_llm"] == 60
    event = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
    assert "límite de 60 llamadas" in event["motivo"]


def test_llm_call_limit_defaults_to_configured_hard_cap(monkeypatch) -> None:
    monkeypatch.delenv("MAX_LLM_CALLS", raising=False)

    assert configured_max_llm_calls() == 60


def test_openrouter_is_enabled_by_default_and_config_errors_are_audited(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("llm:\n  enabled: true\n", encoding="utf-8")
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr("app.load_dotenv", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "app.py",
            "--config",
            str(config_path),
            "--audit",
            str(audit_path),
        ],
    )
    monkeypatch.setenv("AGENTMAIL_API_KEY", "test-agentmail-key")
    monkeypatch.setenv("AGENTMAIL_INBOX_ID", "inbox@example.test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("USE_LLM", raising=False)

    assert main() == 1
    assert "OpenRouter está activado" in capsys.readouterr().out
    event = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
    assert event["decision"] == "error"
    assert event["nodo"] == "validar_configuracion"


def test_rules_only_config_runs_without_openrouter_credentials(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("llm:\n  enabled: false\n", encoding="utf-8")
    client = Mock(inbox_id="inbox@example.test")
    run_terminal = Mock()
    monkeypatch.setattr("app.load_dotenv", lambda: None)
    monkeypatch.setattr("app.AgentMailClient", Mock(return_value=client))
    monkeypatch.setattr("app.run_terminal_agent", run_terminal)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "app.py",
            "--config",
            str(config_path),
            "--audit",
            str(tmp_path / "audit.jsonl"),
            "--state",
            str(tmp_path / "state.json"),
        ],
    )
    monkeypatch.setenv("AGENTMAIL_API_KEY", "test-agentmail-key")
    monkeypatch.setenv("AGENTMAIL_INBOX_ID", "inbox@example.test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("USE_LLM", raising=False)

    assert main() == 0

    run_terminal.assert_called_once()
    assert run_terminal.call_args.kwargs["llm"] is None
    assert "OpenRouter está desactivado" in capsys.readouterr().out


def test_use_llm_environment_variable_overrides_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("llm:\n  enabled: true\n", encoding="utf-8")
    client = Mock(inbox_id="inbox@example.test")
    run_terminal = Mock()
    monkeypatch.setattr("app.load_dotenv", lambda: None)
    monkeypatch.setattr("app.AgentMailClient", Mock(return_value=client))
    monkeypatch.setattr("app.run_terminal_agent", run_terminal)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "app.py",
            "--config",
            str(config_path),
            "--audit",
            str(tmp_path / "audit.jsonl"),
        ],
    )
    monkeypatch.setenv("AGENTMAIL_API_KEY", "test-agentmail-key")
    monkeypatch.setenv("AGENTMAIL_INBOX_ID", "inbox@example.test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("USE_LLM", "false")

    assert main() == 0

    assert run_terminal.call_args.kwargs["llm"] is None


def test_cli_simulation_writes_final_sla_report(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    report_path = tmp_path / "final-report.json"
    report = {
        "modo": "sin_llm",
        "motivo_finalizacion": "sin_solicitudes",
        "horas_simuladas": 0,
        "solicitudes": [],
    }
    simulate = Mock(return_value=report)
    monkeypatch.setattr("app.load_dotenv", lambda: None)
    monkeypatch.setattr("app.AgentMailClient", Mock(return_value=Mock()))
    monkeypatch.setattr("app.run_simulation", simulate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "app.py",
            "--simulate",
            "--report",
            str(report_path),
            "--audit",
            str(tmp_path / "audit.jsonl"),
        ],
    )
    monkeypatch.setenv("AGENTMAIL_API_KEY", "test-agentmail-key")
    monkeypatch.setenv("AGENTMAIL_INBOX_ID", "inbox@example.test")
    monkeypatch.setenv("USE_LLM", "false")

    assert main() == 0
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert "Reporte SLA:" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["-1", "61", "unlimited"])
def test_llm_call_limit_rejects_invalid_values(monkeypatch, value: str) -> None:
    monkeypatch.setenv("MAX_LLM_CALLS", value)

    with pytest.raises(ValueError, match="MAX_LLM_CALLS"):
        configured_max_llm_calls()


def test_rf03_rf04_rf10_extract_assign_notify_and_audit(tmp_path: Path) -> None:
    assignments_path = tmp_path / "asignacion.csv"
    assignments_path.write_text(
        "cliente,PM,director\nAcme,PM Ana,Directora Sol\n",
        encoding="utf-8",
    )
    assignments = load_assignments(assignments_path)
    message = make_message(
        "m1",
        "Solicitud de personal",
        "Cliente: ACME\nRol solicitado: Ingeniera de datos\nCantidad: 2",
    )
    audit_path = tmp_path / "auditoria.jsonl"

    result = run_phase_one(
        FakeAgentMailClient([[message]]),
        audit_path=audit_path,
        assignments=assignments,
        pm_profiles={"PM Ana": {"perfil": "normal"}},
    )

    request = result["solicitudes"][0]
    assert request["cliente"] == "ACME"
    assert request["rol"] == "Ingeniera de datos"
    assert request["cantidad_posiciones"] == 2
    assert request["pm_responsable"] == "PM Ana"
    assert request["hora_notificacion"] == 0
    assert request["estado"] == "Notificada"
    assert result["hora_actual"] == 1

    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert all(
        {"hora_simulada", "solicitud", "nodo", "decision", "motivo"} <= event.keys()
        for event in events
    )
    classification = next(
        event
        for event in events
        if event["nodo"] == "clasificar_por_reglas"
    )
    notification = next(
        event
        for event in events
        if event["decision"] == "notificacion_simulada"
    )
    assert classification["datos"]["pm_responsable"] == "PM Ana"
    assert notification["hora_simulada"] == request["hora_notificacion"] == 0
    assert notification["datos"]["estado"] == "Notificada"
    assert all(
        event["decision"] not in {"sondeo_completado", "hora_avanzada", "correo_leido"}
        for event in events
    )


def test_rf03_extracts_prose_and_table_requests_without_llm(
    tmp_path: Path,
) -> None:
    messages = [
        make_message(
            "prose-1",
            "High priority client requirement",
            "Good morning. We have a high priority requirement for Acme for "
            "4 Full Stack Engineers. 2 should be frontend-focused and 2 backend.",
        ),
        make_message(
            "table-2",
            "Latest project requirements",
            "Client: Globex\n"
            "Role High-level Skillset Location Total positions\n"
            "Mainframe Dev AS400 US-Onshore 8\n"
            "Full stack Dev Java full-stack Offshore 5\n"
            "Data engineers Python, SQL Offshore 2",
        ),
    ]
    pm_profiles = {
        "pm ana": {
            "nombre": "PM Ana",
            "perfil": "normal",
            "horas_confirmacion": None,
            "elegible": True,
        },
        "pm luis": {
            "nombre": "PM Luis",
            "perfil": "normal",
            "horas_confirmacion": None,
            "elegible": True,
        },
    }

    state = run_phase_one(
        FakeAgentMailClient([messages]),
        audit_path=tmp_path / "audit.jsonl",
        assignments={"Acme": "PM Ana", "Globex": "PM Luis"},
        pm_profiles=pm_profiles,
        config=AgentConfig(testing_enabled=False),
    )

    prose, table = state["solicitudes"]
    assert prose["cliente"] == "Acme"
    assert prose["rol"] == "Full Stack Engineers"
    assert prose["cantidad_posiciones"] == 4
    assert prose["pm_responsable"] == "PM Ana"
    assert prose["metodo_clasificacion"] == "reglas"

    assert table["cliente"] == "Globex"
    assert table["rol"] == "Mainframe Dev / Full stack Dev / Data engineers"
    assert table["cantidad_posiciones"] == 15
    assert table["pm_responsable"] == "PM Luis"
    assert table["posiciones"] == [
        {
            "rol": "Mainframe Dev",
            "cantidad_posiciones": 8,
            "ubicacion": "US-Onshore",
            "skills": "AS400",
        },
        {
            "rol": "Full stack Dev",
            "cantidad_posiciones": 5,
            "ubicacion": "Offshore",
            "skills": "Java full-stack",
        },
        {
            "rol": "Data engineers",
            "cantidad_posiciones": 2,
            "ubicacion": "Offshore",
            "skills": "Python, SQL",
        },
    ]
    assert state["llamadas_llm"] == 0


def test_rule_assignment_skips_ineligible_pm(tmp_path: Path) -> None:
    state = run_phase_one(
        FakeAgentMailClient(
            [[
                make_message(
                    "m1",
                    "Solicitud de personal",
                    "Cliente: Acme\nRol: Analista\nCantidad: 1",
                )
            ]]
        ),
        audit_path=tmp_path / "audit.jsonl",
        assignments={"Acme": "PM Ana", "Beta": "PM Zoe"},
        pm_profiles={
            "pm ana": {
                "nombre": "PM Ana",
                "perfil": "normal",
                "elegible": False,
            },
            "pm zoe": {
                "nombre": "PM Zoe",
                "perfil": "normal",
                "elegible": True,
            },
        },
        config=AgentConfig(testing_enabled=False),
    )

    assert state["solicitudes"][0]["pm_responsable"] == "PM Zoe"


def test_new_request_goes_to_pm_with_fewest_open_projects(tmp_path: Path) -> None:
    state = initial_agent_state()
    state["solicitudes"] = [
        {
            "id": "existing-1",
            "estado": "Notificada",
            "pm_responsable": "PM Ana",
            "hora_notificacion": 0,
        }
    ]
    request_email = make_message(
        "new-1",
        "Solicitud de personal",
        "Cliente: Acme\nRol: Analista\nCantidad: 1",
    )

    result = run_phase_one(
        FakeAgentMailClient([[request_email]]),
        audit_path=tmp_path / "audit.jsonl",
        state=state,
        assignments={"Acme": "PM Ana", "Beta": "PM Luis"},
        pm_profiles={
            "PM Ana": {"perfil": "normal"},
            "PM Luis": {"perfil": "normal"},
        },
        config=AgentConfig(testing_enabled=False),
    )

    assert result["solicitudes"][1]["pm_responsable"] == "PM Luis"
    assignment_event = next(
        event
        for event in result["eventos"]
        if event.get("decision") == "pm_asignado_automaticamente"
    )
    assert assignment_event["datos"]["cargas_abiertas"] == {
        "PM Ana": 1,
        "PM Luis": 0,
    }


def test_no_pm_available_writes_assignment_problem_report(tmp_path: Path) -> None:
    messages = [
        make_message(
            "req-1",
            "Solicitud de personal",
            "Cliente: Acme\nRol: Analista\nCantidad: 1",
        ),
        make_message(
            "req-2",
            "Solicitud de personal",
            "Cliente: Beta\nRol: Ingeniero\nCantidad: 1",
        ),
        make_message(
            "req-3",
            "Solicitud de personal",
            "Cliente: Gamma\nRol: Tester\nCantidad: 1",
        ),
    ]
    audit_path = tmp_path / "audit.jsonl"

    result = run_phase_one(
        FakeAgentMailClient([messages]),
        audit_path=audit_path,
        assignments={"Acme": "PM Ana", "Beta": "PM Luis", "Gamma": "PM Ana"},
        pm_profiles={
            "PM Ana": {"perfil": "normal"},
            "PM Luis": {"perfil": "normal"},
        },
        config=AgentConfig(
            max_open_projects_per_pm=1,
            testing_enabled=False,
        ),
    )

    assert [request["pm_responsable"] for request in result["solicitudes"]] == [
        "PM Ana",
        "PM Luis",
        None,
    ]
    assignment_report = tmp_path / ASSIGNMENT_REPORT_PATH.name
    issue = json.loads(assignment_report.read_text(encoding="utf-8").splitlines()[0])
    assert issue["decision"] == "sin_pm_disponible"
    assert issue["solicitud"] == "req-3"
    assert issue["datos"]["cargas_abiertas"] == {"PM Ana": 1, "PM Luis": 1}
    assert issue["datos"]["limite_proyectos_abiertos"] == 1


def test_manual_pm_assignment_cannot_exceed_open_project_cap(tmp_path: Path) -> None:
    state = initial_agent_state()
    state["solicitudes"] = [
        {
            "id": f"open-{index}",
            "estado": "Notificada",
            "pm_responsable": "PM Ana",
        }
        for index in range(10)
    ]
    state["solicitudes"].append(
        {
            "id": "new",
            "asunto": "Nueva solicitud",
            "cliente": "Acme",
            "rol": "Analista",
            "cantidad_posiciones": 1,
            "estado": "Nueva",
        }
    )

    with pytest.raises(ValueError, match="límite es 10"):
        assign_pm_to_request(
            state,
            request_number=11,
            pm_name="PM Ana",
            audit=JsonlAuditLog(tmp_path / "audit.jsonl"),
            config=AgentConfig(),
        )


def test_missing_pm_is_audited_as_pending_not_as_notified(tmp_path: Path) -> None:
    message = make_message(
        "m1",
        "Solicitud de puesto",
        "Cliente: Desconocido\nRol: Analista\nCantidad: 1",
    )

    result = run_phase_one(
        FakeAgentMailClient([[message]]),
        audit_path=tmp_path / "auditoria.jsonl",
        assignments={},
        pm_profiles={},
    )

    assert result["solicitudes"][0]["pm_responsable"] is None
    assert result["solicitudes"][0]["hora_notificacion"] is None
    assert result["solicitudes"][0]["estado"] == "Nueva"
    events = [
        json.loads(line)
        for line in (tmp_path / "auditoria.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["decision"] == "notificacion_pendiente" for event in events)


def test_pm_can_be_assigned_to_one_request_and_notified_from_terminal(
    tmp_path: Path,
    capsys,
) -> None:
    message = make_message(
        "m1",
        "Solicitud de personal",
        "Cliente: Acme\nRol: Analista\nCantidad: 1",
    )
    audit_path = tmp_path / "audit.jsonl"
    state = run_phase_one(
        FakeAgentMailClient([[message]]),
        audit_path=audit_path,
        state={
            **initial_agent_state(),
            "simular_confirmacion_automatica": False,
        },
        assignments={},
        pm_profiles={},
    )

    same_state = assign_pm_to_request(
        state,
        request_number=1,
        pm_name="PM Ana",
        audit=JsonlAuditLog(audit_path),
        pm_profiles={
            "PM Ana": {"perfil": "rapido", "horas_confirmacion": 1}
        },
    )

    request = state["solicitudes"][0]
    assert same_state is state
    assert request["pm_responsable"] == "PM Ana"
    assert request["estado"] == "Nueva"
    assert request["hora_notificacion"] == state["hora_actual"]
    assert request["datos_faltantes"] == []
    assert request["perfil_pm"] == "rapido"
    assert "ASIGNACIÓN" in capsys.readouterr().out

    mark_request_opened(
        state,
        identifier="m1",
        audit=JsonlAuditLog(audit_path),
    )
    assert request["estado"] == "Notificada"

    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["decision"] == "pm_asignado_por_usuario" for event in events)
    assert any(event["decision"] == "notificacion_simulada" for event in events)


def test_pm_profile_delays_are_deterministic_and_within_profile_ranges() -> None:
    request = {"id": "m-1", "pm_responsable": "PM Ana"}
    for profile, minimum, maximum in (
        ("rapido", 1, 2),
        ("normal", 3, 6),
        ("lento", 7, 12),
    ):
        settings = {"perfil": profile, "horas_confirmacion": None}
        first = simulated_confirmation_delay(request, settings, seed=42)
        assert first == simulated_confirmation_delay(request, settings, seed=42)
        assert minimum <= first <= maximum
    assert simulated_confirmation_delay(
        request,
        {"perfil": "ausente", "horas_confirmacion": None},
        seed=42,
    ) is None


def test_load_pm_profiles_validates_and_loads_csv(tmp_path: Path) -> None:
    profiles_path = tmp_path / "pms.csv"
    profiles_path.write_text(
        "pm,perfil,horas_confirmacion\nPM Ana,rapido,2\nPM Luis,ausente,\n",
        encoding="utf-8",
    )

    profiles = load_pm_profiles(profiles_path)

    assert profiles["pm ana"] == {
        "nombre": "PM Ana",
        "perfil": "rapido",
        "horas_confirmacion": 2,
        "elegible": True,
    }
    assert profiles["pm luis"]["elegible"] is True
    assert profiles["pm luis"]["perfil"] == "ausente"
    assert profiles["pm luis"]["horas_confirmacion"] is None


def test_load_pm_profiles_reads_eligibility_column(tmp_path: Path) -> None:
    profiles_path = tmp_path / "pms.csv"
    profiles_path.write_text(
        "pm,perfil,horas_confirmacion,elegible\n"
        "PM Ana,normal,,no\n"
        "PM Luis,normal,,si\n",
        encoding="utf-8",
    )

    profiles = load_pm_profiles(profiles_path)

    assert profiles["pm ana"]["elegible"] is False
    assert profiles["pm luis"]["elegible"] is True


def test_pm_profiles_link_assignment_emails_and_detect_absence_replies(
    tmp_path: Path,
) -> None:
    profiles_path = tmp_path / "pms.csv"
    profiles_path.write_text(
        "pm,perfil,horas_confirmacion,elegible,correo\n"
        "PM Ana,normal,,si,\n"
        "PM Luis,normal,,si,\n",
        encoding="utf-8",
    )
    profiles = load_pm_profiles(
        profiles_path,
        pm_addresses=("pm.ana@empresa.com", "pm.luis@empresa.com"),
    )

    absence = detect_pm_unavailability(
        make_message(
            "auto-reply-1",
            "Automatic reply: out of office",
            "I am on vacation and will return next week.",
            sender="PM Ana <pm.ana@empresa.com>",
        ),
        profiles,
    )

    assert profiles["pm ana"]["emails"] == ["pm.ana@empresa.com"]
    assert absence == {
        "pm_responsable": "PM Ana",
        "correo_pm": "pm.ana@empresa.com",
        "motivo": "respuesta de ausencia por vacaciones o permiso",
    }

    unknown_sender = detect_pm_unavailability(
        make_message(
            "auto-reply-2",
            "Out of office - sick leave",
            "I am sick and unavailable.",
            sender="unknown@empresa.com",
        ),
        profiles,
    )
    assert unknown_sender is None


def test_pm_reply_with_explicit_unavailable_until_date_is_detected() -> None:
    profiles = {
        "pm ana": {
            "nombre": "PM Ana",
            "emails": ["pm.ana@empresa.com"],
        }
    }

    absence = detect_pm_unavailability(
        make_message(
            "pm-reply-unavailable",
            "Re: Solicitud activa",
            "No estaría presente hasta el 15 de noviembre.",
            sender="PM Ana <pm.ana@empresa.com>",
        ),
        profiles,
    )

    assert absence == {
        "pm_responsable": "PM Ana",
        "correo_pm": "pm.ana@empresa.com",
        "motivo": (
            "respuesta del PM indica que no estará disponible hasta una fecha"
        ),
    }

    no_return_date = detect_pm_unavailability(
        make_message(
            "pm-reply-unavailable-today",
            "Re: Solicitud activa",
            "No estaré disponible hoy para revisar esta solicitud.",
            sender="PM Ana <pm.ana@empresa.com>",
        ),
        profiles,
    )
    assert no_return_date is None


def test_pm_reply_saying_not_working_and_returning_is_detected() -> None:
    profiles = {
        "pm santiago": {
            "nombre": "PM Santiago",
            "emails": ["santiagovargasquesada17@gmail.com"],
        }
    }

    absence = detect_pm_unavailability(
        make_message(
            "pm-santiago-return",
            "Re: Recordatorio de seguimiento",
            "No estoy trabajando, vuelvo el 14 de noviembre.",
            sender="Santiago <santiagovargasquesada17@gmail.com>",
            in_reply_to="request-previously-notified",
        ),
        profiles,
    )

    assert absence == {
        "pm_responsable": "PM Santiago",
        "correo_pm": "santiagovargasquesada17@gmail.com",
        "motivo": (
            "respuesta del PM indica que no estará disponible hasta una fecha"
        ),
    }


def test_pm_email_autoreply_disables_pm_and_reassigns_open_requests(
    tmp_path: Path,
) -> None:
    profiles_path = tmp_path / "pms.csv"
    profiles_path.write_text(
        "pm,perfil,horas_confirmacion,elegible,correo\n"
        "PM Ana,normal,,si,\n"
        "PM Luis,normal,,si,\n",
        encoding="utf-8",
    )
    assignments = {"Acme": "pm.ana@empresa.com"}
    profiles = load_pm_profiles(
        profiles_path,
        pm_addresses=assignments.values(),
    )
    initial_state = initial_agent_state()
    initial_state["solicitudes"] = [
        {
            "id": "request-active",
            "asunto": "Solicitud activa",
            "cliente": "Acme",
            "rol": "Analista",
            "cantidad_posiciones": 1,
            "pm_responsable": "pm.ana@empresa.com",
            "estado": "Notificada",
            "hora_notificacion": 0,
            "fecha_notificacion": datetime.now(timezone.utc).isoformat(),
            "datos_faltantes": [],
        }
    ]
    away_reply = make_message(
        "pm-away-1",
        "Re: Recordatorio de seguimiento",
        "No estoy trabajando, vuelvo el 14 de noviembre.",
        sender="PM Ana <pm.ana@empresa.com>",
        in_reply_to="request-active",
    )
    initial_state["correos_procesados"].append("pm-away-1")
    initial_state["correos_clasificados"].append(
        {**away_reply, "categoria": "actualizacion_solicitud"}
    )
    stop_event = threading.Event()
    client = FakeAgentMailClient(
        [],
        events=[away_reply],
        stop_event=stop_event,
    )

    def stop_after_processing(state) -> None:
        if "pm-away-1" in state["correos_procesados"]:
            stop_event.set()

    result = run_agent_forever(
        client,
        audit_path=tmp_path / "audit.jsonl",
        stop_event=stop_event,
        on_state=stop_after_processing,
        assignments=assignments,
        pm_profiles=profiles,
        initial_state=initial_state,
        state_path=tmp_path / "state.json",
        profiles_path=profiles_path,
        config=AgentConfig(testing_enabled=False),
    )

    assert result["correos_procesados"] == ["pm-away-1"]
    assert result["solicitudes"][0]["pm_responsable"] == "pm.ana@empresa.com"
    assert result["solicitudes"][0]["responsable_actual"] == "Erick"
    assert result["solicitudes"][0]["estado"] == "Escalada"
    assert profiles["pm ana"]["elegible"] is False
    persisted_profiles = load_pm_profiles(
        profiles_path,
        pm_addresses=assignments.values(),
    )
    assert persisted_profiles["pm ana"]["elegible"] is False
    assert persisted_profiles["pm luis"]["elegible"] is True
    assert not any(
        request["id"] == "pm-away-1"
        for request in result["solicitudes"]
    )
    events = [
        json.loads(line)
        for line in (tmp_path / "audit.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert any(
        event["decision"] == "pm_marcado_no_elegible_por_ausencia"
        for event in events
    )
    absence_event = next(
        event
        for event in events
        if event["decision"] == "pm_marcado_no_elegible_por_ausencia"
    )
    assert absence_event["datos"]["solicitudes_escaladas"] == [
        "request-active"
    ]

    next_request = run_phase_one(
        FakeAgentMailClient(
            [[
                make_message(
                    "request-new",
                    "Solicitud de personal",
                    "Cliente: Acme\nRol: Ingeniero\nCantidad: 1",
                )
            ]]
        ),
        audit_path=tmp_path / "new-request-audit.jsonl",
        assignments=assignments,
        pm_profiles=profiles,
        config=AgentConfig(testing_enabled=False),
    )
    assert next_request["solicitudes"][0]["pm_responsable"] == "PM Luis"


def test_pm_eligibility_toggle_reassigns_open_requests_and_persists(
    tmp_path: Path,
) -> None:
    profiles_path = tmp_path / "pms.csv"
    profiles_path.write_text(
        "pm,perfil,horas_confirmacion,elegible\n"
        "PM Ana,normal,,si\n"
        "PM Luis,normal,,si\n",
        encoding="utf-8",
    )
    profiles = load_pm_profiles(profiles_path)
    state = initial_agent_state()
    state["solicitudes"] = [
        {
            "id": "request-1",
            "asunto": "Solicitud activa",
            "cliente": "Acme",
            "rol": "Analista",
            "cantidad_posiciones": 1,
            "pm_responsable": "PM Ana",
            "estado": "Confirmada",
            "confirmacion_usuario": True,
            "hora_confirmacion": 1,
        }
    ]
    audit_path = tmp_path / "audit.jsonl"
    audit = JsonlAuditLog(audit_path)

    set_pm_eligibility(
        state,
        pm_name="PM Ana",
        eligible=False,
        pm_profiles=profiles,
        profiles_path=profiles_path,
        audit=audit,
        config=AgentConfig(testing_enabled=False),
    )

    request = state["solicitudes"][0]
    assert profiles["pm ana"]["elegible"] is False
    assert load_pm_profiles(profiles_path)["pm ana"]["elegible"] is False
    assert request["pm_responsable"] == "PM Luis"
    assert request["estado"] == "Notificada"
    assert request["confirmacion_usuario"] is None
    assert request["hora_confirmacion"] is None

    set_pm_eligibility(
        state,
        pm_name="PM Ana",
        eligible=True,
        pm_profiles=profiles,
        profiles_path=profiles_path,
        audit=audit,
        config=AgentConfig(testing_enabled=False),
    )

    assert profiles["pm ana"]["elegible"] is True
    assert load_pm_profiles(profiles_path)["pm ana"]["elegible"] is True
    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["decision"] == "solicitud_reasignada" for event in events)
    assert [
        event["datos"]["elegible"]
        for event in events
        if event["decision"] == "pm_elegibilidad_actualizada"
    ] == [False, True]


def test_manual_assignment_rejects_an_ineligible_pm(tmp_path: Path) -> None:
    state = initial_agent_state()
    state["solicitudes"] = [
        {
            "id": "request-1",
            "asunto": "Solicitud activa",
            "estado": "Nueva",
        }
    ]

    with pytest.raises(ValueError, match="no elegible"):
        assign_pm_to_request(
            state,
            request_number=1,
            pm_name="PM Zed",
            audit=JsonlAuditLog(tmp_path / "audit.jsonl"),
            pm_profiles={
                "pm zed": {
                    "nombre": "PM Zed",
                    "perfil": "normal",
                    "elegible": False,
                }
            },
        )


def test_terminal_pm_eligibility_command_handles_multiword_pm_name(
    tmp_path: Path,
    monkeypatch,
) -> None:
    profiles_path = tmp_path / "pms.csv"
    profiles_path.write_text(
        "pm,perfil,horas_confirmacion,elegible\n"
        "PM Ana,normal,,si\n"
        "PM Luis,normal,,si\n",
        encoding="utf-8",
    )
    profiles = load_pm_profiles(profiles_path)
    profiles_path.write_text(
        "pm,perfil,horas_confirmacion,elegible\n"
        "PM Ana,normal,,si\n"
        "PM Luis,normal,,si\n"
        "PM Omar,ausente,,si\n",
        encoding="utf-8",
    )
    state = initial_agent_state()
    state["simular_confirmacion_automatica"] = False
    state["solicitudes"] = [
        {
            "id": "request-1",
            "asunto": "Solicitud activa",
            "cliente": "Acme",
            "rol": "Analista",
            "cantidad_posiciones": 1,
            "pm_responsable": "PM Ana",
            "estado": "Notificada",
            "hora_notificacion": 0,
            "fecha_notificacion": datetime.now(timezone.utc).isoformat(),
        }
    ]
    state_path = tmp_path / "state.json"
    save_agent_state(state_path, state)

    class IdleAgentMailClient(FakeAgentMailClient):
        async def message_events(self):
            self.stream_calls += 1
            yield {"type": "subscribed"}
            await asyncio.Event().wait()

    class FakePromptSession:
        def __init__(self) -> None:
            self.commands = iter(
                [
                    "pm-eligibility PM Ana no",
                    "pm-eligibility Omar no",
                    "stop",
                ]
            )

        def prompt(self, prompt_text: str) -> str:
            assert prompt_text == "agentmail> "
            return next(self.commands)

    import terminal as terminal_module

    monkeypatch.setattr(terminal_module, "PromptSession", FakePromptSession)
    monkeypatch.setattr(terminal_module, "patch_stdout", lambda: nullcontext())

    run_terminal_agent(
        IdleAgentMailClient([]),
        audit_path=tmp_path / "audit.jsonl",
        reconnect_interval=0.01,
        assignments={},
        pm_profiles=profiles,
        state_path=state_path,
        profiles_path=profiles_path,
        config=AgentConfig(testing_enabled=False),
    )

    assert profiles["pm ana"]["elegible"] is False
    assert profiles["pm omar"]["elegible"] is False
    assert load_pm_profiles(profiles_path)["pm ana"]["elegible"] is False
    assert load_pm_profiles(profiles_path)["pm omar"]["elegible"] is False
    assert load_agent_state(state_path)["solicitudes"][0]["pm_responsable"] == (
        "PM Luis"
    )


def test_simulated_pm_confirms_on_profile_hour(tmp_path: Path, capsys) -> None:
    message = make_message(
        "m1",
        "Solicitud de personal",
        "Cliente: Acme\nRol: Analista\nCantidad: 1",
    )
    client = FakeAgentMailClient([[message]])
    settings = {"PM Ana": {"perfil": "rapido", "horas_confirmacion": 1}}

    after_notification = run_phase_one(
        client,
        audit_path=tmp_path / "audit.jsonl",
        assignments={"Acme": "PM Ana"},
        pm_profiles=settings,
    )
    request_after_first_turn = after_notification["solicitudes"][0]
    assert request_after_first_turn["estado"] == "Notificada"
    assert request_after_first_turn["hora_notificacion"] == 0
    assert request_after_first_turn["perfil_pm"] == "rapido"
    assert after_notification["hora_actual"] == 1

    after_confirmation = run_phase_one(
        client,
        audit_path=tmp_path / "audit.jsonl",
        state=after_notification,
        assignments={"Acme": "PM Ana"},
        pm_profiles=settings,
    )
    request = after_confirmation["solicitudes"][0]
    assert request["estado"] == "Confirmada"
    assert request["hora_confirmacion"] == 1
    assert request["confirmacion_simulada"] is True
    events = [
        json.loads(line)
        for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["decision"] == "pm_confirmo_simulado" for event in events)
    assert "CONFIRMACIÓN SIMULADA" in capsys.readouterr().out


def test_user_can_confirm_or_reject_without_a_new_email(tmp_path: Path) -> None:
    message = make_message(
        "m1",
        "Solicitud de personal",
        "Cliente: Acme\nRol: Analista\nCantidad: 1",
    )
    audit_path = tmp_path / "audit.jsonl"
    state = run_phase_one(
        FakeAgentMailClient([[message]]),
        audit_path=audit_path,
        assignments={"Acme": "PM Allisson"},
    )

    apply_human_confirmation(
        state,
        identifier="m1",
        confirmed=False,
        audit=JsonlAuditLog(audit_path),
    )
    assert state["solicitudes"][0]["estado"] == "Notificada"
    assert state["solicitudes"][0]["confirmacion_usuario"] is False

    apply_human_confirmation(
        state,
        identifier="m1",
        confirmed=True,
        audit=JsonlAuditLog(audit_path),
    )
    request = state["solicitudes"][0]
    assert request["estado"] == "Confirmada"
    assert request["hora_confirmacion"] == state["hora_actual"]
    assert request["confirmacion_usuario"] is True
    assert completed_requests(state) == []
    assert count_open_projects(state["solicitudes"], "PM Allisson") == 1

    complete_request(
        state,
        request_number=1,
        reason="El PM confirmó que las posiciones fueron cubiertas.",
        audit=JsonlAuditLog(audit_path),
    )
    assert request["estado"] == "Cerrada"
    assert request["razon_finalizacion"] == (
        "El PM confirmó que las posiciones fueron cubiertas."
    )
    assert completed_requests(state)[0][1]["id"] == "m1"
    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["decision"] == "pm_confirmado_por_usuario" for event in events)
    assert any(
        event["decision"] == "pm_no_confirmado_por_usuario"
        for event in events
    )
    assert any(
        event["decision"] == "solicitud_completada" for event in events
    )
    completion_event = next(
        event
        for event in events
        if event["decision"] == "solicitud_completada"
    )
    assert completion_event["motivo"] == request["razon_finalizacion"]
    assert (
        completion_event["datos"]["razon_finalizacion"]
        == request["razon_finalizacion"]
    )


def test_graph_processes_one_inbox_poll_per_invocation(tmp_path: Path) -> None:
    first = make_message("m1", "Solicitud de posición", "Solicitamos un ingeniero.")
    client = FakeAgentMailClient([[first]])

    result = run_phase_one(client, audit_path=tmp_path / "audit.jsonl")

    assert result["correos_procesados"] == ["m1"]
    assert client.list_calls == 1
    assert result["hora_actual"] == 1


def test_empty_polls_do_not_write_audit_or_print_json(tmp_path: Path, capsys) -> None:
    audit_path = tmp_path / "audit.jsonl"

    result = run_phase_one(FakeAgentMailClient([[]]), audit_path=audit_path)

    assert result["hora_actual"] == 1
    assert not audit_path.exists()
    assert capsys.readouterr().out == ""


def test_service_repeats_graph_until_stopped_and_emits_heartbeat(
    tmp_path: Path,
    capsys,
) -> None:
    first = make_message("m1", "Solicitud de posición", "Solicitamos un ingeniero.")
    second = make_message("m2", "Solicitud de puesto", "Requerimos una analista.")
    stop_event = threading.Event()
    client = FakeAgentMailClient(
        [],
        events=[first, second],
        stop_event=stop_event,
    )
    published_states = []

    def publish(state) -> None:
        published_states.append(state)
        if len(published_states) == 2:
            stop_event.set()

    result = run_agent_forever(
        client,
        audit_path=tmp_path / "audit.jsonl",
        stop_event=stop_event,
        on_state=publish,
        reconnect_interval=0.001,
        heartbeat_interval=1e-9,
        pm_profiles={},
    )

    assert client.list_calls == 1
    assert client.stream_calls == 1
    assert result["hora_actual"] == 2
    assert result["correos_procesados"] == ["m1", "m2"]
    assert len(published_states) == 2
    assert "." in capsys.readouterr().out


def test_service_remains_idle_without_polling_until_stopped(
    tmp_path: Path,
) -> None:
    stop_event = threading.Event()
    client = FakeAgentMailClient([], stop_event=stop_event)
    timer = threading.Timer(0.2, stop_event.set)
    timer.start()
    initial_state = initial_agent_state()
    initial_state["llamadas_llm"] = 59

    def publish(state) -> None:
        assert isinstance(state["hora_actual"], int)

    result = run_agent_forever(
        client,
        audit_path=tmp_path / "audit.jsonl",
        stop_event=stop_event,
        on_state=publish,
        heartbeat_interval=60,
        initial_state=initial_state,
    )
    timer.join()

    assert client.list_calls == 1
    assert client.stream_calls == 1
    assert result["correos_procesados"] == []
    assert result["llamadas_llm"] == 0
    assert not (tmp_path / "audit.jsonl").exists()


def test_service_rejects_nonpositive_reconnect_interval(tmp_path: Path) -> None:
    def publish(state) -> None:
        assert isinstance(state["hora_actual"], int)

    with pytest.raises(ValueError, match="reconnect_interval"):
        run_agent_forever(
            FakeAgentMailClient([[]]),
            audit_path=tmp_path / "audit.jsonl",
            stop_event=threading.Event(),
            on_state=publish,
            reconnect_interval=0,
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


def test_service_backfills_old_messages_and_restores_active_state(
    tmp_path: Path,
    capsys,
) -> None:
    old_message = make_message(
        "old-1",
        "Solicitud de posición",
        "Cliente: Acme\nRol: Analista\nCantidad: 2",
    )
    state_path = tmp_path / "agent-state.json"
    stop_event = threading.Event()
    client = FakeAgentMailClient([[old_message]], stop_event=stop_event)
    states = []

    def stop_after_backfill(state) -> None:
        states.append(state)
        if "old-1" in state["correos_procesados"]:
            stop_event.set()

    result = run_agent_forever(
        client,
        audit_path=tmp_path / "audit.jsonl",
        stop_event=stop_event,
        on_state=stop_after_backfill,
        assignments={},
        pm_profiles={},
        state_path=state_path,
    )

    assert client.list_calls == 1
    assert result["correos_procesados"] == ["old-1"]
    assert result["solicitudes"][0]["estado"] == "Nueva"
    assert "CORREO ANTIGUO" in capsys.readouterr().out
    restored = load_agent_state(state_path)
    assert restored["correos_procesados"] == ["old-1"]
    assert restored["solicitudes"][0]["id"] == "old-1"


def test_service_reclassifies_saved_staffing_emails_previously_marked_unrelated(
    tmp_path: Path,
) -> None:
    missed_message = make_message(
        "missed-1",
        "Solicitud urgente de 4 desarrolladores Frontend",
        "Tenemos una nueva solicitud por parte de nuestro cliente para "
        "encontrar 4 desarrolladores frontend. Cliente: Acme",
    )
    initial_state = initial_agent_state()
    initial_state["correos_clasificados"] = [
        {**missed_message, "categoria": "no_relacionado"}
    ]
    initial_state["correos_procesados"] = ["missed-1"]
    stop_event = threading.Event()
    client = FakeAgentMailClient([], stop_event=stop_event)

    def stop_after_reclassification(state) -> None:
        if any(request["id"] == "missed-1" for request in state["solicitudes"]):
            stop_event.set()

    result = run_agent_forever(
        client,
        audit_path=tmp_path / "audit.jsonl",
        stop_event=stop_event,
        on_state=stop_after_reclassification,
        reconnect_interval=0.01,
        assignments={"Acme": "PM Ana"},
        pm_profiles={},
        initial_state=initial_state,
        state_path=tmp_path / "state.json",
    )

    assert [request["id"] for request in result["solicitudes"]] == ["missed-1"]
    assert next(
        item
        for item in result["correos_clasificados"]
        if item["message_id"] == "missed-1"
    )["categoria"] == "solicitud_posicion"
    audit_events = [
        json.loads(line)
        for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        event["decision"] == "correo_no_relacionado_reevaluado"
        for event in audit_events
    )


def test_service_answers_natural_language_case_query_from_shared_mailbox(
    tmp_path: Path,
) -> None:
    shared_email = "team@example.test"
    profiles = {
        "PM Santiago": {
            "nombre": "PM Santiago",
            "perfil": "normal",
            "elegible": True,
            "emails": [shared_email],
        },
        "PM Allisson": {
            "nombre": "PM Allisson",
            "perfil": "normal",
            "elegible": True,
            "emails": [shared_email],
        },
    }
    state = initial_agent_state()
    state["solicitudes"] = [
        {
            "id": "case-santiago",
            "asunto": "Solicitud de analistas",
            "pm_responsable": "PM Santiago",
            "responsable_actual": "PM Santiago",
            "estado": "Notificada",
            "fecha_notificacion": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": "case-allisson",
            "asunto": "Solicitud de ingenieros",
            "pm_responsable": "PM Allisson",
            "responsable_actual": "PM Allisson",
            "estado": "Notificada",
            "fecha_notificacion": datetime.now(timezone.utc).isoformat(),
        },
    ]
    query = make_message(
        "query-cases",
        "Consulta",
        "¿Me puedes decir qué casos tengo pendientes?",
        sender=f"Erick <{shared_email}>",
    )
    stop_event = threading.Event()
    client = RecordingAgentMailClient(
        events=[query],
        stop_event=stop_event,
    )

    def stop_after_reply(updated_state) -> None:
        if "query-cases" in updated_state["correos_procesados"]:
            stop_event.set()

    result = run_agent_forever(
        client,
        audit_path=tmp_path / "audit.jsonl",
        stop_event=stop_event,
        on_state=stop_after_reply,
        assignments={
            "Acme": "PM Santiago",
            "Globex": "PM Allisson",
        },
        pm_profiles=profiles,
        initial_state=state,
        state_path=tmp_path / "state.json",
        config=AgentConfig(
            llm_enabled=False,
            manager_name="Erick",
            manager_email=shared_email,
        ),
    )

    assert result["correos_clasificados"][-1]["categoria"] == "consulta_casos"
    assert len(client.sent_messages) == 1
    recipient, subject, body = client.sent_messages[0]
    assert recipient == shared_email
    assert subject == "Solicitudes pendientes asignadas"
    assert "case-santiago" in body
    assert "case-allisson" in body


def test_manager_email_reply_accepts_absent_pm_escalation(
    tmp_path: Path,
) -> None:
    shared_email = "team@example.test"
    profiles = {
        "PM Santiago": {
            "nombre": "PM Santiago",
            "perfil": "normal",
            "elegible": True,
            "emails": [shared_email],
        }
    }
    state = initial_agent_state()
    state["solicitudes"] = [
        {
            "id": "case-approval",
            "asunto": "Solicitud urgente",
            "correo_solicitante": "requester@example.test",
            "pm_responsable": "PM Santiago",
            "jefe_responsable": "Erick",
            "responsable_actual": "Erick",
            "escalada_a_jefe": True,
            "escalamiento_ausencia_pendiente": True,
            "aprobacion_director_message_id": "approval-email-1",
            "estado": "Escalada",
            "fecha_notificacion": datetime.now(timezone.utc).isoformat(),
        }
    ]
    approval = make_message(
        "approval-reply-1",
        "Re: [Aprobación de escalamiento solicitada] case-approval",
        "Sí, acepto hacerme cargo.",
        sender=f"Erick <{shared_email}>",
        in_reply_to="approval-email-1",
    )
    stop_event = threading.Event()
    client = RecordingAgentMailClient(
        events=[approval],
        stop_event=stop_event,
    )

    def stop_after_approval(updated_state) -> None:
        if "approval-reply-1" in updated_state["correos_procesados"]:
            stop_event.set()

    result = run_agent_forever(
        client,
        audit_path=tmp_path / "audit.jsonl",
        stop_event=stop_event,
        on_state=stop_after_approval,
        assignments={"Acme": "PM Santiago"},
        pm_profiles=profiles,
        initial_state=state,
        state_path=tmp_path / "state.json",
        config=AgentConfig(
            llm_enabled=False,
            manager_name="Erick",
            manager_email=shared_email,
        ),
    )

    request = result["solicitudes"][0]
    assert request["responsable_actual"] == "Erick"
    assert request["jefe_responsable"] == "Erick"
    assert request["estado"] == "Confirmada"
    assert request["escalamiento_ausencia_pendiente"] is False
    assert len(client.sent_messages) == 1
    assert client.sent_messages[0][0] == "requester@example.test"
    assert "[Solicitud aceptada por el jefe]" in client.sent_messages[0][1]


def test_shared_mailbox_pm_reply_is_not_mistaken_for_manager_approval(
    tmp_path: Path,
) -> None:
    shared_email = "team@example.test"
    profiles = {
        "PM Santiago": {
            "nombre": "PM Santiago",
            "perfil": "normal",
            "elegible": True,
            "emails": [shared_email],
        },
        "PM Allisson": {
            "nombre": "PM Allisson",
            "perfil": "normal",
            "elegible": True,
            "emails": [shared_email],
        },
    }
    state = initial_agent_state()
    state["solicitudes"] = [
        {
            "id": "case-shared-thread",
            "asunto": "Solicitud de analistas",
            "correo_solicitante": "requester@example.test",
            "pm_responsable": "PM Santiago",
            "jefe_responsable": "Erick",
            "responsable_actual": "Erick",
            "escalada_a_jefe": True,
            "escalamiento_ausencia_pendiente": True,
            "aprobacion_director_pendiente": True,
            "aprobacion_director_message_id": "approval-email-2",
            "notificacion_pm_message_id": "pm-email-2",
            "estado": "Escalada",
            "fecha_notificacion": datetime.now(timezone.utc).isoformat(),
        }
    ]
    pm_reply = make_message(
        "pm-reply-shared-thread",
        "Re: [Nueva solicitud asignada] case-shared-thread",
        "Me hago cargo de esta solicitud.",
        sender=f"PM Santiago <{shared_email}>",
        in_reply_to="pm-email-2",
    )
    stop_event = threading.Event()
    client = RecordingAgentMailClient(
        events=[pm_reply],
        stop_event=stop_event,
    )

    def stop_after_pm_reply(updated_state) -> None:
        if "pm-reply-shared-thread" in updated_state["correos_procesados"]:
            stop_event.set()

    result = run_agent_forever(
        client,
        audit_path=tmp_path / "audit.jsonl",
        stop_event=stop_event,
        on_state=stop_after_pm_reply,
        assignments={"Acme": "PM Santiago"},
        pm_profiles=profiles,
        initial_state=state,
        state_path=tmp_path / "state.json",
        config=AgentConfig(
            llm_enabled=False,
            manager_name="Erick",
            manager_email=shared_email,
        ),
    )

    request = result["solicitudes"][0]
    assert request["estado"] == "Confirmada"
    assert request["responsable_actual"] == "Erick"
    assert request["escalamiento_ausencia_pendiente"] is False
    assert request.get("respuesta_aprobacion_director") is None
    assert not any(
        event["decision"] == "jefe_acepto_escalamiento"
        for event in result["eventos"]
    )


def test_ambiguous_absence_from_shared_mailbox_does_not_disable_every_pm(
    tmp_path: Path,
) -> None:
    shared_email = "team@example.test"
    profiles = {
        name: {
            "nombre": name,
            "perfil": "normal",
            "elegible": True,
            "emails": [shared_email],
        }
        for name in ("PM Santiago", "PM Allisson")
    }
    state = initial_agent_state()
    state["solicitudes"] = [
        {
            "id": "case-santiago",
            "asunto": "Caso Santiago",
            "pm_responsable": "PM Santiago",
            "responsable_actual": "PM Santiago",
            "estado": "Notificada",
            "fecha_notificacion": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": "case-allisson",
            "asunto": "Caso Allisson",
            "pm_responsable": "PM Allisson",
            "responsable_actual": "PM Allisson",
            "estado": "Notificada",
            "fecha_notificacion": datetime.now(timezone.utc).isoformat(),
        },
    ]
    absence = make_message(
        "ambiguous-away",
        "Automatic reply: Vacation",
        "I am out of office on vacation.",
        sender=f"Shared inbox <{shared_email}>",
    )
    stop_event = threading.Event()
    client = FakeAgentMailClient(
        [],
        events=[absence],
        stop_event=stop_event,
    )

    def stop_after_review(updated_state) -> None:
        if "ambiguous-away" in updated_state["correos_procesados"]:
            stop_event.set()

    result = run_agent_forever(
        client,
        audit_path=tmp_path / "audit.jsonl",
        stop_event=stop_event,
        on_state=stop_after_review,
        assignments={
            "Acme": "PM Santiago",
            "Globex": "PM Allisson",
        },
        pm_profiles=profiles,
        initial_state=state,
        state_path=tmp_path / "state.json",
        profiles_path=tmp_path / "pms.csv",
        config=AgentConfig(
            llm_enabled=False,
            manager_name="Erick",
            manager_email=shared_email,
        ),
    )

    assert all(profile["elegible"] for profile in profiles.values())
    assert all(
        request["estado"] == "Notificada"
        for request in result["solicitudes"]
    )
    assert result["correos_clasificados"][-1]["categoria"] == (
        "respuesta_ausencia_pm_ambigua"
    )


def test_service_migrates_active_requests_to_current_pm_roster(
    tmp_path: Path,
) -> None:
    profiles = {
        "PM Santiago": {
            "nombre": "PM Santiago",
            "perfil": "normal",
            "elegible": True,
            "emails": ["santiago@example.test"],
        },
        "PM Allisson": {
            "nombre": "PM Allisson",
            "perfil": "normal",
            "elegible": True,
            "emails": ["allisson@example.test"],
        },
    }
    state = initial_agent_state()
    state["solicitudes"] = [
        {
            "id": "legacy-case",
            "asunto": "Solicitud heredada",
            "pm_responsable": "PM Ana",
            "responsable_actual": "PM Ana",
            "estado": "Notificada",
            "fecha_notificacion": datetime.now(timezone.utc).isoformat(),
        }
    ]
    stop_event = threading.Event()
    client = RecordingAgentMailClient(stop_event=stop_event)

    def stop_after_migration(updated_state) -> None:
        if updated_state["solicitudes"][0]["pm_responsable"] != "PM Ana":
            stop_event.set()

    result = run_agent_forever(
        client,
        audit_path=tmp_path / "audit.jsonl",
        stop_event=stop_event,
        on_state=stop_after_migration,
        assignments={
            "Acme": "PM Santiago",
            "Globex": "PM Allisson",
        },
        pm_profiles=profiles,
        initial_state=state,
        state_path=tmp_path / "state.json",
        config=AgentConfig(llm_enabled=False),
    )

    request = result["solicitudes"][0]
    assert request["pm_anterior"] == "PM Ana"
    assert request["pm_responsable"] == "PM Allisson"
    assert request["estado"] == "Notificada"
    assert len(client.sent_messages) == 1
    assert client.sent_messages[0][0] == "allisson@example.test"
    assert "[Solicitud reasignada]" in client.sent_messages[0][1]


def test_due_reminder_is_sent_once_after_configured_four_hours(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    message = make_message(
        "m1",
        "Solicitud de personal",
        "Cliente: Acme\nRol: Analista\nCantidad: 1",
    )
    audit_path = tmp_path / "audit.jsonl"
    state = run_phase_one(
        FakeAgentMailClient([[message]]),
        audit_path=audit_path,
        assignments={"Acme": "PM Ana"},
        pm_profiles={"PM Ana": {"perfil": "normal"}},
    )
    request = state["solicitudes"][0]
    notified_at = datetime.fromisoformat(
        request["fecha_notificacion"].replace("Z", "+00:00")
    )
    reminder_time = notified_at + timedelta(hours=4)

    sent = process_due_reminders(
        state,
        audit=JsonlAuditLog(audit_path),
        config=AgentConfig(testing_enabled=False),
        now=reminder_time,
    )

    assert sent == 1
    assert request["estado"] == "Recordatorio enviado"
    assert request["numero_notificacion_seguimiento"] == 1
    reminder_output = capsys.readouterr().out
    assert "[NOTIFICACIÓN DE SEGUIMIENTO SIMULADA #1]" in reminder_output
    assert "Para: PM Ana" in reminder_output
    assert "PM asignado: PM Ana" in reminder_output
    assert process_due_reminders(
        state,
        audit=JsonlAuditLog(audit_path),
        config=AgentConfig(testing_enabled=False),
        now=reminder_time + timedelta(hours=1),
    ) == 0
    assert request["numero_notificacion_seguimiento"] == 1
    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    reminder = next(event for event in events if event["decision"] == "recordatorio_enviado")
    assert reminder["datos"]["reloj_simulado"] is False
    assert reminder["datos"]["notificacion_simulada"] is True
    assert reminder["datos"]["umbral"] == 4
    assert reminder["datos"]["numero_notificacion"] == 1
    assert reminder["datos"]["destinatario"] == "PM Ana"
    assert reminder["datos"]["pm_responsable"] == "PM Ana"


def test_first_due_reminder_sends_one_real_email_with_generic_template(
    tmp_path: Path,
) -> None:
    state = initial_agent_state()
    notified_at = datetime.fromisoformat("2026-10-29T12:00:00+00:00")
    request = {
        "id": "request-1",
        "asunto": "Solicitud de Analista",
        "cliente": "Acme",
        "rol": "Analista",
        "cantidad_posiciones": 2,
        "pm_responsable": "PM Ana",
        "estado": "Notificada",
        "fecha_notificacion": notified_at.isoformat(),
    }
    state["solicitudes"] = [request]
    client = RecordingAgentMailClient()
    config = AgentConfig(
        reminder_email_enabled=True,
        reminder_email_recipient="santiagovargasquesada17@gmail.com",
    )
    audit_path = tmp_path / "audit.jsonl"

    process_due_reminders(
        state,
        audit=JsonlAuditLog(audit_path),
        config=config,
        pm_profiles={
            "PM Ana": {
                "nombre": "PM Ana",
                "emails": ["santiagovargasquesada17@gmail.com"],
            }
        },
        mail_client=client,
        now=notified_at + timedelta(hours=4),
    )

    assert len(client.sent_messages) == 1
    recipient, subject, body = client.sent_messages[0]
    assert recipient == "santiagovargasquesada17@gmail.com"
    assert subject == (
        "[Recordatorio de solicitud] request-1: Solicitud de Analista"
    )
    assert "Cliente: Acme" in body
    assert "PM responsable: PM Ana" in body
    assert request["correo_recordatorio_real_enviado"] is True

    process_due_reminders(
        state,
        audit=JsonlAuditLog(audit_path),
        config=config,
        pm_profiles={
            "PM Ana": {
                "nombre": "PM Ana",
                "emails": ["santiagovargasquesada17@gmail.com"],
            }
        },
        mail_client=client,
        now=notified_at + timedelta(hours=5),
    )
    assert len(client.sent_messages) == 1


def test_existing_first_reminder_does_not_send_email_again_on_startup(
    tmp_path: Path,
) -> None:
    state = initial_agent_state()
    notified_at = datetime.fromisoformat("2026-10-29T12:00:00+00:00")
    state["solicitudes"] = [
        {
            "id": "request-existing-reminder",
            "asunto": "Solicitud ya notificada",
            "pm_responsable": "PM Ana",
            "estado": "Recordatorio enviado",
            "fecha_notificacion": notified_at.isoformat(),
            "primer_recordatorio_enviado": True,
            "recordatorio_enviado": True,
            "numero_notificacion_seguimiento": 1,
        }
    ]
    client = RecordingAgentMailClient()

    process_due_reminders(
        state,
        audit=JsonlAuditLog(tmp_path / "audit.jsonl"),
        config=AgentConfig(
            reminder_email_enabled=True,
            reminder_email_recipient="santiagovargasquesada17@gmail.com",
        ),
        mail_client=client,
        now=notified_at + timedelta(hours=5),
    )

    assert client.sent_messages == []
    assert state["solicitudes"][0]["numero_notificacion_seguimiento"] == 1


def test_first_due_reminder_uses_llm_written_body_when_available(
    tmp_path: Path,
) -> None:
    class ReminderWriter(OpenRouterClient):
        def write_lifecycle_email(
            self,
            event_type: str,
            request: dict[str, Any],
            context: dict[str, Any],
        ) -> str:
            assert event_type == "recordatorio"
            return f"Seguimiento personalizado para {request['asunto']}."

    llm = ReminderWriter("test-key", "provider/test-model")
    state = initial_agent_state()
    notified_at = datetime.fromisoformat("2026-10-29T12:00:00+00:00")
    state["solicitudes"] = [
        {
            "id": "request-llm",
            "asunto": "Solicitud de Analista",
            "pm_responsable": "PM Ana",
            "estado": "Notificada",
            "fecha_notificacion": notified_at.isoformat(),
        }
    ]
    client = RecordingAgentMailClient()

    process_due_reminders(
        state,
        audit=JsonlAuditLog(tmp_path / "audit.jsonl"),
        config=AgentConfig(
            reminder_email_enabled=True,
            reminder_email_recipient="santiagovargasquesada17@gmail.com",
        ),
        pm_profiles={
            "PM Ana": {
                "nombre": "PM Ana",
                "emails": ["santiagovargasquesada17@gmail.com"],
            }
        },
        mail_client=client,
        llm=llm,
        now=notified_at + timedelta(hours=4),
    )

    assert len(client.sent_messages) == 1
    assert client.sent_messages[0][2] == (
        "Seguimiento personalizado para Solicitud de Analista."
    )
    assert state["llamadas_llm"] == 1


def test_first_due_reminder_retries_llm_and_uses_template_on_failure(
    tmp_path: Path,
) -> None:
    class FailingReminderWriter(OpenRouterClient):
        def __init__(self) -> None:
            super().__init__("test-key", "provider/test-model")
            self.calls = 0

        def write_lifecycle_email(
            self,
            event_type: str,
            request: dict[str, Any],
            context: dict[str, Any],
        ) -> str:
            assert event_type == "recordatorio"
            self.calls += 1
            raise OpenRouterError("salida inválida")

    state = initial_agent_state()
    notified_at = datetime.fromisoformat("2026-10-29T12:00:00+00:00")
    state["solicitudes"] = [
        {
            "id": "request-2",
            "asunto": "Solicitud de datos",
            "cliente": None,
            "rol": None,
            "cantidad_posiciones": None,
            "pm_responsable": "PM Luis",
            "estado": "Notificada",
            "fecha_notificacion": notified_at.isoformat(),
        }
    ]
    client = RecordingAgentMailClient()
    llm = FailingReminderWriter()
    audit_path = tmp_path / "audit.jsonl"

    process_due_reminders(
        state,
        audit=JsonlAuditLog(audit_path),
        config=AgentConfig(
            reminder_email_enabled=True,
            reminder_email_recipient="santiagovargasquesada17@gmail.com",
        ),
        pm_profiles={
            "PM Luis": {
                "nombre": "PM Luis",
                "emails": ["santiagovargasquesada17@gmail.com"],
            }
        },
        mail_client=client,
        llm=llm,
        now=notified_at + timedelta(hours=4),
    )

    assert llm.calls == 2
    assert state["llamadas_llm"] == 2
    assert len(client.sent_messages) == 1
    assert "Solicitud: request-2" in client.sent_messages[0][2]
    assert "PM responsable: PM Luis" in client.sent_messages[0][2]
    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    llm_errors = [
        event
        for event in events
        if event["nodo"] == "redactar_correo_ciclo_vida_llm"
    ]
    assert len(llm_errors) == 2
    sent_event = next(
        event for event in events
        if event["decision"] == "correo_ciclo_vida_enviado"
    )
    assert sent_event["datos"]["metodo_redaccion"] == "plantilla"


def test_testing_sla_uses_configured_minute_milestones(tmp_path: Path) -> None:
    request = {
        "id": "m-test",
        "asunto": "Solicitud de prueba",
        "cliente": "Acme",
        "pm_responsable": "PM Ana",
        "estado": "Notificada",
        "fecha_notificacion": "2026-10-29T12:00:00+00:00",
        "hora_notificacion": 0,
        "recordatorio_enviado": False,
        "escalado_director": False,
    }
    state = initial_agent_state()
    state["solicitudes"] = [request]
    audit = JsonlAuditLog(tmp_path / "audit.jsonl")
    config = AgentConfig(testing_enabled=True)

    reminder_actions = process_due_reminders(
        state,
        audit=audit,
        config=config,
        now=datetime.fromisoformat("2026-10-29T12:01:00+00:00"),
    )
    assert reminder_actions == 1
    assert request["estado"] == "Recordatorio enviado"
    assert request["nivel_maximo_escalamiento"] == "PM"

    second_reminder_actions = process_due_reminders(
        state,
        audit=audit,
        config=config,
        now=datetime.fromisoformat("2026-10-29T12:02:00+00:00"),
        directors={"Acme": "Directora Sol"},
    )
    assert second_reminder_actions == 1
    assert request["estado"] == "Recordatorio enviado"

    escalation_actions = process_due_reminders(
        state,
        audit=audit,
        config=config,
        now=datetime.fromisoformat("2026-10-29T12:04:00+00:00"),
        directors={"Acme": "Directora Sol"},
        approval_handler=lambda _: {
            "approved": True,
            "reason": "Aprobado para la prueba.",
        },
    )
    assert escalation_actions == 1
    assert request["estado"] == "Escalada"
    assert request["nivel_maximo_escalamiento"] == "Director"
    assert request["numero_notificacion_seguimiento"] == 3
    events = [
        json.loads(line)
        for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    milestone_events = [
        event
        for event in events
        if event["decision"]
        in {
            "recordatorio_enviado",
            "segundo_recordatorio_enviado",
            "escalamiento_director_aprobado",
        }
    ]
    assert [event["datos"]["umbral"] for event in milestone_events] == [1, 2, 4]
    assert [event["datos"]["numero_notificacion"] for event in milestone_events] == [
        1,
        2,
        3,
    ]
    assert [event["datos"]["destinatario"] for event in milestone_events] == [
        "PM Ana",
        "PM Ana",
        "Directora Sol",
    ]
    assert all(
        event["datos"]["pm_responsable"] == "PM Ana"
        for event in milestone_events
    )
    assert [event["datos"]["unidad_umbral"] for event in events] == [
        "minutos",
        "minutos",
        "minutos",
        "minutos",
    ]


def test_director_escalation_requires_approval_at_sixteen_hours(
    tmp_path: Path,
) -> None:
    message = make_message(
        "m1",
        "Solicitud de personal",
        "Cliente: Acme\nRol: Analista\nCantidad: 1",
    )
    audit_path = tmp_path / "audit.jsonl"
    state = run_phase_one(
        FakeAgentMailClient([[message]]),
        audit_path=audit_path,
        assignments={"Acme": "PM Ana"},
    )
    request = state["solicitudes"][0]
    notified_at = datetime.fromisoformat(
        request["fecha_notificacion"].replace("Z", "+00:00")
    )

    processed_at_eight_hours = process_due_reminders(
        state,
        audit=JsonlAuditLog(audit_path),
        config=AgentConfig(testing_enabled=False),
        now=notified_at + timedelta(hours=8),
        directors={"Acme": "Directora Sol"},
    )

    assert processed_at_eight_hours == 2
    assert request["estado"] == "Recordatorio enviado"
    assert request["escalado_director"] is False

    processed_at_sixteen_hours = process_due_reminders(
        state,
        audit=JsonlAuditLog(audit_path),
        config=AgentConfig(testing_enabled=False),
        now=notified_at + timedelta(hours=16),
        directors={"Acme": "Directora Sol"},
        approval_handler=lambda _: {
            "approved": True,
            "reason": "Autorizado por la prueba.",
        },
    )

    assert processed_at_sixteen_hours == 1
    assert request["estado"] == "Escalada"
    assert request["nivel_maximo_escalamiento"] == "Director"
    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    escalation = next(
        event
        for event in events
        if event["decision"] == "escalamiento_director_aprobado"
    )
    assert escalation["datos"]["director"] == "Directora Sol"
    assert escalation["datos"]["umbral"] == 16


def test_rejected_director_escalation_is_audited_and_retried(
    tmp_path: Path,
) -> None:
    request = {
        "id": "m-approval",
        "asunto": "Solicitud de prueba",
        "cliente": "Acme",
        "pm_responsable": "PM Ana",
        "estado": "Notificada",
        "fecha_notificacion": "2026-10-29T12:00:00+00:00",
        "hora_notificacion": 0,
    }
    state = initial_agent_state()
    state["solicitudes"] = [request]
    audit_path = tmp_path / "audit.jsonl"
    audit = JsonlAuditLog(audit_path)
    graph = build_director_approval_graph()
    responses = [
        {"approved": False, "reason": "Falta validar presupuesto."},
        {"approved": True, "reason": "Presupuesto validado."},
    ]
    approval_payloads: list[dict[str, Any]] = []

    def approval_handler(payload: dict[str, Any]) -> dict[str, Any]:
        approval_payloads.append(payload)
        return responses.pop(0)

    first_review = datetime.fromisoformat(
        "2026-10-30T04:00:00+00:00"
    )
    process_due_reminders(
        state,
        audit=audit,
        config=AgentConfig(testing_enabled=False),
        now=first_review,
        directors={"Acme": "Directora Sol"},
        approval_handler=approval_handler,
        approval_graph=graph,
    )
    assert request["estado"] == "Recordatorio enviado"
    assert request.get("escalado_director", False) is False

    process_due_reminders(
        state,
        audit=audit,
        config=AgentConfig(testing_enabled=False),
        now=first_review + timedelta(minutes=5),
        directors={"Acme": "Directora Sol"},
        approval_handler=approval_handler,
        approval_graph=graph,
    )
    assert request["estado"] == "Escalada"
    assert request["escalado_director"] is True
    assert request["intentos_aprobacion_director"] == 2
    assert [payload["intento"] for payload in approval_payloads] == [1, 2]

    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    rejection = next(
        event
        for event in events
        if event["decision"] == "escalamiento_director_rechazado"
    )
    assert rejection["motivo"] == "Falta validar presupuesto."
    assert sum(
        event["decision"] == "aprobacion_director_solicitada"
        for event in events
    ) == 2


def test_unanswered_director_approval_is_repeated_with_next_sla_notification(
    tmp_path: Path,
) -> None:
    request = {
        "id": "m-unanswered-approval",
        "asunto": "Solicitud de prueba",
        "cliente": "Acme",
        "pm_responsable": "PM Ana",
        "estado": "Notificada",
        "fecha_notificacion": "2026-10-29T12:00:00+00:00",
        "hora_notificacion": 0,
    }
    state = initial_agent_state()
    state["solicitudes"] = [request]
    audit_path = tmp_path / "audit.jsonl"
    audit = JsonlAuditLog(audit_path)
    graph = build_director_approval_graph()
    approval_attempts: list[int] = []

    def approval_handler(payload: dict[str, Any]) -> dict[str, Any] | None:
        attempt = int(payload["intento"])
        approval_attempts.append(attempt)
        if attempt == 1:
            return None
        return {"approved": True, "reason": "Aprobado en el segundo aviso."}

    first_review = datetime.fromisoformat("2026-10-30T04:00:00+00:00")
    config = AgentConfig(testing_enabled=False)
    process_due_reminders(
        state,
        audit=audit,
        config=config,
        now=first_review,
        directors={"Acme": "Directora Sol"},
        approval_handler=approval_handler,
        approval_graph=graph,
    )
    assert request["aprobacion_director_pendiente"] is True
    assert request.get("escalado_director", False) is False

    actions_before_repeat = len(state["eventos"])
    process_due_reminders(
        state,
        audit=audit,
        config=config,
        now=first_review + timedelta(hours=2) - timedelta(minutes=1),
        directors={"Acme": "Directora Sol"},
        approval_handler=approval_handler,
        approval_graph=graph,
    )
    assert request["aprobacion_director_pendiente"] is True
    assert len(state["eventos"]) == actions_before_repeat

    process_due_reminders(
        state,
        audit=audit,
        config=config,
        now=first_review + timedelta(hours=2),
        directors={"Acme": "Directora Sol"},
        approval_handler=approval_handler,
        approval_graph=graph,
    )

    assert request["estado"] == "Escalada"
    assert request["escalado_director"] is True
    assert request["aprobacion_director_pendiente"] is False
    assert approval_attempts == [1, 1, 1, 2]
    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert sum(
        event["decision"] == "aprobacion_director_solicitada"
        for event in events
    ) == 2
    assert sum(
        event["decision"] == "aprobacion_director_sin_respuesta"
        for event in events
    ) == 1
    repeated_notice = next(
        event
        for event in events
        if event["decision"] == "repetir_solicitud_aprobacion_director"
    )
    assert repeated_notice["datos"]["destinatario"] == "Directora Sol"
    assert repeated_notice["datos"]["numero_notificacion"] >= 1


def test_terminal_can_answer_director_approval_without_blocking_state_lock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    approval_prompted = threading.Event()
    request = {
        "id": "m-terminal-approval",
        "asunto": "Solicitud de prueba",
        "cliente": "Acme",
        "pm_responsable": "PM Ana",
        "estado": "Notificada",
        "fecha_notificacion": (
            datetime.now(timezone.utc) - timedelta(hours=17)
        ).isoformat(),
        "hora_notificacion": 0,
    }
    initial_state = initial_agent_state()
    initial_state["solicitudes"] = [request]
    state_path = tmp_path / "state.json"
    save_agent_state(state_path, initial_state)

    class FakePromptSession:
        def prompt(self, prompt_text: str) -> str:
            if prompt_text != "agentmail> ":
                return ""
            if not approval_prompted.wait(timeout=3):
                return "stop"
            if not hasattr(self, "approval_sent"):
                self.approval_sent = True
                return "approve 1 sí"
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if (
                    load_agent_state(state_path)["solicitudes"][0]["estado"]
                    == "Escalada"
                ):
                    return "stop"
                time.sleep(0.01)
            return "stop"

    import terminal as terminal_module

    original_print = builtins.print

    def observe_approval_prompt(*args: Any, **kwargs: Any) -> None:
        if any("[APROBACIÓN REQUERIDA]" in str(arg) for arg in args):
            approval_prompted.set()
        original_print(*args, **kwargs)

    monkeypatch.setattr(terminal_module, "PromptSession", FakePromptSession)
    monkeypatch.setattr(
        terminal_module,
        "patch_stdout",
        lambda: nullcontext(),
    )
    monkeypatch.setattr(
        terminal_module,
        "print",
        observe_approval_prompt,
        raising=False,
    )

    run_terminal_agent(
        FakeAgentMailClient([]),
        audit_path=tmp_path / "audit.jsonl",
        reconnect_interval=0.01,
        assignments={"Acme": "PM Ana"},
        directors={"Acme": "Directora Sol"},
        state_path=state_path,
        config=AgentConfig(testing_enabled=False),
    )

    saved_state = load_agent_state(state_path)
    assert saved_state["solicitudes"][0]["estado"] == "Escalada"
    assert saved_state["solicitudes"][0]["escalado_director"] is True
    assert any(
        json.loads(line)["decision"] == "escalamiento_director_aprobado"
        for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    )


def test_simulation_report_is_reproducible_without_llm(tmp_path: Path) -> None:
    messages = [
        make_message(
            "m1",
            "Solicitud de personal",
            "Cliente: Acme\nRol: Analista\nCantidad: 1",
        ),
        make_message(
            "m2",
            "Solicitud de personal",
            "Cliente: Beta\nRol: Ingeniero\nCantidad: 2",
        ),
    ]
    config = AgentConfig(max_simulated_hours=37)
    profiles = {
        "PM Ana": {"perfil": "ausente", "horas_confirmacion": None},
        "PM Luis": {"perfil": "normal", "horas_confirmacion": None},
    }
    assignments = {"Acme": "PM Ana", "Beta": "PM Luis"}
    directors = {"Acme": "Directora Sol", "Beta": "Director Max"}
    reports = []
    for run in (1, 2):
        reports.append(
            run_simulation(
                FakeAgentMailClient([messages]),
                audit_path=tmp_path / f"audit-{run}.jsonl",
                assignments=assignments,
                directors=directors,
                pm_profiles=profiles,
                simulation_seed=42,
                config=config,
            )
        )

    assert reports[0] == reports[1]
    assert reports[0]["modo"] == "sin_llm"
    assert reports[0]["motivo_finalizacion"] == "maximo_horas_simuladas"
    assert reports[0]["horas_simuladas"] == 37
    no_response, simulated_response = reports[0]["solicitudes"]
    assert no_response["hora_confirmacion"] is None
    assert no_response["nivel_maximo_escalamiento"] == "Comercial/Delivery"
    assert no_response["cumplio_sla"] is False
    expected_delay = simulated_confirmation_delay(
        {"id": "m2", "pm_responsable": "PM Luis"},
        profiles["PM Luis"],
        seed=42,
    )
    assert simulated_response["hora_confirmacion"] == expected_delay
    assert simulated_response["cumplio_sla"] is True
    assert reports[0]["limites"]["llamadas_llm_realizadas"] == 0
    audit_events = [
        json.loads(line)
        for line in (tmp_path / "audit-1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert next(
        event for event in audit_events
        if event["decision"] == "recordatorio_enviado"
        and event["solicitud"] == "m1"
    )["hora_simulada"] == 4
    assert next(
        event for event in audit_events
        if event["decision"] == "escalamiento_director_simulado"
        and event["solicitud"] == "m1"
    )["hora_simulada"] == 16
    assert next(
        event for event in audit_events
        if event["decision"] == "segundo_recordatorio_enviado"
        and event["solicitud"] == "m1"
    )["hora_simulada"] == 8
    assert next(
        event for event in audit_events
        if event["decision"] == "recordatorio_siguiente_nivel_enviado"
        and event["solicitud"] == "m1"
    )["hora_simulada"] == 24
    assert next(
        event for event in audit_events
        if event["decision"] == "involucrar_comercial_y_delivery"
        and event["solicitud"] == "m1"
    )["hora_simulada"] == 32
    assert [
        event["hora_simulada"]
        for event in audit_events
        if event["decision"] == "repetir_recordatorio_escalamiento"
        and event["solicitud"] == "m1"
    ] == [34, 36]
    followup_events = [
        event
        for event in audit_events
        if event["solicitud"] == "m1"
        and "numero_notificacion" in event["datos"]
    ]
    assert [
        event["datos"]["numero_notificacion"] for event in followup_events
    ] == list(range(1, 8))
    assert [
        event["datos"]["destinatario"] for event in followup_events
    ] == [
        "PM Ana",
        "PM Ana",
        "Directora Sol",
        "Directora Sol",
        "Equipo Comercial; Responsable de Delivery",
        "Director; Equipo Comercial; Responsable de Delivery",
        "Director; Equipo Comercial; Responsable de Delivery",
    ]
    assert all(
        event["datos"]["pm_responsable"] == "PM Ana"
        for event in followup_events
    )

    report_paths = (tmp_path / "report-1.json", tmp_path / "report-2.json")
    for path, report in zip(report_paths, reports):
        write_sla_report(path, report)
    assert report_paths[0].read_bytes() == report_paths[1].read_bytes()


def test_simulation_stops_early_when_requests_finish_and_meets_sla(
    tmp_path: Path,
) -> None:
    message = make_message(
        "m1",
        "Solicitud de personal",
        "Cliente: Acme\nRol: Analista\nCantidad: 1",
    )
    report = run_simulation(
        FakeAgentMailClient([[message]]),
        audit_path=tmp_path / "audit.jsonl",
        assignments={"Acme": "PM Ana"},
        pm_profiles={
            "PM Ana": {"perfil": "rapido", "horas_confirmacion": 2}
        },
        simulation_seed=42,
        config=AgentConfig(sla_target_hours=3, max_simulated_hours=12),
    )

    assert report["motivo_finalizacion"] == "todas_confirmadas_o_cerradas"
    assert report["horas_simuladas"] == 2
    assert report["solicitudes"][0]["hora_confirmacion"] == 2
    assert report["solicitudes"][0]["cumplio_sla"] is True
    assert report["solicitudes"][0]["pm_confirmo"] is True
    assert report["solicitudes"][0]["trabajo_finalizado"] is False
    assert report["solicitudes"][0]["nivel_maximo_escalamiento"] == "Ninguno"


def test_director_escalation_does_not_prevent_late_pm_confirmation(
    tmp_path: Path,
) -> None:
    message = make_message(
        "m1",
        "Solicitud de personal",
        "Cliente: Acme\nRol: Analista\nCantidad: 1",
    )
    report = run_simulation(
        FakeAgentMailClient([[message]]),
        audit_path=tmp_path / "audit.jsonl",
        assignments={"Acme": "PM Ana"},
        directors={"Acme": "Directora Sol"},
        pm_profiles={
            "PM Ana": {"perfil": "lento", "horas_confirmacion": 18}
        },
        simulation_seed=42,
        config=AgentConfig(max_simulated_hours=20),
    )

    request = report["solicitudes"][0]
    assert report["motivo_finalizacion"] == "todas_confirmadas_o_cerradas"
    assert report["horas_simuladas"] == 18
    assert request["estado_final"] == "Confirmada"
    assert request["hora_confirmacion"] == 18
    assert request["nivel_maximo_escalamiento"] == "Director"
    assert request["cumplio_sla"] is False


def test_config_yaml_loads_thresholds_and_enforces_hard_limits(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "sla:\n"
        "  reminder_after_hours: 3\n"
        "  second_reminder_after_hours: 5\n"
        "  escalate_after_hours: 7\n"
        "  next_level_reminder_after_hours: 9\n"
        "  commercial_delivery_after_hours: 11\n"
        "  target_hours: 7\n"
        "simulation:\n"
        "  max_hours: 20\n"
        "  recursion_limit: 24\n"
        "llm:\n"
        "  max_calls: 40\n"
        "reminder_email:\n"
        "  enabled: true\n"
        "  recipient: notices@example.test\n",
        encoding="utf-8",
    )

    config = load_agent_config(config_path)
    assert config == AgentConfig(
        reminder_after_hours=3,
        second_reminder_after_hours=5,
        escalate_after_hours=7,
        next_level_reminder_after_hours=9,
        commercial_delivery_after_hours=11,
        sla_target_hours=7,
        max_simulated_hours=20,
        recursion_limit=24,
        max_llm_calls=40,
        reminder_email_enabled=True,
        reminder_email_recipient="notices@example.test",
    )
    assignments_path = tmp_path / "asignacion.csv"
    assignments_path.write_text(
        "cliente,PM,director\nAcme,PM Ana,Directora Sol\n",
        encoding="utf-8",
    )
    assert load_director_assignments(assignments_path) == {
        "acme": "Directora Sol"
    }

    config_path.write_text(
        "simulation:\n  max_hours: 49\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="simulation.max_hours"):
        load_agent_config(config_path)
    config_path.write_text(
        "simulation:\n  recursion_limit: 5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="recursion_limit"):
        load_agent_config(config_path)
    with pytest.raises(ValueError, match="max_hours"):
        AgentConfig(max_simulated_hours=49)


def test_run_simulation_uses_configured_recursion_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seen_limits = []

    class OneTickGraph:
        def invoke(self, state, *, config):
            seen_limits.append(config["recursion_limit"])
            next_state = dict(state)
            next_state["hora_actual"] += 1
            return next_state

    monkeypatch.setattr("service.build_graph", lambda *args, **kwargs: OneTickGraph())
    report = run_simulation(
        FakeAgentMailClient(),
        audit_path=tmp_path / "audit.jsonl",
        config=AgentConfig(max_simulated_hours=12, recursion_limit=9),
    )

    assert seen_limits == [9]
    assert report["motivo_finalizacion"] == "sin_solicitudes"


def test_simulation_hard_caps_llm_calls_at_sixty(tmp_path: Path) -> None:
    class CountingOpenRouter:
        model = "test/model"

        def __init__(self) -> None:
            self.calls = 0

        def classify_message(self, message, known_requests):
            self.calls += 1
            return {"categoria": "no_relacionado", "motivo": "No relacionado."}

    messages = [
        make_message(
            f"m{index:02}",
            "Aviso general",
            "Reunión de equipo.",
        )
        for index in range(61)
    ]
    llm = CountingOpenRouter()
    report = run_simulation(
        FakeAgentMailClient([messages]),
        audit_path=tmp_path / "audit.jsonl",
        llm=llm,
        config=AgentConfig(),
    )

    assert llm.calls == 60
    assert report["limites"]["llamadas_llm_realizadas"] == 60


def test_opened_request_and_terminal_close_transition_are_audited(
    tmp_path: Path,
) -> None:
    message = make_message(
        "m1",
        "Solicitud de personal",
        "Cliente: Acme\nRol: Analista\nCantidad: 1",
    )
    audit_path = tmp_path / "audit.jsonl"
    state = run_phase_one(
        FakeAgentMailClient([[message]]),
        audit_path=audit_path,
        state={
            **initial_agent_state(),
            "simular_confirmacion_automatica": False,
        },
        assignments={},
        pm_profiles={},
    )
    request = state["solicitudes"][0]
    assert request["estado"] == "Nueva"

    mark_request_opened(
        state,
        identifier="m1",
        audit=JsonlAuditLog(audit_path),
    )
    assert request["estado"] == "Notificada"

    close_request(
        state,
        request_number=1,
        reason="Cliente canceló la solicitud.",
        audit=JsonlAuditLog(audit_path),
    )
    assert request["estado"] == "Cerrada"
    assert request["razon_finalizacion"] == "Cliente canceló la solicitud."
    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["decision"] == "solicitud_abierta" for event in events)
    assert any(event["decision"] == "solicitud_cerrada" for event in events)
    closure_event = next(
        event for event in events if event["decision"] == "solicitud_cerrada"
    )
    assert closure_event["motivo"] == "Cliente canceló la solicitud."
    assert (
        closure_event["datos"]["razon_finalizacion"]
        == "Cliente canceló la solicitud."
    )


def test_pm_reply_to_reminder_updates_existing_request_instead_of_creating_one(
    tmp_path: Path,
) -> None:
    original = make_message(
        "request-1",
        "Solicitud de personal",
        "Cliente: Acme\nRol: Analista\nCantidad: 1",
        sender="cliente@example.test",
    )
    reply = make_message(
        "reply-1",
        "Re: Solicitud de personal",
        (
            "No estaré presente hasta el 20 de octubre.\n\n"
            "> Solicitud de personal\n> Cliente: Acme\n> Rol: Analista"
        ),
        sender="ana@example.test",
        in_reply_to="reminder-outbound-1",
    )
    client = FakeAgentMailClient([[original], [reply]])
    profiles = {
        "PM Ana": {
            "emails": ["ana@example.test"],
            "perfil": "normal",
        }
    }
    audit_path = tmp_path / "audit.jsonl"

    state = run_phase_one(
        client,
        audit_path=audit_path,
        assignments={"Acme": "PM Ana"},
        pm_profiles=profiles,
    )
    state = run_phase_one(
        client,
        audit_path=audit_path,
        state=state,
        assignments={"Acme": "PM Ana"},
        pm_profiles=profiles,
    )

    assert len(state["solicitudes"]) == 1
    assert state["solicitudes"][0]["id"] == "request-1"
    assert state["correos_clasificados"][-1]["categoria"] == (
        "actualizacion_solicitud"
    )
    update_event = next(
        event
        for event in reversed(state["eventos"])
        if event["solicitud"] == "reply-1"
        and event["nodo"] == "actualizar_solicitud_desde_correo"
    )
    assert update_event["datos"]["solicitud_relacionada_id"] == "request-1"


def test_messages_sent_from_agent_inbox_are_not_classified_as_requests(
    tmp_path: Path,
) -> None:
    sent_reminder = make_message(
        "sent-reminder-1",
        "Recordatorio: solicitud de personal",
        "Solicitamos un analista para Acme.",
        sender="test@agentmail.to",
    )

    state = run_phase_one(
        FakeAgentMailClient([[sent_reminder]]),
        audit_path=tmp_path / "audit.jsonl",
    )

    assert state["solicitudes"] == []
    assert state["correos_clasificados"][0]["categoria"] == (
        "correo_saliente_ignorado"
    )


def test_terminal_close_and_complete_save_reasons_and_show_them_in_history(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    state = initial_agent_state()
    state["solicitudes"] = [
        {
            "id": "request-1",
            "asunto": "Solicitud cancelada",
            "estado": "Notificada",
        },
        {
            "id": "request-2",
            "asunto": "Solicitud completada",
            "estado": "Notificada",
        },
    ]
    state_path = tmp_path / "state.json"
    save_agent_state(state_path, state)

    class IdleAgentMailClient(FakeAgentMailClient):
        async def message_events(self):
            self.stream_calls += 1
            yield {"type": "subscribed"}
            await asyncio.Event().wait()

    class FakePromptSession:
        def __init__(self) -> None:
            self.commands = iter(
                [
                    "close #1",
                    "close #1 Cliente canceló la solicitud.",
                    "complete 2",
                    "complete 2 Las posiciones fueron cubiertas.",
                    "history",
                    "stop",
                ]
            )

        def prompt(self, prompt_text: str) -> str:
            assert prompt_text == "agentmail> "
            return next(self.commands)

    import terminal as terminal_module

    monkeypatch.setattr(terminal_module, "PromptSession", FakePromptSession)
    monkeypatch.setattr(terminal_module, "patch_stdout", lambda: nullcontext())

    run_terminal_agent(
        IdleAgentMailClient([]),
        audit_path=tmp_path / "audit.jsonl",
        reconnect_interval=0.01,
        assignments={},
        state_path=state_path,
        config=AgentConfig(testing_enabled=False),
    )

    output = capsys.readouterr().out
    saved_requests = load_agent_state(state_path)["solicitudes"]
    assert "Uso: close <número de solicitud> <razón en texto>" in output
    assert "Uso: complete <número de solicitud> <razón en texto>" in output
    assert "[CERRADA] Solicitud #1" in output
    assert "Cliente canceló la solicitud." in output
    assert "Las posiciones fueron cubiertas." in output
    assert saved_requests[0]["estado"] == "Cerrada"
    assert saved_requests[0]["razon_finalizacion"] == (
        "Cliente canceló la solicitud."
    )
    assert saved_requests[1]["estado"] == "Cerrada"
    assert saved_requests[1]["razon_finalizacion"] == (
        "Las posiciones fueron cubiertas."
    )
    audit_events = [
        json.loads(line)
        for line in (tmp_path / "audit.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert any(
        event.get("motivo") == "Cliente canceló la solicitud."
        for event in audit_events
    )
    assert any(
        event.get("motivo") == "Las posiciones fueron cubiertas."
        for event in audit_events
    )


def test_terminal_close_and_complete_send_lifecycle_emails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state = initial_agent_state()
    state["solicitudes"] = [
        {
            "id": "request-close",
            "asunto": "Solicitud cancelada",
            "cliente": "Acme",
            "pm_responsable": "PM Santiago",
            "estado": "Notificada",
            "correo_solicitante": "requester1@example.test",
        },
        {
            "id": "request-complete",
            "asunto": "Solicitud completada",
            "cliente": "Globex",
            "pm_responsable": "PM Santiago",
            "estado": "Notificada",
            "correo_solicitante": "requester2@example.test",
        },
    ]
    state_path = tmp_path / "state.json"
    save_agent_state(state_path, state)
    shared_pm_address = "pm@example.test"

    class IdleRecordingAgentMailClient(RecordingAgentMailClient):
        async def message_events(self):
            self.stream_calls += 1
            yield {"type": "subscribed"}
            await asyncio.Event().wait()

    class FakePromptSession:
        def __init__(self) -> None:
            self.commands = iter(
                [
                    "close 1 Cliente canceló por presupuesto.",
                    "complete 2 Se cubrieron las posiciones.",
                    "stop",
                ]
            )

        def prompt(self, prompt_text: str) -> str:
            assert prompt_text == "agentmail> "
            return next(self.commands)

    import terminal as terminal_module

    monkeypatch.setattr(terminal_module, "PromptSession", FakePromptSession)
    monkeypatch.setattr(terminal_module, "patch_stdout", lambda: nullcontext())
    client = IdleRecordingAgentMailClient()

    run_terminal_agent(
        client,
        audit_path=tmp_path / "audit.jsonl",
        reconnect_interval=0.01,
        assignments={
            "Acme": "PM Santiago",
            "Globex": "PM Santiago",
        },
        pm_profiles={
            "PM Santiago": {
                "nombre": "PM Santiago",
                "perfil": "normal",
                "elegible": True,
                "emails": [shared_pm_address],
            }
        },
        state_path=state_path,
        config=AgentConfig(llm_enabled=False),
    )

    assert len(client.sent_messages) == 4
    recipients = {
        message[0]
        for message in client.sent_messages
    }
    assert recipients == {
        "requester1@example.test",
        "requester2@example.test",
        shared_pm_address,
    }
    assert any(
        "Cliente canceló por presupuesto." in message[2]
        for message in client.sent_messages
    )
    assert any(
        "Se cubrieron las posiciones." in message[2]
        for message in client.sent_messages
    )


def test_pm_acceptance_and_completion_have_distinct_statuses(
    tmp_path: Path,
) -> None:
    message = make_message(
        "pm-acceptance",
        "Solicitud de personal",
        "Cliente: Acme\nRol: Analista\nCantidad: 1",
    )
    audit_path = tmp_path / "audit.jsonl"
    state = run_phase_one(
        FakeAgentMailClient([[message]]),
        audit_path=audit_path,
        assignments={"Acme": "PM Ana"},
        pm_profiles={"PM Ana": {"perfil": "normal"}},
    )

    apply_pm_update(
        state,
        request_number=1,
        response_text="I will take ownership of this request.",
        audit=JsonlAuditLog(audit_path),
        llm=None,
        assignments={"Acme": "PM Ana"},
    )
    request = state["solicitudes"][0]
    assert request["estado"] == "Confirmada"
    assert request["hora_confirmacion"] == state["hora_actual"]
    assert completed_requests(state) == []
    assert count_open_projects(state["solicitudes"], "PM Ana") == 1

    apply_pm_update(
        state,
        request_number=1,
        response_text="The positions are filled.",
        audit=JsonlAuditLog(audit_path),
        llm=None,
        assignments={"Acme": "PM Ana"},
    )
    assert request["estado"] == "Cerrada"
    assert completed_requests(state)[0][1]["id"] == "pm-acceptance"


def test_pm_status_update_escalates_or_confirms_request(tmp_path: Path) -> None:
    original = make_message(
        "m1",
        "Solicitud de personal",
        "Cliente: Acme\nRol: Analista\nCantidad: 1",
    )
    reply = make_message(
        "m2",
        "Re: Solicitud de personal",
        "We are working on it now.",
        in_reply_to="m1",
    )
    audit_path = tmp_path / "audit.jsonl"
    state = run_phase_one(
        FakeAgentMailClient([[original]]),
        audit_path=audit_path,
        assignments={"Acme": "PM Ana"},
    )

    updated = run_phase_one(
        FakeAgentMailClient([[original, reply]]),
        audit_path=audit_path,
        state=state,
        assignments={"Acme": "PM Ana"},
    )

    assert updated["solicitudes"][0]["estado"] == "Escalada"
    reply_email = next(
        email for email in updated["correos_clasificados"]
        if email["message_id"] == "m2"
    )
    assert reply_email["categoria"] == "actualizacion_solicitud"


def test_state_store_rejects_corrupt_state_file(tmp_path: Path) -> None:
    state_path = tmp_path / "agent-state.json"
    state_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(AgentStateStoreError, match="No se pudo leer"):
        load_agent_state(state_path)


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
        inbox_responses: list[list[dict[str, Any]]] | None = None,
        *,
        events: list[dict[str, Any]] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.inbox_id = "test@agentmail.to"
        self.inbox_responses = inbox_responses or []
        self.events = events or []
        self.list_calls = 0
        self.stream_calls = 0
        self.stop_event = stop_event
        self.message_by_id = {
            message["message_id"]: message
            for page in self.inbox_responses
            for message in page
        }

    def list_messages(self) -> list[dict[str, Any]]:
        if not self.inbox_responses:
            self.list_calls += 1
            return []
        index = min(self.list_calls, len(self.inbox_responses) - 1)
        self.list_calls += 1
        return list(self.inbox_responses[index])

    def get_message(self, identifier: str) -> dict[str, Any]:
        return dict(self.message_by_id[identifier])

    async def message_events(self):
        self.stream_calls += 1
        yield {"type": "subscribed"}
        for message in self.events:
            yield {
                "type": "message_received",
                "event_type": "message.received",
                "event_id": f"evt-{message['message_id']}",
                "message": message,
            }
        while self.stop_event is not None and not self.stop_event.is_set():
            await asyncio.sleep(0.01)


class RecordingAgentMailClient(FakeAgentMailClient):
    def __init__(
        self,
        inbox_responses: list[list[dict[str, Any]]] | None = None,
        *,
        events: list[dict[str, Any]] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        super().__init__(
            inbox_responses,
            events=events,
            stop_event=stop_event,
        )
        self.sent_messages: list[tuple[str, str, str]] = []

    def send(self, recipient: str, subject: str, text: str) -> None:
        self.sent_messages.append((recipient, subject, text))
