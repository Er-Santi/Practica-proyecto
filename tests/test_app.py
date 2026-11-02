import asyncio
import json
import sys
import threading
from datetime import datetime, timedelta
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
    write_sla_report,
)
from state_store import load_agent_state


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
    prompt_data = json.loads(
        request.call_args.kwargs["json"]["messages"][1]["content"]
    )
    assert prompt_data["message_id"] == "m1"


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
    assert "NOTIFICACIÓN SIMULADA" in capsys.readouterr().out


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
        assignments={"Acme": "PM Ana"},
        llm=FakeOpenRouter(),
    )

    assert result["correos_clasificados"][0]["categoria"] == "solicitud_posicion"
    request = result["solicitudes"][0]
    assert request["cliente"] == "Acme"
    assert request["rol"] == "Analista de datos"
    assert request["cantidad_posiciones"] == 2
    assert request["pm_responsable"] == "PM Ana"
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
    assert "Para: PM Ana" in capsys.readouterr().out


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
                    "pregunta_confirmacion": "¿El PM recibió la solicitud para Acme?",
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
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr("app.load_dotenv", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.py", "--audit", str(audit_path)],
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
        "perfil": "rapido",
        "horas_confirmacion": 2,
    }
    assert profiles["pm luis"]["perfil"] == "ausente"
    assert profiles["pm luis"]["horas_confirmacion"] is None


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
        assignments={"Acme": "PM Ana"},
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
    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["decision"] == "pm_confirmado_por_usuario" for event in events)
    assert any(
        event["decision"] == "pm_no_confirmado_por_usuario"
        for event in events
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
        state_path=state_path,
    )

    assert client.list_calls == 1
    assert result["correos_procesados"] == ["old-1"]
    assert result["solicitudes"][0]["estado"] == "Nueva"
    assert "CORREO ANTIGUO" in capsys.readouterr().out
    restored = load_agent_state(state_path)
    assert restored["correos_procesados"] == ["old-1"]
    assert restored["solicitudes"][0]["id"] == "old-1"


def test_due_reminder_is_sent_once_after_configured_four_hours(tmp_path: Path) -> None:
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
    reminder_time = notified_at + timedelta(hours=4)

    sent = process_due_reminders(
        state,
        audit=JsonlAuditLog(audit_path),
        config=AgentConfig(testing_enabled=False),
        now=reminder_time,
    )

    assert sent == 1
    assert request["estado"] == "Recordatorio enviado"
    assert process_due_reminders(
        state,
        audit=JsonlAuditLog(audit_path),
        config=AgentConfig(testing_enabled=False),
        now=reminder_time + timedelta(hours=1),
    ) == 0
    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    reminder = next(event for event in events if event["decision"] == "recordatorio_enviado")
    assert reminder["datos"]["reloj_simulado"] is False
    assert reminder["datos"]["notificacion_simulada"] is True
    assert reminder["datos"]["umbral"] == 4


def test_testing_sla_uses_one_and_two_minute_thresholds(tmp_path: Path) -> None:
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

    escalation_actions = process_due_reminders(
        state,
        audit=audit,
        config=config,
        now=datetime.fromisoformat("2026-10-29T12:02:00+00:00"),
        directors={"Acme": "Directora Sol"},
    )
    assert escalation_actions == 1
    assert request["estado"] == "Escalada"
    assert request["nivel_maximo_escalamiento"] == "Director"
    events = [
        json.loads(line)
        for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["datos"]["umbral"] for event in events] == [1, 2]
    assert [event["datos"]["unidad_umbral"] for event in events] == [
        "minutos",
        "minutos",
    ]


def test_director_escalation_is_triggered_at_eight_hours(tmp_path: Path) -> None:
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

    processed = process_due_reminders(
        state,
        audit=JsonlAuditLog(audit_path),
        config=AgentConfig(testing_enabled=False),
        now=notified_at + timedelta(hours=8),
        directors={"Acme": "Directora Sol"},
    )

    assert processed == 2
    assert request["estado"] == "Escalada"
    assert request["nivel_maximo_escalamiento"] == "Director"
    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    escalation = next(
        event
        for event in events
        if event["decision"] == "escalamiento_director_simulado"
    )
    assert escalation["datos"]["director"] == "Directora Sol"
    assert escalation["datos"]["umbral"] == 8


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
    config = AgentConfig(max_simulated_hours=10)
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
    assert reports[0]["horas_simuladas"] == 10
    no_response, simulated_response = reports[0]["solicitudes"]
    assert no_response["hora_confirmacion"] is None
    assert no_response["nivel_maximo_escalamiento"] == "Director"
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
    )["hora_simulada"] == 8

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

    assert report["motivo_finalizacion"] == "todas_finalizadas"
    assert report["horas_simuladas"] == 2
    assert report["solicitudes"][0]["hora_confirmacion"] == 2
    assert report["solicitudes"][0]["cumplio_sla"] is True
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
            "PM Ana": {"perfil": "lento", "horas_confirmacion": 12}
        },
        simulation_seed=42,
        config=AgentConfig(max_simulated_hours=15),
    )

    request = report["solicitudes"][0]
    assert report["motivo_finalizacion"] == "todas_finalizadas"
    assert report["horas_simuladas"] == 12
    assert request["estado_final"] == "Confirmada"
    assert request["hora_confirmacion"] == 12
    assert request["nivel_maximo_escalamiento"] == "Director"
    assert request["cumplio_sla"] is False


def test_config_yaml_loads_thresholds_and_enforces_hard_limits(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "sla:\n"
        "  reminder_after_hours: 3\n"
        "  escalate_after_hours: 7\n"
        "  target_hours: 7\n"
        "simulation:\n"
        "  max_hours: 20\n"
        "  recursion_limit: 24\n"
        "llm:\n"
        "  max_calls: 40\n",
        encoding="utf-8",
    )

    config = load_agent_config(config_path)
    assert config == AgentConfig(3, 7, 7, 20, 24, 40)
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
        audit=JsonlAuditLog(audit_path),
    )
    assert request["estado"] == "Cerrada"
    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["decision"] == "solicitud_abierta" for event in events)
    assert any(event["decision"] == "solicitud_cerrada" for event in events)


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
