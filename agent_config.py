from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from settings import CONFIG_PATH


@dataclass(frozen=True)
class AgentConfig:
    reminder_after_hours: int = 4
    escalate_after_hours: int = 8
    sla_target_hours: int = 8
    max_simulated_hours: int = 48
    recursion_limit: int = 16
    max_llm_calls: int = 60
    max_open_projects_per_pm: int = 10
    testing_enabled: bool = False
    test_reminder_after_minutes: int = 1
    test_escalate_after_minutes: int = 2
    test_sla_target_minutes: int = 3
    sla_check_interval_seconds: int = 5

    def __post_init__(self) -> None:
        _positive_integer(
            self.reminder_after_hours,
            name="sla.reminder_after_hours",
            maximum=48,
        )
        _positive_integer(
            self.escalate_after_hours,
            name="sla.escalate_after_hours",
            maximum=48,
        )
        _positive_integer(
            self.sla_target_hours,
            name="sla.target_hours",
            maximum=48,
        )
        _positive_integer(
            self.max_simulated_hours,
            name="simulation.max_hours",
            maximum=48,
        )
        _positive_integer(
            self.recursion_limit,
            name="simulation.recursion_limit",
            maximum=1000,
        )
        _positive_integer(
            self.max_llm_calls,
            name="llm.max_calls",
            maximum=60,
        )
        _positive_integer(
            self.max_open_projects_per_pm,
            name="assignment.max_open_projects_per_pm",
            maximum=1000,
        )
        _positive_integer(
            self.test_reminder_after_minutes,
            name="testing.reminder_after_minutes",
            maximum=48 * 60,
        )
        _positive_integer(
            self.test_escalate_after_minutes,
            name="testing.escalate_after_minutes",
            maximum=48 * 60,
        )
        _positive_integer(
            self.test_sla_target_minutes,
            name="testing.sla_target_minutes",
            maximum=48 * 60,
        )
        _positive_integer(
            self.sla_check_interval_seconds,
            name="testing.check_interval_seconds",
            maximum=3600,
        )
        if self.recursion_limit < 6:
            raise ValueError(
                "simulation.recursion_limit debe ser al menos 6 para ejecutar "
                "una vuelta completa del grafo."
            )
        if self.reminder_after_hours >= self.escalate_after_hours:
            raise ValueError(
                "sla.reminder_after_hours debe ser menor que "
                "sla.escalate_after_hours."
            )
        if self.sla_target_hours > self.max_simulated_hours:
            raise ValueError(
                "sla.target_hours no puede superar simulation.max_hours."
            )
        if self.test_reminder_after_minutes >= self.test_escalate_after_minutes:
            raise ValueError(
                "testing.reminder_after_minutes debe ser menor que "
                "testing.escalate_after_minutes."
            )
def _read_section(
    payload: dict[str, Any],
    name: str,
    allowed_keys: set[str],
) -> dict[str, Any]:
    section = payload.get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"La sección {name!r} de config.yaml debe ser un objeto.")
    unknown_keys = set(section) - allowed_keys
    if unknown_keys:
        raise ValueError(
            f"Claves desconocidas en {name!r}: {', '.join(sorted(unknown_keys))}."
        )
    return section


def _positive_integer(value: Any, *, name: str, maximum: int) -> int:
    if type(value) is not int or value < 1 or value > maximum:
        raise ValueError(f"{name} debe ser un entero entre 1 y {maximum}.")
    return value


def load_agent_config(path: Path = CONFIG_PATH) -> AgentConfig:
    try:
        with path.open("r", encoding="utf-8") as source:
            payload = yaml.safe_load(source)
    except OSError as exc:
        raise ValueError(f"No se pudo leer la configuración {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML inválido en {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"La configuración {path} debe contener un objeto YAML.")
    unknown_sections = set(payload) - {
        "sla",
        "simulation",
        "llm",
        "assignment",
        "testing",
    }
    if unknown_sections:
        raise ValueError(
            f"Secciones desconocidas en {path}: {', '.join(sorted(unknown_sections))}."
        )

    sla = _read_section(
        payload,
        "sla",
        {"reminder_after_hours", "escalate_after_hours", "target_hours"},
    )
    simulation = _read_section(
        payload,
        "simulation",
        {"max_hours", "recursion_limit"},
    )
    llm = _read_section(payload, "llm", {"max_calls"})
    assignment = _read_section(
        payload,
        "assignment",
        {"max_open_projects_per_pm"},
    )
    testing = _read_section(
        payload,
        "testing",
        {
            "enabled",
            "reminder_after_minutes",
            "escalate_after_minutes",
            "sla_target_minutes",
            "check_interval_seconds",
        },
    )
    testing_enabled = testing.get("enabled", False)
    if not isinstance(testing_enabled, bool):
        raise ValueError("testing.enabled debe ser true o false.")

    config = AgentConfig(
        reminder_after_hours=_positive_integer(
            sla.get("reminder_after_hours", 4),
            name="sla.reminder_after_hours",
            maximum=48,
        ),
        escalate_after_hours=_positive_integer(
            sla.get("escalate_after_hours", 8),
            name="sla.escalate_after_hours",
            maximum=48,
        ),
        sla_target_hours=_positive_integer(
            sla.get("target_hours", 8),
            name="sla.target_hours",
            maximum=48,
        ),
        max_simulated_hours=_positive_integer(
            simulation.get("max_hours", 48),
            name="simulation.max_hours",
            maximum=48,
        ),
        recursion_limit=_positive_integer(
            simulation.get("recursion_limit", 16),
            name="simulation.recursion_limit",
            maximum=1000,
        ),
        max_llm_calls=_positive_integer(
            llm.get("max_calls", 60),
            name="llm.max_calls",
            maximum=60,
        ),
        max_open_projects_per_pm=_positive_integer(
            assignment.get("max_open_projects_per_pm", 10),
            name="assignment.max_open_projects_per_pm",
            maximum=1000,
        ),
        testing_enabled=testing_enabled,
        test_reminder_after_minutes=_positive_integer(
            testing.get("reminder_after_minutes", 1),
            name="testing.reminder_after_minutes",
            maximum=48 * 60,
        ),
        test_escalate_after_minutes=_positive_integer(
            testing.get("escalate_after_minutes", 2),
            name="testing.escalate_after_minutes",
            maximum=48 * 60,
        ),
        test_sla_target_minutes=_positive_integer(
            testing.get("sla_target_minutes", 3),
            name="testing.sla_target_minutes",
            maximum=48 * 60,
        ),
        sla_check_interval_seconds=_positive_integer(
            testing.get("check_interval_seconds", 5),
            name="testing.check_interval_seconds",
            maximum=3600,
        ),
    )
    return config
