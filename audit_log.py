
import json
import threading
from pathlib import Path
from typing import Any

class JsonlAuditLog:
    """Escribe cada evento como una línea JSON independiente."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, Any]) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(event, ensure_ascii=False) + "\n")


def audit_error(
    audit: JsonlAuditLog,
    error: Exception | str,
    *,
    node: str,
    hora_simulada: int = 0,
    solicitud: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    error_type = type(error).__name__ if isinstance(error, Exception) else "Error"
    event: dict[str, Any] = {
        "hora_simulada": hora_simulada,
        "solicitud": solicitud,
        "nodo": node,
        "decision": "error",
        "motivo": f"{error_type}: {error}",
    }
    if details:
        event["datos"] = details
    audit.append(event)
    return event
