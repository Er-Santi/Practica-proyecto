import json
import os
from pathlib import Path

from agent_state import AgentState, initial_agent_state


class AgentStateStoreError(RuntimeError):
    """Raised when persisted agent state cannot be read or written."""


def load_agent_state(path: Path) -> AgentState:
    if not path.exists():
        return initial_agent_state()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentStateStoreError(
            f"No se pudo leer el estado persistido en {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise AgentStateStoreError(
            f"El estado persistido en {path} no contiene un objeto JSON."
        )

    state = initial_agent_state()
    for key, default in state.items():
        value = payload.get(key, default)
        if not isinstance(value, type(default)):
            raise AgentStateStoreError(
                f"El campo {key!r} del estado persistido tiene un tipo inválido."
            )
        state[key] = value
    for request in state["solicitudes"]:
        if request.get("estado") == "Pendiente de asignación":
            request["estado"] = "Nueva"
        elif request.get("estado") == "Notificada (datos pendientes)":
            request["estado"] = "Notificada"
    return state


def save_agent_state(path: Path, state: AgentState) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError) as exc:
        raise AgentStateStoreError(
            f"No se pudo guardar el estado del agente en {path}: {exc}"
        ) from exc
