import csv
import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping

from settings import (
    ASSIGNMENTS_PATH,
    HIRING_ROLE_MARKERS,
    PM_PROFILE_HOUR_RANGES,
    PM_PROFILES_PATH,
    REQUEST_MARKERS,
)

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


def load_assignments(path: Path = ASSIGNMENTS_PATH) -> dict[str, str]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        headers = {
            normalize_text(header.strip()): header
            for header in (reader.fieldnames or [])
            if header
        }
        if "cliente" not in headers or "pm" not in headers:
            raise ValueError(f"{path} debe contener las columnas cliente y pm.")

        assignments: dict[str, str] = {}
        for row_number, row in enumerate(reader, start=2):
            client = str(row.get(headers["cliente"]) or "").strip()
            pm = str(row.get(headers["pm"]) or "").strip()
            if not client and not pm:
                continue
            if not client or not pm:
                raise ValueError(
                    f"Fila {row_number} de {path}: cliente y pm son obligatorios."
                )
            client_key = normalize_text(client)
            if client_key in assignments and assignments[client_key] != pm:
                raise ValueError(
                    f"Fila {row_number} de {path}: cliente {client} tiene varios PM."
                )
            assignments[client_key] = pm
    return assignments


def load_pm_profiles(path: Path = PM_PROFILES_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    profile_aliases = {
        "rapido": "rapido",
        "fast": "rapido",
        "normal": "normal",
        "lento": "lento",
        "slow": "lento",
        "ausente": "ausente",
        "absent": "ausente",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        headers = {
            normalize_text(header.strip()): header
            for header in (reader.fieldnames or [])
            if header
        }
        if "pm" not in headers or "perfil" not in headers:
            raise ValueError(f"{path} debe contener las columnas pm y perfil.")

        profiles: dict[str, dict[str, Any]] = {}
        for row_number, row in enumerate(reader, start=2):
            pm = str(row.get(headers["pm"]) or "").strip()
            profile_text = normalize_text(str(row.get(headers["perfil"]) or ""))
            if not pm and not profile_text:
                continue
            profile = profile_aliases.get(profile_text)
            if not pm or profile is None:
                raise ValueError(
                    f"Fila {row_number} de {path}: indica un PM y un perfil "
                    "rapido, normal, lento o ausente."
                )

            hours_header = headers.get("horas_confirmacion")
            hours_text = str(row.get(hours_header) or "").strip() if hours_header else ""
            hours: int | None = None
            if hours_text:
                try:
                    hours = int(hours_text)
                except ValueError as exc:
                    raise ValueError(
                        f"Fila {row_number} de {path}: horas_confirmacion debe ser entero."
                    ) from exc
                if profile == "ausente":
                    raise ValueError(
                        f"Fila {row_number} de {path}: el perfil ausente no confirma."
                    )
                minimum, maximum = PM_PROFILE_HOUR_RANGES[profile]
                if not minimum <= hours <= maximum:
                    raise ValueError(
                        f"Fila {row_number} de {path}: {profile} confirma entre "
                        f"{minimum} y {maximum} horas."
                    )

            key = normalize_text(pm)
            if key in profiles:
                raise ValueError(f"Fila {row_number} de {path}: PM duplicado {pm}.")
            profiles[key] = {"perfil": profile, "horas_confirmacion": hours}
    return profiles


def simulated_confirmation_delay(
    request: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    seed: int,
) -> int | None:
    profile_name = str(profile.get("perfil") or "normal")
    if profile_name == "ausente":
        return None
    configured_hours = profile.get("horas_confirmacion")
    if configured_hours is not None:
        return int(configured_hours)
    minimum, maximum = PM_PROFILE_HOUR_RANGES.get(
        profile_name, PM_PROFILE_HOUR_RANGES["normal"]
    )
    stable_key = (
        f"{seed}:{request.get('id', '')}:"
        f"{normalize_text(str(request.get('pm_responsable') or ''))}"
    )
    stable_value = int(hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:8], 16)
    return minimum + stable_value % (maximum - minimum + 1)


def _extract_labeled_value(text: str, labels: tuple[str, ...]) -> str | None:
    label_pattern = "|".join(re.escape(label) for label in labels)
    match = re.search(
        rf"(?im)(?:^|[,;\n])\s*(?:{label_pattern})\s*[:=]\s*"
        r"(.+?)(?=\s*(?:[,;\n]|$))",
        text,
    )
    if not match:
        return None
    value = match.group(1).strip(" \t.")
    return value or None


def extract_request_details(
    message: dict[str, Any],
    *,
    assignments: Mapping[str, str],
) -> dict[str, Any]:
    text = message_text(message)
    client = _extract_labeled_value(
        text, ("cliente", "empresa", "customer", "client")
    )
    role = _extract_labeled_value(
        text,
        (
            "rol solicitado",
            "puesto solicitado",
            "rol",
            "puesto",
            "perfil",
            "cargo",
            "role",
            "position",
        ),
    )
    quantity_text = _extract_labeled_value(
        text,
        (
            "cantidad de posiciones",
            "numero de posiciones",
            "número de posiciones",
            "cantidad de vacantes",
            "cantidad solicitada",
            "numero de vacantes",
            "número de vacantes",
            "cantidad",
            "posiciones",
            "vacantes",
            "headcount",
        ),
    )
    quantity_match = re.search(r"\b\d+\b", quantity_text or "")
    quantity = int(quantity_match.group()) if quantity_match else None
    pm = _extract_labeled_value(
        text, ("pm responsable", "pm asignado", "project manager", "pm")
    )
    if not pm and client:
        pm = assignments.get(normalize_text(client))

    missing = [
        field
        for field, value in (
            ("cliente", client),
            ("rol", role),
            ("cantidad_posiciones", quantity),
            ("pm_responsable", pm),
        )
        if value is None
    ]
    return {
        "cliente": client,
        "rol": role,
        "cantidad_posiciones": quantity,
        "pm_responsable": pm,
        "datos_faltantes": missing,
    }


def confirmation_question(request: Mapping[str, Any]) -> str:
    quantity = request.get("cantidad_posiciones")
    role = request.get("rol")
    client = request.get("cliente")
    description_parts: list[str] = []
    if quantity is not None:
        noun = "posición" if quantity == 1 else "posiciones"
        description_parts.append(f"{quantity} {noun}")
    if role:
        description_parts.append(f"de {role}")
    if client:
        description_parts.append(f"para {client}")
    work_description = " ".join(description_parts) or str(
        request.get("asunto") or "esta solicitud de trabajo"
    )
    pm = request.get("pm_responsable") or "el PM asignado"
    return f"¿Confirmas que {pm} recibió la solicitud de {work_description}?"


def clarification_questions(request: Mapping[str, Any]) -> list[str]:
    field_questions = {
        "cliente": "¿De qué cliente es la solicitud?",
        "rol": "¿Qué rol o perfil se necesita?",
        "cantidad_posiciones": "¿Cuántas posiciones se solicitan?",
        "pm_responsable": "¿Qué PM debe recibir la solicitud?",
    }
    missing = request.get("datos_faltantes") or []
    return [
        field_questions[field]
        for field in missing
        if field in field_questions
    ]


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

    has_role = any(marker in normalized_body for marker in HIRING_ROLE_MARKERS)
    has_quantity = bool(
        re.search(r"\b\d{1,3}\b", normalized_body)
        or re.search(
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten|"
            r"uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)\b",
            normalized_body,
        )
    )
    has_explicit_intent = any(
        marker in normalized_body for marker in REQUEST_MARKERS
    ) or "requirement" in normalized_body
    has_positions_table = (
        "total positions" in normalized_body
        or "total de posiciones" in normalized_body
        or ("role" in normalized_body and "positions" in normalized_body)
    )
    is_position_request = (
        any(marker in normalized_body for marker in REQUEST_MARKERS)
        or (
            has_role
            and has_quantity
            and (has_explicit_intent or has_positions_table)
        )
        or (
            has_role
            and has_positions_table
            and has_quantity
        )
    )
    if not is_position_request:
        return "no_relacionado", (
            "El contenido no presenta suficiente evidencia de que se busque personal."
        )

    fingerprint = request_fingerprint(message)
    if fingerprint in known_fingerprints:
        return "duplicado", "El contenido coincide con una solicitud ya clasificada."
    if identifier and identifier in known_request_ids:
        return "duplicado", "El identificador ya pertenece a una solicitud registrada."
    if has_role and has_quantity and not any(
        marker in normalized_body for marker in REQUEST_MARKERS
    ):
        reason = "El correo enumera roles de trabajo y cantidades, aunque no use una frase formal de solicitud."
    else:
        reason = "El correo expresa una necesidad de incorporar o buscar personal."
    return "solicitud_posicion", reason


def _ordered_by_timestamp(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def timestamp_key(message: dict[str, Any]) -> str:
        timestamp = message.get("timestamp") or message.get("received_at") or ""
        return str(timestamp)

    return sorted(messages, key=timestamp_key)
