import csv
import hashlib
import re
import unicodedata
from email.utils import parseaddr
from pathlib import Path
from typing import Any, Iterable, Mapping

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


def get_sender_address(message: Mapping[str, Any]) -> str:
    sender = message.get("from")
    if isinstance(sender, Mapping):
        address = sender.get("email") or sender.get("address")
        return str(address or "").strip().casefold()
    if isinstance(sender, list):
        for item in sender:
            if isinstance(item, Mapping):
                address = item.get("email") or item.get("address")
                if address:
                    return str(address).strip().casefold()
    _, address = parseaddr(str(sender or ""))
    return address.strip().casefold()


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


def load_director_assignments(path: Path = ASSIGNMENTS_PATH) -> dict[str, str]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        headers = {
            normalize_text(header.strip()): header
            for header in (reader.fieldnames or [])
            if header
        }
        director_header = headers.get("director")
        if director_header is None:
            return {}
        client_header = headers.get("cliente")
        if client_header is None:
            raise ValueError(f"{path} debe contener la columna cliente.")

        directors: dict[str, str] = {}
        for row_number, row in enumerate(reader, start=2):
            client = str(row.get(client_header) or "").strip()
            director = str(row.get(director_header) or "").strip()
            if not client and not director:
                continue
            if not client:
                raise ValueError(
                    f"Fila {row_number} de {path}: cliente es obligatorio."
                )
            if director:
                client_key = normalize_text(client)
                if client_key in directors and directors[client_key] != director:
                    raise ValueError(
                        f"Fila {row_number} de {path}: cliente {client} tiene "
                        "varios directores."
                    )
                directors[client_key] = director
    return directors


def load_pm_profiles(
    path: Path = PM_PROFILES_PATH,
    *,
    pm_addresses: Iterable[str] = (),
) -> dict[str, dict[str, Any]]:
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
        email_header = (
            headers.get("correo")
            or headers.get("email")
            or headers.get("correo electronico")
        )

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
            eligibility_header = headers.get("elegible")
            eligibility_text = (
                normalize_text(str(row.get(eligibility_header) or ""))
                if eligibility_header
                else ""
            )
            eligibility_values = {
                "true": True,
                "1": True,
                "yes": True,
                "si": True,
                "elegible": True,
                "false": False,
                "0": False,
                "no": False,
                "no elegible": False,
                "inelegible": False,
            }
            if eligibility_text and eligibility_text not in eligibility_values:
                raise ValueError(
                    f"Fila {row_number} de {path}: elegible debe ser si o no."
                )
            is_eligible = (
                eligibility_values.get(eligibility_text, True)
                if eligibility_text
                else True
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
            profile_data: dict[str, Any] = {
                "nombre": pm,
                "perfil": profile,
                "horas_confirmacion": hours,
                "elegible": is_eligible,
            }
            email = (
                str(row.get(email_header) or "").strip().casefold()
                if email_header
                else ""
            )
            if email:
                if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
                    raise ValueError(
                        f"Fila {row_number} de {path}: correo no es una dirección válida."
                    )
                profile_data["emails"] = [email]
            profiles[key] = profile_data
    associate_pm_email_addresses(profiles, pm_addresses)
    return profiles


def _email_matches_profile_name(
    email_address: str,
    profile_name: str,
) -> bool:
    local_part = email_address.split("@", 1)[0]
    email_tokens = set(re.findall(r"[a-z0-9]+", normalize_text(local_part)))
    email_tokens.discard("pm")
    name_tokens = set(re.findall(r"[a-z0-9]+", normalize_text(profile_name)))
    name_tokens.discard("pm")
    return bool(name_tokens) and email_tokens == name_tokens


def associate_pm_email_addresses(
    pm_profiles: dict[str, dict[str, Any]],
    addresses: Iterable[str],
) -> None:
    email_rows = [
        (key, profile)
        for key, profile in pm_profiles.items()
        if "@" in str(profile.get("nombre") or "")
    ]
    for alias_key, alias_profile in email_rows:
        address = str(alias_profile["nombre"]).strip().casefold()
        candidates = [
            (key, profile)
            for key, profile in pm_profiles.items()
            if key != alias_key
            and "@" not in str(profile.get("nombre") or "")
            and _email_matches_profile_name(
                address,
                str(profile.get("nombre") or key),
            )
        ]
        if len(candidates) != 1:
            continue
        _, profile = candidates[0]
        emails = profile.setdefault("emails", [])
        if address not in emails:
            emails.append(address)
        del pm_profiles[alias_key]

    for raw_address in addresses:
        _, address = parseaddr(str(raw_address or ""))
        address = address.strip().casefold()
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", address):
            continue
        candidates = [
            profile
            for profile in pm_profiles.values()
            if "@" not in str(profile.get("nombre") or "")
            and _email_matches_profile_name(
                address,
                str(profile.get("nombre") or ""),
            )
        ]
        if len(candidates) != 1:
            continue
        emails = candidates[0].setdefault("emails", [])
        if address not in emails:
            emails.append(address)


def detect_pm_unavailability(
    message: Mapping[str, Any],
    pm_profiles: Mapping[str, Mapping[str, Any]],
) -> dict[str, str] | None:
    sender_address = get_sender_address(message)
    if not sender_address:
        return None

    matched_profile = next(
        (
            profile
            for profile in pm_profiles.values()
            if sender_address
            in {
                str(address).strip().casefold()
                for address in profile.get("emails", [])
            }
            or normalize_text(str(profile.get("nombre") or ""))
            == normalize_text(sender_address)
        ),
        None,
    )
    if matched_profile is None:
        return None

    content = normalize_text(message_text(dict(message)))
    headers = message.get("headers")
    header_values = (
        " ".join(
            f"{key} {value}"
            for key, value in headers.items()
        )
        if isinstance(headers, Mapping)
        else ""
    )
    auto_reply_pattern = (
        r"\b(?:automatic reply|automated response|auto[- ]?reply|"
        r"autoreply|auto[- ]?submitted|auto[- ]?replied|"
        r"out of office|ooo|respuesta automatica|"
        r"respuesta automatizada|fuera de oficina)\b"
    )
    vacation_pattern = (
        r"\b(?:vacation|vacations|holiday|holidays|annual leave|"
        r"paid time off|pto|time off|on leave|vacaciones|"
        r"de vacaciones|permiso vacacional|licencia)\b"
    )
    illness_pattern = (
        r"\b(?:sick|illness|ill|medical leave|medical absence|"
        r"hospitalized|hospitalised|recovery|enfermo|enferma|"
        r"enfermedad|baja medica|licencia medica|reposo medico|"
        r"incapacidad|hospitalizado|hospitalizada)\b"
    )
    automatic_absence = bool(
        re.search(auto_reply_pattern, normalize_text(f"{content} {header_values}"))
        and re.search(f"{vacation_pattern}|{illness_pattern}", content)
    )
    unavailable_pattern = (
        r"\b(?:no\s+estoy\s+trabajando|"
        r"no\s+estoy\s+disponible|"
        r"no\s+(?:estare|estaria|voy\s+a\s+estar|podre|podria)\s+"
        r"(?:present[ea]|disponible|en\s+la\s+oficina|atender)|"
        r"(?:estare|estaria|voy\s+a\s+estar)\s+(?:fuera|ausente)|"
        r"no\s+me\s+(?:encontrare|encontraria)\s+disponible|"
        r"i\s+(?:(?:will|would)\s+not|won't|wouldn't)\s+"
        r"(?:be\s+)?(?:available|present|in\s+the\s+office|"
        r"able\s+to\s+work)|"
        r"i\s+(?:(?:will|would)\s+be|am)\s+"
        r"(?:unavailable|out|away|absent))\b"
    )
    return_until_pattern = (
        r"\b(?:hasta|until|through|through\s+the|back\s+on|"
        r"vuelvo|volvere|estare\s+de\s+vuelta|"
        r"regreso\s+el|regresare\s+el|return(?:ing)?\s+on)\s+"
        r"(?:(?:el|la|the)\s+)?"
        r"(?:\d{1,2}(?:st|nd|rd|th|º|°)?"
        r"(?:[/-]\d{1,2})?(?:[/-]\d{2,4})?"
        r"|(?:proxim[oa]|siguiente|next|this)\s+"
        r"(?:semana|week|mes|month|lunes|martes|miercoles|jueves|"
        r"viernes|sabado|domingo|monday|tuesday|wednesday|thursday|"
        r"friday|saturday|sunday)"
        r"|(?:lunes|martes|miercoles|jueves|viernes|sabado|domingo|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
        r"|(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
        r"septiembre|octubre|noviembre|diciembre|january|february|"
        r"march|april|may|june|july|august|september|october|"
        r"november|december)\b)"
    )
    explicit_return_date = bool(
        re.search(unavailable_pattern, content)
        and re.search(return_until_pattern, content)
    )
    if not automatic_absence and not explicit_return_date:
        return None

    if re.search(vacation_pattern, content):
        reason = "respuesta de ausencia por vacaciones o permiso"
    elif re.search(illness_pattern, content):
        reason = "respuesta de ausencia por enfermedad"
    else:
        reason = "respuesta del PM indica que no estará disponible hasta una fecha"

    return {
        "pm_responsable": str(matched_profile.get("nombre") or sender_address),
        "correo_pm": sender_address,
        "motivo": reason,
    }


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
    if not client:
        normalized_body = normalize_text(text)
        known_clients = sorted(
            (str(name) for name in assignments),
            key=len,
            reverse=True,
        )
        client = next(
            (
                name
                for name in known_clients
                if re.search(
                    rf"(?<!\w){re.escape(normalize_text(name))}(?!\w)",
                    normalized_body,
                )
            ),
            None,
        )
        if client:
            client = client.title()
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

    positions = _extract_position_groups(text)
    if positions:
        if role is None:
            role = " / ".join(
                dict.fromkeys(position["rol"] for position in positions)
            )
        if quantity is None:
            quantity = sum(
                position["cantidad_posiciones"] for position in positions
            )

    if role is None or quantity is None:
        inferred_quantity, inferred_role = _extract_prose_position(text)
        role = role or inferred_role
        quantity = quantity if quantity is not None else inferred_quantity

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
        "posiciones": positions,
        "datos_faltantes": missing,
    }


_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "uno": 1,
    "una": 1,
    "un": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
}


def _quantity_value(value: str) -> int | None:
    token = normalize_text(value).strip()
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)


def _extract_position_groups(text: str) -> list[dict[str, Any]]:
    lines = [line.strip().strip("|") for line in text.splitlines()]
    for header_index, line in enumerate(lines):
        normalized_header = normalize_text(line)
        if (
            re.search(r"\b(?:role|rol|puesto|position)\b", normalized_header)
            and re.search(
                r"\b(?:total positions|total de posiciones|positions|posiciones)\b",
                normalized_header,
            )
        ):
            whitespace_positions = _extract_whitespace_table_rows(
                lines[header_index + 1 :]
            )
            if whitespace_positions:
                return whitespace_positions

        cells = [cell.strip() for cell in re.split(r"\s*\|\s*|\t+", line)]
        normalized_cells = [normalize_text(cell) for cell in cells]
        role_index = next(
            (
                index
                for index, cell in enumerate(normalized_cells)
                if cell in {"role", "rol", "puesto", "position"}
            ),
            None,
        )
        quantity_index = next(
            (
                index
                for index, cell in enumerate(normalized_cells)
                if cell
                in {
                    "total positions",
                    "total de posiciones",
                    "positions",
                    "posiciones",
                    "vacantes",
                    "cantidad",
                }
            ),
            None,
        )
        if (
            role_index is None
            or quantity_index is None
            or role_index == quantity_index
            or len(cells) < 2
        ):
            continue

        location_index = next(
            (
                index
                for index, cell in enumerate(normalized_cells)
                if cell in {"location", "ubicacion", "localidad"}
            ),
            None,
        )
        skills_index = next(
            (
                index
                for index, cell in enumerate(normalized_cells)
                if "skill" in cell or "habilidad" in cell
            ),
            None,
        )
        positions: list[dict[str, Any]] = []
        for data_line in lines[header_index + 1 :]:
            row = [cell.strip() for cell in re.split(r"\s*\|\s*|\t+", data_line)]
            if len(row) <= max(role_index, quantity_index):
                if positions:
                    break
                continue
            role_value = row[role_index].strip(" -")
            quantity_value = _quantity_value(row[quantity_index])
            if not role_value or quantity_value is None or quantity_value < 1:
                if positions:
                    break
                continue
            position: dict[str, Any] = {
                "rol": role_value,
                "cantidad_posiciones": quantity_value,
            }
            if location_index is not None and location_index < len(row):
                position["ubicacion"] = row[location_index]
            if skills_index is not None and skills_index < len(row):
                position["skills"] = row[skills_index]
            positions.append(position)
        if positions:
            return positions
    return []


def _extract_whitespace_table_rows(
    lines: list[str],
) -> list[dict[str, Any]]:
    location_pattern = (
        r"US(?:[-\s])?(?:Onshore|Offshore)|"
        r"Onshore|Offshore|LATAM|US"
    )
    role_end_pattern = (
        r"(?:mainframe\s+dev(?:eloper)?s?|full[\s-]?stack\s+dev(?:eloper)?s?|"
        r"data\s+engineers?|software\s+engineers?|quality\s+engineers?|"
        r"engineers?|developers?|analysts?|testers?|designers?|"
        r"architects?|consultants?|scientists?|programmers?|programadores?|"
        r"desarrolladores?|ingenieros?|analistas?|qe)"
    )
    positions: list[dict[str, Any]] = []
    for line in lines:
        row_match = re.match(
            rf"^(?P<before>.+?)\s+(?P<location>{location_pattern})\s+"
            r"(?P<count>\d{1,3})\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if row_match is None:
            if positions and not line.strip("- "):
                break
            continue
        role_skill_text = row_match.group("before").strip()
        role_match = re.match(
            rf"(?P<role>.*?\b{role_end_pattern})\b(?:\s+(?P<skills>.+))?$",
            role_skill_text,
            flags=re.IGNORECASE,
        )
        if role_match is None:
            continue
        role = role_match.group("role").strip()
        position: dict[str, Any] = {
            "rol": role,
            "cantidad_posiciones": int(row_match.group("count")),
            "ubicacion": row_match.group("location"),
        }
        skills = (role_match.group("skills") or "").strip(" ,;")
        if skills:
            position["skills"] = skills
        positions.append(position)
    return positions


def _extract_prose_position(text: str) -> tuple[int | None, str | None]:
    quantity_pattern = (
        r"(?P<count>\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"uno|una|un|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)"
    )
    intent_pattern = (
        r"(?:need(?:s|ed)?|looking for|seek(?:ing)?|recruit(?:ing)?|"
        r"hiring|hire|requirement for|require(?:s|d)?|"
        r"buscamos|busca|buscan|necesitamos|necesita|necesitan|"
        r"solicitamos|solicita|requerimos|requiere|requieren|"
        r"se requieren|contratar|contratando)"
    )
    match = re.search(
        rf"\b{intent_pattern}\s+(?:the\s+)?{quantity_pattern}\s+"
        r"(?P<role>[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9][A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9 +/&().-]*?)"
        r"(?=\s+(?:for|with|in|to|should|must|ideally)\b|[,;.\n]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        match = re.search(
            rf"\b{quantity_pattern}\s+"
            r"(?P<role>(?:senior\s+|junior\s+|full[\s-]?stack\s+|"
            r"front[\s-]?end\s+|back[\s-]?end\s+|data\s+|software\s+|"
            r"quality\s+|mainframe\s+)?"
            r"(?:engineers?|developers?|devs?|analysts?|testers?|"
            r"designers?|architects?|consultants?|scientists?|"
            r"ingenieros?|desarrolladores?|analistas?|"
            r"probadores?|programadores?))\b",
            text,
            flags=re.IGNORECASE,
        )
    if match is None:
        return None, None
    count = _quantity_value(match.group("count"))
    role = re.sub(r"\s+", " ", match.group("role")).strip(" .,-")
    return count, role or None


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
    return (
        f"¿Confirmas que {pm} se hará cargo de la solicitud de "
        f"{work_description}?"
    )


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


def infer_request_status(text: str) -> str | None:
    normalized = normalize_text(text)
    if any(
        phrase in normalized
        for phrase in (
            "we are working on it",
            "we're working on it",
            "working on this",
            "currently working",
            "estamos trabajando",
            "estoy trabajando",
            "ya estamos trabajando",
            "en proceso",
            "in progress",
        )
    ):
        return "escalada"
    if any(
        phrase in normalized
        for phrase in (
            "cancelled",
            "canceled",
            "request is closed",
            "closing this request",
            "position is no longer needed",
            "solicitud cerrada",
            "solicitud cancelada",
            "cerramos la solicitud",
            "cancelamos la solicitud",
        )
    ):
        return "cerrada"
    if any(
        phrase in normalized
        for phrase in (
            "request is complete",
            "request has been completed",
            "positions are filled",
            "position is filled",
            "all positions are filled",
            "the role is filled",
            "solicitud completada",
            "solicitud atendida",
            "posiciones cubiertas",
            "puesto cubierto",
            "ya fue resuelta",
            "ya se cubrio",
            "ya se cubrió",
        )
    ):
        return "completada"
    if any(
        phrase in normalized
        for phrase in (
            "i will take ownership",
            "i'll take ownership",
            "i will handle this",
            "i'll handle this",
            "i will handle it",
            "i'll handle it",
            "i will take care of this",
            "i'll take care of this",
            "i will take this",
            "i'll take this",
            "i can take this",
            "i can handle this",
            "i am taking ownership",
            "i'm taking ownership",
            "happy to take this",
            "me hare cargo",
            "me haré cargo",
            "yo me hago cargo",
            "me hago cargo",
            "yo me encargo",
            "me encargare",
            "me encargaré",
            "me encargo",
            "me responsabilizo",
            "me hago responsable",
            "cuenta conmigo",
            "lo tomo",
            "la tomo",
            "lo tomo yo",
            "la tomo yo",
            "confirmo que me hago cargo",
            "confirmo que me hare cargo",
            "confirmo que me haré cargo",
        )
    ):
        return "confirmada"
    return None


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
        if infer_request_status(body):
            return (
                "actualizacion_solicitud",
                "El PM informó un cambio de estado para una solicitud relacionada.",
            )
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
    ) or bool(
        re.search(
            r"\b(?:need|needs|needed|seeking|seeks|recruit|recruits|"
            r"recruiting|hiring|hire|hires|buscar|busca|buscan|buscando|"
            r"necesita|necesitan|requerir|requiere|requieren|contratar|"
            r"contratando)\b",
            normalized_body,
        )
    )
    has_positions_table = (
        "total positions" in normalized_body
        or "total de posiciones" in normalized_body
        or ("role" in normalized_body and "positions" in normalized_body)
    )
    has_explicit_staffing_request = any(
        phrase in normalized_body
        for phrase in (
            "solicitud de personal",
            "solicitud de puesto",
            "solicitud de posicion",
            "solicitud de recurso",
            "position request",
            "resource request",
            "staffing request",
            "headcount request",
            "request for personnel",
        )
    )
    is_position_request = has_explicit_staffing_request or (
        has_role
        and (has_explicit_intent or (has_quantity and has_positions_table))
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
    if has_role and has_quantity and not has_explicit_intent:
        reason = "El correo enumera roles de trabajo y cantidades, aunque no use una frase formal de solicitud."
    else:
        reason = "El correo expresa una necesidad de incorporar o buscar personal."
    return "solicitud_posicion", reason


def _ordered_by_timestamp(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def timestamp_key(message: dict[str, Any]) -> str:
        timestamp = message.get("timestamp") or message.get("received_at") or ""
        return str(timestamp)

    return sorted(messages, key=timestamp_key)
