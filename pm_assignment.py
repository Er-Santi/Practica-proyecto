from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from email_utils import normalize_text

FINAL_STATES = {"Cerrada"}


def canonical_pm_name(
    pm_name: str,
    pm_profiles: Mapping[str, Mapping[str, Any]] | None,
) -> str:
    key = normalize_text(pm_name)
    matches = [
        str(profile.get("nombre") or name).strip()
        for name, profile in (pm_profiles or {}).items()
        if key
        in {
            normalize_text(str(name)),
            normalize_text(str(profile.get("nombre") or "")),
            *(
                normalize_text(str(address))
                for address in profile.get("emails", [])
            ),
        }
    ]
    if len(matches) == 1:
        return matches[0]

    requested_alias = key.removeprefix("pm ")
    abbreviated_matches = [
        str(profile.get("nombre") or name).strip()
        for name, profile in (pm_profiles or {}).items()
        if requested_alias
        == normalize_text(str(profile.get("nombre") or name)).removeprefix("pm ")
    ]
    return abbreviated_matches[0] if len(abbreviated_matches) == 1 else pm_name.strip()


def is_pm_eligible(
    pm_name: str,
    pm_profiles: Mapping[str, Mapping[str, Any]] | None,
) -> bool:
    key = normalize_text(pm_name)
    canonical_name = normalize_text(canonical_pm_name(pm_name, pm_profiles))
    profile = next(
        (
            candidate
            for name, candidate in (pm_profiles or {}).items()
            if normalize_text(str(name)) in {key, canonical_name}
            or normalize_text(str(candidate.get("nombre") or ""))
            in {key, canonical_name}
            or key
            in {
                normalize_text(str(address))
                for address in candidate.get("emails", [])
            }
        ),
        None,
    )
    return profile is None or profile.get("elegible", True) is not False


def build_pm_roster(
    assignments: Mapping[str, str] | None,
    pm_profiles: Mapping[str, Mapping[str, Any]] | None,
    *,
    eligible_only: bool = True,
) -> tuple[str, ...]:
    names: dict[str, str] = {}
    for pm_name in (assignments or {}).values():
        if pm_name.strip():
            canonical_name = canonical_pm_name(pm_name, pm_profiles)
            names.setdefault(normalize_text(canonical_name), canonical_name)
    for key, profile in (pm_profiles or {}).items():
        name = str(profile.get("nombre") or key).strip()
        if name:
            names.setdefault(normalize_text(name), name)
    return tuple(
        names[key]
        for key in sorted(names)
        if not eligible_only or is_pm_eligible(names[key], pm_profiles)
    )


def save_pm_eligibility(
    path: Path,
    *,
    pm_name: str,
    eligible: bool,
    pm_aliases: Iterable[str] = (),
) -> None:
    if not path.exists():
        raise ValueError(f"No existe el archivo de perfiles de PM: {path}.")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            fieldnames = list(reader.fieldnames or [])
            headers = {
                normalize_text(field.strip()): field
                for field in fieldnames
                if field
            }
            pm_header = headers.get("pm")
            if pm_header is None:
                raise ValueError(f"{path} debe contener la columna pm.")
            profile_header = headers.get("perfil")
            if profile_header is None:
                raise ValueError(f"{path} debe contener la columna perfil.")
            eligibility_header = headers.get("elegible")
            if eligibility_header is None:
                eligibility_header = "elegible"
                fieldnames.append(eligibility_header)
            rows = list(reader)
    except OSError as exc:
        raise ValueError(f"No se pudo leer el archivo de PM {path}: {exc}") from exc

    matching_names = {
        normalize_text(pm_name),
        *(normalize_text(alias) for alias in pm_aliases),
    }
    matching_rows = [
        row
        for row in rows
        if normalize_text(str(row.get(pm_header) or "")) in matching_names
    ]
    match = next(
        (
            row
            for row in matching_rows
            if normalize_text(str(row.get(pm_header) or ""))
            == normalize_text(pm_name)
        ),
        matching_rows[0] if matching_rows else None,
    )
    if match is None:
        match = {field: "" for field in fieldnames}
        match[pm_header] = pm_name
        match[profile_header] = "normal"
        rows.append(match)
    for row in matching_rows or [match]:
        row[eligibility_header] = "si" if eligible else "no"

    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    except OSError as exc:
        raise ValueError(f"No se pudo guardar el archivo de PM {path}: {exc}") from exc


def count_open_projects(
    requests: Iterable[Mapping[str, Any]],
    pm_name: str,
    *,
    exclude_request_id: str | None = None,
    pm_profiles: Mapping[str, Mapping[str, Any]] | None = None,
) -> int:
    pm_key = normalize_text(canonical_pm_name(pm_name, pm_profiles))
    return sum(
        1
        for request in requests
        if request.get("id") != exclude_request_id
        and request.get("estado") not in FINAL_STATES
        and request.get("pm_responsable")
        and normalize_text(
            canonical_pm_name(str(request["pm_responsable"]), pm_profiles)
        )
        == pm_key
    )


def select_least_loaded_pm(
    requests: Iterable[Mapping[str, Any]],
    pm_roster: Iterable[str],
    *,
    maximum_open_projects: int,
    pm_profiles: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[str | None, dict[str, int]]:
    roster_by_key = {
        normalize_text(canonical_pm_name(pm, pm_profiles)): canonical_pm_name(
            pm,
            pm_profiles,
        )
        for pm in pm_roster
        if pm.strip()
    }
    open_counts = {key: 0 for key in roster_by_key}
    for request in requests:
        pm = request.get("pm_responsable")
        if not pm or request.get("estado") in FINAL_STATES:
            continue
        pm_key = normalize_text(canonical_pm_name(str(pm), pm_profiles))
        if pm_key in open_counts:
            open_counts[pm_key] += 1

    public_loads = {
        roster_by_key[key]: count
        for key, count in sorted(
            open_counts.items(),
            key=lambda item: (roster_by_key[item[0]].casefold(), item[0]),
        )
    }
    eligible = [
        (key, count)
        for key, count in open_counts.items()
        if count < maximum_open_projects
        and is_pm_eligible(roster_by_key[key], pm_profiles)
    ]
    if not eligible:
        return None, public_loads

    selected_key, _ = min(
        eligible,
        key=lambda item: (
            item[1],
            roster_by_key[item[0]].casefold(),
            item[0],
        ),
    )
    return roster_by_key[selected_key], public_loads
