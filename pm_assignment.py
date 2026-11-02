from __future__ import annotations

from typing import Any, Iterable, Mapping

from email_utils import normalize_text

FINAL_STATES = {"Confirmada", "Cerrada"}


def count_open_projects(
    requests: Iterable[Mapping[str, Any]],
    pm_name: str,
    *,
    exclude_request_id: str | None = None,
) -> int:
    pm_key = normalize_text(pm_name)
    return sum(
        1
        for request in requests
        if request.get("id") != exclude_request_id
        and request.get("estado") not in FINAL_STATES
        and request.get("pm_responsable")
        and normalize_text(str(request["pm_responsable"])) == pm_key
    )


def select_least_loaded_pm(
    requests: Iterable[Mapping[str, Any]],
    pm_roster: Iterable[str],
    *,
    maximum_open_projects: int,
) -> tuple[str | None, dict[str, int]]:
    roster_by_key = {
        normalize_text(pm): pm.strip()
        for pm in pm_roster
        if pm.strip()
    }
    open_counts = {key: 0 for key in roster_by_key}
    for request in requests:
        pm = request.get("pm_responsable")
        if not pm or request.get("estado") in FINAL_STATES:
            continue
        pm_key = normalize_text(str(pm))
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
