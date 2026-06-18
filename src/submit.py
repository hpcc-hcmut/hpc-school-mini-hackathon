# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Reference workload or workflow infrastructure; teams should not modify this file.

"""Submit hackathon payloads to a backend endpoint."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any


def submit_to_backend(
    *,
    path: str | Path,
    backend_url: str,
    team_id: str,
    team_token: str,
    timeout: int = 60,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{backend_url.rstrip('/')}/api/submissions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-Team-ID": team_id,
            "X-Team-Token": team_token,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))
