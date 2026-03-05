from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
import wrds


def load_wrds_credentials(env_candidates: list[Path] | None = None) -> tuple[str, str]:
    """Load WRDS_USERNAME and WRDS_PASSWORD from the first .env file found."""
    if env_candidates is None:
        env_candidates = [
            Path(__file__).resolve().parents[1] / ".env",  # team_t/.env
            Path.cwd() / ".env",
        ]

    env_path = next((p for p in env_candidates if p.exists()), None)
    if env_path:
        load_dotenv(env_path)

    username = os.getenv("WRDS_USERNAME")
    password = os.getenv("WRDS_PASSWORD")

    if not username:
        raise ValueError(
            "WRDS_USERNAME missing. Add it to your .env file:\n"
            "  WRDS_USERNAME=your_wrds_username\n"
            "  WRDS_PASSWORD=your_wrds_password"
        )
    return username, password


def connect_wrds(env_candidates: list[Path] | None = None) -> wrds.Connection:
    """Return an open WRDS connection using credentials from .env."""
    username, password = load_wrds_credentials(env_candidates)
    return wrds.Connection(wrds_username=username, wrds_password=password)