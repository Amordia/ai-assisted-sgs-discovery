from __future__ import annotations

import os


def get_jhtdb_token() -> str:
    token = os.environ.get("JHTDB_AUTH_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "JHTDB_AUTH_TOKEN is not set. Export it before running any JHTDB download script."
        )
    return token


def mask_token(token: str) -> str:
    if len(token) <= 8:
        return "*" * len(token)
    return f"{token[:8]}...{token[-4:]}"
