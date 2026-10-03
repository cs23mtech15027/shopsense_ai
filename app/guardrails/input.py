import re

from fastapi import HTTPException


_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"reveal\s+(the\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"bypass\s+(all\s+)?(security|safety)\s+(rules|checks)", re.IGNORECASE),
)


def validate_message(message: str, max_chars: int) -> str:
    cleaned = _CONTROL_CHARACTERS.sub("", message).strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="Message must not be empty.")
    if len(cleaned) > max_chars:
        raise HTTPException(status_code=413, detail="Message exceeds the allowed size.")
    if any(pattern.search(cleaned) for pattern in _INJECTION_PATTERNS):
        raise HTTPException(
            status_code=400,
            detail="The request could not be processed. Please rephrase your question.",
        )
    return cleaned
