from fastapi import HTTPException


def validate_answer(answer: str) -> str:
    cleaned = answer.strip()
    if not cleaned:
        raise HTTPException(
            status_code=502,
            detail="The assistant could not produce a verified response.",
        )
    return cleaned
