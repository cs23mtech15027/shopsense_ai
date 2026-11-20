import re
from typing import Any

from app.guardrails.input import redact_pii

_RESTRICTED_RETURN_PROMISE = re.compile(
    r"(?:(?:innerwear|underwear|lingerie|cosmetics|makeup).{0,120}"
    r"(?:returnable|eligible|can be returned|may be returned|accept(?:ed)? for return)"
    r"|(?:returnable|eligible|can be returned|may be returned|accept(?:ed)? for return)"
    r".{0,120}(?:innerwear|underwear|lingerie|cosmetics|makeup))",
    re.IGNORECASE,
)
_REFUND_ACTION_CLAIM = re.compile(
    r"\b(?:instant(?:ly)?|immediate(?:ly)?|automatic(?:ally)?|already)\b.{0,80}"
    r"\b(?:refund(?:ed)?|credited|processed|initiated)\b"
    r"|\b(?:refund(?:ed)?|credited|processed|initiated)\b.{0,80}"
    r"\b(?:instantly|immediately|automatically|already)\b",
    re.IGNORECASE,
)


def validate_answer(answer: str, facts: Any = None) -> str:
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("The assistant could not produce a verified response.")

    cleaned = redact_pii(answer.strip())
    verified = facts if isinstance(facts, dict) else {}
    item_name = str(verified.get("item_name", ""))
    restricted_order = any(
        category in item_name.casefold()
        for category in ("innerwear", "underwear", "lingerie", "cosmetic", "makeup")
    )
    if restricted_order or _RESTRICTED_RETURN_PROMISE.search(cleaned):
        return (
            "Hygiene-restricted innerwear and cosmetics are non-returnable. "
            "I can help check another order or explain the policy."
        )

    amount_minor = verified.get("amount_minor", verified.get("approval_amount_minor"))
    currency = verified.get("currency", verified.get("approval_currency"))
    amount_exceeds_limit = (
        isinstance(amount_minor, int)
        and amount_minor > 500_000
        and currency == "INR"
    )
    requires_tier2 = bool(verified.get("requires_supervisor_approval")) or (
        amount_exceeds_limit
    )
    if requires_tier2 and _REFUND_ACTION_CLAIM.search(cleaned):
        if isinstance(amount_minor, int) and currency == "INR":
            amount = f"₹{amount_minor / 100:,.2f}"
            return (
                f"This refund request is for {amount} and requires Tier-2 "
                "supervisor approval. It has not been initiated."
            )
        return (
            "This refund request requires Tier-2 supervisor approval. "
            "It has not been initiated."
        )
    return cleaned
