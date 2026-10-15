import logging


logger = logging.getLogger("shopsense.agents.intent")


def on_agent_complete(intent: str, has_order_id: bool) -> None:
    logger.info(
        "intent_agent completed intent=%s order_context_selected=%s",
        intent,
        has_order_id,
    )
