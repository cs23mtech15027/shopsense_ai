import logging


logger = logging.getLogger("shopsense.agents.order")


def on_agent_complete(status: str) -> None:
    logger.info("order_agent completed status=%s", status)
