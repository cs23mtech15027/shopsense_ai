import logging


logger = logging.getLogger("shopsense.agents.refund")


def on_agent_complete(status: str) -> None:
    logger.info("refund_agent completed status=%s", status)
