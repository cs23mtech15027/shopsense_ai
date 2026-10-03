import logging


logger = logging.getLogger("shopsense.agents.care")


def on_agent_complete(status: str, source_count: int) -> None:
    logger.info("care_agent completed status=%s source_count=%d", status, source_count)
