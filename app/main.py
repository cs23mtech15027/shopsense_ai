from dataclasses import dataclass
import logging
from pathlib import Path

from langfuse import Langfuse

from app.agents.care_agent.agent import CareAgent
from app.agents.care_agent.tools.policy_tool import PolicyTool
from app.agents.order_agent.agent import OrderAgent
from app.agents.refund_agent.agent import RefundAgent
from app.config import (
    CHECKPOINT_PATH,
    EMBEDDING_MODEL,
    POLICY_DIRECTORY,
    Settings,
    get_settings,
)
from app.llm.openai_compatible import OpenAICompatibleClient
from app.observability import get_client
from app.persistence.checkpoints import checkpoint_context
from app.persistence.memory import MemoryStore
from app.persistence.supermemory import SupermemoryStore
from app.retrieval.policies import PolicyStore
from app.workflow.router import SupportWorkflow

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
logger = logging.getLogger("shopsense.runtime")


@dataclass
class RuntimeServices:
    workflow: SupportWorkflow
    memory: MemoryStore | None
    langfuse: Langfuse | None


def create_runtime(settings: Settings | None = None) -> RuntimeServices:
    settings = settings or get_settings()
    llm = OpenAICompatibleClient(
        _GROQ_BASE_URL,
        settings.chat_model,
        settings.groq_api_key.get_secret_value(),
    )
    memory = None
    if settings.supermemory_enabled and settings.supermemory_api_key is not None:
        try:
            memory = SupermemoryStore(
                api_key=settings.supermemory_api_key.get_secret_value(),
                search_limit=settings.supermemory_search_limit,
                timeout_seconds=5.0,
            )
        except Exception as error:
            logger.warning(
                "Supermemory initialization failed error_type=%s",
                type(error).__name__,
            )
    langfuse = get_client(settings)
    care_agent = CareAgent(
        PolicyTool(
            store_factory=lambda: PolicyStore(
                qdrant_url=settings.qdrant_url,
                collection_name=settings.qdrant_collection,
                policy_directory=Path(POLICY_DIRECTORY),
                api_key=settings.qdrant_api_key.get_secret_value(),
                embedding_model=EMBEDDING_MODEL,
            ),
            top_k=settings.top_k,
        ),
        llm,
    )
    workflow = SupportWorkflow(
        care_agent=care_agent,
        order_agent=OrderAgent(llm),
        refund_agent=RefundAgent(llm),
        memory=memory,
        llm=llm,
    )
    return RuntimeServices(workflow, memory, langfuse)


def checkpoint_file() -> str:
    return str(CHECKPOINT_PATH)


__all__ = ["RuntimeServices", "checkpoint_context", "create_runtime", "checkpoint_file"]
