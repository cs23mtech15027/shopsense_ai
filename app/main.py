import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agents.care_agent.agent import CareAgent
from app.agents.care_agent.tools.policy_tool import PolicyTool
from app.agents.order_agent.agent import OrderAgent
from app.agents.order_agent.tools.order_tool import OrderTool
from app.agents.refund_agent.agent import RefundAgent
from app.agents.refund_agent.tools.refund_tool import RefundTool
from app.api.routes import router
from app.config import Settings, get_settings
from app.llm.ollama import OllamaClient
from app.logging_config import configure_file_logging
from app.persistence.checkpoints import quarantine_corrupt_checkpoint
from app.persistence.database import Database
from app.persistence.memory import AgentMemory
from app.persistence.orders import OrderRepository
from app.persistence.users import UserRepository
from app.retrieval.policies import PolicyStore
from app.workflow.router import SupportWorkflow

logger = logging.getLogger("shopsense")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    database = Database(settings.database_path)
    repository = OrderRepository(database)
    user_repository = UserRepository(database)
    ollama = OllamaClient(settings.ollama_url, settings.chat_model)
    order_tool = OrderTool(repository)
    memory = AgentMemory(
        settings.memory_directory,
        dreaming_interval=settings.memory_dreaming_interval,
    )
    agents = {
        "care": CareAgent(
            PolicyTool(
                store_factory=lambda: PolicyStore(
                    qdrant_url=settings.qdrant_url,
                    collection_name=settings.qdrant_collection,
                    embedding_model=settings.embedding_model,
                    policy_directory=Path(__file__).parent.parent / "data" / "policies",
                ),
                top_k=settings.top_k,
            ),
            ollama,
            order_tool,
        ),
        "order": OrderAgent(order_tool, ollama),
        "refund": RefundAgent(
            order_tool,
            RefundTool(repository),
            ollama,
        ),
    }
    workflow: SupportWorkflow | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        configure_file_logging(settings.log_directory)
        database.initialize()
        Path(settings.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        quarantine_corrupt_checkpoint(settings.checkpoint_path)
        logger.info("ShopSense API started")
        async with AsyncSqliteSaver.from_conn_string(settings.checkpoint_path) as checkpointer:
            await checkpointer.setup()
            if workflow is None:
                raise RuntimeError("ShopSense workflow services are not initialized.")
            workflow.compile(checkpointer)
            yield

    application = FastAPI(
        title="ShopSense",
        description="Local-first shopping support assistant prototype.",
        version="0.1.0",
        lifespan=lifespan,
    )
    workflow = SupportWorkflow(
        care_agent=agents["care"],
        order_agent=agents["order"],
        refund_agent=agents["refund"],
        memory=memory,
        llm=ollama,
    )
    application.state.settings = settings
    application.state.order_repository = repository
    application.state.user_repository = user_repository
    application.state.ollama = ollama
    application.state.memory = memory
    application.state.workflow = workflow
    application.include_router(router)
    return application


app = create_app()
