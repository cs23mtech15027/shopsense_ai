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
from app.persistence.database import Database
from app.persistence.orders import OrderRepository
from app.retrieval.policies import PolicyStore
from app.workflow.router import SupportWorkflow

logger = logging.getLogger("shopsense")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    database = Database(settings.database_path)
    repository = OrderRepository(database)
    order_tool = OrderTool(repository)
    policy_tool = PolicyTool(
        store_factory=lambda: PolicyStore(
            qdrant_url=settings.qdrant_url,
            collection_name=settings.qdrant_collection,
            embedding_model=settings.embedding_model,
            policy_directory=Path(__file__).parent.parent / "data" / "policies",
        ),
        top_k=settings.top_k,
    )
    ollama = OllamaClient(settings.ollama_url, settings.chat_model)
    workflow = SupportWorkflow(
        care_agent=CareAgent(policy_tool, ollama),
        order_agent=OrderAgent(order_tool),
        refund_agent=RefundAgent(order_tool, RefundTool(repository)),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        database.initialize()
        Path(settings.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info("ShopSense API started")
        async with AsyncSqliteSaver.from_conn_string(settings.checkpoint_path) as checkpointer:
            await checkpointer.setup()
            workflow.compile(checkpointer)
            yield

    application = FastAPI(
        title="ShopSense",
        description="Local-first shopping support assistant prototype.",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.state.order_repository = repository
    application.state.workflow = workflow
    application.include_router(router)
    return application


app = create_app()
