import asyncio
import csv
import logging
import os
import uuid
from pathlib import Path

import jwt
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

from app.config import (
    CHECKPOINT_PATH,
    EMBEDDING_MODEL,
    JWT_AUDIENCE,
    JWT_ISSUER,
    POLICY_DIRECTORY,
    get_settings,
)
from app.evaluation_metrics import compute_rag_metrics
from app.guardrails.input import (
    InputValidationError,
    assess_message,
    redact_pii,
    validate_message,
)
from app.guardrails.output import validate_answer
from app.main import create_runtime
from app.observability import (
    flush_observability,
    get_callback_handler,
    redact_sensitive_data,
    trace_turn,
)
from app.persistence.checkpoints import checkpoint_context
from app.retrieval.policies import PolicyStore

_DATASET = PROJECT_ROOT / "data" / "golden_dataset" / "golden_dataset.csv"
_RESULTS = PROJECT_ROOT / "data" / "golden_dataset" / "golden_test_results.csv"
logger = logging.getLogger("shopsense.evaluation")


def _write_results(
    results: list[dict[str, str]],
    output_path: Path = _RESULTS,
) -> Path:
    if not results:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_results_file(results, output_path)
    except PermissionError:
        fallback_path = output_path.with_name(
            f"{output_path.stem}_{uuid.uuid4().hex[:8]}{output_path.suffix}"
        )
        _write_results_file(results, fallback_path)
        logger.warning(
            "Could not write evaluation results to %s; saved to %s instead. "
            "Close the file if you want to overwrite the default results file.",
            output_path,
            fallback_path,
        )
        return fallback_path
    return output_path


def _write_results_file(results: list[dict[str, str]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def _verified_user_id(token: str, secret: str) -> str:
    claims = jwt.decode(
        token,
        secret,
        algorithms=["HS256"],
        issuer=JWT_ISSUER,
        audience=JWT_AUDIENCE,
        options={"require": ["sub", "exp", "iat", "jti", "iss", "aud"]},
    )
    user_id = claims.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise ValueError("USER_JWT must contain a shopper subject.")
    return user_id


async def _run() -> None:
    settings = get_settings()
    user_jwt = os.getenv("USER_JWT", "").strip()
    if not user_jwt:
        raise RuntimeError(
            "Set USER_JWT in .env to a token from scripts/get_user_jwt.py."
        )
    user_id = _verified_user_id(
        user_jwt,
        settings.jwt_secret.get_secret_value(),
    )
    runtime = create_runtime(settings)
    policy_store = PolicyStore(
        qdrant_url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value(),
        collection_name=settings.qdrant_collection,
        policy_directory=Path(POLICY_DIRECTORY),
        embedding_model=EMBEDDING_MODEL,
    )
    with _DATASET.open(encoding="utf-8", newline="") as dataset_file:
        rows = list(csv.DictReader(dataset_file))
    results: list[dict[str, str]] = []
    results_path = _RESULTS

    for row in rows:
        trace_id = uuid.uuid4().hex
        thread_id = uuid.uuid4().hex[:12]
        query = row["query_text"]
        expected_guardrail = row["guardrail_expected"]
        predicted_intent = ""
        status = "blocked"
        answer = ""
        error_type = ""
        precision = ""
        recall = ""
        retrieved_sources: list[str] = []
        guardrail_pass = False

        try:
            clean_query = validate_message(query, settings.max_message_chars)
        except InputValidationError:
            guardrail_pass = expected_guardrail == "blocked"
        else:
            safe_query = redact_pii(clean_query)
            assessment_result: bool | None = None
            if expected_guardrail != "blocked":
                try:
                    assessment_result = await assess_message(
                        safe_query,
                        runtime.workflow.llm,
                    )
                except Exception as error:
                    error_type = type(error).__name__
                    status = "error"
                    logger.warning(
                        "Evaluation input guardrail failed query_id=%s "
                        "error_type=%s trace_id=%s",
                        row["query_id"],
                        error_type,
                        trace_id,
                    )
            if expected_guardrail == "blocked":
                guardrail_pass = False
            elif assessment_result is None:
                pass
            elif not assessment_result:
                guardrail_pass = expected_guardrail == "blocked"
            else:
                guardrail_pass = expected_guardrail != "blocked"
                passages = await asyncio.to_thread(
                    policy_store.search,
                    safe_query,
                    3,
                )
                retrieved_sources = [
                    item["source"].split(":", 1)[0] for item in passages
                ]
                expected_sources = row["expected_source_files"]
                if expected_sources:
                    precision_value, recall_value, _, _ = compute_rag_metrics(
                        passages,
                        expected_sources,
                        "",
                        k=3,
                    )
                    precision = f"{precision_value:.4f}"
                    recall = f"{recall_value:.4f}"

                workflow_input = {
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "conversation_id": str(uuid.uuid4()),
                    "session_id": str(uuid.uuid4()),
                    "turn_id": str(uuid.uuid4()),
                    "message": safe_query,
                    "query_type": "auto",
                }
                graph_config = {
                    "configurable": {"thread_id": thread_id},
                    "metadata": {
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "trace_id": trace_id,
                    },
                }
                callback = get_callback_handler(runtime.langfuse)
                if callback is not None:
                    graph_config["callbacks"] = [callback]
                with trace_turn(
                    runtime.langfuse,
                    user_id=user_id,
                    thread_id=thread_id,
                    trace_id=trace_id,
                    message=safe_query,
                ) as trace:
                    try:
                        async with checkpoint_context(str(CHECKPOINT_PATH)) as checkpointer:
                            graph = runtime.workflow.builder.compile(
                                checkpointer=checkpointer
                            )
                            result = await graph.ainvoke(
                                workflow_input,
                                graph_config,
                                context={
                                    "user_jwt": user_jwt,
                                    "trace_id": trace_id,
                                },
                            )
                        predicted_intent = result.get("intent", "")
                        status = result.get("status", "unknown")
                        answer = validate_answer(
                            result.get("answer", ""),
                            result.get("guardrail_facts"),
                        )
                        if trace is not None:
                            trace.update(
                                output=redact_sensitive_data(
                                    {
                                        "intent": predicted_intent,
                                        "status": status,
                                        "answer": answer,
                                    }
                                )
                            )
                    except Exception as error:
                        error_type = type(error).__name__
                        status = "error"
                        answer = "This evaluation case failed during workflow execution."
                        logger.warning(
                            "Evaluation workflow failed query_id=%s error_type=%s "
                            "trace_id=%s",
                            row["query_id"],
                            error_type,
                            trace_id,
                        )
                        if trace is not None:
                            trace.update(
                                level="ERROR",
                                status_message=error_type,
                                output={
                                    "status": "error",
                                    "error_type": error_type,
                                },
                            )

        expected_intent = row["expected_intent"]
        results.append(
            {
                "query_id": row["query_id"],
                "query_text": query,
                "expected_intent": expected_intent,
                "predicted_intent": predicted_intent,
                "intent_match": str(
                    predicted_intent == expected_intent
                    if expected_intent != "blocked"
                    else guardrail_pass
                ),
                "expected_source_files": row["expected_source_files"],
                "retrieved_source_files": ", ".join(retrieved_sources),
                "precision_at_3": precision,
                "recall_at_3": recall,
                "guardrail_expected": expected_guardrail,
                "guardrail_pass": str(guardrail_pass),
                "status": status,
                "trace_id": trace_id,
                "answer": redact_pii(answer),
                "error_type": error_type,
            }
        )
        results_path = _write_results(results, results_path)
        logger.info(
            "Evaluation case complete query_id=%s intent=%s status=%s "
            "guardrail_pass=%s trace_id=%s",
            row["query_id"],
            predicted_intent or "blocked",
            status,
            guardrail_pass,
            trace_id,
        )

    intent_cases = [item for item in results if item["expected_intent"] != "blocked"]
    intent_accuracy = (
        sum(item["intent_match"] == "True" for item in intent_cases)
        / len(intent_cases)
        if intent_cases
        else 0
    )
    guardrail_accuracy = (
        sum(item["guardrail_pass"] == "True" for item in results)
        / len(results)
        if results
        else 0
    )
    print(
        f"Evaluated {len(results)} cases; intent accuracy={intent_accuracy:.3f}; "
        f"guardrail pass rate={guardrail_accuracy:.3f}; "
        f"workflow errors={sum(bool(item['error_type']) for item in results)}."
    )
    print(f"Results: {results_path}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(_run())
    finally:
        flush_observability()


if __name__ == "__main__":
    main()
