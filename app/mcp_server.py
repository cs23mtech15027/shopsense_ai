import json
import logging
import re
from contextvars import Token
from urllib.parse import urlsplit

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import ASGIApp, Receive, Scope, Send

from app.auth_context import authenticated_user_id, authenticated_user_role
from app.config import (
    DATABASE_PATH,
    JWT_AUDIENCE,
    JWT_ISSUER,
    LOG_DIRECTORY,
    MAX_AUTO_REFUND_MINOR,
    MCP_SERVER_URL,
    get_settings,
)
from app.logging_config import bind_thread_id, bind_trace_id, configure_file_logging
from app.persistence.database import Database
from app.persistence.orders import OrderRepository
from app.persistence.users import UserRepository

_ORDER_ID = re.compile(r"^ORD-\d{4,}$")
_THREAD_ID = re.compile(r"^[A-Za-z0-9_-]{12}$")
_TRACE_ID = re.compile(r"^[a-f0-9]{32}$")
_DAMAGE_CLAIM = re.compile(r"\b(damag(?:e|ed)|broken|cracked|defect(?:ive)?)\b", re.I)
_HYGIENE_RESTRICTED = ("innerwear", "underwear", "lingerie", "cosmetic", "makeup")
logger = logging.getLogger("shopsense.mcp")


class LocalMCPAuthMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        user_repository: UserRepository,
        jwt_secret: str,
    ):
        self.app = app
        self.user_repository = user_repository
        self.jwt_secret = jwt_secret

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        thread_header = headers.get(b"x-shopsense-thread-id", b"").decode("latin-1")
        trace_header = headers.get(b"x-shopsense-trace-id", b"").decode("latin-1")
        thread_id = thread_header if _THREAD_ID.fullmatch(thread_header) else "-"
        trace_id = trace_header if _TRACE_ID.fullmatch(trace_header) else "-"
        origin = headers.get(b"origin", b"").decode("latin-1")
        host = headers.get(b"host", b"").decode("latin-1")
        with bind_thread_id(thread_id), bind_trace_id(trace_id):
            if not _is_loopback_origin(origin) or not _is_loopback_host(host):
                logger.warning("MCP request rejected status=403 reason=origin_or_host")
                await _json_error(send, 403, "MCP accepts only loopback browser origins and hosts.")
                return
            authorization = headers.get(b"authorization", b"").decode("latin-1")
            if not authorization.startswith("Bearer "):
                logger.warning("MCP request rejected status=401 reason=missing_token")
                await _json_error(send, 401, "A valid user bearer token is required.")
                return
            jwt_token = authorization.removeprefix("Bearer ").strip()
            settings_claims = self.user_repository.get_token_claims(
                jwt_token,
                self.jwt_secret,
                JWT_ISSUER,
                JWT_AUDIENCE,
            )
            if settings_claims is None:
                logger.warning("MCP request rejected status=401 reason=invalid_token")
                await _json_error(send, 401, "The user token is invalid or expired.")
                return
            user_token: Token[str | None] = authenticated_user_id.set(settings_claims["sub"])
            role_token: Token[str | None] = authenticated_user_role.set(settings_claims["role"])
            state = scope.setdefault("state", {})
            state["authenticated_user_id"] = settings_claims["sub"]
            state["authenticated_user_role"] = settings_claims["role"]
            state["thread_id"] = thread_id
            state["trace_id"] = trace_id
            logger.info("MCP request authenticated")
            try:
                await self.app(scope, receive, send)
            finally:
                authenticated_user_role.reset(role_token)
                authenticated_user_id.reset(user_token)


def _is_loopback_origin(origin: str) -> bool:
    if not origin:
        return True
    try:
        parsed = urlsplit(origin)
        return (
            parsed.scheme == "http"
            and parsed.hostname in {"localhost", "127.0.0.1"}
            and parsed.port in {None, 8501}
            and parsed.path in {"", "/"}
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        return False


def _is_loopback_host(host: str) -> bool:
    return host.lower() in {"127.0.0.1:8001", "localhost:8001"}


async def _json_error(send: Send, status_code: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _normalize_order_id(order_id: str) -> str:
    normalized = order_id.strip().upper().replace("_", "-").replace(" ", "-")
    if not _ORDER_ID.fullmatch(normalized):
        raise ValueError("Invalid order ID format.")
    return normalized


def _is_hygiene_restricted(item_name: str) -> bool:
    normalized = item_name.casefold()
    return any(category in normalized for category in _HYGIENE_RESTRICTED)


def _evaluate_and_initiate_refund(
    order_repository: OrderRepository,
    user_id: str,
    order_id: str,
    confirmed_order_id: str,
    reason: str,
) -> dict:
    eligibility = _check_refund_eligibility(
        order_repository,
        user_id,
        order_id,
        confirmed_order_id,
        reason,
    )
    if not eligibility["eligible"]:
        return {"created": False, "reason": eligibility["reason"]}

    normalized = _normalize_order_id(order_id)
    needs_tier2 = bool(eligibility["requires_supervisor_approval"])
    if not needs_tier2:
        refund = order_repository.initiate_refund(normalized, user_id)
        if refund is None:
            return {"created": False, "reason": "Order not found for this account."}
        if refund["status"] == "ineligible":
            return {"created": False, "reason": "Order is not eligible for refund."}
        if refund["status"] == "already_initiated":
            return {
                "created": False,
                "reason": "A refund has already been initiated for this order.",
            }
        return {
            "created": True,
            "status": "initiated",
            "order_id": normalized,
            "amount_minor": refund["amount_minor"],
            "currency": refund["currency"],
            "refund_initiated": True,
        }

    approval_id = order_repository.create_approval(normalized, user_id)
    return {
        "created": True,
        "status": "pending_tier2_approval",
        "approval_id": approval_id,
        "order_id": normalized,
        "amount_minor": eligibility["amount_minor"],
        "currency": eligibility["currency"],
        "requires_supervisor_approval": True,
        "automatic_limit_minor": MAX_AUTO_REFUND_MINOR,
        "refund_initiated": False,
    }


def _check_refund_eligibility(
    order_repository: OrderRepository,
    user_id: str,
    order_id: str,
    confirmed_order_id: str,
    reason: str,
) -> dict:
    normalized = _normalize_order_id(order_id)
    if _normalize_order_id(confirmed_order_id) != normalized:
        return {"eligible": False, "reason": "Exact order ID confirmation is required."}
    order = order_repository.get_owned_order(normalized, user_id)
    if order is None:
        return {"eligible": False, "reason": "Order not found for this account."}
    if _is_hygiene_restricted(order["item_name"]):
        return {
            "eligible": False,
            "reason": "Hygiene-restricted innerwear and cosmetics are non-returnable.",
        }
    if (
        order["open_box_delivery"]
        and order["otp_verified"]
        and _DAMAGE_CLAIM.search(reason)
    ):
        return {
            "eligible": False,
            "reason": "Physical damage cannot be claimed after Open-Box Delivery OTP acceptance.",
        }
    if not order_repository.is_eligible(order):
        return {"eligible": False, "reason": "Order is not eligible for refund."}
    return {
        "eligible": True,
        "order_id": normalized,
        "item_name": order["item_name"],
        "amount_minor": order["amount_minor"],
        "currency": order["currency"],
        "requires_supervisor_approval": (
            order["currency"] != "INR"
            or order["amount_minor"] > MAX_AUTO_REFUND_MINOR
        ),
        "automatic_limit_minor": MAX_AUTO_REFUND_MINOR,
    }


def create_mcp_app(
    order_repository: OrderRepository,
    user_repository: UserRepository,
    jwt_secret: str,
):
    mcp = FastMCP(
        "ShopSense local OMS",
        instructions="Use verified, customer-owned order and refund tools only.",
        stateless_http=True,
        streamable_http_path="/mcp",
        transport_security=TransportSecuritySettings(
            allowed_hosts=["127.0.0.1:8001", "localhost:8001"],
            allowed_origins=["http://127.0.0.1:8501", "http://localhost:8501"],
        ),
    )

    def authenticated_customer_id() -> str:
        user_id = authenticated_user_id.get()
        if not user_id:
            raise PermissionError("A verified user token is required.")
        return user_id

    @mcp.tool()
    def get_order_details(order_id: str) -> dict:
        """Return details only for an order owned by the authenticated shopper."""
        user_id = authenticated_customer_id()
        order = order_repository.get_owned_order(_normalize_order_id(order_id), user_id)
        if order is None:
            return {"found": False, "reason": "Order not found for this account."}
        return_assessment = order_repository.return_eligibility(order)
        return {
            "found": True,
            "order_id": order["order_id"],
            "item_name": order["item_name"],
            "status": order["status"],
            "delivered_at": order["delivered_at"],
            "estimated_delivery_at": order["estimated_delivery_at"],
            "amount_minor": order["amount_minor"],
            "currency": order["currency"],
            "return_window_days": order["return_window_days"],
            "item_unused": bool(order["item_unused"]),
            "tags_attached": bool(order["tags_attached"]),
            "final_sale": bool(order["final_sale"]),
            "refund_status": order["refund_status"],
            "return_eligibility": return_assessment,
            "open_box_delivery": bool(order["open_box_delivery"]),
            "otp_verified": bool(order["otp_verified"]),
            "refund_initiated": bool(order["refund_initiated"]),
            "refund_initiated_date": order["refund_initiated_date"],
        }

    @mcp.tool()
    def list_customer_orders() -> list[dict]:
        """List order summaries belonging to the authenticated shopper."""
        user_id = authenticated_customer_id()
        return [
            {
                "order_id": order["order_id"],
                "item_name": order["item_name"],
                "status": order["status"],
                "amount_minor": order["amount_minor"],
                "currency": order["currency"],
            }
            for order in order_repository.list_customer_orders(user_id)
        ]

    @mcp.tool()
    def cancel_order(order_id: str, confirmed_order_id: str) -> dict:
        """Cancel an owned, cancellable order after exact-ID confirmation."""
        user_id = authenticated_customer_id()
        normalized = _normalize_order_id(order_id)
        if _normalize_order_id(confirmed_order_id) != normalized:
            return {"cancelled": False, "reason": "Exact order ID confirmation is required."}
        status = order_repository.cancel_owned_order(
            normalized,
            user_id,
            confirmed_order_id,
        )
        if status is None:
            return {"cancelled": False, "reason": "Order not found for this account."}
        return {"cancelled": status == "cancelled", "status": status}

    @mcp.tool()
    def check_refund_eligibility(
        order_id: str,
        confirmed_order_id: str,
        reason: str = "",
    ) -> dict:
        """Check refund eligibility without initiating or creating approval."""
        user_id = authenticated_customer_id()
        return _check_refund_eligibility(
            order_repository,
            user_id,
            order_id,
            confirmed_order_id,
            reason,
        )

    @mcp.tool()
    def evaluate_and_initiate_refund(
        order_id: str,
        confirmed_order_id: str,
        reason: str = "",
    ) -> dict:
        """Initiate an eligible low-value refund or escalate for Tier-2 approval."""
        user_id = authenticated_customer_id()
        return _evaluate_and_initiate_refund(
            order_repository,
            user_id,
            order_id,
            confirmed_order_id,
            reason,
        )

    @mcp.tool()
    def request_return(order_id: str, confirmed_order_id: str) -> dict:
        """Record a return request after the authenticated shopper confirms."""
        user_id = authenticated_customer_id()
        normalized = _normalize_order_id(order_id)
        order = order_repository.get_owned_order(normalized, user_id)
        if order is None:
            return {"status": "not_found"}
        if _is_hygiene_restricted(order["item_name"]):
            return {
                "status": "ineligible",
                "reason": "Hygiene-restricted innerwear and cosmetics are non-returnable.",
            }
        result = order_repository.create_return_request(
            normalized,
            user_id,
            confirmed_order_id,
        )
        if result is None:
            return {"status": "not_found"}
        return result

    app = mcp.streamable_http_app()
    return LocalMCPAuthMiddleware(
        app,
        user_repository=user_repository,
        jwt_secret=jwt_secret,
    )


def main() -> None:
    settings = get_settings()
    configure_file_logging(LOG_DIRECTORY, settings.log_level)
    database = Database(str(DATABASE_PATH))
    database.initialize()
    app = create_mcp_app(
        OrderRepository(database),
        UserRepository(database),
        settings.jwt_secret.get_secret_value(),
    )
    logger.info("Starting local MCP endpoint at %s", MCP_SERVER_URL)
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")


if __name__ == "__main__":
    main()
