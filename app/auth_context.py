from contextvars import ContextVar


authenticated_user_id: ContextVar[str | None] = ContextVar(
    "shopsense_authenticated_user_id",
    default=None,
)
authenticated_user_role: ContextVar[str | None] = ContextVar(
    "shopsense_authenticated_user_role",
    default=None,
)
