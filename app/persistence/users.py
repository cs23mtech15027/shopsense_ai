import hashlib
import secrets
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from app.config import JWT_AUDIENCE, JWT_ISSUER

_PASSWORD_ITERATIONS = 310_000
_TOKEN_LIFETIME = timedelta(hours=12)


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _PASSWORD_ITERATIONS,
    )
    return salt.hex(), digest.hex()


def verify_password(password: str, salt_hex: str, expected_hash: str) -> bool:
    if not salt_hex or not expected_hash:
        return False
    try:
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(expected_hash)
    except ValueError:
        return False
    actual = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _PASSWORD_ITERATIONS,
    )
    return secrets.compare_digest(actual, expected)


def seed_demo_customers(connection: sqlite3.Connection) -> None:
    for customer_id in (f"customer-{number}" for number in range(1, 6)):
        existing = connection.execute(
            "SELECT 1 FROM customers WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()
        if existing is None:
            connection.execute(
                """
                INSERT INTO customers (customer_id, username, password_salt, password_hash)
                VALUES (?, ?, ?, ?)
                """,
                (customer_id, customer_id, "", ""),
            )


class UserRepository:
    def __init__(self, database):
        self.database = database

    def login(
        self,
        username: str,
        password: str,
        jwt_secret: str,
        jwt_issuer: str = JWT_ISSUER,
        jwt_audience: str = JWT_AUDIENCE,
    ) -> dict[str, str] | None:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT customer_id, password_salt, password_hash, role
                FROM customers WHERE username = ?
                """,
                (username.strip().lower(),),
            ).fetchone()
        if row is None:
            hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                bytes(16),
                _PASSWORD_ITERATIONS,
            )
            return None
        if not verify_password(password, row["password_salt"], row["password_hash"]):
            return None
        token = self.issue_token(
            row["customer_id"],
            row["role"],
            jwt_secret,
            jwt_issuer,
            jwt_audience,
        )
        return {
            "access_token": token,
            "customer_id": row["customer_id"],
            "role": row["role"],
        }

    def issue_token(
        self,
        customer_id: str,
        role: str,
        jwt_secret: str,
        jwt_issuer: str = JWT_ISSUER,
        jwt_audience: str = JWT_AUDIENCE,
    ) -> str:
        now = datetime.now(UTC)
        expires_at = now + _TOKEN_LIFETIME
        token = jwt.encode(
            {
                "sub": customer_id,
                "role": role,
                "iss": jwt_issuer,
                "aud": jwt_audience,
                "iat": now,
                "exp": expires_at,
                "jti": secrets.token_urlsafe(24),
            },
            jwt_secret,
            algorithm="HS256",
        )
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO auth_sessions (token_hash, customer_id, expires_at)
                VALUES (?, ?, ?)
                """,
                (
                    hashlib.sha256(token.encode("ascii")).hexdigest(),
                    customer_id,
                    expires_at.isoformat(),
                ),
            )
        return token

    def get_token_claims(
        self,
        token: str,
        jwt_secret: str,
        jwt_issuer: str = JWT_ISSUER,
        jwt_audience: str = JWT_AUDIENCE,
    ) -> dict[str, Any] | None:
        try:
            claims = jwt.decode(
                token,
                jwt_secret,
                algorithms=["HS256"],
                issuer=jwt_issuer,
                audience=jwt_audience,
                options={"require": ["sub", "exp", "iat", "jti", "iss", "aud"]},
            )
            customer_id = claims["sub"]
            if not isinstance(customer_id, str) or not claims.get("jti"):
                return None
            token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
        except (jwt.InvalidTokenError, KeyError, TypeError, UnicodeEncodeError):
            return None
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT c.customer_id, c.role
                FROM auth_sessions AS s
                JOIN customers AS c ON c.customer_id = s.customer_id
                WHERE s.token_hash = ? AND s.customer_id = ? AND s.expires_at > ?
                """,
                (token_hash, customer_id, datetime.now(UTC).isoformat()),
            ).fetchone()
        if row is None:
            return None
        return {
            "sub": row["customer_id"],
            "role": row["role"],
            "jti": claims["jti"],
        }

    def revoke_token(self, token: str) -> None:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self.database.connect() as connection:
            connection.execute(
                "DELETE FROM auth_sessions WHERE token_hash = ?",
                (token_hash,),
            )
