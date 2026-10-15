import logging
import sqlite3
from pathlib import Path
from uuid import uuid4


logger = logging.getLogger("shopsense.checkpoints")
_SIDECAR_SUFFIXES = ("-wal", "-shm")
_CORRUPTION_ERRORS = ("malformed", "not a database", "disk image")


def quarantine_corrupt_checkpoint(path: str) -> Path | None:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return None

    try:
        database_uri = checkpoint_path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(database_uri, uri=True, timeout=1)
        try:
            check_results = connection.execute("PRAGMA quick_check").fetchall()
        finally:
            connection.close()
    except sqlite3.DatabaseError as exc:
        if not any(marker in str(exc).lower() for marker in _CORRUPTION_ERRORS):
            raise RuntimeError(
                "Could not verify conversation checkpoint storage."
            ) from exc
        check_results = [(str(exc),)]

    if check_results == [("ok",)]:
        return None

    archive_path = checkpoint_path.with_name(
        f"{checkpoint_path.name}.corrupt-{uuid4().hex[:12]}"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    files_to_archive = [
        (checkpoint_path, archive_path),
        *(
            (
                Path(f"{checkpoint_path}{suffix}"),
                Path(f"{archive_path}{suffix}"),
            )
            for suffix in _SIDECAR_SUFFIXES
        ),
    ]
    moved: list[tuple[Path, Path]] = []
    try:
        for source, destination in files_to_archive:
            if source.exists():
                source.rename(destination)
                moved.append((source, destination))
    except OSError as exc:
        for source, destination in reversed(moved):
            if destination.exists():
                destination.rename(source)
        raise RuntimeError(
            "Could not preserve corrupt conversation checkpoint storage."
        ) from exc

    logger.error(
        "Corrupt conversation checkpoint archived path=%s; a new checkpoint "
        "database will be created",
        archive_path,
    )
    return archive_path
