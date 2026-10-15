import sqlite3

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.persistence.checkpoints import quarantine_corrupt_checkpoint


def test_corrupt_checkpoint_and_sidecars_are_preserved(tmp_path):
    checkpoint_path = tmp_path / "checkpoints.db"
    checkpoint_path.write_bytes(b"not a sqlite database")
    wal_path = tmp_path / "checkpoints.db-wal"
    wal_path.write_bytes(b"wal data")
    shm_path = tmp_path / "checkpoints.db-shm"
    shm_path.write_bytes(b"shm data")

    archive_path = quarantine_corrupt_checkpoint(str(checkpoint_path))

    assert archive_path is not None
    assert archive_path.read_bytes() == b"not a sqlite database"
    assert (tmp_path / f"{archive_path.name}-wal").read_bytes() == b"wal data"
    assert (tmp_path / f"{archive_path.name}-shm").exists()
    assert not checkpoint_path.exists()
    assert not wal_path.exists()
    assert not shm_path.exists()


def test_healthy_checkpoint_is_not_moved(tmp_path):
    checkpoint_path = tmp_path / "checkpoints.db"
    with sqlite3.connect(checkpoint_path) as connection:
        connection.execute("CREATE TABLE retained (value TEXT)")
        connection.execute("INSERT INTO retained VALUES ('conversation state')")

    assert quarantine_corrupt_checkpoint(str(checkpoint_path)) is None
    with sqlite3.connect(checkpoint_path) as connection:
        assert connection.execute("SELECT value FROM retained").fetchone() == (
            "conversation state",
        )


def test_api_starts_with_new_checkpoint_after_archiving_corrupt_store(tmp_path):
    checkpoint_path = tmp_path / "checkpoints.db"
    checkpoint_path.write_bytes(b"corrupt checkpoint")
    wal_path = tmp_path / "checkpoints.db-wal"
    wal_path.write_bytes(b"preserve wal")
    application = create_app(
        Settings(
            database_path=str(tmp_path / "orders.db"),
            checkpoint_path=str(checkpoint_path),
            memory_directory=str(tmp_path / "memories"),
        )
    )

    with TestClient(application) as client:
        response = client.get("/health")

    assert response.status_code == 200
    archive_path = next(tmp_path.glob("checkpoints.db.corrupt-*"))
    assert archive_path.read_bytes() == b"corrupt checkpoint"
    assert (tmp_path / f"{archive_path.name}-wal").read_bytes() == b"preserve wal"
    with sqlite3.connect(checkpoint_path) as connection:
        assert connection.execute("PRAGMA quick_check").fetchone() == ("ok",)
