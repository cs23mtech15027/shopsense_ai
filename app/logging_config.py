import logging
from contextlib import contextmanager
from contextvars import ContextVar, Token
from logging.handlers import RotatingFileHandler
from pathlib import Path
import re
from typing import Iterator


_thread_id: ContextVar[str] = ContextVar("shopsense_thread_id", default="-")
_FILE_HANDLER_MARKER = "_shopsense_file_handler"
_THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{12}$")


class _ThreadIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.thread_id = _thread_id.get()
        return True


class _ThreadPartitionedRotatingFileHandler(logging.Handler):
    def __init__(self, directory: Path):
        super().__init__()
        self.directory = directory

    def emit(self, record: logging.LogRecord) -> None:
        thread_id = getattr(record, "thread_id", "-")
        filename = (
            f"{thread_id}.log"
            if isinstance(thread_id, str) and _THREAD_ID_PATTERN.fullmatch(thread_id)
            else "shopsense.log"
        )
        try:
            path = self.directory / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            destination = RotatingFileHandler(
                path,
                maxBytes=5_000_000,
                backupCount=3,
                encoding="utf-8",
            )
            try:
                destination.setFormatter(self.formatter)
                destination.handle(record)
            finally:
                destination.close()
        except OSError:
            self.handleError(record)


@contextmanager
def bind_thread_id(thread_id: str) -> Iterator[None]:
    token: Token[str] = _thread_id.set(thread_id)
    try:
        yield
    finally:
        _thread_id.reset(token)


def configure_file_logging(directory: str | Path) -> None:
    log_directory = Path(directory).resolve()
    logger = logging.getLogger("shopsense")
    for handler in tuple(logger.handlers):
        if not getattr(handler, _FILE_HANDLER_MARKER, False):
            continue
        if getattr(handler, "directory", None) == log_directory:
            return
        logger.removeHandler(handler)
        handler.close()

    log_directory.mkdir(parents=True, exist_ok=True)
    handler = _ThreadPartitionedRotatingFileHandler(log_directory)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s thread_id=%(thread_id)s %(message)s"
        )
    )
    handler.addFilter(_ThreadIdFilter())
    setattr(handler, _FILE_HANDLER_MARKER, True)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
