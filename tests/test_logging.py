import logging
from logging.handlers import RotatingFileHandler

from transcripto_core.logging_utils import setup_logging


def test_setup_logging_uses_rotating_file(tmp_path):
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    log_path = tmp_path / "logs" / "app.log"

    try:
        logger = setup_logging(log_path, console=False, max_bytes=1024, backups=2)
        rotating = [
            handler for handler in logger.handlers if isinstance(handler, RotatingFileHandler)
        ]

        assert rotating and rotating[0].maxBytes == 1024
        assert rotating[0].backupCount == 2

        logging.info("hello from the test")
        for handler in logger.handlers:
            handler.flush()

        assert "hello from the test" in log_path.read_text(encoding="utf-8")
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        for handler in previous_handlers:
            root.addHandler(handler)


def test_setup_logging_replaces_previous_handlers(tmp_path):
    root = logging.getLogger()
    previous_handlers = list(root.handlers)

    try:
        setup_logging(tmp_path / "one.log", console=False)
        setup_logging(tmp_path / "two.log", console=False)

        settings = [
            getattr(handler, "baseFilename", "") for handler in root.handlers
        ]
        assert any(str(tmp_path / "two.log") in name for name in settings)
        assert not any(str(tmp_path / "one.log") in name for name in settings)
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        for handler in previous_handlers:
            root.addHandler(handler)
