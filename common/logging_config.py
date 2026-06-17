import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logging(service_name: str, log_to_file: bool = True) -> logging.Logger:
    from common.config import get_config

    log_level_str = get_config("logging.level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    log_format = get_config(
        "logging.format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    formatter = logging.Formatter(log_format)

    root_logger = logging.getLogger()

    if root_logger.handlers:
        root_logger.handlers.clear()

    root_logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    if log_to_file:
        logs_dir = Path(get_config("paths.logs", "logs"))
        logs_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            logs_dir / f"{service_name}.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)

    _configure_third_party_loggers()

    service_logger = logging.getLogger(service_name)
    service_logger.info(f"{service_name} logger initialized (level={log_level_str})")

    return service_logger


def _configure_third_party_loggers():
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
