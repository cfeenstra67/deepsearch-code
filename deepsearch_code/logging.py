import os
from logging.config import dictConfig


def setup_logging(level: str | None = None) -> None:
    if level is None:
        level = os.getenv("LOG_LEVEL", "ERROR")
    dictConfig(
        {
            "version": 1,
            "formatters": {
                "main": {"format": "%(asctime)s - %(name)s [%(levelname)s] %(message)s"}
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "main",
                },
            },
            "loggers": {
                "deepsearch_code": {
                    "level": level,
                    "handlers": ["console"],
                    "propagate": False,
                },
                "__main__": {
                    "level": level,
                    "handlers": ["console"],
                    "propagate": False,
                },
            },
        }
    )
