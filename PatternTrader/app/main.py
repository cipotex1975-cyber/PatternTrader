from __future__ import annotations

import uvicorn

from app.api.main import create_app
from app.core.config.settings import get_settings
from app.core.logger import get_logger, setup_logger


def main() -> None:
    setup_logger()
    logger = get_logger("Main")

    settings = get_settings()
    app = create_app()

    logger.info(f"Starting {settings.application.name} v{settings.application.version}")

    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        workers=settings.server.workers,
        reload=settings.server.reload,
    )


if __name__ == "__main__":
    main()
