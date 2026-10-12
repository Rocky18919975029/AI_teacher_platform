"""ASGI entry point; configuration is validated during startup."""

from teacher.api import create_app

app = create_app()
