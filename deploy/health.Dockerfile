FROM python:3.12-slim-bookworm
WORKDIR /opt/app
COPY deploy/__init__.py deploy/cluster.py deploy/fleet_health.py ./deploy/
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
USER 65532:65532
CMD ["python", "-m", "deploy.fleet_health"]
