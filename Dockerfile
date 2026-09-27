FROM python:3.13-slim
LABEL qfbench2.interface_version="2.0"
WORKDIR /app
COPY agent.py /app/agent.py
COPY strong_rag /app/strong_rag
ENTRYPOINT ["python", "/app/agent.py"]
