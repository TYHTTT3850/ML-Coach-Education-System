FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

ARG INSTALL_LOCAL_EMBEDDING=false

COPY requirements.txt requirements-local.txt ./
RUN apt-get update \
    && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && if [ "$INSTALL_LOCAL_EMBEDDING" = "true" ]; then \
         pip install -r requirements-local.txt; \
       else \
         pip install -r requirements.txt; \
       fi

COPY app ./app
COPY knowledge ./knowledge
COPY docker-entrypoint.sh /usr/local/bin/coursecoach-entrypoint
RUN mkdir -p /app/data/chroma /app/data/uploads \
    && useradd --create-home --uid 10001 coursecoach \
    && chown -R coursecoach:coursecoach /app \
    && chmod +x /usr/local/bin/coursecoach-entrypoint

EXPOSE 8000

ENTRYPOINT ["coursecoach-entrypoint"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
