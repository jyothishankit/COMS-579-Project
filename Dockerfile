FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml ./
COPY legalrag ./legalrag
RUN pip install --no-cache-dir -e ".[ui]"

# Inside compose, Pinecone Local is reachable by service name rather than localhost.
ENV PINECONE_LOCAL_HOST=http://pinecone:5080 \
    VECTOR_BACKEND=pinecone

ENTRYPOINT ["legalrag"]
CMD ["--help"]
