FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN mkdir -p /data

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY streamlit_app.py ./
COPY .streamlit ./.streamlit
COPY README.md ./

EXPOSE 8080

CMD ["sh", "-c", "streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port ${PORT:-8080} --server.headless true"]
