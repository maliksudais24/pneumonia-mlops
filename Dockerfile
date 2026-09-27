FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=120 --retries=5 -r requirements.txt

COPY pneumonia_model.h5 .
COPY src/app.py .
COPY src/templates/ templates/

EXPOSE 8080

RUN useradd --create-home appuser
USER appuser

CMD ["python", "app.py"]
