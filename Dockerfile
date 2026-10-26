FROM python:3.12-slim
WORKDIR /app
COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt
COPY . .
ENV GREENLIT_DATA_DIR=/app/data
EXPOSE 8765
CMD ["python", "-m", "uvicorn", "greenlit.app:app", "--host", "0.0.0.0", "--port", "8765", "--workers", "1"]
