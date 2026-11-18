FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN pip install streamlit==1.67.0 pandas==3.0.6 requests==2.34.2 plotly
COPY dashboard/app.py .
COPY data/sample ./sample
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
