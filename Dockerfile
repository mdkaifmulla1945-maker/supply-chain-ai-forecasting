FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# expose both ports
EXPOSE 8000 8501

# run API + Streamlit together
CMD uvicorn api:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app/app.py --server.port 8501 --server.address 0.0.0.0