FROM python:3.7.13

EXPOSE 8501

WORKDIR /spp-app

COPY *py .

COPY Dockerfile .

RUN pip install -r requirements.txt

CMD python train.py

ENTRYPOINT ["streamlit", "run", "web-app.py", "--server.port=8501", "--server.address=0.0.0.0"]