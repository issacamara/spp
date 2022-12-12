FROM continuumio/miniconda3

WORKDIR /spp-app

#COPY environment.yml .
RUN conda update --name base conda
#RUN conda env create -f environment.yml
RUN conda install python=3.10.8
RUN conda update -n base -c defaults conda
RUN conda install numpy -y
RUN conda install tensorflow -y
RUN conda install streamlit -y
RUN pip install cufflinks
RUN pip install sqlalchemy
RUN pip install scikit-learn
RUN pip install yfinance
RUN pip install bs4

#RUN echo "source activate spp-env2"
#    ~/.bashrc
ENV PATH /opt/conda/envs/spp-env2/bin:$PATH
#ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH

EXPOSE 8501

COPY . .
#
#COPY configuration.yml .
#
#COPY tickers.csv .
#
#COPY create_db.sql .
#
##COPY requirements.txt .
#
#COPY *py .
#
#RUN pip install -r requirements.txt
#
#CMD streamlit run web-app.py
#
ENTRYPOINT ["streamlit", "run", "web-app.py", "--server.port=8501", "--server.address=0.0.0.0"]