FROM python:3.10-slim
WORKDIR /app
COPY my_streamlit_site.py .
COPY dags/user_inputs.csv .
COPY model.py .
COPY symptom_classifier_model.joblib .
COPY Symptom2Disease.csv .
COPY requirements.txt .
COPY frequency_dictionary_en_82_765.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet
EXPOSE 8501
CMD ["streamlit", "run", "my_streamlit_site.py", "--server.port=8501", "--server.address=0.0.0.0"]
