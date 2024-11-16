from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def process_user_data():
    input_file = '/opt/airflow/dags/user_inputs.csv'
    output_file = '/opt/airflow/dags/processed_user_inputs.csv'
    
    df = pd.read_csv(input_file, names=['text'])
    
    if df.empty:
        print("Input file is empty.")
        return
    
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        processed_df = pd.read_csv(output_file)
        new_data = df[~df['text'].isin(processed_df['text'])]
    else:
        new_data = df.copy()

    if new_data.empty:
        print("No new data to process.")
        return
    
    stop_words = set(stopwords.words('english'))
    stop_words.update(('and', 'i', "i'm", 'im', "i've", "ive", 'also', 'a', 'http', 'and', 'so', 'arnt', 
                                "arn't", 'this', 'when', 'it', 'many', 'cant', "can't", 'yes', 'no', 'these', 
                                'mailto', 'regards', 'ayanna', 'like'))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalpha()]
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

    new_data['processed_text'] = new_data['text'].apply(clean_text)

    if os.path.exists(output_file):
        new_data.to_csv(output_file, mode='a', header=False, index=False)
    else:
        new_data.to_csv(output_file, index=False)


with DAG(
    'load_user_inputs',
    default_args={'retries': 1},
    description='A DAG for processing user input data',
    schedule_interval='0 4 * * *',
    start_date=datetime(2024, 10, 29),
    catchup=False,
) as dag:
    
    start = DummyOperator(task_id='start')
    process_task = PythonOperator(
        task_id='process_user_data',
        python_callable=process_user_data
    )
    end = DummyOperator(task_id='end')

    start >> process_task >> end
