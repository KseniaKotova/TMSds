import streamlit as st
import pandas as pd
import re
import os
import joblib
from symspellpy import SymSpell, Verbosity
from model import SymptomClassifier
from langdetect import detect, LangDetectException
import nltk
from nltk.corpus import stopwords


nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", 0, 1)

MODEL_PATH = "symptom_classifier_model.joblib"

if os.path.exists(MODEL_PATH):
    classifier = joblib.load(MODEL_PATH)
else:
    classifier = SymptomClassifier('Symptom2Disease.csv')
    classifier.load_and_clean_data()
    classifier.train_model()
    joblib.dump(classifier, MODEL_PATH)


def correct_spelling(text):
    corrected_text = []
    for word in text.split():
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            corrected_text.append(suggestions[0].term)
        else:
            corrected_text.append(word)
    return ' '.join(corrected_text)


def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9.,?!\'\s]', '', text)
    return text.strip()

def preprocess_text(input_text):
    input_text = clean_text(input_text)
    input_text = correct_spelling(input_text)
    return input_text


def contains_common_words(text):
    words = text.split()
    common_words = [word for word in words if word.lower() in stop_words]
    return len(common_words) > 0


def is_text_informative(text):
    word_count = len(text.split())
    return word_count >= 5


def is_valid_sentence(text):
    words = text.split()
    uncommon_words = [word for word in words if not sym_spell.lookup(word.lower(), Verbosity.CLOSEST, max_edit_distance=2)]
    return len(uncommon_words) < len(words) * 0.5


def is_text_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False


def is_meaningful_text(text):
    return (is_text_informative(text) and 
            contains_common_words(text) and 
            is_valid_sentence(text))



st.title('Medical Diagnosis Prediction Survey')

user_input = st.text_area("Describe your symptoms:", "")

if st.button("Predict Diagnosis"):
    if user_input:
        if not is_text_english(user_input) or not is_meaningful_text(user_input):
            st.warning("Please use English or provide a meaningful description to describe your symptoms")
        else:
            processed_input = preprocess_text(user_input)
            predictions = classifier.predict(processed_input, top_n=5)

            st.subheader("These diagnoses are just suppositional, please visit your doctor.")
            st.subheader("Top-5 most likely diagnoses based on your input:")
            for label, probability in predictions:
                probability = probability.replace('%', '')
                probability = float(probability)
                st.write(f"{label}: {probability:.2f}%")
            
            if st.button("Reset Survey"):
                csv_file_path = 'dags/user_inputs.csv'
                file_exists = os.path.isfile(csv_file_path)
                
                if user_input:
                    with open(csv_file_path, mode='a', newline='') as file:
                        writer = pd.DataFrame([[user_input]], columns=["User Input"])
                        writer.to_csv(file, header=not file_exists, index=False)
                    st.success("Your symptoms have been saved!")
                st.experimental_set_query_params()
    else:
        st.write("Please enter a description of your symptoms to get a prediction.")


## Code below is for BERT MODEL, which is not using now, because my laptop Asus ZenBook can't manage to do it

# import streamlit as st
# from model_bert import SymptomClassifier
# from symspellpy import SymSpell, Verbosity
# import re
# import os
# import pandas as pd


# sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
# sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", 0, 1) 

# classifier = SymptomClassifier('Symptom2Disease.csv')
# classifier.load_data()
# classifier.prepare_datasets()
# classifier.build_model()
# classifier.model.load_weights('bert_model_weights.h5')

# def correct_spelling(text):
#     corrected_text = []
#     for word in text.split():
#         suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
#         if suggestions:
#             corrected_text.append(suggestions[0].term)
#         else:
#             corrected_text.append(word)
#     return ' '.join(corrected_text)


# def clean_text(text):
#     text = re.sub(r'[^a-zA-Z0-9.,?!\'\s]', '', text)  # Оставляем только текст и пунктуацию
#     return text.strip()

# def preprocess_text(input_text):
#     input_text = clean_text(input_text)
#     input_text = correct_spelling(input_text)
#     return input_text

# def is_text_informative(text):
#     word_count = len(text.split())
#     return word_count >= 5

# st.title('Medical Diagnosis Prediction Survey')
# st.header('Please describe your symptoms below.')

# user_input = st.text_area("Describe your symptoms:", "")

# if st.button("Predict Diagnosis"):
#     if user_input:
#         if is_text_informative(user_input):
#             processed_input = preprocess_text(user_input)
#             predictions = classifier.predict(processed_input, top_n=5)

#             st.subheader("Top-5 most likely diagnoses based on your input:")
#             for label, probability in predictions:
#                 probability = probability.replace('%', '')
#                 probability = float(probability)
#                 st.write(f"{label}: {probability:.2f}%")
#         else:
#             st.warning("Please provide a more detailed description of your symptoms (e.g., 'I have a rash and headache that has been persistent for two days.').")
#     else:
#         st.write("Please enter a description of your symptoms to get a prediction.")

# if st.button("Reset Survey"):
#     if user_input:
#         csv_file_path = 'user_inputs.csv'
#         file_exists = os.path.isfile(csv_file_path)
        
#         with open(csv_file_path, mode='a', newline='') as file:
#             writer = pd.DataFrame([[user_input]], columns=["User Input"])
#             writer.to_csv(file, header=not file_exists, index=False)

#         st.success("Your symptoms have been saved!")

#     st.experimental_set_query_params()