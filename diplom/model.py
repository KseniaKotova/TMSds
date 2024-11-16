import pandas as pd
import string
import nltk
import os
import joblib
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class SymptomClassifier:
    """
    SymptomClassifier is used to classify medical symptoms into specific diagnoses
    using a trained Random Forest model and a TF-IDF vectorizer for text features.

    Attributes:
        csv_path (str): Path to the CSV file containing training data.
        model_path (str): Path to save or load the trained model.
        stop_words (set): Set of stopwords for text preprocessing.
        lemmatizer (WordNetLemmatizer): Lemmatizer for word normalization.
        punct (str): String of punctuation to remove from text.
        model (RandomForestClassifier): Random Forest model for symptom classification.
        vectorizer (TfidfVectorizer): TF-IDF vectorizer for text feature extraction.
    """
    def __init__(self, csv_path: str, model_path='symptom_classifier_model.joblib'):
        self.csv_path = csv_path
        self.model_path = model_path 
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(('and', 'i', "i'm", 'im', "i've", "ive", 'also', 'a', 'http', 'and', 'so', 'arnt', 
                                "arn't", 'this', 'when', 'it', 'many', 'cant', "can't", 'yes', 'no', 'these', 
                                'mailto', 'regards', 'ayanna', 'like'))
        self.lemmatizer = WordNetLemmatizer()
        self.punct = string.punctuation + "’‘”“"
        self.model = RandomForestClassifier(n_estimators=100, max_depth=49, min_samples_split=10, 
                                            min_samples_leaf=2, random_state=42)
        self.vectorizer = TfidfVectorizer(max_features=1500)

    def load_and_clean_data(self):
        """
        Loads and cleans data from the CSV file by removing duplicates, unnecessary labels,
        and punctuation, and performs tokenization and preprocessing.

        The cleaned data is stored in `self.data`.
        """
        data = pd.read_csv(self.csv_path)
        data.drop(columns='Unnamed: 0', axis=0, inplace=True)
        data = data[data['label'] != 'Dimorphic Hemorrhoids']
        data = data.sample(frac=1).reset_index(drop=True)
        data = data.drop_duplicates()

        data['text'] = data['text'].apply(lambda x: x.lower())
        data['text'] = data['text'].str.replace(f"[{string.punctuation}]", "", regex=True)
        
        data['text'] = data['text'].apply(WordPunctTokenizer().tokenize)

        data['text'] = data['text'].apply(self.preprocess)

        data['text'] = data['text'].apply(self.delete_punct)

        data['X'] = data['text'].apply(lambda x: ' '.join(x))
        
        self.data = data

    def preprocess(self, tokens: list) -> list:
        """
        Lemmatizes and removes stopwords from tokens.

        Args:
            tokens (list): List of tokens to preprocess.

        Returns:
            list: List of processed tokens.
        """
        return [self.lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in self.stop_words]

    def delete_punct(self, lemmas: list) -> list:
        """
        Removes punctuation from a list of lemmatized words.

        Args:
            lemmas (list): List of lemmatized words.

        Returns:
            list: List of words without punctuation.
        """
        return [x for x in lemmas if x not in self.punct]

    def train_model(self):
        """
        Trains the Random Forest model on the cleaned text data, splits the data
        into training and test sets, evaluates the model on the test set,
        and saves the model to `self.model_path`.
        """
        tfidf_features = self.vectorizer.fit_transform(self.data['X']).toarray()
        
        X_train, X_test, y_train, y_test = train_test_split(tfidf_features, self.data['label'], test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
    
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        
        print(f'Accuracy: {accuracy:.2f}')
        print(report)

        joblib.dump((self.model, self.vectorizer), self.model_path)

    def load_model(self):
        """
        Loads a saved model and vectorizer from `self.model_path`. If the model
        is not found, a message prompts the user to train it first.
        """
        if os.path.exists(self.model_path):
            self.model, self.vectorizer = joblib.load(self.model_path)
            print("Model loaded successfully.")
        else:
            print("Model file not found. Train the model first by calling `train_model` method.")

    def predict(self, text: str, top_n=5) -> list:
        """
        Predicts the top diagnoses for a given text input.

        Args:
            text (str): Symptom description to predict diagnosis for.
            top_n (int): Number of top predictions to return.

        Returns:
            list: List of tuples with predicted diagnoses and their probabilities.
        """
        clean_text = self.preprocess(WordPunctTokenizer().tokenize(text.lower()))
        clean_text = self.delete_punct(clean_text)
        clean_text = ' '.join(clean_text)

        vectorized_text = self.vectorizer.transform([clean_text]).toarray()

        probabilities = self.model.predict_proba(vectorized_text)[0]
        labels = self.model.classes_

        results = sorted(zip(labels, probabilities), key=lambda x: x[1], reverse=True)

        top_results = results[:top_n]
        total_prob = sum(prob for _, prob in top_results)
        top_results_normalized = [(label, f"{(prob / total_prob) * 100:.2f}%") for label, prob in top_results]

        return top_results_normalized

## Using:
# if __name__ == "__main__":
#     classifier = SymptomClassifier('Symptom2Disease.csv')
#     classifier.load_and_clean_data()
#     classifier.train_model()

#     symptom_input = "I’ve been experiencing a lot of neck pain lately, also my chest feels tight, my muscles are sore from coughing all the time, and my throat is also quite sore and red. Also my temperature is high."
#     predictions = classifier.predict(symptom_input, top_n=5)
    
#     print("Top-5 predicted probabilities for each diagnosis (recalculated to 100%):")
#     for label, probability in predictions:
#         print(f"{label}: {probability}")

