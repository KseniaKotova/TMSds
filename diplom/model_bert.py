import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

class SymptomClassifier:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.model = None

    def load_data(self):
        data = pd.read_csv(self.csv_path)
        data.drop(columns='Unnamed: 0', axis=0, inplace=True)
        data = data[data['label'] != 'Dimorphic Hemorrhoids']
        data = data.sample(frac=1).reset_index(drop=True)
        data = data.drop_duplicates()
        
        self.data = data

    def prepare_datasets(self, test_size=0.1):
        X, y = self.data['text'].values, self.data['label'].values
        self.label2int = {label: i for i, label in enumerate(self.data['label'].unique())}
        self.int2label = {i: label for label, i in self.label2int.items()}
        
        self.data['label'] = self.data['label'].map(self.label2int)
        y = self.data['label'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
        
        train_encodings = self.tokenizer(list(X_train), padding="max_length", truncation=True)
        test_encodings = self.tokenizer(list(X_test), padding="max_length", truncation=True)

        train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train)).batch(8)
        test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test)).batch(8)

        return train_dataset, test_dataset

    def build_model(self):
        num_classes = len(self.int2label)
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            "bert-base-cased",
            num_labels=num_classes,
            id2label=self.int2label,
            label2id=self.label2int
        )
        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
            metrics=['accuracy']
        )


    def train_model(self, train_dataset, val_dataset, epochs=3):
        self.model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)


    def predict(self, text, top_n=5):
        encoding = self.tokenizer(text, return_tensors='tf', padding="max_length", truncation=True)
        predictions = self.model.predict(encoding).logits
        probabilities = tf.nn.softmax(predictions, axis=1).numpy()[0]
        
        top_indices = probabilities.argsort()[-top_n:][::-1]
        top_labels = [self.int2label[idx] for idx in top_indices]
        top_probabilities = probabilities[top_indices]
        
        total_prob = sum(top_probabilities)
        top_results_normalized = [(label, f"{(prob / total_prob) * 100:.2f}%") for label, prob in zip(top_labels, top_probabilities)]
        
        return top_results_normalized


classifier = SymptomClassifier('Symptom2Disease.csv')
classifier.load_data()
train_dataset, val_dataset = classifier.prepare_datasets()
classifier.build_model()

classifier.train_model(train_dataset, val_dataset, epochs=4)
classifier.model.save_weights('bert_model_weights.h5')

classifier.model.load_weights('bert_model_weights.h5')

symptom_input = "I have red spots and my skin is itching"
predictions = classifier.predict(symptom_input, top_n=5)

print("Top-5:")
for label, probability in predictions:
    print(f"{label}: {probability}")