from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib

class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.vectorizer = CountVectorizer()
        self.classifier = MultinomialNB()

    def train_model(self):
        X_train, _, y_train, _ = train_test_split(self.X, self.y, random_state=104, test_size=0.2, shuffle=True)

        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_vectorized, y_train)

    def save_model(self, vectorizer_path, classifier_path):
        # Save the trained vectorizer and classifier

        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.classifier, classifier_path)

class Predictor:
    def __init__(self, test_data, vectorizer, classifier):
        self.test_data = test_data
        self.vectorizer = vectorizer
        self.classifier = classifier

    def make_predictions(self):
        X_test = self.vectorizer.transform(self.test_data['text'])
        y_pred = self.classifier.predict_proba(X_test)[:, 1]
        return y_pred