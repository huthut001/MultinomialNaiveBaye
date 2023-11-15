from data_handler import DataHandler, DataPreprocessor
from model_trainer import ModelTrainer, Predictor
import pandas as pd

# Data Handling
data_handler = DataHandler("data/raw/dataset.csv")
data_handler.load_data()

# Data Preprocessing
data_preprocessor = DataPreprocessor()
data_handler.df = data_preprocessor.preprocess_data(data_handler.df)
data_handler.save_data("data/processed/preprocessed_dataset.csv")

# Model Training
model_trainer = ModelTrainer(data_handler.df["text"], data_handler.df["label"])
model_trainer.train_model()
model_trainer.save_model("results/trained_vectorizer.joblib", "results/trained_classifier.joblib")

# Making Predictions
test_data = pd.read_csv("path/to/test_data.csv")
predictor = Predictor(test_data, model_trainer.vectorizer, model_trainer.classifier)
predictions = predictor.make_predictions()

print(predictions)

# sample_submission["generated"] = predictions
# sample_submission.to_csv('results/submission.csv', index=False)
