import pandas as pd

class DataHandler:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)

    def save_data(self, output_path):
        self.df.to_csv(output_path, index=False)

class DataPreprocessor:
    @staticmethod
    def preprocess_data(df):
        df["text"] = df["text"].str.replace('\n', '')
        return df