import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from tabular_data import *

class AirbnbLogisticRegression:
    def __init__(self):
        self.model = None

    def load_data(self, file_path, label):
        Airbnb_data = pd.read_csv(file_path)

        label_encoder = LabelEncoder()
        Airbnb_data['Category'] = label_encoder.fit_transform(Airbnb_data['Category'])

        columns_to_drop = ['Unnamed: 0', 'ID', 'url']
        Airbnb_data.drop(columns=columns_to_drop, inplace=True)

        text_columns = ['Title', 'Description', 'Amenities']
        for col in text_columns:
            Airbnb_data[col] = Airbnb_data[col].str.lower()

        Airbnb_data.dropna(inplace=True)

        features, labels = load_airbnb_class(Airbnb_data, label)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    def preprocess_data(self):
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)

    def train_model(self):
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(self.X_train_scaled, self.y_train)

    def evaluate_model(self):
        train_accuracy = self.model.score(self.X_train_scaled, self.y_train)
        test_accuracy = self.model.score(self.X_test_scaled, self.y_test)
        print("Training Accuracy:", train_accuracy)
        print("Testing Accuracy:", test_accuracy)

if __name__ == "__main__":
    airbnb_lr = AirbnbLogisticRegression()
    airbnb_lr.load_data('/Users/ryanhughes/Desktop/Aicore/Airbnb/Airbnb/AirbnbData/Raw_Data/tabular_data/listing.csv', "Category")
    airbnb_lr.preprocess_data()
    airbnb_lr.train_model()
    airbnb_lr.evaluate_model()
