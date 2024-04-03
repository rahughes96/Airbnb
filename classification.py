import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import re

class AirbnbLogisticRegression:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_valid = None
        self.y_valid = None
        self.best_hyperparameters = None
        self.best_model = None
        self.metrics = {}

    def load_data(self):
        # Load data
        dataframe = pd.read_csv(self.filepath)

        # Preprocess data
        columns_to_drop = ['Unnamed: 0', 'ID', 'url']
        dataframe.drop(columns=columns_to_drop, inplace=True)

        label_encoder = LabelEncoder()
        columns_to_encode = ['Category', 'Title', 'Description', 'Location']

        for column in columns_to_encode:
            dataframe[column] = label_encoder.fit_transform(dataframe[column])

        amenities_encoder = LabelEncoder()
        dataframe['Amenities'] = dataframe['Amenities'].apply(lambda x: '|'.join(map(str, x)))  # Convert list to string
        dataframe['Amenities'] = amenities_encoder.fit_transform(dataframe['Amenities'])
        #print('1st')
        #print(dataframe)

        dataframe = dataframe.select_dtypes(['float64', 'int64'])

        dataframe[['Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating','Description']] = dataframe[['Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating','Description']].dropna(axis=1)
        #print("2nd")
        #print(dataframe)
        # Ensure the DataFrame is not empty
        if len(dataframe) == 0:
            raise ValueError("DataFrame is empty. Please check the data.")

        # Split data into features and target variable
        X = dataframe.drop(columns=['Category'])
        y = dataframe['Category']
        #print("TITS")
        #print(train_test_split(X, y, test_size=0.2, random_state=42))

        # Split data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train, self.y_train, test_size=0.25, random_state=42)
        #print("TITS")
        #print(self.y_train)

        self.data = dataframe

        return self.data

    def train_model(self):
        #print(self.X_train)
        # Train logistic regression model
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(self.X_train, self.y_train)

    def preprocess_data(self):
        scaler = StandardScaler()

        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
    
    def evaluate_model(self):
        # Predictions on training set
        train_predictions = self.model.predict(self.X_train_scaled)

        # Predictions on test set
        test_predictions = self.model.predict(self.X_test_scaled)

        # Compute accuracy for training set
        train_accuracy = accuracy_score(self.y_train, train_predictions)

        # Compute accuracy for test set
        test_accuracy = accuracy_score(self.y_test, test_predictions)

        # Compute precision for training set
        train_precision = precision_score(self.y_train, train_predictions, average='weighted')

        # Compute precision for test set
        test_precision = precision_score(self.y_test, test_predictions, average='weighted')

        # Compute recall for training set
        train_recall = recall_score(self.y_train, train_predictions, average='weighted')

        # Compute recall for test set
        test_recall = recall_score(self.y_test, test_predictions, average='weighted')

        # Compute F1 score for training set
        train_f1_score = f1_score(self.y_train, train_predictions, average='weighted')

        # Compute F1 score for test set
        test_f1_score = f1_score(self.y_test, test_predictions, average='weighted')
        """""
        # Print the performance measures
        print("Training Set Metrics:")
        print("Accuracy:", train_accuracy)
        print("Precision:", train_precision)
        print("Recall:", train_recall)
        print("F1 Score:", train_f1_score)
        print("\nTest Set Metrics:")
        print("Accuracy:", test_accuracy)
        print("Precision:", test_precision)
        print("Recall:", test_recall)
        print("F1 Score:", test_f1_score)

        """""

    def tune_classification_model_hyperparameters(model_class, xtrain, ytrain, xvalid, yvalid, param_grid):
        grid_search = GridSearchCV(
            estimator=model_class(),
            param_grid=param_grid,
            scoring='accuracy',  # Use accuracy as the scoring metric
            cv=5
        )

        grid_search.fit(xtrain, ytrain)

        best_model = grid_search.best_estimator_
        best_hyperparameters = grid_search.best_params_

        # Calculate validation accuracy
        validation_accuracy = accuracy_score(yvalid, best_model.predict(xvalid))

        # Additional steps for performance evaluation if needed
        ypred_valid = grid_search.predict(xvalid)
        validation_classification_report = classification_report(yvalid, ypred_valid, output_dict=True)

        performance_metrics = {
            "validation_accuracy": validation_accuracy,
            "validation_classification_report": validation_classification_report,
            # Add other metrics if needed
        }

        return best_model, best_hyperparameters, performance_metrics
    
    def tune_classification_model_hyperparameters_1(self, model_class, xtrain, ytrain, xvalid, yvalid, param_grid):
        grid_search = GridSearchCV(
            estimator=model_class(),
            param_grid=param_grid,
            scoring='accuracy',  # Use accuracy for classification
            cv=5
        )

        grid_search.fit(xtrain, ytrain)

        self.best_model = grid_search.best_estimator_
        self.best_hyperparameters = grid_search.best_params_
        
        # Evaluate performance on validation set
        ypred_valid = grid_search.predict(xvalid)
        validation_accuracy = accuracy_score(yvalid, ypred_valid)
        self.metrics["validation_accuracy"] = validation_accuracy

        # Additional performance metrics
        precision = precision_score(yvalid, ypred_valid, average='macro')
        recall = recall_score(yvalid, ypred_valid, average='macro')
        f1 = f1_score(yvalid, ypred_valid, average='macro')

        performance_metrics = {
            "validation_accuracy": validation_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        print(performance_metrics)
        return self.best_model, self.best_hyperparameters, performance_metrics

if __name__ == "__main__":
    filepath = 'AirbnbData/Raw_Data/tabular_data/listing.csv'
    airbnb_lr = AirbnbLogisticRegression(filepath)
    print("load data...")
    airbnb_lr.load_data()
    #print(data)
    print("preprocess data...")
    airbnb_lr.preprocess_data()
    print("train model...")
    airbnb_lr.train_model()
    print("evaluate model...")
    airbnb_lr.evaluate_model()

    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    print("perfomance metrics:")
    best_model, best_hyperparameters, performance_metrics = airbnb_lr.tune_classification_model_hyperparameters_1(LogisticRegression, airbnb_lr.X_train, airbnb_lr.y_train, airbnb_lr.X_valid, airbnb_lr.y_valid, param_grid)


