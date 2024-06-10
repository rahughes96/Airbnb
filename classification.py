import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
from tabular_data import load_data_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import re

class AirbnbLogisticRegression:
    def __init__(self):
        self.data = None
        self.model = None
        self.best_hyperparameters = None
        self.best_model = None
        self.metrics = {}
        self.train_split = {}
        self.train_accuracy_history = []  # Store training accuracy history
        self.valid_accuracy_history = []  # Store validation accuracy history

    def load_data(self, filepath):
        dataframe = load_data_classification(filepath)

        # Split data into features and target variable
        X = dataframe.drop(columns=['Category'])
        y = dataframe['Category']
        #print(train_test_split(X, y, test_size=0.2, random_state=42))

        # Split data into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.5, random_state=42)
        #print(self.y_train)

        self.train_split = {
            'xtrain': x_train,
            'xtest': x_test,
            'ytrain': y_train,
            'ytest': y_test,
            'xvalid': x_valid,
            'yvalid': y_valid
        }

        self.data = dataframe
        #self.data.to_csv('/Users/ryanhughes/Desktop/Aicore/Airbnb/Airbnb/AirbnbData/Raw_Data/tabular_data/classification_data_test.csv')

        return self.data

    def train_model(self):
        #print(self.X_train)
        # Train logistic regression model
        self.model = LogisticRegression(solver= 'liblinear', max_iter=9000)
        self.model.fit(self.train_split["xtrain"], self.train_split["ytrain"])

    def preprocess_data(self):
        scaler = StandardScaler()

        self.train_split["xtrain"] = scaler.fit_transform(self.train_split['xtrain'])
        self.train_split["xtest"] = scaler.transform(self.train_split['xtest'])
        self.train_split['xvalid'] = scaler.transform(self.train_split['xvalid'])

        #self.X_train = scaler.fit_transform(self.train_split['xtrain'])
        #self.X_test = scaler.transform(self.train_split['xtest'])
        #self.X_valid = scaler.transform(self.train_split['xvalid'])
    
    def evaluate_model(self):
        # Predictions on training set
        train_predictions = self.model.predict(self.train_split["xtrain"])

        # Predictions on test set
        test_predictions = self.model.predict(self.train_split["xtest"])

        # Compute accuracy for training set
        train_accuracy = accuracy_score(self.train_split["ytrain"], train_predictions)

        # Compute accuracy for test set
        test_accuracy = accuracy_score(self.train_split["ytest"], test_predictions)

        # Compute precision for training set
        train_precision = precision_score(self.train_split["ytrain"], train_predictions, average='weighted')

        # Compute precision for test set
        test_precision = precision_score(self.train_split["ytest"], test_predictions, average='weighted')

        # Compute recall for training set
        train_recall = recall_score(self.train_split["ytrain"], train_predictions, average='weighted')

        # Compute recall for test set
        test_recall = recall_score(self.train_split["ytest"], test_predictions, average='weighted')

        # Compute F1 score for training set
        train_f1_score = f1_score(self.train_split["ytrain"], train_predictions, average='weighted')

        # Compute F1 score for test set
        test_f1_score = f1_score(self.train_split["ytest"], test_predictions, average='weighted')
        
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
    
    def tune_classification_model_hyperparameters(self, model_class, xtrain, ytrain, xvalid, yvalid, param_grid):
        grid_search = GridSearchCV(
            estimator=model_class(),
            param_grid=param_grid,
            scoring='accuracy',  # Use accuracy for classification
            cv=5, 
            error_score='raise'
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
    
    def compute_learning_curves(self, model, train_sizes):
        performance_metrics = {'train': [], 'validation': []}

        for size in train_sizes:
            size = int(size)
            # Select a subset of the training data with the current size
            subset_x_train = self.train_split['xtrain'][:size]
            subset_y_train = self.train_split['ytrain'][:size]

            # Train the model on the subset
            model.fit(subset_x_train, subset_y_train)

            # Evaluate the model on the subset and validation set
            train_predictions = model.predict(subset_x_train)
            train_accuracy = accuracy_score(subset_y_train, train_predictions)

            validation_predictions = model.predict(self.train_split['xvalid'])
            validation_accuracy = accuracy_score(self.train_split['yvalid'], validation_predictions)

            # Record the performance metrics
            performance_metrics['train'].append(train_accuracy)
            performance_metrics['validation'].append(validation_accuracy)

        return performance_metrics
    
    def plot_learning_curves(self, train_sizes, performance_metrics):
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, performance_metrics['train'], label='Training')
        plt.plot(train_sizes, performance_metrics['validation'], label='Validation')
        plt.xlabel('Training Set Size')
        plt.ylabel('Performance Metric')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def save_model(self, folder, best_model=None, best_hyperparameters=None, metrics=None):

        """

        Saves the trained regression model, hyperparameters, and performance metrics to a specified folder.

        Parameters:
            folder (str): The path to the folder where the model, hyperparameters, and metrics will be saved.
            best_model: The trained regression model to be saved (default is None).
            best_hyperparameters (dict): The hyperparameters used for training the model (default is None).
            metrics (dict): The performance metrics of the model (default is None).

        """

        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)

        # Save the trained model using joblib
        model_filename = os.path.join(folder, "model.joblib")
        joblib.dump(best_model, model_filename)

        # Save hyperparameters as JSON
        hyperparameters_filename = os.path.join(folder, "hyperparameters.json")
        with open(hyperparameters_filename, "w") as hyperparameters_file:
            json.dump(best_hyperparameters, hyperparameters_file, indent=4)

        # Save performance metrics as JSON
        metrics_filename = os.path.join(folder, "metrics.json")
        with open(metrics_filename, "w") as metrics_file:
            json.dump(metrics, metrics_file, indent=4)

    def evaluate_all_models(self, path):
        self.load_data(path)
        self.preprocess_data()
        self.train_model()

        print("Evaluate Logistic Regression")
        log_hyperparameters = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [5000]}

        log_best_model, log_best_hyperparams, log_metrics = self.tune_classification_model_hyperparameters(
            model_class=LogisticRegression,
            xtrain=self.train_split['xtrain'],
            ytrain=self.train_split['ytrain'],
            xvalid=self.train_split['xvalid'],
            yvalid=self.train_split['yvalid'],
            param_grid=log_hyperparameters
        )

        self.save_model(folder="models/classification/logistic_regression", best_model=log_best_model,
            best_hyperparameters=log_best_hyperparams, metrics=log_metrics)

        print("Evaluate Decision Tree")
        dt_hyperparameters = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }

        dt_best_model, dt_best_hyperparams, dt_metrics = self.tune_classification_model_hyperparameters(
            model_class=DecisionTreeClassifier,
            xtrain=self.train_split['xtrain'],
            ytrain=self.train_split['ytrain'],
            xvalid=self.train_split['xvalid'],
            yvalid=self.train_split['yvalid'],
            param_grid=dt_hyperparameters
        )

        self.save_model(folder="models/classification/decision_tree", best_model=dt_best_model,
            best_hyperparameters=dt_best_hyperparams, metrics=dt_metrics)


        print("Evaluate Random Forest")
        rf_hyperparameters = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }

        rf_best_model, rf_best_hyperparams, rf_metrics = self.tune_classification_model_hyperparameters(
            model_class=RandomForestClassifier,
            xtrain=self.train_split['xtrain'],
            ytrain=self.train_split['ytrain'],
            xvalid=self.train_split['xvalid'],
            yvalid=self.train_split['yvalid'],
            param_grid=rf_hyperparameters
        )

        self.save_model(folder="models/classification/random_forest", best_model=rf_best_model,
            best_hyperparameters=rf_best_hyperparams, metrics=rf_metrics)


        print("Evaluate Gradient Boosting")
        gb_hyperparameters = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]  # Adjusted max_features values
        }

        gb_best_model, gb_best_hyperparams, gb_metrics = self.tune_classification_model_hyperparameters(
            model_class=GradientBoostingClassifier,
            xtrain=self.train_split['xtrain'],
            ytrain=self.train_split['ytrain'],
            xvalid=self.train_split['xvalid'],
            yvalid=self.train_split['yvalid'],
            param_grid=gb_hyperparameters
        )

        self.save_model(folder="models/classification/gradient_boosting", best_model=gb_best_model,
            best_hyperparameters=gb_best_hyperparams, metrics=gb_metrics)
        
    def find_best_model(self, task_folder):

        """

        Finds the best-performing regression model among a predefined list of models.

        Returns:
            tuple: A tuple containing the best model, its hyperparameters, and performance metrics.

        """

        # Define a list of models and their respective folders
        models_folders = [
            ("Linear Regression", f"{task_folder}logistic_regression"),
            ("Decision Tree", f"{task_folder}decision_tree"),
            ("Random Forest", f"{task_folder}random_forest"),
            ("Gradient Boosting", f"{task_folder}gradient_boosting")
        ]

        best_model_name = None
        best_model = None
        best_hyperparameters = {}
        best_performance_metrics = {}
        best_mean = 0

        for model_name, folder in models_folders:
            # Load the model
            model_filename = os.path.join(folder, "model.joblib")
            loaded_model = joblib.load(model_filename)

            # Load hyperparameters
            hyperparameters_filename = os.path.join(folder, "hyperparameters.json")
            with open(hyperparameters_filename, "r") as hyperparameters_file:
                hyperparameters = json.load(hyperparameters_file)

            # Load performance metrics
            metrics_filename = os.path.join(folder, "metrics.json")
            with open(metrics_filename, "r") as metrics_file:
                performance_metrics = json.load(metrics_file)

            # Check mean performance metric
            Total = performance_metrics['validation_accuracy']+performance_metrics['precision']+performance_metrics['recall']+performance_metrics['f1_score']
            mean = Total / 4

            # Check if the current model has a lower RMSE
            if mean > best_mean:
                best_model_name = model_name
                best_model = loaded_model
                best_hyperparameters = hyperparameters
                best_performance_metrics = performance_metrics

        print("Best Model:", best_model_name)
        print("Best Hyperparameters:", best_hyperparameters)
        print("Best Performance Metrics:", best_performance_metrics)

        return best_model, best_hyperparameters, best_performance_metrics
    
if __name__ == "__main__":
    filepath = 'AirbnbData/Processed_Data/clean_tabular_data/clean_tabular_data.csv'

    airbnb_lr = AirbnbLogisticRegression()
    airbnb_lr.evaluate_all_models(path=filepath)
    best_model, best_hyperparameters, best_performance_metrics = airbnb_lr.find_best_model("models/classification/")


    """""

    correlation_matrix = airbnb_lr.data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
    plt.title('Correlation Heatmap of Numerical Columns')
    plt.show()

    """


    #train_sizes = np.linspace(0.1, 1.0, 10) * len(airbnb_lr.train_split['xtrain'])

    # Compute the learning curves
    #performance_metrics = airbnb_lr.compute_learning_curves(airbnb_lr.model, train_sizes)

    # Plot the learning curves
    #airbnb_lr.plot_learning_curves(train_sizes, performance_metrics)

