from unicodedata import normalize
import pandas as pd
import itertools
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
from tabular_data import load_airbnb
from tabular_data import clean_tabular_data
import os
import joblib
import json

from sklearn import model_selection
from sklearn import linear_model
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

class MGS():

    def __init__(self):
        self.train_split ={}
        self.best_model = None
        self.best_hyperparameters = {}
        self.metrics = {}

    def sgd(self, path, label):
        airbnb_data = pd.read_csv(path)
        cleaned_airbnb_data = clean_tabular_data(airbnb_data)
        cleaned_airbnb_data[['Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating','Description']] = cleaned_airbnb_data[['Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating','Description']].dropna()
        cleaned_airbnb_data.to_csv('/Users/ryanhughes/Desktop/Aicore/Airbnb/Airbnb/AirbnbData/Raw_Data/tabular_data/listing_test.csv')

        print("clean airbnb data:", cleaned_airbnb_data.head())

        #Airbnb_data = pd.read_csv(path)

        (X,y) = load_airbnb(cleaned_airbnb_data, label)
        #print("X and y:")
        #print(X,y)


        #(X,y) = load_airbnb(Airbnb_data, label)

        X = scale(X)
        y = scale(y)
        #print("scale(X) and scale(y):")
        #print(X,y)
        print(X)
        xtrain, xtest, ytrain, ytest = model_selection.train_test_split(X, y, test_size=0.3)

        xvalid, xtest, yvalid, ytest = model_selection.train_test_split(xtest, ytest, test_size=0.5)

        self.train_split = {
            'xtrain' : xtrain,
            'xtest' : xtest,
            'ytrain' : ytrain,
            'ytest' : ytest,
            'xvalid' :xvalid,
            'yvalid' : yvalid
        }
        sgdr = SGDRegressor()
        sgdr.fit(mgs.train_split['xtrain'], mgs.train_split['ytrain'])

        cv_score = cross_val_score(sgdr, X, y, cv = 10)
        print("CV mean score: ", cv_score.mean())
        #def plot_results(self, )

        score = sgdr.score(mgs.train_split['xtrain'], mgs.train_split['ytrain'])
        print("R-squared:", score)

        ypred = sgdr.predict(xtest)

        mse = mean_squared_error(ytest, ypred)
        print("MSE: ", mse)
        print("RMSE: ", mse**(1/2.0))

        x_ax = range(len(ytest))
        plt.plot(x_ax, ytest, linewidth=1, label="original")
        plt.plot(x_ax, ypred, linewidth=1.1, label="predicted")
        plt.title("y-test and y-predicted data")
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend(loc='best', fancybox = True, shadow=True)
        plt.grid(True)
        plt.show()
    
    def _generate_hyperparameter_combinations(self, hyperparameters):
        keys = list(hyperparameters.keys())
        values = list(hyperparameters.values())
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))


    def custom_tune_regression_model_hyperparameters(self, model_class, xtrain, ytrain, xvalid, yvalid, hyperparameters):
        best_model = None
        best_hyperparameters = {}
        best_rmse = float('inf')

        for param_combination in self._generate_hyperparameter_combinations(hyperparameters):
            model = model_class(**param_combination)
            model.fit(xtrain, ytrain)

            ypred_valid = model.predict(xvalid)
            rmse = np.sqrt(mean_squared_error(yvalid, ypred_valid))

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_hyperparameters = param_combination

        performance_metrics = {
            "validation_RMSE": best_rmse,
            # You can add more performance metrics if needed
        }

        return best_model, best_hyperparameters, performance_metrics

    def tune_regression_model_hyperparameters(self, model_class, param_grid, xtrain, ytrain, xvalid, yvalid):
        grid_search = GridSearchCV(
            estimator=model_class(),
            param_grid=param_grid,
            scoring='neg_mean_squared_error',  # Use 'neg_mean_squared_error' for RMSE
            cv=5  # You can adjust the number of cross-validation folds as needed
        )

        grid_search.fit(xtrain, ytrain)

        best_model = grid_search.best_estimator_
        best_hyperparameters = grid_search.best_params_
        validation_rmse = np.sqrt(-grid_search.best_score_)

        return best_model, best_hyperparameters, validation_rmse

    def save_model(self, folder="models/regression/linear_regression"):
        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)

        # Save the trained model using joblib
        model_filename = os.path.join(folder, "model.joblib")
        joblib.dump(self.best_model, model_filename)

        # Save hyperparameters as JSON
        hyperparameters_filename = os.path.join(folder, "hyperparameters.json")
        with open(hyperparameters_filename, "w") as hyperparameters_file:
            json.dump(self.best_hyperparameters, hyperparameters_file, indent=4)

        # Save performance metrics as JSON
        metrics_filename = os.path.join(folder, "metrics.json")
        with open(metrics_filename, "w") as metrics_file:
            json.dump(self.metrics, metrics_file, indent=4)

        
if __name__ == "__main__":

    mgs = MGS()
    mgs.sgd('/Users/ryanhughes/Desktop/Aicore/Airbnb/Airbnb/AirbnbData/Raw_Data/tabular_data/listing.csv', 'Price_Night')

    hyperparameters = {
        'alpha': [0.001, 0.01, 0.1, 1.0],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'max_iter': [1000, 2000, 3000],
    }

    best_model, best_hyperparams, metrics = mgs.custom_tune_regression_model_hyperparameters(
        model_class=SGDRegressor,
        xtrain=mgs.train_split['xtrain'],
        ytrain=mgs.train_split['ytrain'],
        xvalid=mgs.train_split['xvalid'],
        yvalid=mgs.train_split['yvalid'],
        hyperparameters=hyperparameters
    )
    best_model = best_model
    print("Best hyperparameters:", best_hyperparams)
    print("Validation RMSE:", metrics["validation_RMSE"])

    mgs.save_model(folder="models/regression/linear_regression")










