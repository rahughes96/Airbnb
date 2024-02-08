import pandas as pd
import numpy as np
import joblib
import json
import os
import itertools
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tabular_data import load_airbnb
from tabular_data import clean_tabular_data

class MGS():
    def __init__(self):
        self.train_split = {}
        self.best_model = None
        self.best_hyperparameters = {}
        self.metrics = {}

    def sgd(self, path, label, plot = False):
        airbnb_data = pd.read_csv(path)
        cleaned_airbnb_data = clean_tabular_data(airbnb_data)
        cleaned_airbnb_data[['Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating','Description']] = cleaned_airbnb_data[['Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating','Description']].dropna()
        cleaned_airbnb_data.to_csv('/Users/ryanhughes/Desktop/Aicore/Airbnb/Airbnb/AirbnbData/Raw_Data/tabular_data/listing_test.csv')
    

        print("Cleaned Airbnb data:", cleaned_airbnb_data.head())

        (X, y) = load_airbnb(cleaned_airbnb_data, label)
        X = scale(X)
        y = scale(y)

        xtrain, xtest, ytrain, ytest = model_selection.train_test_split(X, y, test_size=0.3)
        xvalid, xtest, yvalid, ytest = model_selection.train_test_split(xtest, ytest, test_size=0.5)

        self.train_split = {
            'xtrain': xtrain,
            'xtest': xtest,
            'ytrain': ytrain,
            'ytest': ytest,
            'xvalid': xvalid,
            'yvalid': yvalid
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


        if plot == True:
            x_ax = range(len(ytest))
            plt.plot(x_ax, ytest, linewidth=1, label="original")
            plt.plot(x_ax, ypred, linewidth=1.1, label="predicted")
            plt.title("y-test and y-predicted data")
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.legend(loc='best', fancybox = True, shadow=True)
            plt.grid(True)
            plt.show()
    
    def plot_predictions(self, model, xtest, ytest, title="y-test and y-predicted data"):
        ypred = model.predict(xtest)

        x_ax = range(len(ytest))
        plt.plot(x_ax, ytest, linewidth=1, label="original")
        plt.plot(x_ax, ypred, linewidth=1.1, label="predicted")
        plt.title(title)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend(loc='best', fancybox=True, shadow=True)
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
        }

        return best_model, best_hyperparameters, performance_metrics

    def tune_regression_model_hyperparameters(self, model_class, param_grid, xtrain, ytrain, xvalid, yvalid):
        grid_search = GridSearchCV(
            estimator=model_class(),
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=5
        )

        grid_search.fit(xtrain, ytrain)

        self.best_model = grid_search.best_estimator_
        self.best_hyperparameters = grid_search.best_params_
        validation_rmse = np.sqrt(-grid_search.best_score_)

        self.metrics["validation_RMSE"] = validation_rmse

        # Additional steps for using xvalid and yvalid if needed
        ypred_valid = grid_search.predict(xvalid)
        valid_rmse = np.sqrt(mean_squared_error(yvalid, ypred_valid))
        self.metrics["validation_RMSE_on_xvalid"] = valid_rmse

    def save_model_test(self, folder="models/regression/linear_regression", best_model=None, best_hyperparameters=None, metrics=None):
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


    def save_model(self, folder):
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
    
    def evaluate_all_models(self, path, label):
            # Load and clean the data
            print("Load and clean the data")
            airbnb_data = pd.read_csv(path)
            cleaned_airbnb_data = clean_tabular_data(airbnb_data)
            cleaned_airbnb_data.to_csv('/Users/ryanhughes/Desktop/Aicore/Airbnb/Airbnb/AirbnbData/Raw_Data/tabular_data/listing_test.csv')

            (X, y) = load_airbnb(cleaned_airbnb_data, label)

            # Train-test split
            print("Train-test split")
            X = scale(X)
            y = scale(y)
            xtrain, xtest, ytrain, ytest = model_selection.train_test_split(X, y, test_size=0.3)
            xvalid, xtest, yvalid, ytest = model_selection.train_test_split(xtest, ytest, test_size=0.5)

            self.train_split = {
                'xtrain': xtrain,
                'xtest': xtest,
                'ytrain': ytrain,
                'ytest': ytest,
                'xvalid': xvalid,
                'yvalid': yvalid
            }

            # Evaluate Linear Regression
            print("Evaluate Linear Regression")
            sgd_hyperparameters={
                    'alpha': [0.001, 0.01, 0.1, 1.0],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'max_iter': [1000, 2000, 3000],
            }
            sgd_best_model, sgd_best_hyperparams, sgd_metrics = self.custom_tune_regression_model_hyperparameters(
                model_class=SGDRegressor,
                xtrain=self.train_split['xtrain'],
                ytrain=self.train_split['ytrain'],
                xvalid=self.train_split['xvalid'],
                yvalid=self.train_split['yvalid'],
                hyperparameters=sgd_hyperparameters
            )

            self.save_model_test(folder="models/regression/decision_tree", best_model=sgd_best_model,
                        best_hyperparameters=sgd_best_hyperparams, metrics=sgd_metrics)
            xtest_sgd = self.train_split['xtest']
            ytest_sgd = self.train_split['ytest']
            self.plot_predictions(model=sgd_best_model, xtest=xtest_sgd, ytest=ytest_sgd, title="Linear Regression Predictions")

            # Evaluate Decision Tree
            print("Evaluate Decision Tree")
            dt_hyperparameters = {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            dt_best_model, dt_best_hyperparams, dt_metrics = self.custom_tune_regression_model_hyperparameters(
                model_class=DecisionTreeRegressor,
                xtrain=self.train_split['xtrain'],
                ytrain=self.train_split['ytrain'],
                xvalid=self.train_split['xvalid'],
                yvalid=self.train_split['yvalid'],
                hyperparameters=dt_hyperparameters
            )
            self.save_model_test(folder="models/regression/decision_tree", best_model=dt_best_model,
                        best_hyperparameters=dt_best_hyperparams, metrics=dt_metrics)
            
            xtest_dt = self.train_split['xtest']
            ytest_dt = self.train_split['ytest']
            self.plot_predictions(model=dt_best_model, xtest=xtest_dt, ytest=ytest_dt, title="Decision Tree Predictions")

            # Evaluate Random Forest
            print("Evaluate Random Forest")
            rf_hyperparameters = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            rf_best_model, rf_best_hyperparams, rf_metrics = self.custom_tune_regression_model_hyperparameters(
                model_class=RandomForestRegressor,
                xtrain=self.train_split['xtrain'],
                ytrain=self.train_split['ytrain'],
                xvalid=self.train_split['xvalid'],
                yvalid=self.train_split['yvalid'],
                hyperparameters=rf_hyperparameters,
            )
            self.save_model_test(folder="models/regression/random_forest", best_model=rf_best_model,
                        best_hyperparameters=rf_best_hyperparams, metrics=rf_metrics)
            
            xtest_rf = self.train_split['xtest']
            ytest_rf = self.train_split['ytest']
            self.plot_predictions(model=rf_best_model, xtest=xtest_rf, ytest=ytest_rf, title="Random Forest Predictions")


            # Evaluate Gradient Boosting
            print("Evaluate Gradient Boosting")
            gb_hyperparameters = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            gb_best_model, gb_best_hyperparams, gb_metrics = self.custom_tune_regression_model_hyperparameters(
                model_class=GradientBoostingRegressor,
                xtrain=self.train_split['xtrain'],
                ytrain=self.train_split['ytrain'],
                xvalid=self.train_split['xvalid'],
                yvalid=self.train_split['yvalid'],
                hyperparameters=gb_hyperparameters
            )
            self.save_model_test(folder="models/regression/gradient_boosting", best_model=gb_best_model,
                            best_hyperparameters=gb_best_hyperparams, metrics=gb_metrics)

            xtest_gb = self.train_split['xtest']
            ytest_gb = self.train_split['ytest']
            self.plot_predictions(model=gb_best_model, xtest=xtest_gb, ytest=ytest_gb, title="Gradient Boosting Predictions")


if __name__ == "__main__":
    mgs = MGS()

    #print("Load and preprocess data")
    #mgs.sgd('/Users/ryanhughes/Desktop/Aicore/Airbnb/Airbnb/AirbnbData/Raw_Data/tabular_data/listing.csv', 'Price_Night', plot=True)

    #print("Define hyperparameter grid")
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1.0],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'max_iter': [1000, 2000, 3000],
    }

    #print("Tune hyperparameters")
    #mgs.tune_regression_model_hyperparameters(model_class=SGDRegressor, param_grid=param_grid)

    #print("Save the model, hyperparameters, and metrics")
    #mgs.save_model(folder="models/regression/linear_regression")

    print("Evaluate all models")
    mgs.evaluate_all_models('/Users/ryanhughes/Desktop/Aicore/Airbnb/Airbnb/AirbnbData/Raw_Data/tabular_data/listing.csv', 'Price_Night')