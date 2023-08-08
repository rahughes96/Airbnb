from unicodedata import normalize
import pandas as pd
import seaborn as sns
from tabular_data import load_airbnb
from tabular_data import clean_tabular_data

from sklearn import model_selection
from sklearn import linear_model
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score

class MGS():

    def __init__(self):
        self.train_split ={}

    def sgd(self, path, label):
        Airbnb_data = pd.read_csv(path)
        cleaned_airbnb_data = clean_tabular_data(Airbnb_data)
        print("clean airbnb data:", cleaned_airbnb_data.head())

        #Airbnb_data = pd.read_csv(path)

        (X,y) = load_airbnb(cleaned_airbnb_data, label)
        print("X and y:")
        print(X,y)


        #(X,y) = load_airbnb(Airbnb_data, label)

        #X = normalize(X, )
        #y = normalize(y)
        print("scale(X) and scale(y):")
        print(X,y)
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

        
    #def plot_results(self, )

if __name__ == "__main__":
    mgs = MGS()
    mgs.sgd('/Users/ryanhughes/Desktop/Aicore/Airbnb/Airbnb/AirbnbData/Raw_Data/tabular_data/listing.csv', 'Price_Night')
    
    print(mgs.train_split)
