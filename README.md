# Airbnb
AIcore Specialisation 

*Contents*
1. Intro
2. Data preparation
3. Regression Model
4. Classification Model
5. Configurable Neural Network
6. Reusing Framework

1. Intro
The aim of this project is to Model Airbnb’s property listing dataset. Build a framework that systematically train, tune, and evaluate models on several tasks that are tackled by the Airbnb team.

To begin with the data is loaded in and cleaned in preparation. This includes the full dataset and pictures that need to be resized for consistency. Then a simple linear regression model is built to predict the price per night feature, then try with different models, including gradient boosting, random forest and decision trees, to beat the baseline model.

Classification
Neural
Reuse

2.Data Preparation
The file AirBnbData.csv is downloaded from the zip file provided. The python file tabular_data.py is used to store all the functionality of cleaning the tabular data. Starting by defining a function called remove_rows_with_missing_ratings which removes the rows with missing values from the columns in the dataset:

<img width="1085" alt="Screenshot 2024-02-12 at 22 50 12" src="https://github.com/rahughes96/Airbnb/assets/102994234/e18df85c-bd7b-4206-905d-7ccbb1cd3295">

Because the "description" column of the dataset contained lists of strings we needed to create a function to combine them into one string:

