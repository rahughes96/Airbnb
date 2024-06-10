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

Because the "description" column of the dataset contained lists of strings we needed to create a function to combine them into one string once they had been cleaned. The cleaning function can be reused:

<img width="1075" alt="Screenshot 2024-06-10 at 17 11 48" src="https://github.com/rahughes96/Airbnb/assets/102994234/f0b60199-c3c8-4381-9a5d-dcf93acd6de6">

<img width="1059" alt="Screenshot 2024-06-10 at 17 12 07" src="https://github.com/rahughes96/Airbnb/assets/102994234/a8d6caac-0275-43ed-ac73-612353f15d72">

The feature values that will be used are beds, bathrooms, bedrooms and guests. However in the raw dataset some of the values in these columns are blank, alluding to one. A funtion is needed to replace these blank values:

<img width="1073" alt="Screenshot 2024-06-10 at 17 16 20" src="https://github.com/rahughes96/Airbnb/assets/102994234/873ed0dc-5e83-4eee-a1a4-0aec4b5f631a">

It is useful to gather all these methods under one function that is more easily called in the future.

<img width="1051" alt="Screenshot 2024-06-10 at 17 21 31" src="https://github.com/rahughes96/Airbnb/assets/102994234/001767e4-5945-4b72-a386-17ca57ccd2a7">

A function is then created to load the dataset, do the appropriate cleaning and form the tuples ready for the regression model to be trained. This function will use the subsequent functions and will be inherited into our regression.py file.

<img width="1044" alt="Screenshot 2024-06-10 at 17 19 50" src="https://github.com/rahughes96/Airbnb/assets/102994234/ec4da60a-7845-4a59-a6d8-5ca05fae9810">

Under the if __name__ == "__main__" block, we pull in the raw dataset, clean it and store it in a file now named "clean_tabular_data.csv"

Under a seperate file, the images also need to be prepared. A class is created named ImageDeveloper, in order to resize the images.

<img width="1007" alt="Screenshot 2024-06-10 at 17 27 07" src="https://github.com/rahughes96/Airbnb/assets/102994234/c57a7dd2-eb14-4248-8573-c600561de507">

The class is then used within a new function to store and alter the image sizes, making it easy to loop through all the old images and process them. 

<img width="1060" alt="Screenshot 2024-06-10 at 17 29 12" src="https://github.com/rahughes96/Airbnb/assets/102994234/86dde7c2-3d88-4381-927f-2a33b6bb1b93">

Within the if __name__ == "__main__" block the python library "Glob" is used to loop through the images folder and resize them.

<img width="1073" alt="Screenshot 2024-06-10 at 17 30 08" src="https://github.com/rahughes96/Airbnb/assets/102994234/f47c7ea1-9712-4c8d-b87d-0d7fcde6f46e">

All the data is now prepped and ready for training.

3.Regression Model

In order to gain a better understanding of how it works, the first thing to do is create a simple regression model. This will pull the data in, split it into training and test sets, train our model, calculate the performance metrics and plot the results. This is done using the built in SGDRegressor provided in SKlearn.

<img width="1075" alt="Screenshot 2024-06-10 at 21 35 48" src="https://github.com/rahughes96/Airbnb/assets/102994234/5e6396c8-1516-4439-b7d8-333dcf217d5d">

<img width="1047" alt="Screenshot 2024-06-10 at 17 43 28" src="https://github.com/rahughes96/Airbnb/assets/102994234/33b3af28-2e76-4de1-8189-6e835df934b6">

 We want to beat this baseline model, using more advanced versions of the regression model.

It is necessary now to create a function that will tune our models hyperparamaters and perform a grid search over a reasonable range of hyperparameter values. We could (and will) use SKLearns GridSearchCV to do this but in order to understand the methods used, it is better to initially create it from scratch. This function will return the best model, a dictionary of its best hyperparameter values, and a dictionary of its performance metrics.

<img width="1054" alt="Screenshot 2024-06-10 at 18 00 24" src="https://github.com/rahughes96/Airbnb/assets/102994234/eea4df5d-2c1b-4869-8d55-d11e971dc531">

<img width="1063" alt="Screenshot 2024-06-10 at 18 00 40" src="https://github.com/rahughes96/Airbnb/assets/102994234/45243aea-8f39-40de-b1af-22bc8e69d883">

Now that we have a function for tuning pur hyperparameters, and are confident ion how it works, we can revert to using the GridSearchCV method.

<img width="1051" alt="Screenshot 2024-06-10 at 18 07 35" src="https://github.com/rahughes96/Airbnb/assets/102994234/549e25f8-9b2f-435f-80cd-d72afc418aeb">

<img width="1041" alt="Screenshot 2024-06-10 at 18 07 55" src="https://github.com/rahughes96/Airbnb/assets/102994234/b4f6798a-ba05-4cb7-9a0a-0fbd29576e30">

Once the model is trained and tuned, we need a method for saving the model. The models are saved using JobLib and the metrics are saved as json files.

<img width="1066" alt="Screenshot 2024-06-10 at 20 54 43" src="https://github.com/rahughes96/Airbnb/assets/102994234/ea72e40f-3973-4f1d-8fc4-1aba0fbc1d91">

We now want to beat opur baseline model with decision trees, random forests, and gradient boosting. Each of these models will be tuned using the function we defined earlier. once each of the models is tuned, it is saved in a corresponding folder, this is all defined in one function called evaluate_all_models, making it easier to call.

<img width="1052" alt="Screenshot 2024-06-10 at 21 01 12" src="https://github.com/rahughes96/Airbnb/assets/102994234/5b1fac36-b258-4ed7-9b14-90c660c65e7c">

<img width="1066" alt="Screenshot 2024-06-10 at 21 01 49" src="https://github.com/rahughes96/Airbnb/assets/102994234/deba5746-b2db-4cc7-be6e-3d0dad036975">

<img width="1057" alt="Screenshot 2024-06-10 at 21 02 14" src="https://github.com/rahughes96/Airbnb/assets/102994234/ecb6dca7-645b-482e-80ab-70e864240398">

<img width="1053" alt="Screenshot 2024-06-10 at 21 02 33" src="https://github.com/rahughes96/Airbnb/assets/102994234/4a47547c-bfac-482d-9487-8d1c17dc5645">

<img width="1051" alt="Screenshot 2024-06-10 at 21 02 58" src="https://github.com/rahughes96/Airbnb/assets/102994234/00f64f30-9733-4b9f-995e-d3f88f0ff275">

Within the if __name__ == "__main__" block, we can now call evaluate_all_models function, in our case we can see that gradient boosting provides the best model. If we set plot=True, we can see this visualised.

<img width="639" alt="Screenshot 2024-06-10 at 21 44 47" src="https://github.com/rahughes96/Airbnb/assets/102994234/6ae85d3a-9a26-4090-948f-70bff6d83112">

4.Classification Model
