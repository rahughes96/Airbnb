# Airbnb
AIcore Specialisation 

*Contents*
1. Intro
2. Data preparation
3. Regression Model
4. Classification Model
5. Configurable Neural Network
6. Reusing Framework

*1. Intro*

The aim of this project is to Model Airbnb’s property listing dataset. Build a framework that systematically train, tune, and evaluate models on several tasks that are tackled by the Airbnb team.

To begin with the data is loaded in and cleaned in preparation. This includes the full dataset and pictures that need to be resized for consistency. Then a simple linear regression model is built to predict the price per night feature, then try with different models, including gradient boosting, random forest and decision trees, to beat the baseline model.

Classification
Neural
Reuse

*2.Data Preparation*

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

*3.Regression Model*

The script begins by defining a class, ewhich will then be used to store certain key values that will be used throughout the training of this model.

<img width="1092" alt="Screenshot 2024-06-11 at 20 32 16" src="https://github.com/rahughes96/Airbnb/assets/102994234/0df00aec-c57c-4a0a-86d7-1724ea789a4e">

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

<img width="1097" alt="Screenshot 2024-06-12 at 21 52 33" src="https://github.com/rahughes96/Airbnb/assets/102994234/5cd89fb0-f774-4a7d-9b86-72184b408d5d">

<img width="639" alt="Screenshot 2024-06-10 at 21 44 47" src="https://github.com/rahughes96/Airbnb/assets/102994234/6ae85d3a-9a26-4090-948f-70bff6d83112">

*4.Classification Model*

First we define our class, which will again be used to store the key vlues used in the training of the classification model.

<img width="1099" alt="Screenshot 2024-06-11 at 20 33 41" src="https://github.com/rahughes96/Airbnb/assets/102994234/e8930b4e-f161-426a-8af9-ec027e327a22">

Similarly as before with the regression model, the data is loaded in but this time "category" is the label. A simple logistic regression model is trained to predict the category from the tabular data. This is done by using a few functions in tandem, as shown. First we load the data and split it into are train and test sets

<img width="1099" alt="Screenshot 2024-06-11 at 20 43 47" src="https://github.com/rahughes96/Airbnb/assets/102994234/61bac610-d43c-448c-a1b7-62dd032d2a74">

We then need to scale our data before training it as follows

<img width="1083" alt="Screenshot 2024-06-11 at 20 46 46" src="https://github.com/rahughes96/Airbnb/assets/102994234/9b7c49fc-5c81-44bb-bf08-39b37dc9c996">
<img width="1084" alt="Screenshot 2024-06-11 at 20 47 04" src="https://github.com/rahughes96/Airbnb/assets/102994234/164b31c5-dd45-4414-9701-a49ac2b9842f">

We can then evaluate the performance of the model by gathering and storing the relevant metrics. What we want to do from here is to see how its performance compares to the models we will train next.

<img width="1081" alt="Screenshot 2024-06-11 at 20 51 56" src="https://github.com/rahughes96/Airbnb/assets/102994234/8b4a2afc-44d7-4014-991b-14ca8a37f757">

As with the regression model, we need a method of tuning the hyperparameters. This time we will be using validation acuracy to decide which is the best model.

<img width="1081" alt="Screenshot 2024-06-11 at 21 06 09" src="https://github.com/rahughes96/Airbnb/assets/102994234/e53cba66-484b-4fd3-8cfd-b23c2e4ccf14">

We will again need to have funtions to save our model and plot the results. This time we will plot a confusion matrix to visualise the acuracy the model has in predicting the category.

<img width="1068" alt="Screenshot 2024-06-12 at 21 22 53" src="https://github.com/rahughes96/Airbnb/assets/102994234/64df487b-529e-4ddd-8f9e-c1544135ff08">

<img width="1083" alt="Screenshot 2024-06-12 at 21 23 32" src="https://github.com/rahughes96/Airbnb/assets/102994234/a7ca3ed2-0ef4-40b9-8343-51f18889a65e">

Once again, we will now try and beat the baseline logistic regression model with by using random forest, decision trees and gradient boosting. As before we will create a funtion called evaluate all models, which will apply the tune_classification_model_hyperparameters function to each of these to tune their hyperparameters before evaluating them, plotting their results and saving the hyperparameters, performance metrics and model in the appropriate folder.

<img width="1086" alt="Screenshot 2024-06-12 at 21 34 08" src="https://github.com/rahughes96/Airbnb/assets/102994234/4258afa9-019d-42ed-8603-dcd6721d0969">
<img width="1082" alt="Screenshot 2024-06-12 at 21 34 41" src="https://github.com/rahughes96/Airbnb/assets/102994234/2366c808-e148-40ce-89d9-0f079e4ea5ee">
<img width="1051" alt="Screenshot 2024-06-12 at 21 35 03" src="https://github.com/rahughes96/Airbnb/assets/102994234/b0aa1870-acb9-4bdb-8238-5db5af1f8de7">
<img width="1054" alt="Screenshot 2024-06-12 at 21 35 25" src="https://github.com/rahughes96/Airbnb/assets/102994234/a443cf0b-ccb8-4380-9546-e6e7c1cc2fda">

In order to find the best model, a function is created that wil loop through the classification models folder and select the model which has the best validation accuracy and return the model name, a dictionary of its hyperparameters and a dictionary of its performance metrics.

<img width="1083" alt="Screenshot 2024-06-12 at 21 49 01" src="https://github.com/rahughes96/Airbnb/assets/102994234/95322c28-837f-4573-9a98-b0152dafc1bd">
<img width="1052" alt="Screenshot 2024-06-12 at 21 49 22" src="https://github.com/rahughes96/Airbnb/assets/102994234/1ecdc18f-8255-468b-99fe-d71679256a63">

Within the if __name__ == "__main__" block, we can now call evaluate_all_models function, in our case we can see that gradient boosting provides the best model. If we set plot=True, we can see this visualised.


<img width="1090" alt="Screenshot 2024-06-12 at 21 50 56" src="https://github.com/rahughes96/Airbnb/assets/102994234/96d93fce-cfc5-41ab-8c9f-d7189918cc35">

<img width="1000" alt="Screenshot 2024-06-11 at 22 11 49" src="https://github.com/rahughes96/Airbnb/assets/102994234/dba9018b-f06f-4663-acd3-8b0084567dcd">