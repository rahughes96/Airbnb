*Airbnb Nightly Price Prediction using Neural Networks*

Table of Contents

1.Project Description
2.Installation Instructions
3.Usage Instructions
4.File Structure


*1.Project Description*
This project aims to predict the nightly price of Airbnb listings using a neural network model. It leverages various features of Airbnb properties, such as the number of guests, amenities, and ratings to make accurate price predictions.

The core objectives of the project include:

-Developing a robust neural network model using PyTorch.
-Experimenting with different hyperparameter configurations to find the optimal model.
-Improving prediction accuracy by tuning network architecture, optimizer, and learning rates.
-Gaining insights into deep learning model training and evaluation techniques.

What I learned:

-The importance of feature normalization in neural networks.
-How to perform model configuration search using a grid of hyperparameters.
-Techniques like dropout and gradient clipping to stabilize model training.
-Best practices for model evaluation, including RMSE and R² metrics.

2.Installation Instructions
To install and run the project locally, follow these steps:

Clone this repository:

git clone https://github.com/rahughes96/Airbnb
cd Airbnb

Set up a Python virtual environment (optional but recommended):

python -m venv env
source env/bin/activate  # For Windows: env\Scripts\activate

Install the required packages:

pip install -r requirements.txt

Ensure that the Airbnb dataset is available in the AirbnbData/Raw_Data/ directory:

The dataset should be in a CSV format, named tabular_data.csv.

3.Usage Instructions

To run the entire script and evaluate every model, run the following command:

python pipeline.py

The training process will automatically evaluate different model configurations, log the metrics via TensorBoard, and save the best-performing model.

Evaluating the Model: You can modify the script to load the saved model from the models/neural_network/best_model directory and evaluate its performance on new data.

The training logs, including loss values and gradients, will be saved to TensorBoard. You can visualize them by running:

tensorboard --logdir runs

*4.File Structure*

.
├── AirbnbData/                         # data saved and stored
│   └── Processed_Data/
│       └──processed_images/
│       └── clean_tabular_data.csv
|   └── Raw_Data/
|       └── Images/
|       └── tabular_data/
|           └── Listing.csv     
├── models/                              # Directory for saving trained models
│   └── neural_network/
│       └── best_model/ 
|           └── best_model.pt
|           └── hyperparameters.json
|           └── metrics.json
|           └── model.pt
|   └── neural_network_classification/
|       └── best_model/
|           └── best_model.pt
|           └── hyperparameters.json
|           └── metrics.json
|           └── model.pt
|   └──regression/
|       └── decision_tree/              #Same structure for all models in regression and classification
|           └── hyperparameters.json
|           └── metrics.json
|           └──model.joblib
|       └── gradient_boosting/
|       └── linear_regression/
|       └── random_forest/
|   └──classification/ 
|       └── decision_tree/              
|       └── gradient_boosting/
|       └── linear_regression/
|       └── random_forest/
├──runs/               
├── .gitignore                      # Script for cleaning raw data
├── tabular_data.py                 # cleans the data
├── regression.py                   # runs regression model
├── classification.py               # runs classification model
├── neural.py                       # runs neural network
├── Neural_classification.py        # runs neural classification script
├── pipeline.py                     # ruins entire project
├── README.md                       # Project README file
└── requirements.txt                # Python dependencies

