from json import load
import pandas as pd
import ast
from sklearn.preprocessing import LabelEncoder

def remove_rows_with_missing_ratings(Dataframe):

    """

    This function removes the rows with blank values in the ratings columns and returns the dataframe

    Attributes:
        Dataframe (df): The raw dataframe of the listings file 
    
    """
    
    rating_columns=['Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating','Description']
    Dataframe.dropna(inplace = True, subset = rating_columns)
    return Dataframe

def clean_strings(string):

    """

    Cleans the columns that have string values by removing superflous information and joins the lines into one line

    Attributes:
        string (str): the string contained in the dataframe columns
    
    """

    try:
        clean_str = ast.literal_eval(string)
        clean_str.remove("About this space")
        clean_str = "".join(clean_str)
        return clean_str
    except Exception as e:
        return str

def clean_description_strings(dataframe):

    """

    Applies the previous function into the dataframe as well as remove more supeflous characters in the description string

    Attributes:
        Dataframe (df): The raw dataframe of the listings file 
    
    """

    dataframe["Description"] = dataframe["Description"].apply(clean_strings)
    dataframe["Description"].replace([r"\\n", "\n", r"\'"], [" "," ",""], regex=True, inplace=True)
    return dataframe

def set_default_feature_values(dataframe):

    """

    Selects the columns that will be used as our features, removes the nan values and replaces them with 1

    Attributes:
        Dataframe (df): The raw dataframe of the listings file 
    
    """

    columns = ["beds", "bathrooms", "bedrooms", "guests"]
    dataframe[columns] = dataframe[columns].fillna(1)
    return dataframe

def clean_tabular_data(dataframe):

    """

    Compiles the three previous functions into one by removing rows with missing values, cleaning the description strings andsetting the default feature values.

    Attributes:
        Dataframe (df): The raw dataframe of the listings file 
    
    """

    #dataframe = dataframe.iloc[: ,:-1]
    dataframe = set_default_feature_values(dataframe)
    dataframe = remove_rows_with_missing_ratings(dataframe)
    dataframe = clean_description_strings(dataframe)
    dataframe = dataframe.loc[:, ~dataframe.columns.str.contains('^Unnamed')]
    return dataframe

def load_airbnb_regression(dataframe,label=None):

    """

    Assigns the features and labels into a tuple, where the labels is the set of values you are wanting to predict (in this case it will be the Price_Night),
    ad the festures is the set of all other numerical data  

    Attributes:
        Dataframe (df): The raw dataframe of the listings file 
        Label (str): String value of the column we ar predicting 
    
    """

    df_features = clean_tabular_data(dataframe)
    #print(df_features)
    df_features = dataframe.select_dtypes(['float64', 'int64'])
    features = df_features.to_numpy()
    #print(df_features)
    labels = df_features[label]
    labels = labels.to_numpy()
    df_features.drop([label],axis=1, inplace=True)
    return (features,labels)

def load_data_classification(filepath):
    # Load data
    dataframe = pd.read_csv(filepath)

    # Preprocess data
    columns_to_drop = ['Unnamed: 0', 'ID', 'url', 'Description', 'guests', 'beds', 'Check-in_rating', 'bathrooms']
    dataframe.drop(columns=columns_to_drop, inplace=True)

    label_encoder = LabelEncoder()
    columns_to_encode = ['Category', 'Title', 'Location']

    for column in columns_to_encode:
        dataframe[column] = label_encoder.fit_transform(dataframe[column])

    amenities_encoder = LabelEncoder()
    dataframe['Amenities'] = dataframe['Amenities'].apply(lambda x: '|'.join(map(str, x)))  # Convert list to string
    dataframe['Amenities'] = amenities_encoder.fit_transform(dataframe['Amenities'])


    columns_to_filter = ["bedrooms"]
    for column in columns_to_filter:
        dataframe = dataframe[pd.to_numeric(dataframe[column], errors='coerce').notnull()]
        dataframe[column] = dataframe[column].astype(int)


    dataframe[['Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Value_rating']] = dataframe[['Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Value_rating']].dropna(axis=1)
    return dataframe

if __name__ == "__main__":
    print("Loading data...")
    Airbnb_data = pd.read_csv('/Users/ryanhughes/Desktop/Aicore/Airbnb/Airbnb/AirbnbData/Raw_Data/tabular_data/listing.csv')
    print("Cleaning...")
    cleaned_airbnb_data = clean_tabular_data(Airbnb_data)
    #try:
    cleaned_airbnb_data.to_csv('/Users/ryanhughes/Desktop/Aicore/Airbnb/Airbnb/AirbnbData/Processed_Data/clean_tabular_data/clean_tabular_data.csv')
    print(cleaned_airbnb_data.loc[845])
    #except FileExistsError:
    
        #pass

    (features, labels) = load_airbnb(Airbnb_data, "Price_Night")
    print("done")