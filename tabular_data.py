import pandas as pd
import ast

def remove_rows_with_missing_ratings(Dataframe):
    rating_columns=['Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating','Description']
    Dataframe.dropna(subset = rating_columns, inplace = True)
    return Dataframe

def clean_strings(str):
    try:
        clean_str = ast.literal_eval(str)
        clean_str.remove("About this space")
        clean_str = "".join(clean_str)
        return clean_str
    except Exception as e:
        return str

def clean_description_strings(dataframe):
    dataframe["Description"] = dataframe["Description"].apply(clean_strings)
    dataframe["Description"].replace([r"\\n", "\n", r"\'"], [" "," ",""], regex=True, inplace=True)
    return dataframe

def set_default_feature_values(dataframe):
    columns = ["beds", "bathrooms", "bedrooms", "guests"]
    dataframe[columns] = dataframe[columns].fillna(1)
    return dataframe

def clean_tabular_data(dataframe):
    dataframe = remove_rows_with_missing_ratings(dataframe)
    dataframe = clean_description_strings(dataframe)
    dataframe = set_default_feature_values(dataframe)
    return dataframe

if __name__ == "__main__":
    Airbnb_data = pd.read_csv('/Users/ryanhughes/Desktop/Aicore/Airbnb/Airbnb/AirbnbData/Raw_Data/tabular_data/listing.csv')
    cleaned_airbnb_data = clean_tabular_data(Airbnb_data)
    try:
        cleaned_airbnb_data.to_csv('/Users/ryanhughes/Desktop/Aicore/Airbnb/Airbnb/AirbnbData/Processed_Data/clean_tabular_data/clean_tabular_data.csv')
    except FileExistsError:
        pass