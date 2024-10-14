from tabular_data import *
from regression import *
from classification import *
from Neural import *
from Neural_classification import *

if __name__ == "__main__":
    raw_data_path = 'AirbnbData/Raw_Data/tabular_data/listing.csv'
    clean_data_path = 'AirbnbData/Processed_Data/clean_tabular_data.csv'
    
    #Load and clean data
    print("Loading data...")
    Airbnb_data = pd.read_csv(raw_data_path)
    print("Cleaning...")
    Airbnb_data = Airbnb_data[Airbnb_data['guests']!= 'Somerford Keynes England United Kingdom']
    cleaned_airbnb_data = clean_tabular_data(Airbnb_data)
    cleaned_airbnb_data.to_csv(clean_data_path)
    
    #Regression
    
    print("REGRESSION")
    mgs = MGS()

    mgs.evaluate_all_models(clean_data_path, 'Price_Night', plot=True)

    best_model, best_hyperparameters, best_performance_metrics = mgs.find_best_model("models/regression/")
    
    #Classification
    print("CLASSIFICATION")
    airbnb_lr = AirbnbLogisticRegression()
    airbnb_lr.evaluate_all_models(path=clean_data_path)
    best_model, best_hyperparameters, best_performance_metrics = airbnb_lr.find_best_model("models/classification/")
    
    #Neural
    print("NEURAL NETWORK")
    data = pd.read_csv('AirbnbData/Processed_Data/clean_tabular_data.csv')
    dataset = AirbnbNightlyPriceRegressionDataset(data)
    dim = dataset.input_dim
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    writer = SummaryWriter()

    best_model, best_metrics, best_config = find_best_nn(dim, train_loader, val_loader, test_loader, epochs=200, writer=writer)

    writer.close()

    print("Best Model Config:", best_config)
    print("Best Model Metrics:", best_metrics)
    
    #Neural classification
    print("NEURAL CLASSIFICATION")
    #data = pd.read_csv('AirbnbData/Processed_Data/clean_tabular_data/clean_tabular_data.csv')
    unique_bedrooms = sorted(cleaned_airbnb_data['bedrooms'].unique())
    bedroom_mapping = {bedroom: idx for idx, bedroom in enumerate(unique_bedrooms)}

    cleaned_airbnb_data['bedrooms_mapped'] = cleaned_airbnb_data['bedrooms'].map(bedroom_mapping)
    class_dataset = AirbnbNightlyPriceClassificationDataset(cleaned_airbnb_data)
    dim = class_dataset.input_dim
    train_size = int(0.7 * len(class_dataset))
    val_size = int(0.15 * len(class_dataset))
    test_size = len(class_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(class_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    writer = SummaryWriter()

    best_model, best_metrics, best_config = find_best_classification_nn(dim, train_loader, val_loader, test_loader, epochs=200, writer=writer)

    writer.close()

    print("Best Model Config:", best_config)
    print("Best Model Metrics:", best_metrics)

    print("FINISHED")

    #https://archive.ics.uci.edu/