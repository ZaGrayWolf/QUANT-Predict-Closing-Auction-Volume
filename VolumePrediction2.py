import sys
import pandas as pd
import requests
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Function to download and load the model, scaler, and imputer
def load_model():
    model_url = "https://raw.githubusercontent.com/ZaGrayWolf/QUANT-Predict-Closing-Auction-Volume/main/trained_nn_model.pkl"
    scaler_url = "https://raw.githubusercontent.com/ZaGrayWolf/QUANT-Predict-Closing-Auction-Volume/main/scaler.pkl"
    imputer_url = "https://raw.githubusercontent.com/ZaGrayWolf/QUANT-Predict-Closing-Auction-Volume/main/imputer.pkl"

    model = pickle.loads(requests.get(model_url).content)
    scaler = pickle.loads(requests.get(scaler_url).content)
    imputer = pickle.loads(requests.get(imputer_url).content)

    return model, scaler, imputer

# Function to preprocess input data
def preprocess_input_data(input_data, scaler, imputer):
    # Drop unnecessary columns (like 'symbol_id' and 'date_id')
    input_data_cleaned = input_data.drop(columns=['close_volume', 'symbol_id', 'date_id'], errors='ignore')

    # Impute missing values
    input_data_imputed = imputer.transform(input_data_cleaned)

    # Scale the data
    input_data_scaled = scaler.transform(input_data_imputed)
    
    return input_data_scaled

# Function to predict close volume and format output
def predict_close_volume(input_file, model, scaler, imputer):
    input_data = pd.read_csv(input_file)

    # Extract symbol_id for output
    symbol_ids = input_data['symbol_id']

    # Preprocess the input data
    input_data_scaled = preprocess_input_data(input_data, scaler, imputer)

    # Generate predictions
    predictions = model.predict(input_data_scaled)

    # Create output DataFrame
    output_df = pd.DataFrame({
        'symbol_id': symbol_ids,
        'close_volume': predictions
    })

    return output_df

def main():
    # Load the models, scaler, and imputer
    mlp_model, scaler, imputer = load_model()

    # Read input file path from stdin
    input_file = sys.stdin.read().strip()

    # Output predictions
    predictions_df = predict_close_volume(input_file, mlp_model, scaler, imputer)
    
    # Print the DataFrame to standard output as CSV format
    print(predictions_df.to_csv(index=False))

if __name__ == "__main__":
    main()
