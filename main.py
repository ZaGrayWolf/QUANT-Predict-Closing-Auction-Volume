#!/bin/python3

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import sys

# Define training data
symbol_id = 1006986679
num_entries = 312  # Total number of entries

# Create columns
date_id = list(range(1, 13)) * (num_entries // 12)  # Adjusting to create 312 entries
seconds_bucket = [
    900, 1800, 3600, 5400, 7200, 9000, 10800, 12600, 14400, 16200,
    18000, 19800, 21600, 23400, 25200, 26400, 27000, 28500, 28800, 
    29400, 30000, 30300, 30600, 31200
] * (num_entries // 24)  # 24 buckets repeated

# Ensure each list has exactly 312 elements
cummulative_continous_volume = [
    17070.0, 30690.0, 79319.0, 118703.0, 230172.0, 283645.0, 341273.0, 
    358738.0, 387375.0, 407451.0, 448944.0, 485475.0, 537571.0, 
    599523.0, 685921.0, 711254.0, 730521.0, 783364.0, 788812.0, 
    798585.0, 811957.0, 819540.0, 841447.0, 139512.0
] * (num_entries // 24)  # Ensure to match buckets
cummulative_continous_volume += [0] * (num_entries - len(cummulative_continous_volume))  # Padding

close_volume = [
    162469.0, 273297.0, 363307.0, 454241.0, 574837.0, 686020.0, 749762.0, 
    906758.0, 966180.0, 1045058.0, 1098044.0, 1138083.0, 1197414.0, 
    1257577.0, 1385425.0, 1449012.0, 1482933.0, 1564531.0, 1582837.0, 
    1606944.0, 1643623.0, 1665780.0, 1683852.0, 273090.0
] * (num_entries // 24)  # Ensure to match buckets
close_volume += [0] * (num_entries - len(close_volume))  # Padding

# Load training data
train_data = {
    'symbol_id': [symbol_id] * num_entries,
    'date_id': date_id,
    'seconds_bucket': seconds_bucket,
    'cummulative_continous_volume': cummulative_continous_volume,
    'close_volume': close_volume
}

# Prepare training DataFrame
df = pd.DataFrame(train_data)

# Prepare features and target variable
X = df[['seconds_bucket', 'cummulative_continous_volume']]
y = df['close_volume']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model with increased iterations
nn_model = MLPRegressor(hidden_layer_sizes=(64, 16), max_iter=150000, random_state=42, activation='relu', solver='adam')
nn_model.fit(X_scaled, y)

# Predict function
def predict_close_volume(input_data):
    # Read CSV input
    test_df = pd.read_csv(input_data.strip())

    # Clean column names
    test_df.columns = test_df.columns.str.strip()

    # Check required columns
    required_columns = ['seconds_bucket', 'cummulative_continous_volume', 'symbol_id']
    missing_columns = [col for col in required_columns if col not in test_df.columns]
    if missing_columns:
        print(f"Error: Missing columns {missing_columns} in the input data.")
        return

    # Filter out rows with NaN values in the required columns
    test_df = test_df.dropna(subset=['seconds_bucket', 'cummulative_continous_volume'])

    # Prepare features for prediction
    X_test = test_df[required_columns[:-1]]  # Exclude 'symbol_id'
    
    if X_test.empty:
        print("Error: No valid data to predict.")
        return
    
    # Check for NaN values again
    if X_test.isnull().any().any():
        print("Error: Input contains NaN values.")
        return

    # Scale features
    X_test_scaled = scaler.transform(X_test)

    # Make predictions
    predictions = nn_model.predict(X_test_scaled)

    # Create output DataFrame with the required schema
    output = pd.DataFrame({
        'symbol_id': test_df['symbol_id'].values,
        'close_volume': predictions
    })

    # Group by symbol_id to get the last predicted close volume for each
    output = output.groupby('symbol_id', as_index=False).last()

    # Ensure output matches expected format
    output = output.round(1)  # Round to one decimal place as expected

    # Return output as CSV format
    return output.to_csv(index=False, header=True).strip()

# Read input file path from stdin
input_file = sys.stdin.read().strip()

# Output predictions
print(predict_close_volume(input_file))
